import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import (APIConnectionError, APIStatusError, APITimeoutError,
                    AuthenticationError, OpenAI, RateLimitError)
from pydantic import BaseModel, Field, field_validator

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from search import build_text_index, build_vector_index, load_docs  # noqa: E402
from tools import (TOOL_DEFINITIONS, ExternalToolServiceError, LyricsData,
                   execute_tool)                                     # noqa: E402

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "logs.db"
DB_PATH = Path(os.environ.get("MUSIC_RAG_DB_PATH", DEFAULT_DB_PATH))
MODEL   = "gpt-4o-mini"
MAX_QUESTION_LENGTH = 2_000
MAX_INTERACTION_ID_LENGTH = 64
MAX_CONVERSATION_TITLE_LENGTH = 80
MAX_CONTEXT_TURNS = 4
MAX_CONTEXT_CHARS = 8_000

logger = logging.getLogger(__name__)

app = FastAPI(title="Music Encyclopedia RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if client is None:
    logger.error("OPENAI_API_KEY is not configured; /ask will return 503 until it is set.")

# --- build indexes once at startup ---
docs         = load_docs()
text_index   = build_text_index(docs)
vector_index, embedder = build_vector_index(docs)

# Shared context passed to tool dispatch
_tool_context = {
    "text_index":   text_index,
    "vector_index": vector_index,
    "embedder":     embedder,
}


# ── DB ────────────────────────────────────────────────────────────────────────

def conversation_title(question: str | None) -> str:
    """Produce a compact, display-safe title without storing another text copy."""
    normalized = " ".join((question or "").split())
    if not normalized:
        return "Imported chat"
    if len(normalized) <= MAX_CONVERSATION_TITLE_LENGTH:
        return normalized
    return normalized[: MAX_CONVERSATION_TITLE_LENGTH - 1].rstrip() + "…"

@contextmanager
def db_connection():
    """Open a short-lived SQLite connection and commit successful writes."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except sqlite3.Error:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            conn.close()


def init_db():
    try:
        with db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id TEXT PRIMARY KEY,
                    question TEXT,
                    answer TEXT,
                    method TEXT,
                    num_chunks INTEGER,
                    response_time REAL,
                    feedback INTEGER,
                    created_at REAL,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    cost REAL DEFAULT 0.0,
                    tools_used TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            existing_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(logs)")
            }
            for col, col_type, default in [
                ("prompt_tokens",     "INTEGER", "0"),
                ("completion_tokens", "INTEGER", "0"),
                ("cost",              "REAL",    "0.0"),
                ("tools_used",        "TEXT",    "''"),
            ]:
                if col not in existing_columns:
                    conn.execute(f"ALTER TABLE logs ADD COLUMN {col} {col_type} DEFAULT {default}")
            if "conversation_id" not in existing_columns:
                conn.execute(
                    "ALTER TABLE logs ADD COLUMN conversation_id TEXT "
                    "REFERENCES conversations(id) ON DELETE CASCADE"
                )
            if "sources" not in existing_columns:
                conn.execute("ALTER TABLE logs ADD COLUMN sources TEXT NOT NULL DEFAULT '[]'")

            legacy_rows = conn.execute(
                "SELECT id, question, created_at FROM logs WHERE conversation_id IS NULL"
            ).fetchall()
            for interaction_id, question, created_at in legacy_rows:
                timestamp = created_at if created_at is not None else time.time()
                conversation_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (conversation_id, conversation_title(question), timestamp, timestamp),
                )
                conn.execute(
                    "UPDATE logs SET conversation_id = ? WHERE id = ?",
                    (conversation_id, interaction_id),
                )

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_updated "
                "ON conversations(updated_at DESC, id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_conversation_created "
                "ON logs(conversation_id, created_at, id)"
            )
    except sqlite3.Error as exc:
        logger.critical("Unable to initialize the SQLite interaction log.", exc_info=True)
        raise RuntimeError("Unable to initialize the interaction log database.") from exc


init_db()


# ── Request/Response models ───────────────────────────────────────────────────

class AskRequest(BaseModel):
    conversation_id: uuid.UUID
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class FeedbackRequest(BaseModel):
    conversation_id: uuid.UUID
    interaction_id: str = Field(min_length=1, max_length=MAX_INTERACTION_ID_LENGTH)
    feedback: Literal[-1, 1]  # 1 = thumbs up, -1 = thumbs down

    @field_validator("interaction_id")
    @classmethod
    def normalize_interaction_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("interaction_id must not be blank")
        return value


class CreateConversationRequest(BaseModel):
    title: str | None = Field(default=None, max_length=MAX_CONVERSATION_TITLE_LENGTH)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        if not value:
            return None
        return value


def conversation_exists(conn: sqlite3.Connection, conversation_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
    ).fetchone() is not None


def decode_string_list(value: str | None) -> list[str]:
    """Read compact JSON metadata while tolerating legacy malformed values."""
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, str)]


def lyrics_payload(lyrics: LyricsData | None) -> dict[str, object] | None:
    """Serialize live LRCLIB data without persisting it in conversation logs."""
    if lyrics is None:
        return None
    return {
        "provider": lyrics.provider,
        "provider_id": lyrics.provider_id,
        "track_title": lyrics.track_title,
        "artist": lyrics.artist,
        "album": lyrics.album,
        "duration_seconds": lyrics.duration_seconds,
        "instrumental": lyrics.instrumental,
        "plain_lyrics": lyrics.plain_lyrics,
        "synced_lyrics": lyrics.synced_lyrics,
    }


def truncate_context(text: str, maximum: int) -> str:
    if len(text) <= maximum:
        return text
    if maximum <= 1:
        return text[:maximum]
    return text[: maximum - 1] + "…"


def load_recent_context(conn: sqlite3.Connection, conversation_id: str) -> list[dict[str, str]]:
    """Return a hard-bounded, chronological window of visible prior turns only."""
    rows = conn.execute(
        """
        SELECT question, answer
        FROM logs
        WHERE conversation_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (conversation_id, MAX_CONTEXT_TURNS),
    ).fetchall()

    selected: list[tuple[str, str]] = []
    used_chars = 0
    for question, answer in rows:
        question = question or ""
        answer = answer or ""
        turn_chars = len(question) + len(answer)
        if used_chars + turn_chars <= MAX_CONTEXT_CHARS:
            selected.append((question, answer))
            used_chars += turn_chars
            continue
        if not selected:
            question = truncate_context(question, min(len(question), MAX_CONTEXT_CHARS // 4))
            answer = truncate_context(answer, MAX_CONTEXT_CHARS - len(question))
            selected.append((question, answer))
        break

    messages: list[dict[str, str]] = []
    for question, answer in reversed(selected):
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    return messages


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are "The Crate" — a knowledgeable, warm music encyclopedia assistant with a love for vinyl records and music history.
You have access to tools to look up information. Always ground your answers in tool results — never make up facts.

## Tool strategy
1. **local_search** first — use it for any music question. It covers well-known artists, genres, and bands.
2. **wikipedia_search** — use when local_search returns insufficient results, or for lesser-known artists, specific albums, record labels, music theory, etc.
3. **lyrics_search** — use `display` when the user directly asks to read full lyrics, `analysis` for themes, meaning, imagery, or a passage, and `display_and_analysis` only when the user explicitly asks for both.
4. You may call **multiple tools** in sequence if building a complete answer requires it.

## Rules
- Base your answer entirely on tool results. If no tool returns relevant information, say so honestly.
- Format answers with **markdown** (bold, bullet lists, headings) for readability.
- Be warm and conversational — you love music and it shows.
- Keep answers thorough but concise; avoid padding.
- Complete song lyrics may be displayed only by the application from structured LRCLIB data. Never reproduce a complete lyric body in your answer.
- After a successful display-mode lookup, the application creates the short introduction and displays the lyrics separately; do not try to summarize or reproduce them.
- Prior conversation messages may help resolve follow-up questions, but they are not factual evidence. Use current tool results for factual claims.
"""


# ── Agentic loop ──────────────────────────────────────────────────────────────

def create_completion(**kwargs):
    """Call OpenAI and convert expected provider failures into stable HTTP errors."""
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="The answer service is not configured. Set OPENAI_API_KEY and restart the server.",
        )

    try:
        return client.chat.completions.create(**kwargs)
    except RateLimitError:
        logger.warning("OpenAI rate limit reached.")
        raise HTTPException(
            status_code=429,
            detail="The answer service is busy. Please try again shortly.",
        ) from None
    except (APITimeoutError, APIConnectionError):
        logger.warning("Unable to reach the OpenAI answer service.", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="The answer service is temporarily unavailable. Please try again.",
        ) from None
    except AuthenticationError:
        logger.error("OpenAI authentication failed; check server configuration.", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="The answer service is not configured correctly.",
        ) from None
    except APIStatusError as exc:
        logger.warning("OpenAI returned status %s.", exc.status_code, exc_info=True)
        raise HTTPException(
            status_code=502,
            detail="The answer service returned an error. Please try again.",
        ) from None

def agentic_answer(question: str, conversation_context: list[dict[str, str]] | None = None):
    """
    Run the ReAct agentic loop:
    1. Send question to LLM with tool definitions
    2. If the LLM calls a tool, execute it and loop back
    3. Repeat until the LLM produces a final text answer (max 5 iterations)

    Returns:
        (answer, sources, num_chunks, prompt_tokens, completion_tokens, cost,
        tools_used, lyrics)
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(conversation_context or []),
        {"role": "user",   "content": question},
    ]

    total_prompt_tokens     = 0
    total_completion_tokens = 0
    sources: list[str]      = []
    tools_called: list[str] = []
    num_chunks = 0
    lyrics_for_display: LyricsData | None = None

    MAX_ITERATIONS = 5

    for iteration in range(MAX_ITERATIONS):
        resp = create_completion(
            model=MODEL,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        usage = resp.usage
        if usage:
            total_prompt_tokens     += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens

        msg = resp.choices[0].message

        # No more tool calls — we have the final answer
        if not msg.tool_calls:
            answer = msg.content or ""
            break

        # Append the assistant's tool-call message
        messages.append(msg)

        # Execute each tool call and append results
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            try:
                tool_result = execute_tool(name, arguments, _tool_context)
                result = tool_result.content
                source = tool_result.source
                num_chunks += tool_result.chunk_count
            except ExternalToolServiceError as exc:
                logger.warning("Tool '%s' is unavailable: %s", name, exc)
                result = str(exc)
                source = None
                tool_result = None

            tools_called.append(name)
            if source and source not in sources:
                sources.append(source)

            request_kind = arguments.get("request_kind") if name == "lyrics_search" else None
            if (
                tool_result is not None
                and tool_result.lyrics is not None
                and request_kind in {"display", "display_and_analysis"}
            ):
                lyrics_for_display = tool_result.lyrics

            if (
                tool_result is not None
                and tool_result.lyrics is not None
                and request_kind == "display"
            ):
                lyrics = tool_result.lyrics
                answer = f'I found “{lyrics.track_title}” by {lyrics.artist}.'
                cost = (total_prompt_tokens * 0.15 + total_completion_tokens * 0.60) / 1_000_000
                return (
                    answer,
                    sources,
                    num_chunks,
                    total_prompt_tokens,
                    total_completion_tokens,
                    cost,
                    tools_called,
                    lyrics_for_display,
                )

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      result,
            })
    else:
        # Safety: max iterations reached — ask for a final answer without tools
        messages.append({"role": "user", "content": "Please provide your final answer now."})
        resp = create_completion(model=MODEL, messages=messages)
        usage = resp.usage
        if usage:
            total_prompt_tokens     += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens
        answer = resp.choices[0].message.content or ""

    # Cost formula: gpt-4o-mini  input $0.15/1M tokens, output $0.60/1M tokens
    cost = (total_prompt_tokens * 0.15 + total_completion_tokens * 0.60) / 1_000_000

    return (
        answer,
        sources,
        num_chunks,
        total_prompt_tokens,
        total_completion_tokens,
        cost,
        tools_called,
        lyrics_for_display,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error while serving %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )


@app.post("/conversations", status_code=status.HTTP_201_CREATED)
def create_conversation(req: CreateConversationRequest):
    conversation_id = str(uuid.uuid4())
    timestamp = time.time()
    title = req.title or "New Chat"
    try:
        with db_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, title, timestamp, timestamp),
            )
    except sqlite3.Error:
        logger.exception("Unable to create conversation.")
        raise HTTPException(
            status_code=503,
            detail="Conversation history is temporarily unavailable. Please try again.",
        ) from None
    return {
        "id": conversation_id,
        "title": title,
        "created_at": timestamp,
        "updated_at": timestamp,
        "turn_count": 0,
    }


@app.get("/conversations")
def list_conversations(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    try:
        with db_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(row) for row in conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(l.id) AS turn_count
                FROM conversations AS c
                LEFT JOIN logs AS l ON l.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC, c.id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()]
    except sqlite3.Error:
        logger.exception("Unable to list conversations.")
        raise HTTPException(
            status_code=503,
            detail="Conversation history is temporarily unavailable. Please try again.",
        ) from None
    return {"conversations": rows}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: uuid.UUID):
    conversation_id_text = str(conversation_id)
    try:
        with db_connection() as conn:
            conn.row_factory = sqlite3.Row
            conversation = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id_text,),
            ).fetchone()
            if conversation is None:
                raise HTTPException(status_code=404, detail="Conversation not found.")
            turns = conn.execute(
                """
                SELECT id, question, answer, feedback, created_at, sources, tools_used
                FROM logs
                WHERE conversation_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (conversation_id_text,),
            ).fetchall()
    except sqlite3.Error:
        logger.exception("Unable to load conversation.")
        raise HTTPException(
            status_code=503,
            detail="Conversation history is temporarily unavailable. Please try again.",
        ) from None

    messages = []
    for turn in turns:
        messages.extend([
            {"role": "user", "content": turn["question"] or "", "created_at": turn["created_at"]},
            {
                "role": "assistant",
                "content": turn["answer"] or "",
                "created_at": turn["created_at"],
                "interaction_id": turn["id"],
                "sources": decode_string_list(turn["sources"]),
                "tools_used": decode_string_list(turn["tools_used"]),
                "feedback": turn["feedback"],
            },
        ])
    return {"conversation": dict(conversation), "messages": messages}


@app.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: uuid.UUID):
    try:
        with db_connection() as conn:
            cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (str(conversation_id),))
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Conversation not found.")
    except sqlite3.Error:
        logger.exception("Unable to delete conversation.")
        raise HTTPException(
            status_code=503,
            detail="Conversation history is temporarily unavailable. Please try again.",
        ) from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/ask")
def ask(req: AskRequest):
    conversation_id = str(req.conversation_id)
    try:
        with db_connection() as conn:
            if not conversation_exists(conn, conversation_id):
                raise HTTPException(status_code=404, detail="Conversation not found.")
            conversation_context = load_recent_context(conn, conversation_id)
    except sqlite3.Error:
        logger.exception("Unable to load conversation context.")
        raise HTTPException(
            status_code=503,
            detail="Conversation history is temporarily unavailable. Please try again.",
        ) from None

    start = time.time()
    (
        answer,
        sources,
        num_chunks,
        prompt_tokens,
        completion_tokens,
        cost,
        tools_called,
        lyrics,
    ) = agentic_answer(req.question, conversation_context)
    elapsed = time.time() - start

    interaction_id = str(uuid.uuid4())
    created_at = time.time()
    try:
        with db_connection() as conn:
            if not conversation_exists(conn, conversation_id):
                raise HTTPException(status_code=404, detail="Conversation not found.")
            conn.execute(
                """
                INSERT INTO logs (
                    id, question, answer, method, num_chunks, response_time,
                    feedback, created_at, prompt_tokens, completion_tokens,
                    cost, tools_used, conversation_id, sources
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    interaction_id,
                    req.question,
                    answer,
                    "agentic",
                    num_chunks,
                    elapsed,
                    None,
                    created_at,
                    prompt_tokens,
                    completion_tokens,
                    cost,
                    json.dumps(tools_called, separators=(",", ":")),
                    conversation_id,
                    json.dumps(sources, separators=(",", ":")),
                ),
            )
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? "
                "WHERE id = ? AND title = 'New Chat'",
                (conversation_title(req.question), created_at, conversation_id),
            )
    except sqlite3.Error:
        logger.exception("Unable to save interaction log entry.")
        raise HTTPException(
            status_code=503,
            detail="The interaction log is temporarily unavailable. Please try again.",
        ) from None

    return {
        "conversation_id": conversation_id,
        "interaction_id": interaction_id,
        "answer":         answer,
        "sources":        sources,
        "response_time":  elapsed,
        "tools_used":     tools_called,
        "lyrics":         lyrics_payload(lyrics),
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    try:
        with db_connection() as conn:
            cursor = conn.execute(
                "UPDATE logs SET feedback = ? WHERE id = ? AND conversation_id = ?",
                (req.feedback, req.interaction_id, str(req.conversation_id)),
            )
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Interaction not found.")
    except sqlite3.Error:
        logger.exception("Unable to save interaction feedback.")
        raise HTTPException(
            status_code=503,
            detail="The interaction log is temporarily unavailable. Please try again.",
        ) from None
    return {"status": "ok"}


@app.get("/stats")
def stats():
    try:
        with db_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(
                """
                SELECT id, method, num_chunks, response_time, feedback, created_at,
                       prompt_tokens, completion_tokens, cost, tools_used
                FROM logs
                ORDER BY created_at DESC
                """
            ).fetchall()]
    except sqlite3.Error:
        logger.exception("Unable to read interaction logs.")
        raise HTTPException(
            status_code=503,
            detail="The interaction log is temporarily unavailable. Please try again.",
        ) from None

    from collections import defaultdict
    from datetime import datetime

    daily_cost = defaultdict(float)
    for r in rows:
        dt = datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d")
        daily_cost[dt] += r.get("cost", 0.0) or 0.0

    cost_data = [{"date": dt, "cost": round(total, 6)} for dt, total in sorted(daily_cost.items())]

    return {"logs": rows, "costs": cost_data}


app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent.parent / "static"), name="static")


@app.get("/")
def root():
    return FileResponse(Path(__file__).resolve().parent.parent / "static" / "index.html")
