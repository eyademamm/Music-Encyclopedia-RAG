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

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import (APIConnectionError, APIStatusError, APITimeoutError,
                    AuthenticationError, OpenAI, RateLimitError)
from pydantic import BaseModel, Field, field_validator

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from search import build_text_index, build_vector_index, load_docs  # noqa: E402
from tools import (TOOL_DEFINITIONS, ExternalToolServiceError,
                   execute_tool)                                     # noqa: E402

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "logs.db"
DB_PATH = Path(os.environ.get("MUSIC_RAG_DB_PATH", DEFAULT_DB_PATH))
MODEL   = "gpt-4o-mini"
MAX_QUESTION_LENGTH = 2_000
MAX_INTERACTION_ID_LENGTH = 64

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

@contextmanager
def db_connection():
    """Open a short-lived SQLite connection and commit successful writes."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
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
    except sqlite3.Error as exc:
        logger.critical("Unable to initialize the SQLite interaction log.", exc_info=True)
        raise RuntimeError("Unable to initialize the interaction log database.") from exc


init_db()


# ── Request/Response models ───────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class FeedbackRequest(BaseModel):
    interaction_id: str = Field(min_length=1, max_length=MAX_INTERACTION_ID_LENGTH)
    feedback: Literal[-1, 1]  # 1 = thumbs up, -1 = thumbs down

    @field_validator("interaction_id")
    @classmethod
    def normalize_interaction_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("interaction_id must not be blank")
        return value


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are "The Crate" — a knowledgeable, warm music encyclopedia assistant with a love for vinyl records and music history.
You have access to tools to look up information. Always ground your answers in tool results — never make up facts.

## Tool strategy
1. **local_search** first — use it for any music question. It covers well-known artists, genres, and bands.
2. **wikipedia_search** — use when local_search returns insufficient results, or for lesser-known artists, specific albums, record labels, music theory, etc.
3. **lyrics_search** — use when the user asks for lyrics, wants to know what a song is about, or asks to quote or analyse a song.
4. You may call **multiple tools** in sequence if building a complete answer requires it.

## Rules
- Base your answer entirely on tool results. If no tool returns relevant information, say so honestly.
- Format answers with **markdown** (bold, bullet lists, headings) for readability.
- Be warm and conversational — you love music and it shows.
- Keep answers thorough but concise; avoid padding.
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

def agentic_answer(question: str):
    """
    Run the ReAct agentic loop:
    1. Send question to LLM with tool definitions
    2. If the LLM calls a tool, execute it and loop back
    3. Repeat until the LLM produces a final text answer (max 5 iterations)

    Returns:
        (answer, sources, num_chunks, prompt_tokens, completion_tokens, cost,
        tools_used)
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]

    total_prompt_tokens     = 0
    total_completion_tokens = 0
    sources: list[str]      = []
    tools_called: list[str] = []
    num_chunks = 0

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

            tools_called.append(name)
            if source and source not in sources:
                sources.append(source)

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
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error while serving %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )

@app.post("/ask")
def ask(req: AskRequest):
    start = time.time()
    (
        answer,
        sources,
        num_chunks,
        prompt_tokens,
        completion_tokens,
        cost,
        tools_called,
    ) = agentic_answer(req.question)
    elapsed = time.time() - start

    interaction_id = str(uuid.uuid4())
    try:
        with db_connection() as conn:
            conn.execute(
                """
                INSERT INTO logs (
                    id, question, answer, method, num_chunks, response_time,
                    feedback, created_at, prompt_tokens, completion_tokens,
                    cost, tools_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    interaction_id,
                    req.question,
                    answer,
                    "agentic",
                    num_chunks,
                    elapsed,
                    None,
                    time.time(),
                    prompt_tokens,
                    completion_tokens,
                    cost,
                    json.dumps(tools_called, separators=(",", ":")),
                ),
            )
    except sqlite3.Error:
        logger.exception("Unable to save interaction log entry.")
        raise HTTPException(
            status_code=503,
            detail="The interaction log is temporarily unavailable. Please try again.",
        ) from None

    return {
        "interaction_id": interaction_id,
        "answer":         answer,
        "sources":        sources,
        "response_time":  elapsed,
        "tools_used":     tools_called,
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    try:
        with db_connection() as conn:
            cursor = conn.execute(
                "UPDATE logs SET feedback = ? WHERE id = ?",
                (req.feedback, req.interaction_id),
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
            rows = [dict(r) for r in conn.execute("SELECT * FROM logs ORDER BY created_at DESC").fetchall()]
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
