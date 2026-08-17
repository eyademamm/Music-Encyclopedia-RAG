import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from search import build_text_index, build_vector_index, hybrid_search, load_docs  # noqa: E402

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "logs.db"
MODEL = "gpt-5.4-mini"

app = FastAPI(title="Music Encyclopedia RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "sk-placeholder-set-env-var")

# --- build indexes once at startup ---
docs = load_docs()
text_index = build_text_index(docs)
vector_index, embedder = build_vector_index(docs)


def init_db():
    conn = sqlite3.connect(DB_PATH)
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
            cost REAL DEFAULT 0.0
        )
    """)
    # Migrate existing databases that lack the new columns
    for col, col_type, default in [
        ("prompt_tokens", "INTEGER", "0"),
        ("completion_tokens", "INTEGER", "0"),
        ("cost", "REAL", "0.0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE logs ADD COLUMN {col} {col_type} DEFAULT {default}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    conn.close()


init_db()


class AskRequest(BaseModel):
    question: str


class FeedbackRequest(BaseModel):
    interaction_id: str
    feedback: int  # 1 = thumbs up, -1 = thumbs down


PROMPT_TEMPLATE = """\
You are a knowledgeable music encyclopedia assistant. Answer the QUESTION
using only the CONTEXT below. If the context doesn't contain the answer,
say you don't have enough information.

CONTEXT:
{context}

QUESTION: {question}
"""


def build_context(chunks):
    return "\n\n".join(f"[{c['topic']}] {c['chunk']}" for c in chunks)


def rag_answer(question: str):
    chunks = hybrid_search(text_index, vector_index, embedder, question, num_results=5)
    prompt = PROMPT_TEMPLATE.format(context=build_context(chunks), question=question)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = resp.choices[0].message.content

    # Extract real token usage from OpenAI response
    usage = resp.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    # Cost formula: input $0.75/1M tokens, output $4.50/1M tokens
    cost = (prompt_tokens * 0.75 + completion_tokens * 4.50) / 1_000_000

    return answer, chunks, prompt_tokens, completion_tokens, cost


@app.post("/ask")
def ask(req: AskRequest):
    start = time.time()
    answer, chunks, prompt_tokens, completion_tokens, cost = rag_answer(req.question)
    elapsed = time.time() - start

    interaction_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (interaction_id, req.question, answer, "hybrid", len(chunks), elapsed, None, time.time(),
         prompt_tokens, completion_tokens, cost),
    )
    conn.commit()
    conn.close()

    seen_sources = []
    for c in chunks:
        if c["topic"] not in seen_sources:
            seen_sources.append(c["topic"])

    return {
        "interaction_id": interaction_id,
        "answer": answer,
        "sources": seen_sources,
        "response_time": elapsed,
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE logs SET feedback = ? WHERE id = ?", (req.feedback, req.interaction_id))
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.get("/stats")
def stats():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM logs ORDER BY created_at DESC").fetchall()]

    # Aggregate real per-row cost by day
    from collections import defaultdict
    from datetime import datetime

    daily_cost = defaultdict(float)
    for r in rows:
        dt = datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d")
        daily_cost[dt] += r.get("cost", 0.0) or 0.0

    cost_data = [{"date": dt, "cost": round(total, 6)} for dt, total in sorted(daily_cost.items())]

    conn.close()
    return {"logs": rows, "costs": cost_data}


app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent.parent / "static"), name="static")


@app.get("/")
def root():
    return FileResponse(Path(__file__).resolve().parent.parent / "static" / "index.html")
