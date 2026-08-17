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
            created_at REAL
        )
    """)
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
    return answer, chunks


@app.post("/ask")
def ask(req: AskRequest):
    start = time.time()
    answer, chunks = rag_answer(req.question)
    elapsed = time.time() - start

    interaction_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (interaction_id, req.question, answer, "hybrid", len(chunks), elapsed, None, time.time()),
    )
    conn.commit()
    conn.close()

    return {
        "interaction_id": interaction_id,
        "answer": answer,
        "sources": [c["topic"] for c in chunks],
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
    conn.close()
    return rows


app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent.parent / "static"), name="static")


@app.get("/")
def root():
    return FileResponse(Path(__file__).resolve().parent.parent / "static" / "index.html")
