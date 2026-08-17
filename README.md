# 🎵 Music Encyclopedia RAG

An end-to-end RAG application that answers questions about music artists, bands,
and genres, built for the LLM Zoomcamp final project.

## Problem description

Music fans and casual listeners often want quick, reliable answers about artists,
bands, and genres (formation dates, members, style, influences) without digging
through long Wikipedia articles or unreliable web search. This project builds a
RAG system over a curated set of Wikipedia music articles, letting users ask
natural-language questions and get grounded, cited answers.

## Dataset

Text is pulled from Wikipedia (CC BY-SA, freely licensed) for ~20 music
genres/artists/bands, chunked into ~180-word passages. See `src/ingest.py`.
Lyrics themselves are intentionally excluded (copyright) — only encyclopedic
prose (history, biography, style) is indexed.

## Architecture

```
ingest.py  -->  data/docs.json  -->  search.py (text / ONNX vector / hybrid RRF)
                                          |
                                     app/main.py (FastAPI)
                                     - /ask       RAG answer + logs interaction
                                     - /feedback  thumbs up/down
                                     - /stats     raw logs for dashboard
                                          |
                                static/index.html (chat UI)
                                static/dashboard.html (monitoring, 5 charts)
```

## Evaluation

- **Retrieval evaluation** (`src/evaluate.py`): text search, vector search (BGE small ONNX embeddings),
  and hybrid search (Reciprocal Rank Fusion) are all evaluated with hit rate and
  MRR against LLM-generated ground-truth questions (`src/generate_ground_truth.py` using structured output).
  Results saved to `data/retrieval_eval.json`.
  - **Text Search**: Hit rate: 64.9% | MRR: 0.547
  - **Vector Search (ONNX)**: Hit rate: 90.5% | MRR: 0.785
  - **Hybrid Search**: Hit rate: 91.0% | MRR: 0.711
  Vector search drastically outperformed text search and is considered the best standalone method. Hybrid search improved the hit rate slightly but heavily degraded MRR because the Reciprocal Rank Fusion (RRF) logic mixes in lower-quality text results without thresholding.
- **LLM evaluation**: two prompt variants are compared (see `src/evaluate_llm.py`)
  for answer relevance/faithfulness. With stratified sampling on the evaluation set, both prompts achieved **96.6% relevance**. The better prompt is used in `app/main.py`.

## Setup

1. Clone and enter the repo.
2. Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`.
3. Install dependencies:
   ```bash
   uv sync
   ```
4. Build the knowledge base (needs internet access to Wikipedia):
   ```bash
   uv run python src/ingest.py
   ```
5. Download the ONNX embedding model:
   ```bash
   uv run python src/download_model.py
   ```
6. Generate ground truth and run retrieval evaluation:
   ```bash
   uv run python src/generate_ground_truth.py
   uv run python src/evaluate.py
   uv run python src/evaluate_llm.py
   ```
7. Run the app:
   ```bash
   uv run uvicorn app.main:app --reload
   ```
   Or with Docker:
   ```bash
   docker compose up --build
   ```
   > [!WARNING]
   > **Server Startup Latency:** The server generates ONNX embeddings for the entire document corpus *in memory on startup* rather than persisting them to disk. This is CPU-intensive and can take 5-10 minutes. The server will not be reachable on port 8000 until the `Application startup complete.` message appears in the logs.

8. Open http://localhost:8000 to ask questions, and
   http://localhost:8000/static/dashboard.html for the monitoring dashboard.

## Interface

FastAPI backend with a bespoke "The Crate" HTML/JS frontend (`static/index.html`).
It features a premium vinyl record player animation to simulate crate-digging, runs anywhere `uvicorn` runs without framework lock-in.

## Monitoring

Every query is logged to `data/logs.db` (SQLite) with response time, real token usage (prompt and completion tokens), calculated cost (based on `gpt-5.4-mini` pricing), retrieval
method, and user feedback (👍/👎). `static/dashboard.html` renders 6 charts:
question volume, estimated daily API cost, feedback distribution, response time per query, retrieval
method breakdown, and feedback over time. Source tracking is deduplicated to ensure clean source citations in the UI.

## Containerization

`Dockerfile` + `docker-compose.yml` run the full app in one command:
`docker compose up --build`.

## Best practices implemented

- [x] Hybrid search (text + vector, combined via Reciprocal Rank Fusion)

## Tech stack

Python, FastAPI, `minsearch` (text), `onnxruntime` (`Xenova/bge-small-en-v1.5` embeddings), OpenAI
(`gpt-5.4-mini`), SQLite, Docker.
