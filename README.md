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
ingest.py  -->  data/docs.json  -->  search.py (text / vector / hybrid RRF)
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

- **Retrieval evaluation** (`src/evaluate.py`): text search, vector search (TF-IDF),
  and hybrid search (Reciprocal Rank Fusion) are all evaluated with hit rate and
  MRR against LLM-generated ground-truth questions (`src/generate_ground_truth.py`).
  Results saved to `data/retrieval_eval.json`. **Hybrid search is used in
  production** based on these results.
- **LLM evaluation**: two prompt variants are compared (see `src/evaluate_llm.py`)
  for answer relevance/faithfulness — the better prompt is used in `app/main.py`.

## Setup

1. Clone and enter the repo.
2. Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Build the knowledge base (needs internet access to Wikipedia):
   ```bash
   python src/ingest.py
   ```
5. Generate ground truth and run retrieval evaluation:
   ```bash
   python src/generate_ground_truth.py
   python src/evaluate.py
   ```
6. Run the app:
   ```bash
   uvicorn app.main:app --reload
   ```
   Or with Docker:
   ```bash
   docker compose up --build
   ```
7. Open http://localhost:8000 to ask questions, and
   http://localhost:8000/static/dashboard.html for the monitoring dashboard.

## Interface

FastAPI backend with a plain HTML/JS frontend (`static/index.html`) — no
framework lock-in, runs anywhere `uvicorn` runs.

## Monitoring

Every query is logged to `data/logs.db` (SQLite) with response time, retrieval
method, and user feedback (👍/👎). `static/dashboard.html` renders 5 charts:
question volume, feedback distribution, response time per query, retrieval
method breakdown, and feedback over time.

## Containerization

`Dockerfile` + `docker-compose.yml` run the full app in one command:
`docker compose up --build`.

## Best practices implemented

- [x] Hybrid search (text + vector, combined via Reciprocal Rank Fusion)

## Tech stack

Python, FastAPI, `minsearch` (text + TF-IDF vector search), OpenAI
(`gpt-5.4-mini`), SQLite, Docker.
