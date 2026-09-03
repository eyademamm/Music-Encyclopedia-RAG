# 🎵 The Crate

An **agentic** music encyclopedia RAG application that answers questions about any artist, band, genre, or song — complete with **live lyrics lookup** and **real-time Wikipedia search**. Built for the LLM Zoomcamp final project.

## Problem Description

Music fans and casual listeners often want quick, reliable answers about artists,
bands, and genres (formation dates, members, style, influences) without digging
through long Wikipedia articles or unreliable web search. Traditional RAG systems
are limited to a fixed, pre-ingested corpus — if the topic isn't in the database,
the system fails silently or hallucinates.

**The Crate** solves this with an **agentic ReAct loop**: instead of a static
knowledge base, the LLM orchestrates multiple tools — a local vector index for
fast lookups, live Wikipedia search for any topic, and a lyrics API for song
lyrics — deciding at query time which tools to call and how to combine the results.

## What Makes This Project Stand Out

- **Agentic tool use** — the LLM decides which tools to call (local search, Wikipedia, lyrics) using OpenAI function calling, not a fixed retrieval pipeline
- **Live lyrics integration** — fetches real song lyrics via LRCLIB (free, no API key) — a unique data source
- **No fixed topic list** — any artist, genre, album, or music concept is answerable via live Wikipedia search
- **Graceful fallback chain** — local index → Wikipedia → honest "I don't know"
- **~1 second startup** — precomputed embeddings eliminate the 10-minute ONNX computation on every restart
- **Cozy Vinyl Den UI** — a custom Burgundy & Dusty Rose design with a tactile "Drop the Needle" turntable button, markdown-rendered answers with typewriter effect, and error toast notifications

## Dataset

Text is pulled from Wikipedia (CC BY-SA, freely licensed) for ~20 music
genres/artists/bands, chunked into ~180-word passages. See `src/ingest.py`.
This forms the **local knowledge base** — the agent's fast path for well-known topics.
For topics not in the local index, the agent searches Wikipedia in real time.

Song lyrics are fetched live from [LRCLIB](https://lrclib.net/) (community-driven, free, no API key required).

## Architecture

```
                         ┌─────────────────────────────────────────┐
                         │           User Question                 │
                         └──────────────────┬──────────────────────┘
                                            ▼
                         ┌──────────────────────────────────────────┐
                         │         LLM Agent (ReAct Loop)           │
                         │                                          │
                         │  1. Think: What does the user need?      │
                         │  2. Act:   Pick a tool (or answer)       │
                         │  3. Observe: Read tool result             │
                         │  4. Repeat or produce final answer       │
                         │                                          │
                         │  Tools:                                  │
                         │  ┌─────────────┐  ┌──────────────────┐   │
                         │  │local_search │  │wikipedia_search  │   │
                         │  │(vector+text)│  │(live MediaWiki)  │   │
                         │  └─────────────┘  └──────────────────┘   │
                         │  ┌─────────────┐                         │
                         │  │lyrics_search│                         │
                         │  │  (LRCLIB)   │                         │
                         │  └─────────────┘                         │
                         └──────────────────┬───────────────────────┘
                                            ▼
                         ┌──────────────────────────────────────────┐
                         │        Grounded Answer + Sources          │
                         └──────────────────────────────────────────┘

Backend:    app/main.py (FastAPI)
              - /ask       Agentic answer + logs interaction
              - /feedback  Thumbs up/down
              - /stats     Raw logs for dashboard

Frontend:   static/index.html     (chat UI — Vinyl Den theme)
            static/dashboard.html (monitoring, 6 charts)
```

### Agent Tools

| Tool | Source | Description |
|---|---|---|
| `local_search` | `data/docs.json` + ONNX embeddings | Hybrid text + vector search (RRF) over the local knowledge base. Fast, no network needed. |
| `wikipedia_search` | MediaWiki API (live) | Fetches any Wikipedia article in real time. Handles artists, albums, genres, labels, etc. not in the local index. |
| `lyrics_search` | [LRCLIB API](https://lrclib.net/) (live) | Fetches plain-text song lyrics. Free, no API key, community-maintained. |

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

The repository includes the reviewed local corpus (`data/docs.json`) and its
matching precomputed embeddings (`data/embeddings.npy`). A fresh checkout does
not need to ingest Wikipedia or compute embeddings before it can run. The only
runtime asset downloaded during setup is the pinned ONNX model (~127 MiB).

### Local development

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), internet access
for the one-time model download, and a valid OpenAI API key for `/ask`.

```bash
git clone https://github.com/eyademamm/Music-Encyclopedia-RAG.git music-rag
cd music-rag
cp .env.example .env
# Set OPENAI_API_KEY in .env
uv sync --frozen
uv run python src/download_model.py
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 to ask questions and
http://localhost:8000/static/dashboard.html for the monitoring dashboard.

### Docker

Prerequisites: Docker Compose, internet access for the image base,
dependencies, and pinned model download, plus a valid OpenAI API key.

```bash
git clone https://github.com/eyademamm/Music-Encyclopedia-RAG.git music-rag
cd music-rag
cp .env.example .env
# Set OPENAI_API_KEY in .env
docker compose up --build
```

The build downloads the exact model revision used by the committed embeddings.
The corpus and embeddings are baked into the image; query logs are persisted in
the `app_runtime` named volume, so they do not hide those retrieval assets.

### Refreshing retrieval assets

Wikipedia content is live and may change. Refresh the corpus only when you
intend to review and version a new retrieval snapshot:

```bash
uv run python src/ingest.py
uv run python src/build_embeddings.py
uv run python src/evaluate.py
```

Commit `data/docs.json` and `data/embeddings.npy` together after review. The
existing evaluation data is optional; generating it requires an OpenAI API key:

```bash
uv run python src/generate_ground_truth.py
uv run python src/evaluate.py
uv run python src/evaluate_llm.py
```

The embedding model remains downloaded rather than committed. Do not commit
`models/`, `data/logs.db`, or generated evaluation outputs.

## Interface

A bespoke **"The Crate" Vinyl Den** frontend (`static/index.html`) featuring:

- **Burgundy & Dusty Rose color palette** — warm, cozy, jazz-lounge aesthetic with dark wood-grain textured background
- **Tactile "Drop the Needle" button** — 3D embossed turntable-style button with inline tonearm SVG icon
- **Markdown-rendered answers** — headings, bold, lists, code blocks, and links via `marked.js`
- **Typewriter effect** — character-by-character answer reveal with variable speed at punctuation
- **Vinyl loading animation** — spinning record + tonearm lowering during search
- **Error toast notifications** — slide-in toasts with auto-dismiss for network/server errors
- **Keyboard UX** — button and input disabled during loading to prevent double-submit
- **Example question chips** — 4 clickable suggestions including lyrics and unknown-artist queries
- **Responsive layout** — mobile-friendly with viewport meta and breakpoints

## Monitoring

Every query is logged to `data/logs.db` by default (or the path configured with
`MUSIC_RAG_DB_PATH`) with response time, real token usage (prompt and completion tokens), calculated cost, tools used,
and user feedback (👍/👎). `static/dashboard.html` renders 6 charts:
question volume, estimated daily API cost, feedback distribution, response time per query, retrieval
method breakdown, and feedback over time. Source tracking is deduplicated to ensure clean source citations in the UI.

## Containerization

`Dockerfile` + `docker-compose.yml` run the full app in one command after the
`.env` prerequisite described above: `docker compose up --build`.

## Best Practices Implemented

- [x] Hybrid search (text + vector, combined via Reciprocal Rank Fusion)
- [x] Agentic tool use (OpenAI function calling with ReAct loop)
- [x] Precomputed embeddings for fast startup
- [x] Auto-loaded environment variables (`python-dotenv`)
- [x] Markdown-rendered answers with typewriter reveal
- [x] Error handling with user-facing toast notifications
- [x] Responsive UI with keyboard UX guards

## Tech Stack

Python, FastAPI, `minsearch` (text search), `onnxruntime` (`bge-small-en-v1.5` embeddings), OpenAI
(`gpt-4o-mini`, function calling), LRCLIB (lyrics), Wikipedia MediaWiki API, SQLite, Docker.
