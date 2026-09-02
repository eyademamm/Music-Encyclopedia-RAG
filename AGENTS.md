# The Crate — Repository Guidance

## Scope and principles

- This is a nearly finished portfolio project. Prefer minimal, high-impact changes and preserve existing behavior unless the task explicitly changes it.
- Inspect the relevant existing code paths and patterns before implementing anything new. Do not modify unrelated files.
- Prefer explicit, simple code over unnecessary abstractions.
- Reuse the existing FastAPI, SQLite, vanilla HTML/CSS/JS, local retrieval, and evaluation architecture.
- Do not introduce new frameworks, databases, ORMs, vector databases, Redis, Celery, LangChain, React, or Vue.
- Preserve The Crate's Burgundy/Dusty Rose Vinyl Den frontend identity, including its tone and visual language.
- Keep changes compatible with WSL/Linux and Docker.

## Security and data handling

- Never read, modify, commit, print, or expose `.env` secrets.
- Treat model output, tool output, API error text, and user input as untrusted when rendering in the browser.
- Do not persist raw Wikipedia extracts, full lyrics, hidden reasoning, or raw tool transcripts.
- Do not silently change retrieval behavior, retrieval ranking, or evaluation metrics. Any intentional change must be evaluated and documented.

## Architecture boundaries

- `app/main.py` owns FastAPI routes, the agent loop, and SQLite interaction logging.
- `src/tools.py` owns local, Wikipedia, and LRCLIB tool behavior.
- `src/search.py` and `src/embedder.py` own retrieval and embeddings.
- `static/index.html` is the main Vanilla JS Vinyl Den UI; `static/dashboard.html` is the monitoring UI.
- Keep SQLite schemas and inserts backward-compatible; use explicit column names when changing persisted data.

## Validation

- Run relevant tests or validation after every implementation task. Add focused regression coverage when behavior changes.
- Do not make network-dependent evaluation or ingestion a default test.
- At minimum, run syntax/static validation appropriate to changed files and check the worktree for unintended edits.

## Project commands

Install dependencies:

```bash
uv sync
```

Build or refresh local retrieval assets (network/API requirements noted in the README):

```bash
uv run python src/ingest.py
uv run python src/download_model.py
uv run python src/build_embeddings.py
```

Run evaluations:

```bash
uv run python src/generate_ground_truth.py
uv run python src/evaluate.py
uv run python src/evaluate_llm.py
```

Run the server:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Run with Docker:

```bash
docker compose up --build
```
