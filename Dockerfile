FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv

WORKDIR /code

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache

COPY src/download_model.py src/download_model.py
RUN uv run python src/download_model.py

COPY data/docs.json data/embeddings.npy data/
COPY src/ src/
COPY app/ app/
COPY static/ static/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
