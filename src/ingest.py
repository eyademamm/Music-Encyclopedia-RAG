"""
Ingestion script for the Music Encyclopedia RAG project.

Pulls plain-text summaries + full extracts for a curated list of music
artists / bands / genres from the Wikipedia REST API (freely licensed,
CC BY-SA, no auth needed), chunks them, and writes data/docs.json.

Usage:
    python src/ingest.py
"""

import json
import re
import time
from pathlib import Path

import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "MusicEncyclopediaRAG/1.0 (LLM Zoomcamp student project; contact: eyademam28@gmail.com)"
}
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "docs.json"

# Seed list of music topics. Feel free to expand — this is intentionally
# broad across genres/eras so retrieval questions have variety.
TOPICS = [
    # Genres
    "Rock music", "Hip hop music", "Jazz", "Blues", "Reggae", "Heavy metal",
    "Punk rock", "Electronic music", "Pop music", "Classical music",
    "Country music", "R&B", "Funk", "Disco", "Soul music", "Afrobeats",
    # Artists / bands (mix of eras/genres for variety)
    "The Beatles", "Michael Jackson", "Bob Marley", "Queen (band)",
    "Nirvana (band)", "Beyoncé", "Kendrick Lamar", "Daft Punk",
    "Fela Kuti", "Umm Kulthum", "Amr Diab", "Miles Davis",
    "Led Zeppelin", "Radiohead", "Taylor Swift", "Kanye West",
    "David Bowie", "Pink Floyd", "The Rolling Stones", "Madonna",
]


def fetch_extract(title: str) -> dict | None:
    """Fetch the full plain-text extract of a Wikipedia page."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json",
        "redirects": 1,
    }
    r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    for _, page in pages.items():
        if "missing" in page:
            return None
        return {
            "title": page.get("title", title),
            "text": page.get("extract", ""),
        }
    return None


def clean_text(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"={2,}.*?={2,}", "", text)  # strip section headers like == History ==
    return text.strip()


def chunk_text(title: str, text: str, max_words: int = 180) -> list[dict]:
    """Simple paragraph-aware chunking, similar to the course's approach."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    buf = []
    word_count = 0
    for para in paragraphs:
        words = para.split()
        if word_count + len(words) > max_words and buf:
            chunks.append(" ".join(buf))
            buf, word_count = [], 0
        buf.append(para)
        word_count += len(words)
    if buf:
        chunks.append(" ".join(buf))

    return [
        {
            "id": f"{title}-{i}",
            "topic": title,
            "chunk": chunk,
        }
        for i, chunk in enumerate(chunks)
        if len(chunk.split()) > 20  # drop tiny fragments
    ]


def main():
    all_docs = []
    for title in TOPICS:
        print(f"Fetching: {title}")
        page = fetch_extract(title)
        if not page or not page["text"]:
            print(f"  -> skipped (not found)")
            continue
        cleaned = clean_text(page["text"])
        chunks = chunk_text(page["title"], cleaned)
        all_docs.extend(chunks)
        print(f"  -> {len(chunks)} chunks")
        time.sleep(0.2)  # be polite to the API

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(all_docs, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(all_docs)} chunks from {len(TOPICS)} topics -> {OUT_PATH}")


if __name__ == "__main__":
    main()
