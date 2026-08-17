"""
Search module: text search, vector search, and hybrid (RRF) search
over the music knowledge base. Same pattern as LLM Zoomcamp HW3.
"""

import json
from pathlib import Path

import numpy as np
from minsearch import Index, VectorSearch
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "docs.json"


def load_docs():
    return json.loads(DATA_PATH.read_text())


def build_text_index(docs):
    index = Index(text_fields=["chunk", "topic"], keyword_fields=[])
    index.fit(docs)
    return index


def build_vector_index(docs):
    """TF-IDF based vector index (no external embedding API needed —
    keeps the project runnable without extra API costs; swap in
    sentence-transformers or OpenAI embeddings if you want higher quality)."""
    texts = [d["chunk"] for d in docs]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(texts).toarray()

    vindex = VectorSearch(keyword_fields=[])
    vindex.fit(X, docs)
    return vindex, vectorizer


def text_search(index, query, num_results=5):
    return index.search(query, num_results=num_results)


def vector_search(vindex, vectorizer, query, num_results=5):
    q_vec = vectorizer.transform([query]).toarray()[0]
    return vindex.search(q_vec, num_results=num_results)


def hybrid_search(index, vindex, vectorizer, query, num_results=5, k=60):
    """Reciprocal Rank Fusion of text + vector results."""
    text_results = text_search(index, query, num_results=10)
    vec_results = vector_search(vindex, vectorizer, query, num_results=10)

    scores = {}
    doc_lookup = {}

    for rank, doc in enumerate(text_results):
        doc_id = doc["id"]
        doc_lookup[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    for rank, doc in enumerate(vec_results):
        doc_id = doc["id"]
        doc_lookup[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:num_results]
    return [doc_lookup[i] for i in ranked_ids]


if __name__ == "__main__":
    docs = load_docs()
    print(f"Loaded {len(docs)} chunks")
    index = build_text_index(docs)
    vindex, vectorizer = build_vector_index(docs)

    q = "Who were the members of the Beatles?"
    print("\nText search:")
    for d in text_search(index, q, 3):
        print(" -", d["topic"], "|", d["chunk"][:80])

    print("\nHybrid search:")
    for d in hybrid_search(index, vindex, vectorizer, q, 3):
        print(" -", d["topic"], "|", d["chunk"][:80])
