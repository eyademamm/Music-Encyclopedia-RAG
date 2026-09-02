"""
Build and persist embeddings for all documents in data/docs.json.

Run this once after ingest.py, or whenever docs.json changes:
    uv run python src/build_embeddings.py

Output: data/embeddings.npy  (~3.5 MB, float32, shape [N, 384])

On subsequent server startups, search.py will load this file instead of
recomputing embeddings from scratch — cutting startup from ~10 min to ~1 sec.
"""

import json
import time
from pathlib import Path

import numpy as np

from embedder import Embedder

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "docs.json"
OUT_PATH  = Path(__file__).resolve().parent.parent / "data" / "embeddings.npy"


def main():
    print(f"Loading documents from {DATA_PATH} ...")
    docs = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    print(f"  {len(docs)} chunks loaded.")

    print("Initialising ONNX embedder ...")
    embedder = Embedder()

    print(f"Computing embeddings (batch size 64) — this may take several minutes ...")
    t0 = time.time()
    texts = [d["chunk"] for d in docs]
    embeddings = embedder.encode_batch(texts, batch_size=64)
    elapsed = time.time() - t0

    print(f"  Done in {elapsed:.1f}s  |  shape: {embeddings.shape}  |  dtype: {embeddings.dtype}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    np.save(OUT_PATH, embeddings)
    size_mb = OUT_PATH.stat().st_size / 1_048_576
    print(f"Saved to {OUT_PATH}  ({size_mb:.1f} MB)")
    print("\nNext server startup will load this file and skip recomputation.")


if __name__ == "__main__":
    main()

