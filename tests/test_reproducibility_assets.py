import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_committed_retrieval_assets_are_aligned():
    docs = json.loads((ROOT / "data" / "docs.json").read_text(encoding="utf-8"))
    embeddings = np.load(ROOT / "data" / "embeddings.npy", mmap_mode="r")

    assert docs
    assert embeddings.dtype == np.float32
    assert embeddings.shape == (len(docs), 384)
