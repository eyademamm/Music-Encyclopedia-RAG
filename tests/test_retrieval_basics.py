import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def test_hybrid_search_uses_rrf_to_rank_shared_top_results(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    fake_embedder = ModuleType("embedder")
    fake_embedder.Embedder = object
    monkeypatch.setitem(sys.modules, "embedder", fake_embedder)

    spec = importlib.util.spec_from_file_location("search_under_test", root / "src" / "search.py")
    search = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(search)

    docs = {
        "a": {"id": "a"},
        "b": {"id": "b"},
        "c": {"id": "c"},
    }
    monkeypatch.setattr(search, "text_search", lambda *args, **kwargs: [docs["a"], docs["b"], docs["c"]])
    monkeypatch.setattr(search, "vector_search", lambda *args, **kwargs: [docs["b"], docs["c"], docs["a"]])

    ranked = search.hybrid_search(object(), object(), object(), "query", num_results=3, k=60)

    assert [doc["id"] for doc in ranked] == ["b", "a", "c"]
