import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import dotenv
import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"


@pytest.fixture
def isolated_modules(monkeypatch, tmp_path):
    """Load the app against fake indexes and an isolated SQLite database."""
    monkeypatch.setenv("MUSIC_RAG_DB_PATH", str(tmp_path / "logs.db"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)

    fake_search = ModuleType("search")
    fake_search.load_docs = lambda: []
    fake_search.build_text_index = lambda docs: object()
    fake_search.build_vector_index = lambda docs: (object(), object())
    fake_search.hybrid_search = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "search", fake_search)
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.syspath_prepend(str(SRC_DIR))

    sys.modules.pop("tools", None)
    tools = importlib.import_module("tools")
    monkeypatch.setitem(sys.modules, "tools", tools)

    sys.modules.pop("app.main", None)
    app = importlib.import_module("app.main")
    yield SimpleNamespace(app=app, tools=tools, db_path=app.DB_PATH)

    sys.modules.pop("app.main", None)
    sys.modules.pop("tools", None)


@pytest.fixture
def api_client(isolated_modules):
    with TestClient(isolated_modules.app.app, raise_server_exceptions=False) as client:
        yield client
