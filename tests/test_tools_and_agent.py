import json
from types import SimpleNamespace

import pytest
import requests


def test_successful_local_search_reports_source_and_chunk_count(isolated_modules, monkeypatch):
    chunks = [
        {"topic": "Queen", "chunk": "Queen were a British rock band."},
        {"topic": "Rock music", "chunk": "Rock is a popular music genre."},
    ]
    monkeypatch.setattr(isolated_modules.tools, "hybrid_search", lambda *args, **kwargs: chunks)

    result = isolated_modules.tools.execute_tool(
        "local_search",
        {"query": "Queen"},
        {"text_index": object(), "vector_index": object(), "embedder": object()},
    )

    assert result.source == "Queen, Rock music"
    assert result.chunk_count == 2
    assert "British rock band" in result.content


def test_empty_or_failed_tools_do_not_create_sources(isolated_modules, monkeypatch):
    tools = isolated_modules.tools
    monkeypatch.setattr(tools, "hybrid_search", lambda *args, **kwargs: [])
    empty_local = tools.execute_tool(
        "local_search",
        {"query": "unknown"},
        {"text_index": object(), "vector_index": object(), "embedder": object()},
    )
    assert empty_local.source is None
    assert empty_local.chunk_count == 0

    class Response:
        status_code = 200

        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    monkeypatch.setattr(
        tools.requests,
        "get",
        lambda *args, **kwargs: Response({"query": {"pages": {"1": {"missing": ""}}}}),
    )
    assert tools.wikipedia_search("missing").source is None

    monkeypatch.setattr(tools.requests, "get", lambda *args, **kwargs: Response({"plainLyrics": ""}))
    assert tools.lyrics_search("Queen", "Unknown").source is None

    monkeypatch.setattr(
        tools.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.ConnectionError("offline")),
    )
    with pytest.raises(tools.ExternalToolServiceError):
        tools.wikipedia_search("Queen")


def test_agent_preserves_order_and_handles_malformed_tool_arguments(isolated_modules, monkeypatch):
    app = isolated_modules.app
    tools = isolated_modules.tools
    tool_calls = [
        SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="local_search", arguments="not-json"),
        ),
        SimpleNamespace(
            id="call-2",
            function=SimpleNamespace(name="local_search", arguments=json.dumps({"query": "Queen"})),
        ),
    ]
    responses = [
        SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=tool_calls, content=None))]),
        SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content="Done"))]),
    ]

    class Completions:
        def create(self, **kwargs):
            return responses.pop(0)

    seen_arguments = []
    results = iter([
        tools.ToolResult("first", source="Queen", chunk_count=2),
        tools.ToolResult("second", source="Queen", chunk_count=1),
    ])
    monkeypatch.setattr(app, "client", SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
    monkeypatch.setattr(
        app,
        "execute_tool",
        lambda name, arguments, context: (seen_arguments.append(arguments), next(results))[1],
    )

    answer, sources, num_chunks, *_rest, tool_names = app.agentic_answer("Queen")

    assert answer == "Done"
    assert seen_arguments == [{}, {"query": "Queen"}]
    assert sources == ["Queen"]
    assert num_chunks == 3
    assert tool_names == ["local_search", "local_search"]


def test_agent_does_not_present_failed_tool_as_a_source(isolated_modules, monkeypatch):
    app = isolated_modules.app
    tools = isolated_modules.tools
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="wikipedia_search", arguments=json.dumps({"query": "Queen"})),
    )
    responses = [
        SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tool_call], content=None))]),
        SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content="Unavailable."))]),
    ]

    class Completions:
        def create(self, **kwargs):
            return responses.pop(0)

    monkeypatch.setattr(app, "client", SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
    monkeypatch.setattr(
        app,
        "execute_tool",
        lambda *args: (_ for _ in ()).throw(
            tools.ExternalToolServiceError("Wikipedia is temporarily unavailable.")
        ),
    )

    answer, sources, num_chunks, *_rest, tool_names = app.agentic_answer("Queen")

    assert answer == "Unavailable."
    assert sources == []
    assert num_chunks == 0
    assert tool_names == ["wikipedia_search"]


def test_agent_stops_after_five_tool_iterations(isolated_modules, monkeypatch):
    app = isolated_modules.app
    tools = isolated_modules.tools
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) <= 5:
                tool_call = SimpleNamespace(
                    id=f"call-{len(calls)}",
                    function=SimpleNamespace(name="local_search", arguments="{}"),
                )
                message = SimpleNamespace(tool_calls=[tool_call], content=None)
            else:
                message = SimpleNamespace(tool_calls=None, content="Final answer")
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(app, "client", SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
    monkeypatch.setattr(app, "execute_tool", lambda *args: tools.ToolResult("no result"))

    answer, _sources, _num_chunks, *_rest, tool_names = app.agentic_answer("Queen")

    assert answer == "Final answer"
    assert len(calls) == 6
    assert "tools" not in calls[-1]
    assert tool_names == ["local_search"] * 5
