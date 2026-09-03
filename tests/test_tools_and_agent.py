import json
from types import SimpleNamespace

import pytest
import requests


def lyrics_data(tools, *, plain_lyrics="A private lyric body", instrumental=False):
    return tools.LyricsData(
        provider="LRCLIB",
        provider_id=42,
        track_title="Monks",
        artist="Frank Ocean",
        album="channel ORANGE",
        duration_seconds=200.0,
        instrumental=instrumental,
        plain_lyrics=plain_lyrics,
        synced_lyrics="[00:00.00] A private lyric body",
    )


def lrclib_payload(**overrides):
    payload = {
        "id": 42,
        "trackName": "Monks",
        "artistName": "Frank Ocean",
        "albumName": "channel ORANGE",
        "duration": 200,
        "instrumental": False,
        "plainLyrics": "A private lyric body",
        "syncedLyrics": "[00:00.00] A private lyric body",
    }
    payload.update(overrides)
    return payload


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
    assert tools.lyrics_search("Queen", "Unknown", "analysis").source is None

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

    answer, sources, num_chunks, *_rest, tool_names, lyrics = app.agentic_answer("Queen")

    assert answer == "Done"
    assert seen_arguments == [{}, {"query": "Queen"}]
    assert sources == ["Queen"]
    assert num_chunks == 3
    assert tool_names == ["local_search", "local_search"]
    assert lyrics is None


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

    answer, sources, num_chunks, *_rest, tool_names, lyrics = app.agentic_answer("Queen")

    assert answer == "Unavailable."
    assert sources == []
    assert num_chunks == 0
    assert tool_names == ["wikipedia_search"]
    assert lyrics is None


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

    answer, _sources, _num_chunks, *_rest, tool_names, lyrics = app.agentic_answer("Queen")

    assert answer == "Final answer"
    assert len(calls) == 6
    assert "tools" not in calls[-1]
    assert tool_names == ["local_search"] * 5
    assert lyrics is None


def test_display_mode_short_circuits_and_keeps_lyrics_out_of_model_messages(isolated_modules, monkeypatch):
    app = isolated_modules.app
    tools = isolated_modules.tools
    calls = []
    tool_call = SimpleNamespace(
        id="call-lyrics",
        function=SimpleNamespace(
            name="lyrics_search",
            arguments=json.dumps({
                "artist_name": "Frank Ocean",
                "track_name": "Monks",
                "request_kind": "display",
            }),
        ),
    )

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tool_call], content=None))],
            )

    payload = lyrics_data(tools, plain_lyrics="FULL LYRICS MUST STAY STRUCTURED")
    monkeypatch.setattr(app, "client", SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
    monkeypatch.setattr(
        app,
        "execute_tool",
        lambda *args: tools.ToolResult(
            "Lyrics are displayed separately.",
            source="LRCLIB: Monks by Frank Ocean",
            lyrics=payload,
        ),
    )

    answer, sources, _chunks, *_rest, tool_names, lyrics = app.agentic_answer(
        "What were the lyrics of Monks by Frank Ocean?"
    )

    assert answer == "I found “Monks” by Frank Ocean."
    assert sources == ["LRCLIB: Monks by Frank Ocean"]
    assert tool_names == ["lyrics_search"]
    assert lyrics == payload
    assert len(calls) == 1
    assert "FULL LYRICS MUST STAY STRUCTURED" not in str(calls[0]["messages"])


def test_analysis_mode_keeps_lyrics_as_grounding_only(isolated_modules, monkeypatch):
    app = isolated_modules.app
    tools = isolated_modules.tools
    calls = []
    tool_call = SimpleNamespace(
        id="call-analysis",
        function=SimpleNamespace(
            name="lyrics_search",
            arguments=json.dumps({
                "artist_name": "Frank Ocean",
                "track_name": "Monks",
                "request_kind": "analysis",
            }),
        ),
    )
    responses = [
        SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tool_call], content=None))]),
        SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content="The song uses restless imagery."))]),
    ]

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return responses.pop(0)

    payload = lyrics_data(tools, plain_lyrics="GROUNDING LYRICS")
    monkeypatch.setattr(app, "client", SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
    monkeypatch.setattr(
        app,
        "execute_tool",
        lambda *args: tools.ToolResult("Lyrics for analysis:\n\nGROUNDING LYRICS", lyrics=payload),
    )

    answer, _sources, _chunks, *_rest, _tool_names, lyrics = app.agentic_answer("What does Monks mean?")

    assert answer == "The song uses restless imagery."
    assert lyrics is None
    assert any(
        (message.get("content") if isinstance(message, dict) else getattr(message, "content", None))
        == "Lyrics for analysis:\n\nGROUNDING LYRICS"
        for message in calls[1]["messages"]
    )


def test_display_and_analysis_returns_lyrics_and_analysis(isolated_modules, monkeypatch):
    app = isolated_modules.app
    tools = isolated_modules.tools
    tool_call = SimpleNamespace(
        id="call-mixed",
        function=SimpleNamespace(
            name="lyrics_search",
            arguments=json.dumps({
                "artist_name": "Frank Ocean",
                "track_name": "Monks",
                "request_kind": "display_and_analysis",
            }),
        ),
    )
    responses = [
        SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tool_call], content=None))]),
        SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content="The imagery is nocturnal and tense."))]),
    ]

    class Completions:
        def create(self, **kwargs):
            return responses.pop(0)

    payload = lyrics_data(tools)
    monkeypatch.setattr(app, "client", SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
    monkeypatch.setattr(app, "execute_tool", lambda *args: tools.ToolResult("Grounding", lyrics=payload))

    answer, _sources, _chunks, *_rest, _tool_names, lyrics = app.agentic_answer(
        "Show the lyrics and explain Monks."
    )

    assert answer == "The imagery is nocturnal and tense."
    assert lyrics == payload


def test_lyrics_search_validates_provider_data_and_special_cases(isolated_modules, monkeypatch):
    tools = isolated_modules.tools

    class Response:
        status_code = 200

        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    monkeypatch.setattr(tools.requests, "get", lambda *args, **kwargs: Response(lrclib_payload()))
    display = tools.lyrics_search("Frank Ocean", "Monks", "display")
    assert display.lyrics is not None
    assert display.lyrics.track_title == "Monks"
    assert "A private lyric body" not in display.content

    monkeypatch.setattr(
        tools.requests,
        "get",
        lambda *args, **kwargs: Response(lrclib_payload(trackName="Different Song")),
    )
    assert tools.lyrics_search("Frank Ocean", "Monks", "display").source is None

    monkeypatch.setattr(
        tools.requests,
        "get",
        lambda *args, **kwargs: Response(lrclib_payload(plainLyrics="", instrumental=True)),
    )
    instrumental = tools.lyrics_search("Frank Ocean", "Monks", "display")
    assert instrumental.lyrics is None
    assert instrumental.source == "LRCLIB: Monks by Frank Ocean"
    assert "instrumental" in instrumental.content

    monkeypatch.setattr(tools.requests, "get", lambda *args, **kwargs: Response({"id": "bad"}))
    assert tools.lyrics_search("Frank Ocean", "Monks", "display").source is None


def test_lyrics_search_handles_not_found_and_provider_failure(isolated_modules, monkeypatch):
    tools = isolated_modules.tools

    class MissingResponse:
        status_code = 404

    monkeypatch.setattr(tools.requests, "get", lambda *args, **kwargs: MissingResponse())
    assert tools.lyrics_search("Frank Ocean", "Missing", "display").source is None

    monkeypatch.setattr(
        tools.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.ConnectionError("offline")),
    )
    with pytest.raises(tools.ExternalToolServiceError):
        tools.lyrics_search("Frank Ocean", "Monks", "display")
