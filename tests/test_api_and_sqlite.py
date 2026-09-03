import json
import sqlite3


def create_conversation(api_client, title=None):
    payload = {} if title is None else {"title": title}
    response = api_client.post("/conversations", json=payload)
    assert response.status_code == 201
    return response.json()


def test_ask_rejects_blank_invalid_and_missing_conversation_input(api_client):
    conversation = create_conversation(api_client)
    assert api_client.post("/ask", json={"conversation_id": conversation["id"], "question": "   "}).status_code == 422
    assert api_client.post("/ask", json={"conversation_id": conversation["id"], "question": ["not", "text"]}).status_code == 422
    assert api_client.post("/ask", json={"conversation_id": conversation["id"], "question": "x" * 2_001}).status_code == 422
    assert api_client.post("/ask", json={"question": "Who formed Queen?"}).status_code == 422


def test_ask_reports_missing_openai_configuration_for_existing_conversation(api_client):
    conversation = create_conversation(api_client)
    response = api_client.post(
        "/ask", json={"conversation_id": conversation["id"], "question": "Who formed Queen?"}
    )
    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The answer service is not configured. Set OPENAI_API_KEY and restart the server."
    )


def test_missing_or_deleted_conversations_are_cleanly_rejected(api_client):
    missing_id = "00000000-0000-4000-8000-000000000000"
    response = api_client.post("/ask", json={"conversation_id": missing_id, "question": "Queen"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found."}

    conversation = create_conversation(api_client)
    assert api_client.delete(f"/conversations/{conversation['id']}").status_code == 204
    response = api_client.post("/ask", json={"conversation_id": conversation["id"], "question": "Queen"})
    assert response.status_code == 404


def test_feedback_validation_and_scoping(api_client, isolated_modules, monkeypatch):
    first = create_conversation(api_client)
    second = create_conversation(api_client)
    monkeypatch.setattr(
        isolated_modules.app,
        "agentic_answer",
        lambda question, context: ("Answer", [], 0, 1, 1, 0.0, []),
    )
    interaction_id = api_client.post(
        "/ask", json={"conversation_id": first["id"], "question": "Queen"}
    ).json()["interaction_id"]

    invalid = api_client.post(
        "/feedback",
        json={"conversation_id": first["id"], "interaction_id": interaction_id, "feedback": 0},
    )
    wrong_conversation = api_client.post(
        "/feedback",
        json={"conversation_id": second["id"], "interaction_id": interaction_id, "feedback": 1},
    )
    accepted = api_client.post(
        "/feedback",
        json={"conversation_id": first["id"], "interaction_id": interaction_id, "feedback": 1},
    )

    assert invalid.status_code == 422
    assert wrong_conversation.status_code == 404
    assert accepted.json() == {"status": "ok"}


def test_ask_logs_turn_telemetry_and_sources(api_client, isolated_modules, monkeypatch):
    conversation = create_conversation(api_client)
    seen_context = []
    monkeypatch.setattr(
        isolated_modules.app,
        "agentic_answer",
        lambda question, context: (
            seen_context.append(context), "A grounded answer.", ["Queen"], 3, 11, 7,
            0.00000585, ["local_search", "wikipedia_search", "local_search"],
        )[1:],
    )

    response = api_client.post(
        "/ask",
        json={"conversation_id": conversation["id"], "question": "  Tell me about Queen.  "},
    )

    assert response.status_code == 200
    assert seen_context == [[]]
    interaction_id = response.json()["interaction_id"]
    with sqlite3.connect(isolated_modules.db_path) as conn:
        row = conn.execute(
            """
            SELECT question, num_chunks, prompt_tokens, completion_tokens, cost,
                   tools_used, conversation_id, sources
            FROM logs WHERE id = ?
            """,
            (interaction_id,),
        ).fetchone()

    assert row[:5] == ("Tell me about Queen.", 3, 11, 7, 0.00000585)
    assert json.loads(row[5]) == ["local_search", "wikipedia_search", "local_search"]
    assert row[6] == conversation["id"]
    assert json.loads(row[7]) == ["Queen"]


def test_conversations_are_listed_reopened_and_isolated(api_client, isolated_modules, monkeypatch):
    first = create_conversation(api_client, "First")
    second = create_conversation(api_client, "Second")
    monkeypatch.setattr(
        isolated_modules.app,
        "agentic_answer",
        lambda question, context: (f"Answer: {question}", ["Source"], 1, 1, 1, 0.0, ["local_search"]),
    )
    api_client.post("/ask", json={"conversation_id": first["id"], "question": "First question"})
    api_client.post("/ask", json={"conversation_id": second["id"], "question": "Second question"})

    first_history = api_client.get(f"/conversations/{first['id']}").json()
    second_history = api_client.get(f"/conversations/{second['id']}").json()
    listed = api_client.get("/conversations").json()["conversations"]

    assert [message["content"] for message in first_history["messages"]] == ["First question", "Answer: First question"]
    assert [message["content"] for message in second_history["messages"]] == ["Second question", "Answer: Second question"]
    assert {conversation["id"] for conversation in listed} == {first["id"], second["id"]}
    assert all("question" not in row and "answer" not in row for row in api_client.get("/stats").json()["logs"])


def test_deleting_conversation_cascades_turns_and_preserves_others(api_client, isolated_modules, monkeypatch):
    first = create_conversation(api_client)
    second = create_conversation(api_client)
    monkeypatch.setattr(
        isolated_modules.app,
        "agentic_answer",
        lambda question, context: ("Answer", [], 0, 1, 1, 0.0, []),
    )
    api_client.post("/ask", json={"conversation_id": first["id"], "question": "One"})
    api_client.post("/ask", json={"conversation_id": second["id"], "question": "Two"})

    assert api_client.delete(f"/conversations/{first['id']}").status_code == 204
    assert api_client.get(f"/conversations/{first['id']}").status_code == 404
    assert len(api_client.get(f"/conversations/{second['id']}").json()["messages"]) == 2
    with sqlite3.connect(isolated_modules.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM logs WHERE conversation_id = ?", (first["id"],)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM logs WHERE conversation_id = ?", (second["id"],)).fetchone()[0] == 1


def test_context_is_bounded_and_never_crosses_conversation(api_client, isolated_modules):
    app = isolated_modules.app
    first = create_conversation(api_client)
    second = create_conversation(api_client)
    with sqlite3.connect(isolated_modules.db_path) as conn:
        for index in range(6):
            conn.execute(
                """
                INSERT INTO logs (id, question, answer, method, num_chunks, response_time, feedback,
                    created_at, prompt_tokens, completion_tokens, cost, tools_used, conversation_id, sources)
                VALUES (?, ?, ?, 'agentic', 0, 0, NULL, ?, 0, 0, 0, '[]', ?, '[]')
                """,
                (f"first-{index}", "Q" * 2_000, "A" * 2_000, float(index), first["id"]),
            )
        conn.execute(
            """
            INSERT INTO logs (id, question, answer, method, num_chunks, response_time, feedback,
                created_at, prompt_tokens, completion_tokens, cost, tools_used, conversation_id, sources)
            VALUES ('second', 'SECRET', 'OTHER SECRET', 'agentic', 0, 0, NULL, 1, 0, 0, 0, '[]', ?, '[]')
            """,
            (second["id"],),
        )
        conn.commit()
        conn.row_factory = sqlite3.Row
        context = app.load_recent_context(conn, first["id"])

    assert len(context) <= app.MAX_CONTEXT_TURNS * 2
    assert sum(len(message["content"]) for message in context) <= app.MAX_CONTEXT_CHARS
    assert all("SECRET" not in message["content"] for message in context)


def test_legacy_sqlite_schema_is_migrated_without_losing_logs(isolated_modules, tmp_path):
    legacy_path = tmp_path / "legacy.db"
    with sqlite3.connect(legacy_path) as conn:
        conn.execute(
            """
            CREATE TABLE logs (
                id TEXT PRIMARY KEY, question TEXT, answer TEXT, method TEXT,
                num_chunks INTEGER, response_time REAL, feedback INTEGER, created_at REAL
            )
            """
        )
        conn.execute("INSERT INTO logs VALUES ('legacy-1', 'Legacy question', 'Legacy answer', 'agentic', 2, 1.5, 1, 12.0)")

    original_path = isolated_modules.app.DB_PATH
    isolated_modules.app.DB_PATH = legacy_path
    try:
        isolated_modules.app.init_db()
        isolated_modules.app.init_db()
    finally:
        isolated_modules.app.DB_PATH = original_path

    with sqlite3.connect(legacy_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(logs)")}
        log = conn.execute("SELECT question, answer, feedback, conversation_id FROM logs WHERE id = 'legacy-1'").fetchone()
        conversation = conn.execute("SELECT title FROM conversations WHERE id = ?", (log[3],)).fetchone()

    assert {"prompt_tokens", "completion_tokens", "cost", "tools_used", "conversation_id", "sources"} <= columns
    assert log[:3] == ("Legacy question", "Legacy answer", 1)
    assert conversation == ("Legacy question",)
