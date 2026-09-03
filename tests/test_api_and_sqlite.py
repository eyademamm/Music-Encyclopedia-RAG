import json
import sqlite3


def test_ask_rejects_blank_and_invalid_input(api_client):
    assert api_client.post("/ask", json={"question": "   "}).status_code == 422
    assert api_client.post("/ask", json={"question": ["not", "text"]}).status_code == 422
    assert api_client.post("/ask", json={"question": "x" * 2_001}).status_code == 422


def test_ask_reports_missing_openai_configuration(api_client):
    response = api_client.post("/ask", json={"question": "Who formed Queen?"})

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The answer service is not configured. Set OPENAI_API_KEY and restart the server."
    )


def test_feedback_validation_and_unknown_interaction(api_client):
    invalid = api_client.post("/feedback", json={"interaction_id": "missing", "feedback": 0})
    missing = api_client.post("/feedback", json={"interaction_id": "missing", "feedback": 1})

    assert invalid.status_code == 422
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Interaction not found."}


def test_ask_logs_true_chunk_count_and_ordered_tool_calls(api_client, isolated_modules, monkeypatch):
    monkeypatch.setattr(
        isolated_modules.app,
        "agentic_answer",
        lambda question: (
            "A grounded answer.",
            ["Queen"],
            3,
            11,
            7,
            0.00000585,
            ["local_search", "wikipedia_search", "local_search"],
        ),
    )

    response = api_client.post("/ask", json={"question": "  Tell me about Queen.  "})

    assert response.status_code == 200
    interaction_id = response.json()["interaction_id"]
    with sqlite3.connect(isolated_modules.db_path) as conn:
        row = conn.execute(
            """
            SELECT question, num_chunks, prompt_tokens, completion_tokens, cost, tools_used
            FROM logs WHERE id = ?
            """,
            (interaction_id,),
        ).fetchone()

    assert row[:5] == ("Tell me about Queen.", 3, 11, 7, 0.00000585)
    assert json.loads(row[5]) == ["local_search", "wikipedia_search", "local_search"]


def test_legacy_sqlite_schema_is_migrated(isolated_modules, tmp_path):
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

    original_path = isolated_modules.app.DB_PATH
    isolated_modules.app.DB_PATH = legacy_path
    try:
        isolated_modules.app.init_db()
    finally:
        isolated_modules.app.DB_PATH = original_path

    with sqlite3.connect(legacy_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(logs)")}

    assert {"prompt_tokens", "completion_tokens", "cost", "tools_used"} <= columns
