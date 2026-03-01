from io import BytesIO

import pytest

from app import create_app
from app import routes


class TestConfig:
    TESTING = True
    DEBUG = False
    SECRET_KEY = "test-secret"
    OPENAI_API_KEY = ""
    OPENAI_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 50
    RETRIEVAL_K = 3


@pytest.fixture()
def client():
    app = create_app(TestConfig)
    with app.app_context():
        routes.document_service.clear()
        routes.chat_service.memory.clear()
    with app.test_client() as test_client:
        yield test_client
    with app.app_context():
        routes.document_service.clear()
        routes.chat_service.memory.clear()


def post_json(client, path, payload):
    return client.post(path, json=payload)


def test_health_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_stats_initially_empty(client):
    response = client.get("/api/documents/stats")
    body = response.get_json()
    assert response.status_code == 200
    assert body["documents"] == 0
    assert body["chunks"] == 0
    assert body["sources"] == []


def test_upload_text_success(client):
    response = post_json(
        client,
        "/api/documents/upload",
        {"source": "note.txt", "text": "Flask powers this backend."},
    )
    body = response.get_json()
    assert response.status_code == 201
    assert body["source"] == "note.txt"
    assert body["chunks_added"] >= 1


def test_upload_text_empty_fails(client):
    response = post_json(client, "/api/documents/upload", {"source": "x", "text": ""})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_upload_text_defaults_source(client):
    response = post_json(client, "/api/documents/upload", {"text": "abc"})
    assert response.status_code == 201
    assert response.get_json()["source"] == "manual-input"


def test_upload_non_pdf_file_rejected(client):
    response = client.post(
        "/api/documents/upload",
        data={"file": (BytesIO(b"abc"), "not_pdf.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_upload_pdf_uses_service(monkeypatch, client):
    called = {}

    def fake_add_pdf(path, source):
        called["path"] = path
        called["source"] = source
        return {"source": source, "chunks_added": 2}

    monkeypatch.setattr(routes.document_service, "add_pdf", fake_add_pdf)

    response = client.post(
        "/api/documents/upload",
        data={"file": (BytesIO(b"%PDF-1.4"), "sample.pdf")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    assert response.get_json()["chunks_added"] == 2
    assert called["source"] == "sample.pdf"


def test_upload_pdf_failure_returns_500(monkeypatch, client):
    def fake_add_pdf(path, source):
        raise RuntimeError("loader failed")

    monkeypatch.setattr(routes.document_service, "add_pdf", fake_add_pdf)

    response = client.post(
        "/api/documents/upload",
        data={"file": (BytesIO(b"%PDF-1.4"), "sample.pdf")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 500
    assert "loader failed" in response.get_json()["error"]


def test_stats_after_upload(client):
    post_json(client, "/api/documents/upload", {"source": "a.txt", "text": "one two"})
    post_json(client, "/api/documents/upload", {"source": "b.txt", "text": "three four"})
    response = client.get("/api/documents/stats")
    body = response.get_json()
    assert body["documents"] == 2
    assert "a.txt" in body["sources"]
    assert "b.txt" in body["sources"]


def test_clear_documents(client):
    post_json(client, "/api/documents/upload", {"source": "a.txt", "text": "one two"})
    clear_response = client.post("/api/documents/clear")
    stats_response = client.get("/api/documents/stats")
    assert clear_response.status_code == 200
    assert stats_response.get_json()["chunks"] == 0


def test_chat_requires_question(client):
    response = post_json(client, "/api/chat", {"question": ""})
    assert response.status_code == 400


def test_chat_fallback_without_documents(client):
    response = post_json(client, "/api/chat", {"question": "What is this?"})
    assert response.status_code == 200
    assert "enough context" in response.get_json()["answer"].lower()


def test_chat_with_uploaded_context(client):
    post_json(
        client,
        "/api/documents/upload",
        {"source": "about.txt", "text": "RAG means retrieval augmented generation."},
    )
    response = post_json(client, "/api/chat", {"question": "What does RAG mean?"})
    body = response.get_json()
    assert response.status_code == 200
    assert body["sources"]
    assert "about.txt" in body["sources"]


def test_chat_session_memory_tracks_turns(client):
    post_json(client, "/api/chat", {"question": "First?", "session_id": "s1"})
    post_json(client, "/api/chat", {"question": "Second?", "session_id": "s1"})
    assert len(routes.chat_service.memory["s1"]) == 4


def test_chat_different_sessions_are_isolated(client):
    post_json(client, "/api/chat", {"question": "Hello", "session_id": "s1"})
    post_json(client, "/api/chat", {"question": "Hi", "session_id": "s2"})
    assert len(routes.chat_service.memory["s1"]) == 2
    assert len(routes.chat_service.memory["s2"]) == 2


def test_chat_default_session_used(client):
    post_json(client, "/api/chat", {"question": "Hello"})
    assert len(routes.chat_service.memory["default"]) == 2


def test_chat_custom_session_used(client):
    post_json(client, "/api/chat", {"question": "Hello", "session_id": "abc"})
    assert len(routes.chat_service.memory["abc"]) == 2


def test_upload_accepts_whitespace_around_text(client):
    response = post_json(client, "/api/documents/upload", {"source": "x", "text": "   data  "})
    assert response.status_code == 201


def test_multiple_uploads_increase_chunks(client):
    post_json(client, "/api/documents/upload", {"source": "x", "text": "one"})
    post_json(client, "/api/documents/upload", {"source": "y", "text": "two"})
    stats = client.get("/api/documents/stats").get_json()
    assert stats["chunks"] >= 2


def test_clear_is_idempotent(client):
    first = client.post("/api/documents/clear")
    second = client.post("/api/documents/clear")
    assert first.status_code == 200
    assert second.status_code == 200


def test_chat_returns_sources_key(client):
    response = post_json(client, "/api/chat", {"question": "Any data?"})
    assert "sources" in response.get_json()


def test_upload_handles_missing_json_body(client):
    response = client.post("/api/documents/upload")
    assert response.status_code == 400


def test_chat_handles_missing_json_body(client):
    response = client.post("/api/chat")
    assert response.status_code == 400


def test_stats_contains_vectorstore_flag(client):
    response = client.get("/api/documents/stats")
    assert "vectorstore_enabled" in response.get_json()


def test_sources_are_unique_by_name(client):
    post_json(client, "/api/documents/upload", {"source": "same.txt", "text": "one"})
    post_json(client, "/api/documents/upload", {"source": "same.txt", "text": "two"})
    stats = client.get("/api/documents/stats").get_json()
    assert stats["documents"] == 1


def test_chat_response_shape(client):
    response = post_json(client, "/api/chat", {"question": "Shape?"})
    body = response.get_json()
    assert isinstance(body["answer"], str)
    assert isinstance(body["sources"], list)


def test_upload_response_shape(client):
    response = post_json(client, "/api/documents/upload", {"source": "shape.txt", "text": "x"})
    body = response.get_json()
    assert "source" in body
    assert "chunks_added" in body


def test_health_method_not_allowed_for_post(client):
    response = client.post("/api/health")
    assert response.status_code == 405
