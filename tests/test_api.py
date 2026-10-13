"""
Integration tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture(scope="module")
def client():
    with patch("app.vector_store.VectorStoreManager.__init__", return_value=None):
        with patch("app.vector_store.VectorStoreManager.list_sources", return_value=[]):
            from app.main import app
            return TestClient(app)


def test_health_check(client):
    with patch("app.vector_store.get_vector_store") as mock:
        mock.return_value.list_sources.return_value = []
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()


def test_list_documents_empty(client):
    with patch("app.routers.documents.get_vector_store") as mock:
        mock.return_value.list_sources.return_value = []
        response = client.get("/api/documents/")
    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_ingest_text(client):
    with patch("app.routers.documents.ingest_text") as mock_ingest:
        mock_ingest.return_value = {"source": "test", "chunks_stored": 3}
        response = client.post(
            "/api/documents/text",
            json={"text": "Apple Inc. Q4 2024 earnings report.", "source_name": "test"},
        )
    assert response.status_code == 201
    assert response.json()["chunks_stored"] == 3


def test_query_empty_string(client):
    response = client.post("/api/query", json={"query": "   "})
    assert response.status_code == 422
