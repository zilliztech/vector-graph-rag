"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import tempfile
import os

from vector_graph_rag.api.app import create_app
from vector_graph_rag.config import Settings
from vector_graph_rag.graph.graph import Graph


@pytest.fixture
def test_settings():
    """Create test settings."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_uri = f.name

    yield Settings(
        milvus_uri=temp_uri,
        openai_api_key="test-api-key",
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        collection_prefix="api_test",
    )

    if os.path.exists(temp_uri):
        os.unlink(temp_uri)


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    mock = MagicMock()
    mock.embed.return_value = [0.1] * 1536
    mock.embed_batch.return_value = [[0.1] * 1536]
    mock.dimension = 1536
    return mock


@pytest.fixture
def app(test_settings, mock_embedding_model):
    """Create test app with initialized collections."""
    application = create_app(test_settings)

    # Initialize graph with mock embedding model and create collections
    graph = Graph(settings=test_settings, embedding_model=mock_embedding_model)
    graph.create_collections(drop_existing=True)
    application.state.graph_instances["default"] = graph

    return application


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns ok."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestListGraphsEndpoint:
    """Tests for listing graphs endpoint."""

    def test_list_graphs_empty(self, client):
        """Test listing graphs when empty."""
        response = client.get("/graphs")

        assert response.status_code == 200
        data = response.json()
        assert "graphs" in data
        assert isinstance(data["graphs"], list)


class TestDocumentCRUDEndpoints:
    """Tests for document CRUD endpoints."""

    def test_get_document_not_found(self, client):
        """Test getting non-existent document returns 404."""
        response = client.get("/documents/nonexistent_id")

        assert response.status_code == 404

    def test_list_documents_without_query(self, client):
        """Test listing documents without query returns empty."""
        response = client.get("/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0

    def test_delete_document_not_found(self, client):
        """Test deleting non-existent document returns 404."""
        response = client.delete("/documents/nonexistent_id")

        assert response.status_code == 404

    def test_update_document_not_found(self, client):
        """Test updating non-existent document returns 404."""
        response = client.put(
            "/documents/nonexistent_id",
            json={"text": "New text"},
        )

        assert response.status_code == 404


class TestAddDocumentsEndpoint:
    """Tests for add documents endpoint."""

    @patch("vector_graph_rag.rag.VectorGraphRAG.add_documents")
    def test_add_documents_basic(self, mock_add, client):
        """Test adding documents endpoint."""
        # Mock the add_documents method
        mock_result = MagicMock()
        mock_result.documents = []
        mock_result.entities = []
        mock_result.relations = []
        mock_add.return_value = mock_result

        response = client.post(
            "/add_documents",
            json={
                "documents": ["Test document 1", "Test document 2"],
                "extract_triplets": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "num_documents" in data


class TestQueryEndpoint:
    """Tests for query endpoint."""

    @patch("vector_graph_rag.rag.VectorGraphRAG.query")
    def test_query_basic(self, mock_query, client):
        """Test query endpoint."""
        # Mock query result
        mock_result = MagicMock()
        mock_result.query = "Test question"
        mock_result.answer = "Test answer"
        mock_result.query_entities = []
        mock_result.retrieved_passages = []
        mock_result.retrieved_relations = []
        mock_result.expanded_relations = []
        mock_result.reranked_relations = []
        mock_query.return_value = mock_result

        response = client.post(
            "/query",
            json={"question": "Test question"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "Test question"
        assert "answer" in data
