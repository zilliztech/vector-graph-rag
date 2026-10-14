"""Tests for API endpoints."""

import os
import tempfile
from contextlib import suppress
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.conftest import close_milvus_store
from vector_graph_rag.api.app import create_app
from vector_graph_rag.config import Settings
from vector_graph_rag.graph.graph import Graph


def _attach_span_exporter(exporter: InMemorySpanExporter) -> None:
    """Attach a test exporter to the active OpenTelemetry provider."""
    processor = SimpleSpanProcessor(exporter)
    provider = trace.get_tracer_provider()
    if hasattr(provider, "add_span_processor"):
        provider.add_span_processor(processor)
        return

    provider = TracerProvider()
    provider.add_span_processor(processor)
    try:
        trace.set_tracer_provider(provider)
    except Exception:
        current_provider = trace.get_tracer_provider()
        if not hasattr(current_provider, "add_span_processor"):
            raise
        current_provider.add_span_processor(processor)


@pytest.fixture
def test_settings():
    """Create test settings."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_uri = f.name

    yield Settings(
        milvus_uri=temp_uri,
        openai_api_key="test-api-key",
        llm_model="gpt-6-luna",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        collection_prefix="api_test",
    )

    for path in [temp_uri, f"{temp_uri}.lock"]:
        if os.path.exists(path):
            os.unlink(path)


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

    yield application

    with suppress(Exception):
        graph.drop_collections()
    close_milvus_store(graph._store)


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def span_exporter():
    """Create an in-memory span exporter for API middleware assertions."""
    exporter = InMemorySpanExporter()
    _attach_span_exporter(exporter)
    yield exporter
    exporter.clear()


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns ok."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestObservabilityMiddleware:
    """Tests for API request tracing."""

    def test_request_context_headers_are_added_to_span(self, client, span_exporter):
        """Test request headers and graph context are attached to API spans."""
        response = client.get(
            "/health?graph_name=finance",
            headers={
                "x-request-id": "req-api",
                "x-tenant-id": "tenant-api",
            },
        )

        assert response.status_code == 200

        spans = [
            span for span in span_exporter.get_finished_spans() if span.name == "vgrag.api.request"
        ]
        assert spans
        span = spans[-1]
        assert span.attributes["vgrag.request_id"] == "req-api"
        assert span.attributes["vgrag.tenant_id"] == "tenant-api"
        assert span.attributes["vgrag.graph_name"] == "finance"
        assert span.attributes["http.request.method"] == "GET"
        assert span.attributes["http.response.status_code"] == 200
        assert span.attributes["vgrag.has_request_id"] is True
        assert span.attributes["vgrag.has_tenant_id"] is True


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

    @patch("vector_graph_rag.rag.VectorGraphRAG.rebuild_texts")
    def test_add_documents_basic(self, mock_add, client):
        """Test adding documents endpoint."""
        # Mock the full-rebuild ingestion method
        mock_result = MagicMock()
        mock_result.documents = []
        mock_result.entities = []
        mock_result.relations = []
        mock_add.return_value = mock_result

        response = client.post(
            "/add_documents",
            json={
                "documents": ["Test document 1", "Test document 2"],
                "metadatas": [{"source": "test1"}, {"source": "test2"}],
                "extract_triplets": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "num_documents" in data
        args, kwargs = mock_add.call_args
        assert args[0] == ["Test document 1", "Test document 2"]
        assert kwargs["metadatas"] == [{"source": "test1"}, {"source": "test2"}]


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
            json={"question": "Test question", "filter": 'source == "test"'},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "Test question"
        assert "answer" in data
        _, kwargs = mock_query.call_args
        assert kwargs["filter"] == 'source == "test"'
