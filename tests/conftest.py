"""Pytest configuration and fixtures for Vector Graph RAG tests."""

import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch

from vector_graph_rag.config import Settings
from vector_graph_rag.storage.milvus import MilvusStore
from vector_graph_rag.graph.graph import Graph


@pytest.fixture
def temp_milvus_uri():
    """Create a temporary Milvus database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def mock_settings(temp_milvus_uri):
    """Create test settings with mocked API keys."""
    return Settings(
        milvus_uri=temp_milvus_uri,
        openai_api_key="test-api-key",
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        collection_prefix="test",
    )


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model that returns fixed-dimension vectors."""
    mock = MagicMock()
    # Return a 1536-dimensional vector (same as text-embedding-3-small)
    mock.embed.return_value = [0.1] * 1536
    mock.embed_batch.return_value = [[0.1] * 1536]
    mock.dimension = 1536
    return mock


@pytest.fixture
def milvus_store(mock_settings, mock_embedding_model):
    """Create a MilvusStore with mock embedding model."""
    store = MilvusStore(
        settings=mock_settings,
        embedding_model=mock_embedding_model,
    )
    store.create_collections(drop_existing=True)
    yield store
    store.drop_collections()


@pytest.fixture
def graph(mock_settings, mock_embedding_model):
    """Create a Graph instance with mock embedding model."""
    g = Graph(
        settings=mock_settings,
        embedding_model=mock_embedding_model,
    )
    g.create_collections(drop_existing=True)
    yield g
    g.drop_collections()
