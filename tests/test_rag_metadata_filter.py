"""End-to-end tests for metadata filtering in VectorGraphRAG."""

import os
import tempfile
from unittest.mock import MagicMock, patch

from vector_graph_rag.config import Settings
from vector_graph_rag.graph.retriever import GraphRetriever
from vector_graph_rag.models import Document
from vector_graph_rag.rag import VectorGraphRAG


class FakeEmbeddingModel:
    """Small deterministic embedding model for Milvus Lite tests."""

    dimension = 4

    def embed(self, text: str, text_type: str = "query") -> list[float]:
        text = text.lower()
        if "alpha" in text or "blue" in text:
            return [1.0, 0.0, 0.0, 0.0]
        if "beta" in text or "red" in text:
            return [0.0, 1.0, 0.0, 0.0]
        return [0.5, 0.5, 0.0, 0.0]

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
        text_type: str = "query",
    ) -> list[list[float]]:
        return [self.embed(text, text_type=text_type) for text in texts]


class FakeEntityExtractor:
    """Deterministic entity extractor for query-time graph retrieval."""

    def extract(self, question: str) -> list[str]:
        return ["alpha"]


def create_test_rag(milvus_uri: str, collection_prefix: str) -> VectorGraphRAG:
    """Create a VectorGraphRAG instance with fake embeddings and no LLM calls."""
    settings = Settings(
        milvus_uri=milvus_uri,
        openai_api_key="test-api-key",
        collection_prefix=collection_prefix,
        final_top_k=3,
    )
    fake_embedding = FakeEmbeddingModel()

    with patch("vector_graph_rag.rag.EmbeddingModel", return_value=fake_embedding):
        rag = VectorGraphRAG(settings=settings)

    rag._answer_generator.generate = MagicMock(
        side_effect=lambda question, passages: "\n".join(passages)
    )
    return rag


def test_add_texts_stores_custom_metadata():
    """Store metadata passed through add_texts."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        milvus_uri = f.name

    try:
        rag = create_test_rag(milvus_uri, "metadata_filter_add_texts")

        rag.add_texts(
            ["Alpha metadata passage."],
            ids=["doc_alpha"],
            metadatas=[{"tenant_id": "team_a", "source": "add_texts"}],
            extract_triplets=False,
            show_progress=False,
        )

        assert rag._store.query_passage_ids('source == "add_texts"') == ["doc_alpha"]
    finally:
        if os.path.exists(milvus_uri):
            os.unlink(milvus_uri)


def test_add_documents_stores_custom_metadata():
    """Store metadata passed through Document.metadata."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        milvus_uri = f.name

    try:
        rag = create_test_rag(milvus_uri, "metadata_filter_add_documents")

        rag.add_documents(
            [
                Document(
                    page_content="Alpha document metadata passage.",
                    metadata={"tenant_id": "team_a", "source": "add_documents"},
                    id="doc_alpha",
                )
            ],
            extract_triplets=False,
            show_progress=False,
        )

        assert rag._store.query_passage_ids('source == "add_documents"') == ["doc_alpha"]
    finally:
        if os.path.exists(milvus_uri):
            os.unlink(milvus_uri)


def test_query_filter_uses_custom_metadata_end_to_end():
    """Store document metadata and use it to filter graph query results."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        milvus_uri = f.name

    try:
        rag = create_test_rag(milvus_uri, "metadata_filter_e2e")

        rag.add_documents_with_triplets(
            [
                {
                    "id": "doc_alpha",
                    "passage": "Alpha owns the blue database.",
                    "triplets": [["Alpha", "owns", "blue database"]],
                    "metadata": {"tenant_id": "team_a", "source": "alpha_source"},
                },
                {
                    "id": "doc_beta",
                    "passage": "Alpha owns the red database.",
                    "triplets": [["Alpha", "owns", "red database"]],
                    "metadata": {"tenant_id": "team_b", "source": "beta_source"},
                },
            ],
            show_progress=False,
        )
        rag._retriever = GraphRetriever(
            store=rag._store,
            graph_builder=rag._graph_builder,
            settings=rag.settings,
            embedding_model=rag._embedding_model,
            entity_extractor=FakeEntityExtractor(),
        )

        stored = rag._store.get_passages_by_ids(["doc_alpha", "doc_beta"])
        assert len(stored) == 2
        assert set(rag._store.query_passage_ids('tenant_id == "team_a"')) == {"doc_alpha"}

        result = rag.query(
            "What database does Alpha own?",
            use_reranking=False,
            filter='tenant_id == "team_a"',
        )

        assert result.passages == ["Alpha owns the blue database."]
        assert result.retrieved_passages == ["Alpha owns the blue database."]
        assert "red database" not in result.answer
    finally:
        if os.path.exists(milvus_uri):
            os.unlink(milvus_uri)
