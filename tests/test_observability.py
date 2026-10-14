import os
import tempfile
from contextlib import suppress
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents import Document
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from vector_graph_rag.config import Settings
from vector_graph_rag.observability import (
    get_observability_attributes,
    observability_context,
    start_span,
)
from vector_graph_rag.rag import VectorGraphRAG

_SPAN_EXPORTER = InMemorySpanExporter()


def _attach_span_exporter(exporter: InMemorySpanExporter) -> None:
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


_attach_span_exporter(_SPAN_EXPORTER)


@pytest.fixture(autouse=True)
def clear_finished_spans():
    _SPAN_EXPORTER.clear()
    yield
    _SPAN_EXPORTER.clear()


def _finished_spans():
    return list(_SPAN_EXPORTER.get_finished_spans())


def _span_by_name(name: str):
    for span in _finished_spans():
        if span.name == name:
            return span
    raise AssertionError(f"Span not found: {name}")


class FakeEmbeddingBackend:
    dimension = 8

    def encode(self, texts, text_type="query"):
        del text_type
        if isinstance(texts, str):
            texts = [texts]
        return np.array([[0.1] * self.dimension for _ in texts])


def _fake_response(content: str) -> MagicMock:
    response = MagicMock()
    response.status = "completed"
    response.output = [
        MagicMock(type="message", content=[MagicMock(type="output_text", text=content)])
    ]
    return response


def _create_test_rag(collection_prefix: str) -> tuple[VectorGraphRAG, str]:
    temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_file.close()
    os.unlink(temp_file.name)
    settings = Settings(
        openai_api_key="test-key",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimension=8,
        milvus_uri=temp_file.name,
        collection_prefix=collection_prefix,
        use_llm_cache=False,
    )
    with patch(
        "vector_graph_rag.storage.embeddings.get_provider",
        return_value=FakeEmbeddingBackend(),
    ):
        rag = VectorGraphRAG(settings=settings)
    rag._answer_generator.client.responses.create = MagicMock(
        return_value=_fake_response("Alpha owns Beta.")
    )
    return rag, temp_file.name


def _remove_temp_milvus_file(milvus_uri: str) -> None:
    for path in [milvus_uri, f"{milvus_uri}.lock"]:
        if os.path.exists(path):
            os.remove(path)


def _cleanup_test_rag(rag: VectorGraphRAG, milvus_uri: str) -> None:
    with suppress(Exception):
        rag._store.drop_collections()
    close = getattr(rag._store.client, "close", None)
    if callable(close):
        close()
    _remove_temp_milvus_file(milvus_uri)


def test_observability_context_sets_attributes_on_spans():
    with observability_context(
        request_id="req-123",
        tenant_id="tenant-a",
        graph_name="finance",
        source="file-123",
        attributes={"workspace_id": "workspace-a", "ignored": None},
    ):
        with start_span("vgrag.test", {"vgrag.record_count": 2, "unsafe": {"a": 1}}):
            pass

    span = _span_by_name("vgrag.test")
    assert span.attributes["vgrag.request_id"] == "req-123"
    assert span.attributes["vgrag.tenant_id"] == "tenant-a"
    assert span.attributes["vgrag.graph_name"] == "finance"
    assert span.attributes["vgrag.source"] == "file-123"
    assert span.attributes["workspace_id"] == "workspace-a"
    assert "ignored" not in span.attributes
    assert span.attributes["vgrag.record_count"] == 2
    assert span.attributes["unsafe"] == "{'a': 1}"
    assert get_observability_attributes() == {}


def test_observability_helpers_are_noops_when_opentelemetry_is_unavailable(monkeypatch):
    import vector_graph_rag.observability as observability

    monkeypatch.setattr(observability, "trace", None)

    with observability.observability_context(request_id="req-noop"):
        assert observability.get_observability_attributes()["vgrag.request_id"] == "req-noop"
        with observability.start_span("vgrag.noop", {"vgrag.record_count": 1}) as span:
            assert span is None

    assert observability.get_observability_attributes() == {}


def test_rag_incremental_upsert_emits_source_context_and_milvus_spans():
    rag, milvus_uri = _create_test_rag("otel_incremental_upsert")
    try:
        documents = [
            Document(
                page_content="Alpha owns Beta.",
                id="chunk-1",
                metadata={
                    "source": "file-alpha",
                    "triplets": [["Alpha", "owns", "Beta"]],
                },
            )
        ]

        with observability_context(request_id="req-upsert", tenant_id="tenant-a"):
            result = rag.upsert_documents_by_source(
                documents,
                source="file-alpha",
                extract_triplets=False,
                show_progress=False,
            )

        assert len(result.documents) == 1

        span_names = {span.name for span in _finished_spans()}
        assert "vgrag.upsert_documents_by_source" in span_names
        assert "vgrag.delete_documents_by_source" in span_names
        assert "vgrag.graph.build" in span_names
        assert "vgrag.graph.insert_incremental" in span_names
        assert "vgrag.embedding.batch" in span_names
        assert "vgrag.milvus.insert" in span_names
        assert "vgrag.milvus.query" in span_names

        operation_span = _span_by_name("vgrag.upsert_documents_by_source")
        assert operation_span.attributes["vgrag.request_id"] == "req-upsert"
        assert operation_span.attributes["vgrag.tenant_id"] == "tenant-a"
        assert operation_span.attributes["vgrag.source"] == "file-alpha"
        assert operation_span.attributes["vgrag.document_count"] == 1
        assert operation_span.attributes["vgrag.extract_triplets"] is False

        nested_milvus_span = _span_by_name("vgrag.milvus.insert")
        assert nested_milvus_span.attributes["vgrag.request_id"] == "req-upsert"
        assert nested_milvus_span.attributes["vgrag.tenant_id"] == "tenant-a"
        assert nested_milvus_span.attributes["db.system"] == "milvus"
    finally:
        _cleanup_test_rag(rag, milvus_uri)


def test_rag_query_emits_trace_without_query_text_or_answer():
    rag, milvus_uri = _create_test_rag("otel_query")
    try:
        rag.upsert_documents_by_source(
            [
                Document(
                    page_content="Alpha owns Beta.",
                    id="chunk-1",
                    metadata={
                        "source": "file-alpha",
                        "triplets": [["Alpha", "owns", "Beta"]],
                    },
                )
            ],
            source="file-alpha",
            extract_triplets=False,
            show_progress=False,
        )
        retriever = rag._ensure_retriever()
        retriever.entity_extractor.extract = MagicMock(return_value=["alpha"])
        _SPAN_EXPORTER.clear()

        with observability_context(request_id="req-query", tenant_id="tenant-a"):
            result = rag.query(
                "What does Alpha own?",
                use_reranking=False,
                filter='source == "file-alpha"',
            )

        assert result.answer == "Alpha owns Beta."

        span_names = {span.name for span in _finished_spans()}
        assert "vgrag.query" in span_names
        assert "vgrag.retrieve.graph" in span_names
        assert "vgrag.retrieve.entities" in span_names
        assert "vgrag.retrieve.relations" in span_names
        assert "vgrag.subgraph.expand" in span_names
        assert "vgrag.retrieve.passages_from_relations" in span_names
        assert "vgrag.answer" in span_names

        query_span = _span_by_name("vgrag.query")
        assert query_span.attributes["vgrag.request_id"] == "req-query"
        assert query_span.attributes["vgrag.tenant_id"] == "tenant-a"
        assert query_span.attributes["vgrag.has_filter"] is True
        assert query_span.attributes["vgrag.question_length"] == len("What does Alpha own?")

        for span in _finished_spans():
            attribute_values = [str(value) for value in span.attributes.values()]
            assert "What does Alpha own?" not in attribute_values
            assert "Alpha owns Beta." not in attribute_values
    finally:
        _cleanup_test_rag(rag, milvus_uri)
