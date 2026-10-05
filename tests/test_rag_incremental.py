"""End-to-end tests for incremental source updates in VectorGraphRAG."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from vector_graph_rag.config import Settings
from vector_graph_rag.graph.builder import GraphBuilder
from vector_graph_rag.graph.retriever import GraphRetriever
from vector_graph_rag.models import Document
from vector_graph_rag.rag import VectorGraphRAG


class FakeEmbeddingModel:
    """Small deterministic embedding model for Milvus Lite tests."""

    dimension = 4

    def embed(self, text: str, text_type: str = "query") -> list[float]:
        text = text.lower()
        if "alpha" in text or "blue" in text or "green" in text:
            return [1.0, 0.0, 0.0, 0.0]
        if "beta" in text or "red" in text:
            return [0.0, 1.0, 0.0, 0.0]
        if "gamma" in text or "yellow" in text:
            return [0.0, 0.0, 1.0, 0.0]
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
        embedding_provider="openai",
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


def with_temp_rag(collection_prefix: str):
    """Create a temporary Milvus Lite backed RAG instance."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_file.close()
    rag = create_test_rag(temp_file.name, collection_prefix)
    return rag, temp_file.name


def remove_temp_milvus_file(milvus_uri: str) -> None:
    """Remove a temporary Milvus Lite file if it exists."""
    if os.path.exists(milvus_uri):
        os.unlink(milvus_uri)


def doc(
    text: str,
    triplets: list[list[str]],
    id: str | None = None,
    metadata: dict | None = None,
) -> Document:
    """Build a test document with pre-extracted triplets."""
    return Document(
        page_content=text,
        metadata={"triplets": triplets, **(metadata or {})},
        id=id,
    )


def test_upsert_documents_by_source_adds_without_rebuilding_existing_sources():
    """Upsert a new source without removing sources already in the graph."""
    rag, milvus_uri = with_temp_rag("incremental_add")

    try:
        rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the blue database.",
                    [["Alpha", "owns", "blue database"]],
                )
            ],
            source="file_alpha",
            extract_triplets=False,
            show_progress=False,
        )
        rag.upsert_documents_by_source(
            [
                doc(
                    "Beta owns the red database.",
                    [["Beta", "owns", "red database"]],
                )
            ],
            source="file_beta",
            extract_triplets=False,
            show_progress=False,
        )

        alpha_passages = rag._store.get_passages_by_source("file_alpha")
        beta_passages = rag._store.get_passages_by_source("file_beta")

        assert [p["text"] for p in alpha_passages] == ["Alpha owns the blue database."]
        assert [p["text"] for p in beta_passages] == ["Beta owns the red database."]
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_upsert_documents_by_source_infers_source_and_sets_chunk_metadata():
    """Infer source from metadata and persist source/chunk metadata."""
    rag, milvus_uri = with_temp_rag("incremental_infer_source")

    try:
        result = rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the blue database.",
                    [["Alpha", "owns", "blue database"]],
                    id="alpha_chunk_0",
                    metadata={"source": "file_alpha", "page": 3},
                )
            ],
            extract_triplets=False,
            show_progress=False,
        )

        passages = rag._store.get_passages_by_ids(
            ["alpha_chunk_0"],
            output_fields=[
                "id",
                "text",
                "source",
                "page",
                "chunk_index",
                "document_id",
                "chunk_id",
            ],
        )

        assert result.documents[0].id == "alpha_chunk_0"
        assert result.documents[0].metadata["source"] == "file_alpha"
        assert result.documents[0].metadata["chunk_index"] == 0
        assert passages == [
            {
                "id": "alpha_chunk_0",
                "text": "Alpha owns the blue database.",
                "source": "file_alpha",
                "page": 3,
                "chunk_index": 0,
            }
        ]
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_limit_dynamic_field_metadata_keeps_the_newest_adjacency_ids(caplog):
    """Trim oldest links while preserving user metadata and logging the change."""
    rag = object.__new__(VectorGraphRAG)
    rag.settings = Settings(max_dynamic_field_bytes=500)
    metadata = {
        "source": "file_alpha",
        "relation_ids": [f"relation_{index:03d}" for index in range(100)],
        "passage_ids": [f"passage_{index:03d}" for index in range(100)],
    }

    with caplog.at_level("WARNING", logger="vector_graph_rag.rag"):
        bounded = rag._limit_dynamic_field_metadata(
            metadata,
            ["relation_ids", "passage_ids"],
            "entity",
        )

    assert bounded["source"] == "file_alpha"
    assert bounded["relation_ids"][-1] == metadata["relation_ids"][-1]
    assert bounded["passage_ids"][-1] == metadata["passage_ids"][-1]
    assert len(json.dumps(bounded, ensure_ascii=True).encode("utf-8")) <= 500
    assert len(bounded["relation_ids"]) < len(metadata["relation_ids"])
    assert len(bounded["passage_ids"]) < len(metadata["passage_ids"])
    assert "Trimmed" in caplog.text
    assert len(metadata["relation_ids"]) == 100


def test_incremental_insert_caps_merged_hub_metadata_without_milvus():
    """Exercise the existing-record update path with a mocked store."""
    rag = object.__new__(VectorGraphRAG)
    rag.settings = Settings(max_dynamic_field_bytes=60_000)
    rag._store = MagicMock()
    existing_relation_ids = [f"old_relation_{index:04d}" for index in range(1000)]
    existing_passage_ids = [f"old_passage_{index:04d}" for index in range(1000)]
    rag._store._get_entities_by_texts.return_value = {
        "speaker": {
            "id": "speaker-existing",
            "relation_ids": existing_relation_ids,
            "passage_ids": existing_passage_ids,
        }
    }
    rag._store._get_relations_by_texts.return_value = {}
    rag._store._get_relations_by_ids.return_value = []
    rag._store.get_passages_by_ids.return_value = []

    documents = [
        doc(
            "Second batch of speaker notes.",
            [["Speaker", "mentioned", f"topic_{index:04d}"] for index in range(1000)],
            id="speaker_second",
            metadata={"source": "file_second"},
        )
    ]
    builder = GraphBuilder(settings=rag.settings)
    builder.build_from_documents(documents)
    passage_user_metadatas = {
        item.id: rag._get_user_passage_metadata(item) for item in documents if item.id is not None
    }
    rag._insert_incremental_graph(
        builder,
        passage_user_metadatas,
        [[0.0] * 4 for _ in builder.entity_ids],
        [[0.0] * 4 for _ in builder.relation_ids],
        [[0.0] * 4 for _ in builder.passage_ids],
        source="file_second",
        source_field="source",
        show_progress=False,
    )

    updated_entity = rag._store._upsert_entity_records.call_args.args[0][0]
    speaker_id = next(eid for eid, name in builder.entities.items() if name == "speaker")
    expected_relation_ids = [
        *existing_relation_ids,
        *builder.entity_to_relation_ids[speaker_id],
    ]
    assert (
        updated_entity["relation_ids"]
        == expected_relation_ids[-len(updated_entity["relation_ids"]) :]
    )
    assert updated_entity["passage_ids"][-1] == "speaker_second"
    assert (
        len(
            json.dumps(
                {
                    "relation_ids": updated_entity["relation_ids"],
                    "passage_ids": updated_entity["passage_ids"],
                },
                ensure_ascii=True,
            ).encode("utf-8")
        )
        <= rag.settings.max_dynamic_field_bytes
    )


def test_incremental_upsert_caps_merged_hub_metadata():
    """Keep a high-degree entity below Milvus's dynamic-field byte limit."""
    rag, milvus_uri = with_temp_rag("incremental_metadata_limit")

    try:
        first_triplets = [["Speaker", "mentioned", f"topic_{index:04d}"] for index in range(1000)]
        second_triplets = [
            ["Speaker", "mentioned", f"topic_{index:04d}"] for index in range(1000, 2000)
        ]
        first_result = rag.upsert_documents_by_source(
            [doc("First batch of speaker notes.", first_triplets, id="speaker_first")],
            source="file_first",
            extract_triplets=False,
            show_progress=False,
        )
        second_result = rag.upsert_documents_by_source(
            [doc("Second batch of speaker notes.", second_triplets, id="speaker_second")],
            source="file_second",
            extract_triplets=False,
            show_progress=False,
        )

        entity = rag._store._get_entities_by_texts(["speaker"])["speaker"]
        expected_relation_ids = [
            *first_result.entity_to_relation_ids[entity["id"]],
            *second_result.entity_to_relation_ids[entity["id"]],
        ]
        assert len(expected_relation_ids) == 2000
        assert entity["relation_ids"] == expected_relation_ids[-len(entity["relation_ids"]) :]
        assert len(entity["relation_ids"]) < len(expected_relation_ids)
        entity_metadata = {
            "relation_ids": entity["relation_ids"],
            "passage_ids": entity["passage_ids"],
        }
        assert (
            len(json.dumps(entity_metadata, ensure_ascii=True).encode("utf-8"))
            <= rag.settings.max_dynamic_field_bytes
        )
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_upsert_documents_by_source_replaces_only_target_source_and_cleans_orphans():
    """Replace one source and clean graph records that only belonged to it."""
    rag, milvus_uri = with_temp_rag("incremental_replace")

    try:
        rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the blue database.",
                    [["Alpha", "owns", "blue database"]],
                )
            ],
            source="file_alpha",
            extract_triplets=False,
            show_progress=False,
        )
        alpha_passage_id = rag._store.get_passages_by_source("file_alpha")[0]["id"]

        rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the red database.",
                    [["Alpha", "owns", "red database"]],
                )
            ],
            source="file_beta",
            extract_triplets=False,
            show_progress=False,
        )

        rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the green database.",
                    [["Alpha", "owns", "green database"]],
                )
            ],
            source="file_alpha",
            extract_triplets=False,
            show_progress=False,
        )

        alpha_passages = rag._store.get_passages_by_source("file_alpha")
        beta_passages = rag._store.get_passages_by_source("file_beta")
        assert [p["text"] for p in alpha_passages] == ["Alpha owns the green database."]
        assert [p["text"] for p in beta_passages] == ["Alpha owns the red database."]
        assert alpha_passages[0]["id"] == alpha_passage_id

        relations = rag._store._get_relations_by_texts(
            [
                "alpha owns blue database",
                "alpha owns red database",
                "alpha owns green database",
            ]
        )
        assert "alpha owns blue database" not in relations
        assert set(relations) == {
            "alpha owns red database",
            "alpha owns green database",
        }

        alpha_entity = rag._store._get_entities_by_texts(["alpha"])["alpha"]
        assert set(alpha_entity["passage_ids"]) == {alpha_passages[0]["id"], beta_passages[0]["id"]}
        assert set(alpha_entity["relation_ids"]) == {
            relations["alpha owns red database"]["id"],
            relations["alpha owns green database"]["id"],
        }
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_delete_documents_by_source_cascades_and_preserves_shared_graph_records():
    """Delete one source while preserving shared relation/entity records."""
    rag, milvus_uri = with_temp_rag("incremental_delete")

    try:
        shared_triplet = [["Alpha", "founded", "Acme"]]
        rag.upsert_documents_by_source(
            [doc("Alpha founded Acme in source A.", shared_triplet)],
            source="file_alpha",
            extract_triplets=False,
            show_progress=False,
        )
        rag.upsert_documents_by_source(
            [doc("Alpha founded Acme in source B.", shared_triplet)],
            source="file_beta",
            extract_triplets=False,
            show_progress=False,
        )

        beta_passage_id = rag._store.get_passages_by_source("file_beta")[0]["id"]
        relation = rag._store._get_relations_by_texts(["alpha founded acme"])["alpha founded acme"]
        assert len(relation["passage_ids"]) == 2

        assert rag.delete_documents_by_source("file_alpha") is True
        assert rag._store.get_passages_by_source("file_alpha") == []

        relation = rag._store._get_relations_by_texts(["alpha founded acme"])["alpha founded acme"]
        assert relation["passage_ids"] == [beta_passage_id]
        alpha_entity = rag._store._get_entities_by_texts(["alpha"])["alpha"]
        assert alpha_entity["passage_ids"] == [beta_passage_id]
        assert alpha_entity["relation_ids"] == [relation["id"]]

        assert rag.delete_documents_by_source("file_beta") is True
        assert rag._store._get_relations_by_texts(["alpha founded acme"]) == {}
        assert rag._store._get_entities_by_texts(["alpha"]) == {}
        assert rag.delete_documents_by_source("file_missing") is False
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_delete_documents_by_source_retry_cleans_relations_deleted_before_entity_cleanup():
    """Retry a source delete after relation deletion succeeded but entity cleanup failed."""
    rag, milvus_uri = with_temp_rag("incremental_retry_delete_missing_relation")

    try:
        rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the blue database.",
                    [["Alpha", "owns", "blue database"]],
                )
            ],
            source="file_alpha",
            extract_triplets=False,
            show_progress=False,
        )
        rag.upsert_documents_by_source(
            [
                doc(
                    "Beta owns the red database.",
                    [["Beta", "owns", "red database"]],
                )
            ],
            source="file_beta",
            extract_triplets=False,
            show_progress=False,
        )

        alpha_relation = rag._store._get_relations_by_texts(["alpha owns blue database"])[
            "alpha owns blue database"
        ]
        original_delete_relations = rag._store._delete_relations
        failed_once = False

        def fail_after_relation_delete(relation_ids: list[str]) -> int:
            nonlocal failed_once
            deleted_count = original_delete_relations(relation_ids)
            if not failed_once:
                failed_once = True
                raise RuntimeError("simulated relation delete failure")
            return deleted_count

        with patch.object(
            rag._store,
            "_delete_relations",
            side_effect=fail_after_relation_delete,
        ):
            with pytest.raises(RuntimeError, match="simulated relation delete failure"):
                rag.delete_documents_by_source("file_alpha")

        assert rag._store._get_relations_by_texts(["alpha owns blue database"]) == {}
        alpha_entity = rag._store._get_entities_by_texts(["alpha"])["alpha"]
        assert alpha_relation["id"] in alpha_entity["relation_ids"]

        assert rag.delete_documents_by_source("file_alpha") is True

        assert rag._store.get_passages_by_source("file_alpha") == []
        assert rag._store.get_passages_by_source("file_beta")
        assert rag._store._get_relations_by_texts(["alpha owns blue database"]) == {}
        assert rag._store._get_entities_by_texts(["alpha", "blue database"]) == {}
        assert set(rag._store._get_relations_by_texts(["beta owns red database"])) == {
            "beta owns red database"
        }
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_upsert_documents_by_source_retry_prunes_relation_ids_missing_after_partial_insert():
    """Retry a source upsert after entities were inserted before relation insert failed."""
    rag, milvus_uri = with_temp_rag("incremental_retry_upsert_entity_only")

    try:
        rag.upsert_documents_by_source(
            [
                doc(
                    "Beta owns the red database.",
                    [["Beta", "owns", "red database"]],
                )
            ],
            source="file_beta",
            extract_triplets=False,
            show_progress=False,
        )

        original_insert_entities = rag._store._insert_entities
        failed_once = False

        def fail_after_entity_insert(*args, **kwargs):
            nonlocal failed_once
            inserted_ids = original_insert_entities(*args, **kwargs)
            if not failed_once:
                failed_once = True
                raise RuntimeError("simulated entity insert failure")
            return inserted_ids

        alpha_doc = doc(
            "Alpha owns the blue database.",
            [["Alpha", "owns", "blue database"]],
        )

        with patch.object(
            rag._store,
            "_insert_entities",
            side_effect=fail_after_entity_insert,
        ):
            with pytest.raises(RuntimeError, match="simulated entity insert failure"):
                rag.upsert_documents_by_source(
                    [alpha_doc],
                    source="file_alpha",
                    extract_triplets=False,
                    show_progress=False,
                )

        assert rag._store.get_passages_by_source("file_alpha") == []
        alpha_entity = rag._store._get_entities_by_texts(["alpha"])["alpha"]
        stale_relation_ids = alpha_entity["relation_ids"]
        assert stale_relation_ids
        assert rag._store._get_relations_by_ids(stale_relation_ids) == []

        rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the blue database.",
                    [["Alpha", "owns", "blue database"]],
                )
            ],
            source="file_alpha",
            extract_triplets=False,
            show_progress=False,
        )

        alpha_relation = rag._store._get_relations_by_texts(["alpha owns blue database"])[
            "alpha owns blue database"
        ]
        alpha_entity = rag._store._get_entities_by_texts(["alpha"])["alpha"]

        assert set(alpha_entity["relation_ids"]) == {alpha_relation["id"]}
        assert rag._store.get_passages_by_source("file_alpha")
        assert rag._store.get_passages_by_source("file_beta")
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_upsert_documents_by_source_retry_replaces_partial_passage_insert():
    """Retry a source upsert after the new passage was inserted before failure."""
    rag, milvus_uri = with_temp_rag("incremental_retry_upsert_passage_insert")

    try:
        rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the blue database.",
                    [["Alpha", "owns", "blue database"]],
                )
            ],
            source="file_alpha",
            extract_triplets=False,
            show_progress=False,
        )
        rag.upsert_documents_by_source(
            [
                doc(
                    "Beta owns the red database.",
                    [["Beta", "owns", "red database"]],
                )
            ],
            source="file_beta",
            extract_triplets=False,
            show_progress=False,
        )

        original_insert_passages = rag._store.insert_passages
        failed_once = False

        def fail_after_passage_insert(*args, **kwargs):
            nonlocal failed_once
            inserted_ids = original_insert_passages(*args, **kwargs)
            if not failed_once:
                failed_once = True
                raise RuntimeError("simulated passage insert failure")
            return inserted_ids

        replacement_doc = doc(
            "Alpha owns the green database.",
            [["Alpha", "owns", "green database"]],
        )

        with patch.object(
            rag._store,
            "insert_passages",
            side_effect=fail_after_passage_insert,
        ):
            with pytest.raises(RuntimeError, match="simulated passage insert failure"):
                rag.upsert_documents_by_source(
                    [replacement_doc],
                    source="file_alpha",
                    extract_triplets=False,
                    show_progress=False,
                )

        assert [p["text"] for p in rag._store.get_passages_by_source("file_alpha")] == [
            "Alpha owns the green database."
        ]

        rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the green database.",
                    [["Alpha", "owns", "green database"]],
                )
            ],
            source="file_alpha",
            extract_triplets=False,
            show_progress=False,
        )

        assert [p["text"] for p in rag._store.get_passages_by_source("file_alpha")] == [
            "Alpha owns the green database."
        ]
        assert [p["text"] for p in rag._store.get_passages_by_source("file_beta")] == [
            "Beta owns the red database."
        ]
        assert rag._store._get_relations_by_texts(["alpha owns blue database"]) == {}
        assert set(rag._store._get_relations_by_texts(["alpha owns green database"])) == {
            "alpha owns green database"
        }
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_incremental_exact_text_lookups_are_batched():
    """Batch exact entity/relation lookups instead of querying one text at a time."""
    rag, milvus_uri = with_temp_rag("incremental_batched_lookup")
    rag.settings.batch_size = 2

    try:
        rag._store._insert_entities(
            ["alpha", "beta", "gamma"],
            ids=["entity_alpha", "entity_beta", "entity_gamma"],
            embeddings=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        )
        rag._store._insert_relations(
            [
                "alpha owns beta",
                "beta owns gamma",
                "gamma owns alpha",
            ],
            ids=["relation_alpha_beta", "relation_beta_gamma", "relation_gamma_alpha"],
            embeddings=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        )

        with patch.object(rag._store.client, "query", wraps=rag._store.client.query) as query_spy:
            entities = rag._store._get_entities_by_texts(
                ["alpha", "beta", "alpha", "gamma", "missing"]
            )

        assert set(entities) == {"alpha", "beta", "gamma"}
        assert query_spy.call_count == 2
        assert all("text in [" in call.kwargs["filter"] for call in query_spy.call_args_list)

        with patch.object(rag._store.client, "query", wraps=rag._store.client.query) as query_spy:
            relations = rag._store._get_relations_by_texts(
                [
                    "alpha owns beta",
                    "beta owns gamma",
                    "alpha owns beta",
                    "gamma owns alpha",
                    "missing relation",
                ]
            )

        assert set(relations) == {
            "alpha owns beta",
            "beta owns gamma",
            "gamma owns alpha",
        }
        assert query_spy.call_count == 2
        assert all("text in [" in call.kwargs["filter"] for call in query_spy.call_args_list)
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_upsert_documents_by_source_batches_shared_graph_metadata_updates():
    """Reuse shared graph records without per-record ID lookups on incremental insert."""
    rag, milvus_uri = with_temp_rag("incremental_batched_upsert")

    try:
        shared_triplet = [["Alpha", "founded", "Acme"]]
        rag.upsert_documents_by_source(
            [doc("Alpha founded Acme in source A.", shared_triplet)],
            source="file_alpha",
            extract_triplets=False,
            show_progress=False,
        )

        with (
            patch.object(
                rag._store,
                "_get_entities_by_ids",
                wraps=rag._store._get_entities_by_ids,
            ) as entity_lookup_spy,
            patch.object(
                rag._store,
                "_get_relations_by_ids",
                wraps=rag._store._get_relations_by_ids,
            ) as relation_lookup_spy,
            patch.object(rag._store.client, "upsert", wraps=rag._store.client.upsert) as upsert_spy,
        ):
            rag.upsert_documents_by_source(
                [doc("Alpha founded Acme in source B.", shared_triplet)],
                source="file_beta",
                extract_triplets=False,
                show_progress=False,
            )

        assert entity_lookup_spy.call_count == 0
        assert relation_lookup_spy.call_count == 0

        entity_upserts = [
            call
            for call in upsert_spy.call_args_list
            if call.kwargs["collection_name"] == rag._store.entity_collection
        ]
        relation_upserts = [
            call
            for call in upsert_spy.call_args_list
            if call.kwargs["collection_name"] == rag._store.relation_collection
        ]
        assert len(entity_upserts) == 1
        assert len(entity_upserts[0].kwargs["data"]) == 2
        assert len(relation_upserts) == 1
        assert len(relation_upserts[0].kwargs["data"]) == 1
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_upsert_documents_by_source_supports_metadata_filter_query_end_to_end():
    """Use upserted source metadata to filter graph query results."""
    rag, milvus_uri = with_temp_rag("incremental_query_filter")

    try:
        rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the blue database.",
                    [["Alpha", "owns", "blue database"]],
                )
            ],
            source="file_alpha",
            metadata={"tenant_id": "team_a"},
            extract_triplets=False,
            show_progress=False,
        )
        rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha owns the red database.",
                    [["Alpha", "owns", "red database"]],
                )
            ],
            source="file_beta",
            metadata={"tenant_id": "team_b"},
            extract_triplets=False,
            show_progress=False,
        )
        rag._retriever = GraphRetriever(
            store=rag._store,
            graph_builder=rag._graph_builder,
            settings=rag.settings,
            embedding_model=rag._embedding_model,
            entity_extractor=FakeEntityExtractor(),
        )

        result = rag.query(
            "What database does Alpha own?",
            use_reranking=False,
            filter='tenant_id == "team_a" and source == "file_alpha"',
        )

        assert result.passages == ["Alpha owns the blue database."]
        assert result.retrieved_passages == ["Alpha owns the blue database."]
        assert "red database" not in result.answer
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_upsert_documents_by_source_supports_custom_source_field():
    """Use a custom source metadata field for update and delete."""
    rag, milvus_uri = with_temp_rag("incremental_custom_source_field")

    try:
        rag.upsert_documents_by_source(
            [
                doc(
                    "Gamma owns the yellow database.",
                    [["Gamma", "owns", "yellow database"]],
                    metadata={"file_id": "file_gamma"},
                )
            ],
            source_field="file_id",
            extract_triplets=False,
            show_progress=False,
        )

        passages = rag._store.get_passages_by_source("file_gamma", source_field="file_id")
        assert [p["text"] for p in passages] == ["Gamma owns the yellow database."]
        assert rag.delete_documents_by_source("file_gamma", source_field="file_id") is True
        assert rag._store.get_passages_by_source("file_gamma", source_field="file_id") == []
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_upsert_documents_by_source_validates_source_contract():
    """Reject source-less, multi-source, conflicting, and unsafe source fields."""
    rag, milvus_uri = with_temp_rag("incremental_source_contract")

    try:
        with pytest.raises(ValueError, match='metadata\\["source"\\] or source'):
            rag.upsert_documents_by_source(
                [doc("Alpha owns blue.", [["Alpha", "owns", "blue"]])],
                extract_triplets=False,
                show_progress=False,
            )

        with pytest.raises(ValueError, match="expects one source per call"):
            rag.upsert_documents_by_source(
                [
                    doc(
                        "Alpha owns blue.",
                        [["Alpha", "owns", "blue"]],
                        metadata={"source": "file_alpha"},
                    ),
                    doc(
                        "Beta owns red.",
                        [["Beta", "owns", "red"]],
                        metadata={"source": "file_beta"},
                    ),
                ],
                extract_triplets=False,
                show_progress=False,
            )

        with pytest.raises(ValueError, match="differ from source"):
            rag.upsert_documents_by_source(
                [
                    doc(
                        "Alpha owns blue.",
                        [["Alpha", "owns", "blue"]],
                        metadata={"source": "file_beta"},
                    )
                ],
                source="file_alpha",
                extract_triplets=False,
                show_progress=False,
            )

        with pytest.raises(ValueError, match="simple metadata field name"):
            rag.upsert_documents_by_source(
                [doc("Alpha owns blue.", [["Alpha", "owns", "blue"]])],
                source="file_alpha",
                source_field="source.field",
                extract_triplets=False,
                show_progress=False,
            )

        with pytest.raises(ValueError, match="non-empty string"):
            rag.delete_documents_by_source(" ")
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_legacy_incremental_document_apis_raise_migration_errors():
    """Legacy document_id incremental APIs fail with explicit migration guidance."""
    rag, milvus_uri = with_temp_rag("incremental_legacy_errors")

    try:
        with pytest.raises(RuntimeError, match="upsert_documents_by_source"):
            rag.upsert_documents(
                document_id="file_alpha",
                documents=[doc("Alpha owns blue.", [["Alpha", "owns", "blue"]])],
                extract_triplets=False,
            )

        with pytest.raises(RuntimeError, match="delete_documents_by_source"):
            rag.delete_documents("file_alpha")
    finally:
        remove_temp_milvus_file(milvus_uri)
