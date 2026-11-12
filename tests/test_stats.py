"""Public knowledge-base statistics reflect persisted records."""

from unittest.mock import patch

import pytest
from pymilvus.exceptions import MilvusException

from tests.conftest import close_milvus_store
from tests.test_rag_incremental import create_test_rag, doc


@pytest.fixture
def rag_factory(tmp_path):
    instances = []

    def create(prefix="stats"):
        rag = create_test_rag(str(tmp_path / "stats.db"), prefix)
        instances.append(rag)
        return rag

    yield create

    for rag in reversed(instances):
        close_milvus_store(rag._store)


def test_get_stats_tracks_incremental_updates_and_shared_records(rag_factory):
    rag = rag_factory()
    assert rag.get_stats() == {"entities": 0, "relations": 0, "passages": 0}

    rag.upsert_documents_by_source(
        [
            doc("Alpha owns blue.", [["Alpha", "owns", "blue"]], id="a1"),
            doc("Alpha owns red.", [["Alpha", "owns", "red"]], id="a2"),
        ],
        source="A",
        extract_triplets=False,
        show_progress=False,
    )
    assert rag.get_stats() == {"entities": 3, "relations": 2, "passages": 2}

    for _ in range(2):
        rag.upsert_documents_by_source(
            [
                doc(
                    "Alpha and Beta own blue.",
                    [["Alpha", "owns", "blue"], ["Beta", "owns", "blue"]],
                    id="b1",
                )
            ],
            source="B",
            extract_triplets=False,
            show_progress=False,
        )
        assert rag.get_stats() == {"entities": 4, "relations": 3, "passages": 3}

    rag.upsert_documents_by_source(
        [doc("Alpha owns blue.", [["Alpha", "owns", "blue"]], id="a1")],
        source="A",
        extract_triplets=False,
        show_progress=False,
    )
    assert rag.get_stats() == {"entities": 3, "relations": 2, "passages": 2}

    rag.delete_documents_by_source("B")
    assert rag.get_stats() == {"entities": 2, "relations": 1, "passages": 1}
    rag.delete_documents_by_source("A")
    assert rag.get_stats() == {"entities": 0, "relations": 0, "passages": 0}


def test_get_stats_reads_existing_database_and_other_instances_writes(rag_factory):
    writer = rag_factory()
    writer.rebuild_documents(
        [doc("Alpha owns blue.", [["Alpha", "owns", "blue"]])],
        extract_triplets=False,
        show_progress=False,
    )
    close_milvus_store(writer._store)

    reader = rag_factory()
    assert reader.get_stats() == {"entities": 2, "relations": 1, "passages": 1}

    other_graph = rag_factory("other_graph")
    other_graph.rebuild_documents(
        [doc("Beta owns red.", [["Beta", "owns", "red"]])],
        extract_triplets=False,
        show_progress=False,
    )
    assert reader.get_stats() == {"entities": 2, "relations": 1, "passages": 1}

    writer = rag_factory()
    writer.rebuild_documents(
        [
            doc("Gamma owns yellow.", [["Gamma", "owns", "yellow"]]),
            doc("A passage with no extracted facts.", []),
        ],
        extract_triplets=False,
        show_progress=False,
    )
    assert reader.get_stats() == {"entities": 2, "relations": 1, "passages": 2}
    writer.reset()
    assert reader.get_stats() == {"entities": 0, "relations": 0, "passages": 0}
    assert other_graph.get_stats() == {"entities": 2, "relations": 1, "passages": 1}


def test_get_stats_uses_strong_counts_when_collection_metadata_is_stale(rag_factory):
    rag = rag_factory()
    rag.rebuild_documents(
        [doc("Alpha owns blue.", [["Alpha", "owns", "blue"]])],
        extract_triplets=False,
        show_progress=False,
    )
    client = rag._store.client
    with (
        patch.object(client, "get_collection_stats", return_value={"row_count": 0}),
        patch.object(client, "query", wraps=client.query) as query,
    ):
        assert rag.get_stats() == {"entities": 2, "relations": 1, "passages": 1}

    assert query.call_count == 3
    for call in query.call_args_list:
        assert call.kwargs["output_fields"] == ["count(*)"]
        assert call.kwargs["consistency_level"] == "Strong"


def test_get_stats_treats_missing_collections_as_empty(rag_factory):
    rag = rag_factory()
    rag.rebuild_documents(
        [doc("Alpha owns blue.", [["Alpha", "owns", "blue"]])],
        extract_triplets=False,
        show_progress=False,
    )
    rag._store.client.drop_collection(rag._store.relation_collection)
    assert rag.get_stats() == {"entities": 2, "relations": 0, "passages": 1}
    rag._store.drop_collections()
    assert rag.get_stats() == {"entities": 0, "relations": 0, "passages": 0}


@pytest.mark.parametrize("operation", ["has_collection", "query"])
def test_get_stats_propagates_storage_failures(rag_factory, operation):
    rag = rag_factory()
    with patch.object(
        rag._store.client,
        operation,
        side_effect=MilvusException(code=1, message="Statistics unavailable"),
    ):
        with pytest.raises(MilvusException, match="Statistics unavailable"):
            rag.get_stats()
