"""Regression coverage for ranked relation-to-passage expansion."""

from unittest.mock import MagicMock

from vector_graph_rag.rag import VectorGraphRAG


def test_passages_preserve_relation_rank_with_unordered_filtered_results():
    rag = object.__new__(VectorGraphRAG)
    rag._store = MagicMock()
    rag._store._get_relations_by_ids.return_value = [
        {"id": "low", "passage_ids": ["shared", "low_doc"]},
        {"id": "high", "passage_ids": ["high_doc", "shared", "filtered"]},
    ]
    rag._store.get_passages_by_ids.return_value = [
        {"id": "low_doc", "text": "Low-ranked evidence"},
        {"id": "shared", "text": "Shared evidence"},
        {"id": "high_doc", "text": "High-ranked evidence"},
    ]

    ids, texts = rag._get_passages_from_relations(
        ["high", "missing", "low", "high"], filter='tenant == "a"'
    )

    assert ids == ["high_doc", "shared", "low_doc"]
    assert texts == ["High-ranked evidence", "Shared evidence", "Low-ranked evidence"]
    rag._store.get_passages_by_ids.assert_called_once_with(
        ["high_doc", "shared", "filtered", "low_doc"], filter='tenant == "a"'
    )
