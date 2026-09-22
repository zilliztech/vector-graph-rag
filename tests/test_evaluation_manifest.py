"""The evaluator selects exact source rows, including repeated public question IDs."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vector_graph_rag.config import Settings

spec = importlib.util.spec_from_file_location(
    "evaluation_entry", Path(__file__).parents[1] / "evaluation/evaluate.py"
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def evaluator():
    instance = object.__new__(evaluation.VectorGraphRAGEvaluator)
    instance.dataset_name = "hotpotqa"
    instance.top_k = 10
    instance.questions = [
        {"_id": "same", "question": "first", "supporting_facts": [["A", 0]]},
        {"_id": "same", "question": "second", "supporting_facts": [["B", 0]]},
    ]
    instance.rag = MagicMock()
    instance.rag.settings = Settings()
    instance.rag.retrieve.side_effect = lambda question, **kwargs: SimpleNamespace(
        retrieved_passages=["A\ntext" if question == "first" else "B\ntext"]
    )
    return instance


def write_manifest(tmp_path, rows):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"samples": {"hotpotqa": rows}}))
    return str(path)


def test_manifest_preserves_source_row_indices_and_order(tmp_path):
    instance = evaluator()
    path = write_manifest(
        tmp_path,
        [
            {"index": 1, "id": "same", "query": "second"},
            {"index": 0, "id": "same", "query": "first"},
        ],
    )
    result = instance.evaluate(sample_manifest=path, method="graph", k_list=[5, 10])
    assert [r["index"] for r in result["results"]] == [1, 0]
    assert result["recall_graph_rag"] == {5: 1, 10: 1}
    assert result["protocol"]["reranker_provider"] == "llm"


@pytest.mark.parametrize(
    "row",
    [
        {"index": -1, "id": "same", "query": "second"},
        {"index": 2, "id": "same", "query": "second"},
        {"index": 0, "id": "wrong", "query": "first"},
        {"index": 0, "id": "same", "query": "wrong"},
    ],
)
def test_manifest_rejects_dataset_drift_before_retrieval(tmp_path, row):
    instance = evaluator()
    with pytest.raises(ValueError):
        instance.evaluate(sample_manifest=write_manifest(tmp_path, [row]), method="graph")
    instance.rag.retrieve.assert_not_called()


def test_manifest_rejects_duplicate_rows_and_empty_selection(tmp_path):
    row = {"index": 0, "id": "same", "query": "first"}
    with pytest.raises(ValueError, match="repeated"):
        evaluator().evaluate(sample_manifest=write_manifest(tmp_path, [row, row]))
    with pytest.raises(ValueError, match="No evaluable"):
        evaluator().evaluate(max_samples=0)
