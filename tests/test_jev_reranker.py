"""Offline coverage for optional Jev scoring, batching, and pipeline integration."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from vector_graph_rag.config import Settings
from vector_graph_rag.llm.cache import LLMCache
from vector_graph_rag.llm.jev import JevReranker
from vector_graph_rag.rag import VectorGraphRAG


@pytest.fixture
def ranker():
    return JevReranker(Settings(jev_api_key="test-key", use_llm_cache=False))


def response(scores):
    return {"answers": {key: {"type": "noul", "noul": value} for key, value in scores.items()}}


def test_threshold_order_ties_and_empty_selection(ranker):
    ranker.score_relations = MagicMock(return_value={"a": 0.5, "b": 0.9, "c": 0.5, "d": 0.49})
    assert ranker.rerank("q", list("abcd"), list("ABCD")) == (["b", "a", "c"], ["B", "A", "C"])
    ranker.settings.jev_threshold = 1
    assert ranker.rerank("q", list("abcd"), list("ABCD")) == ([], [])


def test_empty_and_invalid_candidates(ranker):
    assert ranker.rerank("q", [], []) == ([], [])
    with pytest.raises(ValueError, match="equal lengths"):
        ranker.build_requests("q", ["a"], [])
    with pytest.raises(ValueError, match="unique"):
        ranker.build_requests("q", ["a", "a"], ["x", "y"])
    with pytest.raises(ValueError, match="too large"):
        ranker.build_requests("q", ["a"], ["long relation " * 24000])


def test_batching_preserves_all_context_and_target_text(ranker):
    ids = [str(i) for i in range(1000)]
    texts = [f"entity {i} connects to entity {i + 1}" for i in range(1000)]
    payloads = ranker.build_requests("Find a bridge", ids, texts)
    assert len(payloads) > 1
    keys = []
    for payload in payloads:
        assert payload["state"] == payloads[0]["state"]
        assert len(payload["state"]["candidate_graph"].splitlines()) == 1000
        assert ranker._tokens(payload) < 40000
        for key, question in payload["questions"].items():
            keys.append(key)
            assert texts[int(key[1:])] in question["instructions"]
            assert set(question["criteria"]) == {"true", "false"}
    assert keys == ["r" + rid for rid in ids]


@pytest.mark.parametrize(
    "answers",
    [
        {},
        {"ra": {"type": "noul", "noul": 0.5}, "extra": {}},
        {"ra": {"type": "noul", "noul": float("nan")}},
        {"ra": {"type": "noul", "noul": True}},
        {"ra": {"type": "noul", "noul": 1.1}},
        {"ra": {"type": "bool", "noul": 0.9}},
    ],
)
def test_malformed_response_is_an_error(ranker, answers):
    with pytest.raises(ValueError):
        ranker._validate_response({"questions": {"ra": {}}}, {"answers": answers})


def test_http_response_mapping_and_cache_threshold_reuse(ranker, tmp_path):
    ranker._cache = LLMCache(str(tmp_path))
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json=response({"rb": 0.9, "ra": 0.6}))

    client_class = httpx.Client

    def make_client(**kwargs):
        return client_class(transport=httpx.MockTransport(handler), **kwargs)

    with patch.object(ranker._httpx, "Client", side_effect=make_client):
        assert ranker.rerank("q", ["a", "b"], ["A", "B"])[0] == ["b", "a"]
    ranker.settings.jev_threshold = 0.8
    with patch.object(ranker._httpx, "Client", side_effect=AssertionError("No network expected")):
        assert ranker.rerank("q", ["a", "b"], ["A", "B"])[0] == ["b"]
    assert len(calls) == 1


@pytest.mark.parametrize("status, expected_calls", [(429, 3), (503, 3), (401, 1), (402, 1)])
def test_retry_and_error_policy(ranker, status, expected_calls):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with patch("vector_graph_rag.llm.jev.time.sleep"), pytest.raises(httpx.HTTPStatusError):
            ranker._request(client, {"questions": {"ra": {}}})
    assert len(calls) == expected_calls


def test_provider_selection_keeps_default_and_loads_jev_only_when_selected():
    with (
        patch("vector_graph_rag.rag.EmbeddingModel"),
        patch("vector_graph_rag.rag.MilvusStore"),
        patch("vector_graph_rag.rag.GraphBuilder"),
        patch("vector_graph_rag.rag.TripletExtractor"),
        patch("vector_graph_rag.rag.AnswerGenerator"),
        patch("vector_graph_rag.rag.LLMReranker") as llm,
    ):
        rag = VectorGraphRAG(Settings(openai_api_key="test-key"))
        assert rag._reranker is llm.return_value
        rag = VectorGraphRAG(
            Settings(openai_api_key="test-key", reranker_provider="jev", jev_api_key="test-key")
        )
        assert isinstance(rag._reranker, JevReranker)
        assert llm.call_count == 1


def test_query_jev_fallback_preserves_filter_and_graph_first():
    rag = object.__new__(VectorGraphRAG)
    rag.settings = Settings(reranker_provider="jev", final_top_k=3)
    retriever = MagicMock()
    result = retriever.retrieve.return_value
    result.expanded_relation_ids = ["r"]
    result.expanded_relation_texts = ["relation"]
    result.relation_texts = ["relation"]
    result.query_entities = []
    result.entity_ids = []
    result.entity_texts = []
    result.entity_scores = []
    result.relation_scores = []
    result.relation_ids = []
    result.subgraph = None
    result.eviction_occurred = False
    result.eviction_before_count = 1
    result.eviction_after_count = 1
    rag._ensure_retriever = MagicMock(return_value=retriever)
    rag._reranker = MagicMock()
    rag._reranker.rerank.return_value = (["r"], ["relation"])
    rag._get_passages_from_relations = MagicMock(return_value=(["p"], ["graph evidence"]))
    retriever.retrieve_passages_naive.return_value = [
        "graph evidence",
        "fallback one",
        "fallback two",
    ]
    rag._answer_generator = MagicMock()
    rag._answer_generator.generate.return_value = "answer"
    query = rag.query("q", filter='tenant == "a"')
    assert query.retrieved_passages == ["graph evidence", "fallback one", "fallback two"]
    retriever.retrieve_passages_naive.assert_called_once_with("q", top_k=3, filter='tenant == "a"')


def test_key_environment_precedence_and_redaction(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fallback-secret")
    monkeypatch.delenv("VGRAG_JEV_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.jev_api_key.get_secret_value() == "fallback-secret"
    monkeypatch.setenv("VGRAG_JEV_API_KEY", "explicit-secret")
    settings = Settings(_env_file=None)
    assert settings.jev_api_key.get_secret_value() == "explicit-secret"
    assert "explicit-secret" not in repr(settings)
    assert "explicit-secret" not in settings.model_dump_json()


def test_missing_key_and_invalid_configuration():
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        JevReranker(Settings(jev_api_key=None))
    for kwargs in (
        {"jev_threshold": 1.1},
        {"jev_max_concurrency": 0},
        {"reranker_provider": "unknown"},
    ):
        with pytest.raises(ValueError):
            Settings(**kwargs)
