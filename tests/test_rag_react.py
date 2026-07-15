"""Tests for ReAct query loops in VectorGraphRAG."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from vector_graph_rag.config import Settings
from vector_graph_rag.graph.retriever import GraphRetriever
from vector_graph_rag.llm.react import ReActPlanner
from vector_graph_rag.models import Document, ReActAction
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
        if "beta" in question.lower():
            return ["beta"]
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


def create_live_llm_test_rag(milvus_uri: str, collection_prefix: str) -> VectorGraphRAG:
    """Create a test RAG with fake embeddings and live LLM components."""
    settings = Settings(
        milvus_uri=milvus_uri,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        llm_model=os.getenv("VGRAG_LLM_MODEL", "gpt-4o-mini"),
        embedding_provider="openai",
        collection_prefix=collection_prefix,
        final_top_k=2,
        use_llm_cache=False,
    )
    fake_embedding = FakeEmbeddingModel()

    with patch("vector_graph_rag.rag.EmbeddingModel", return_value=fake_embedding):
        rag = VectorGraphRAG(settings=settings)

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


def add_test_documents(rag: VectorGraphRAG) -> None:
    """Index small pre-extracted documents for ReAct tests."""
    rag.rebuild_documents(
        [
            Document(
                page_content="Alpha owns the blue database.",
                metadata={"triplets": [["Alpha", "owns", "blue database"]]},
                id="doc_alpha",
            ),
            Document(
                page_content="Beta owns the red database.",
                metadata={"triplets": [["Beta", "owns", "red database"]]},
                id="doc_beta",
            ),
        ],
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


def add_tenant_documents(rag: VectorGraphRAG) -> None:
    """Index small pre-extracted documents with tenant metadata."""
    rag.rebuild_documents(
        [
            Document(
                page_content="Alpha owns the blue database.",
                metadata={
                    "triplets": [["Alpha", "owns", "blue database"]],
                    "tenant_id": "team_a",
                },
                id="doc_alpha",
            ),
            Document(
                page_content="Beta owns the red database.",
                metadata={
                    "triplets": [["Beta", "owns", "red database"]],
                    "tenant_id": "team_b",
                },
                id="doc_beta",
            ),
        ],
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


def test_query_react_searches_then_finishes():
    """Run one retrieval step and then finish with the planner answer."""
    rag, milvus_uri = with_temp_rag("react_search_finish")

    try:
        add_test_documents(rag)
        rag._react_planner.plan = MagicMock(
            side_effect=[
                ReActAction(
                    thought="Need Alpha evidence.",
                    action="search",
                    query="What database does Alpha own?",
                ),
                ReActAction(
                    thought="The evidence is enough.",
                    action="finish",
                    answer="Alpha owns the blue database.",
                ),
            ]
        )

        result = rag.query_react(
            "What database does Alpha own?",
            max_steps=3,
            use_reranking=False,
        )

        assert result.answer == "Alpha owns the blue database."
        assert result.finished is True
        assert [step.action for step in result.steps] == ["search", "finish"]
        assert "Alpha owns the blue database." in result.steps[0].retrieved_passages
        assert "Passages:" in result.steps[0].observation
        assert "Alpha owns the blue database." in result.passages
        rag._answer_generator.generate.assert_not_called()
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_query_react_can_finish_without_searching():
    """Allow the planner to finish immediately without running retrieval."""
    rag, milvus_uri = with_temp_rag("react_finish_immediately")

    try:
        rag._react_planner.plan = MagicMock(
            return_value=ReActAction(
                thought="No retrieval needed.",
                action="finish",
                answer="Direct answer.",
            )
        )

        result = rag.query_react("Return a direct answer.", max_steps=3)

        assert result.answer == "Direct answer."
        assert result.finished is True
        assert len(result.steps) == 1
        assert result.steps[0].action == "finish"
        assert result.passages == []
        rag._answer_generator.generate.assert_not_called()
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_query_react_runs_multiple_searches_and_deduplicates_observations():
    """Run multiple search actions and keep unique observed passages."""
    rag, milvus_uri = with_temp_rag("react_multi_search")

    try:
        add_test_documents(rag)
        rag._react_planner.plan = MagicMock(
            side_effect=[
                ReActAction(thought="Search Alpha.", action="search", query="Alpha database"),
                ReActAction(thought="Search Beta.", action="search", query="Beta database"),
                ReActAction(
                    thought="Enough evidence.",
                    action="finish",
                    answer="Alpha owns blue and Beta owns red.",
                ),
            ]
        )

        result = rag.query_react(
            "What databases do Alpha and Beta own?",
            max_steps=3,
            use_reranking=False,
        )

        assert result.finished is True
        assert [step.action for step in result.steps] == ["search", "search", "finish"]
        assert "Alpha owns the blue database." in result.passages
        assert "Beta owns the red database." in result.passages
        assert len(result.passages) == len(set(result.passages))
        assert (
            rag._react_planner.plan.call_args_list[1].kwargs["steps"][0].query == "Alpha database"
        )
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_query_react_applies_metadata_filter_to_search_steps():
    """Apply the metadata filter to each ReAct search action."""
    rag, milvus_uri = with_temp_rag("react_filter")

    try:
        add_tenant_documents(rag)
        rag._react_planner.plan = MagicMock(
            side_effect=[
                ReActAction(
                    thought="Search within team A.",
                    action="search",
                    query="What database does Alpha own?",
                ),
                ReActAction(
                    thought="Enough evidence.",
                    action="finish",
                    answer="Alpha owns the blue database.",
                ),
            ]
        )

        result = rag.query_react(
            "What database does Alpha own?",
            max_steps=2,
            use_reranking=False,
            filter='tenant_id == "team_a"',
        )

        assert result.finished is True
        assert result.passages == ["Alpha owns the blue database."]
        assert "red database" not in result.steps[0].observation
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_query_react_passes_retrieval_options_to_each_search_step():
    """Forward per-call retrieval options from query_react to retrieve."""
    rag, milvus_uri = with_temp_rag("react_retrieval_options")

    try:
        rag._react_planner.plan = MagicMock(
            side_effect=[
                ReActAction(thought="Search.", action="search", query="focused query"),
                ReActAction(thought="Done.", action="finish", answer="done"),
            ]
        )
        rag.retrieve = MagicMock(
            return_value=MagicMock(
                passages=["Retrieved passage."],
                retrieved_passages=["Retrieved passage."],
                reranked_relations=["Retrieved relation."],
                retrieved_relations=["Retrieved relation."],
            )
        )

        result = rag.query_react(
            "Original question?",
            max_steps=2,
            use_reranking=False,
            top_k=1,
            filter='tenant_id == "team_a"',
        )

        assert result.answer == "done"
        rag.retrieve.assert_called_once_with(
            "focused query",
            use_reranking=False,
            top_k=1,
            filter='tenant_id == "team_a"',
        )
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_query_react_falls_back_to_answer_generation_after_max_steps():
    """Generate a final answer from observed passages when the planner keeps searching."""
    rag, milvus_uri = with_temp_rag("react_fallback_answer")

    try:
        add_test_documents(rag)
        rag._react_planner.plan = MagicMock(
            return_value=ReActAction(
                thought="Keep searching.",
                action="search",
                query="What database does Alpha own?",
            )
        )
        rag._answer_generator.generate = MagicMock(return_value="Fallback answer.")

        result = rag.query_react(
            "What database does Alpha own?",
            max_steps=2,
            use_reranking=False,
        )

        assert result.answer == "Fallback answer."
        assert result.finished is False
        assert len(result.steps) == 2
        assert "Alpha owns the blue database." in result.passages
        assert len(result.passages) == len(set(result.passages))
        rag._answer_generator.generate.assert_called_once_with(
            "What database does Alpha own?",
            result.passages,
        )
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_query_react_validates_max_steps():
    """Reject invalid max_steps values."""
    rag, milvus_uri = with_temp_rag("react_validate_max_steps")

    try:
        with pytest.raises(ValueError, match="max_steps"):
            rag.query_react("What database does Alpha own?", max_steps=0)
    finally:
        remove_temp_milvus_file(milvus_uri)


def test_react_planner_parses_search_finish_and_invalid_json():
    """Parse planner JSON responses and fall back to a search on invalid JSON."""
    settings = Settings(
        milvus_uri="./react_parse_test.db",
        openai_api_key="test-api-key",
        embedding_provider="openai",
        use_llm_cache=False,
    )
    planner = ReActPlanner(settings=settings)

    search = planner._parse_response(
        '{"thought": "Need evidence.", "action": "search", "query": "alpha database"}',
        question="original question",
    )
    assert search.action == "search"
    assert search.query == "alpha database"

    finish = planner._parse_response(
        '{"thought": "Enough.", "action": "finish", "answer": "Alpha owns blue."}',
        question="original question",
    )
    assert finish.action == "finish"
    assert finish.answer == "Alpha owns blue."

    fallback = planner._parse_response("not json", question="original question")
    assert fallback.action == "search"
    assert fallback.query == "original question"

    missing_answer = planner._parse_response(
        '{"thought": "No answer.", "action": "finish"}',
        question="original question",
    )
    assert missing_answer.action == "search"
    assert missing_answer.query == "original question"


def test_react_planner_formats_history_for_prompt():
    """Include previous thoughts, actions, queries, and observations in planner prompts."""
    history = [
        ReActAction(
            thought="Need Alpha evidence.",
            action="search",
            query="Alpha database",
        )
    ]
    step = history[0]
    formatted = ReActPlanner._format_history(
        [
            MagicMock(
                step=1,
                thought=step.thought,
                action=step.action,
                query=step.query,
                observation="Passages:\n1. Alpha owns the blue database.",
                answer=None,
            )
        ]
    )

    assert "Step 1" in formatted
    assert "Need Alpha evidence." in formatted
    assert "Alpha database" in formatted
    assert "Alpha owns the blue database." in formatted


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_OPENAI_TESTS") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="Set RUN_LIVE_OPENAI_TESTS=1 and OPENAI_API_KEY to run live OpenAI ReAct tests.",
)
def test_query_react_with_live_openai_planner_answers_from_retrieved_context():
    """Smoke test the real OpenAI planner against an end-to-end ReAct query."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_file.close()
    milvus_uri = temp_file.name

    try:
        rag = create_live_llm_test_rag(milvus_uri, "react_live_openai")
        add_tenant_documents(rag)

        result = rag.query_react(
            "Search first, then answer: what color database does Alpha own?",
            max_steps=3,
            use_reranking=False,
            top_k=1,
            filter='tenant_id == "team_a"',
        )

        assert any(step.action == "search" for step in result.steps)
        assert "Alpha owns the blue database." in result.passages
        assert "blue" in result.answer.lower()
        assert "red" not in result.answer.lower()
    finally:
        remove_temp_milvus_file(milvus_uri)
