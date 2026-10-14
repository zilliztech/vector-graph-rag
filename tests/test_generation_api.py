"""Verify default-model routing, response parsing and cache isolation."""

from types import SimpleNamespace as N
from unittest.mock import MagicMock

import pytest

from vector_graph_rag.config import Settings
from vector_graph_rag.llm.extractor import EntityExtractor, TripletExtractor
from vector_graph_rag.llm.generation import generate_text
from vector_graph_rag.llm.reranker import AnswerGenerator, LLMReranker


def response(text):
    return N(
        status="completed",
        output=[
            N(type="reasoning", content=[N(type="output_text", text="not the answer")]),
            N(type="message", content=[N(type="output_text", text=text)]),
        ],
    )


def test_default_cache_identity():
    settings = Settings(openai_api_key="test")
    assert settings.llm_model == "gpt-6-luna"
    assert settings.task_model("answer").model == "gpt-6-luna"
    key = settings.cache_model_name(settings.llm_model)
    assert key != settings.llm_model
    settings.llm_reasoning_effort = "low"
    assert key != settings.cache_model_name(settings.llm_model)
    assert settings.cache_model_name("custom-model") == "custom-model"


@pytest.mark.parametrize("kind", ["triplets", "entities", "reranking", "answer"])
def test_components_use_responses(kind):
    settings = Settings(openai_api_key="test")
    client = MagicMock()
    client.responses.create.return_value = response('{"named_entities": ["Paris"]}')
    if kind == "triplets":
        obj = TripletExtractor(settings=settings, use_cache=False)
        obj.client = client
        obj._call_llm("Paris is in France.")
    elif kind == "entities":
        obj = EntityExtractor(settings=settings, use_cache=False)
        obj.client = client
        assert obj.extract("Where is Paris?") == ["paris"]
    elif kind == "reranking":
        obj = LLMReranker(settings=settings, use_cache=False)
        obj.client = client
        obj._call_llm("Where is Paris?", "[1] Paris is in France")
    else:
        obj = AnswerGenerator(settings=settings, use_cache=False)
        obj.client = client
        obj.generate("Where is Paris?", ["Paris is in France."])
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["model"] == "gpt-6-luna"
    assert kwargs["reasoning"] == {"effort": "medium"}
    assert kwargs["store"] is False
    assert "temperature" not in kwargs and "stop" not in kwargs
    assert ("text" in kwargs) == (kind != "answer")
    client.chat.completions.create.assert_not_called()
    if kind == "reranking":
        assert "useful_relations" in kwargs["input"][-1]["content"]


@pytest.mark.parametrize(
    "status,output",
    [
        ("incomplete", []),
        ("failed", []),
        ("completed", []),
        ("completed", [N(type="message", content=[N(type="refusal")])]),
    ],
)
def test_unusable_response_rejected(status, output):
    client = MagicMock()
    client.responses.create.return_value = N(status=status, output=output)
    with pytest.raises(ValueError):
        generate_text(client, settings=Settings(), model="gpt-6-luna", messages=[])


def test_only_final_message_text():
    client = MagicMock()
    client.responses.create.return_value = response("final answer")
    assert (
        generate_text(client, settings=Settings(), model="gpt-6-luna", messages=[])
        == "final answer"
    )


def test_explicit_chat_model_preserves_contract():
    client = MagicMock()
    client.chat.completions.create.return_value = N(choices=[N(message=N(content='{"ok":true}'))])
    generate_text(
        client, settings=Settings(), model="custom-model", messages=[], json_mode=True, stop=["END"]
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["stop"] == ["END"]
    assert kwargs["temperature"] == 0
    client.responses.create.assert_not_called()
