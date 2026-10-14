"""Generate text using the model's supported API without mixing reasoning and answers."""

from openai import OpenAI

from vector_graph_rag.config import Settings


def generate_text(
    client: OpenAI,
    *,
    settings: Settings,
    model: str,
    messages: list[dict],
    json_mode: bool = False,
    temperature: float = 0,
    stop: list[str] | None = None,
) -> str:
    """Use Responses for GPT-6 and retain Chat Completions for explicit other models."""
    if model.startswith("gpt-6"):
        kwargs = {
            "model": model,
            "input": messages,
            "reasoning": {"effort": settings.llm_reasoning_effort},
            "store": False,
        }
        if json_mode:
            kwargs["text"] = {"format": {"type": "json_object"}}
        response = client.responses.create(**kwargs)
        if response.status != "completed":
            raise ValueError("Model response did not complete; no answer was cached.")
        parts = []
        for item in response.output:
            if item.type != "message":
                continue
            for content in item.content:
                if content.type == "refusal":
                    raise ValueError("Model refused the request; no answer was cached.")
                if content.type == "output_text":
                    parts.append(content.text)
        result = "".join(parts)
        if not result.strip():
            raise ValueError("Model returned no final answer; no answer was cached.")
        return result

    kwargs = {"model": model, "messages": messages}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if not model.startswith("gpt-5"):
        kwargs["temperature"] = temperature
        if stop is not None:
            kwargs["stop"] = stop
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ("{}" if json_mode else "I don't know.")
