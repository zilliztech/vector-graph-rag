"""
ReAct planner for iterative retrieval.
"""

import json
from typing import List, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from vector_graph_rag.config import Settings, get_settings
from vector_graph_rag.llm.cache import LLMCache, get_llm_cache
from vector_graph_rag.models import ReActAction, ReActStep

REACT_SYSTEM_PROMPT = """You control a retrieval loop for answering questions.

At each step, choose one JSON action:

1. Search for more context:
{"thought": "brief reason", "action": "search", "query": "focused search query"}

2. Finish when the observations are enough:
{"thought": "brief reason", "action": "finish", "answer": "final answer"}

Rules:
- Return only one JSON object.
- Use "search" when more evidence is needed.
- Use "finish" only when the observations contain enough evidence.
- Keep "thought" brief and do not include hidden reasoning.
"""

REACT_USER_PROMPT = """Question:
{question}

Previous steps:
{history}

Current step: {step_number} of {max_steps}

Choose the next action."""


class ReActPlanner:
    """
    LLM planner for a basic ReAct search loop.

    The planner emits either a search query or a final answer. Retrieval and
    answer fallback are handled by VectorGraphRAG.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model: Optional[str] = None,
        use_cache: Optional[bool] = None,
        cache: Optional[LLMCache] = None,
    ) -> None:
        """
        Initialize the planner.

        Args:
            settings: Configuration settings.
            model: Override LLM model from settings.
            use_cache: Whether to use LLM response caching.
            cache: Custom cache instance.
        """
        self.settings = settings or get_settings()
        self.settings.validate_settings()

        self.model = model or self.settings.llm_model
        self.client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
        )
        self.use_cache = use_cache if use_cache is not None else self.settings.use_llm_cache
        self.cache = cache or get_llm_cache() if self.use_cache else None

    @staticmethod
    def _format_history(steps: List[ReActStep]) -> str:
        """Format previous ReAct steps for the planner prompt."""
        if not steps:
            return "None"

        sections = []
        for step in steps:
            lines = [
                f"Step {step.step}",
                f"Thought: {step.thought or ''}",
                f"Action: {step.action}",
            ]
            if step.query:
                lines.append(f"Query: {step.query}")
            if step.observation:
                lines.append(f"Observation: {step.observation}")
            if step.answer:
                lines.append(f"Answer: {step.answer}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def _build_prompt(
        self,
        question: str,
        steps: List[ReActStep],
        step_number: int,
        max_steps: int,
    ) -> str:
        """Build the user prompt."""
        return REACT_USER_PROMPT.format(
            question=question,
            history=self._format_history(steps),
            step_number=step_number,
            max_steps=max_steps,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with retry and optional cache."""
        cache_key = f"{REACT_SYSTEM_PROMPT}\n\n{prompt}"
        if self.cache:
            cached = self.cache.get(self.model, cache_key, temperature=0)
            if cached is not None:
                return cached

        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        api_kwargs = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if not self.model.startswith("gpt-5"):
            api_kwargs["temperature"] = 0

        response = self.client.chat.completions.create(**api_kwargs)
        result = response.choices[0].message.content or "{}"

        if self.cache:
            self.cache.set(self.model, cache_key, result, temperature=0)

        return result

    @staticmethod
    def _parse_response(response: str, question: str) -> ReActAction:
        """Parse an LLM response into a planner action."""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return ReActAction(
                thought="Fallback to search because the planner returned invalid JSON.",
                action="search",
                query=question,
            )

        action = str(data.get("action", "")).strip().lower()
        thought = str(data.get("thought", "")).strip()

        if action == "finish":
            answer = str(data.get("answer", "")).strip()
            if answer:
                return ReActAction(
                    thought=thought,
                    action="finish",
                    answer=answer,
                )

        query = str(data.get("query", "")).strip() or question
        return ReActAction(
            thought=thought,
            action="search",
            query=query,
        )

    def plan(
        self,
        question: str,
        steps: List[ReActStep],
        step_number: int,
        max_steps: int,
    ) -> ReActAction:
        """
        Plan the next ReAct action.

        Args:
            question: Original user question.
            steps: Previous ReAct steps.
            step_number: 1-based current step number.
            max_steps: Maximum loop steps.

        Returns:
            Parsed planner action.
        """
        prompt = self._build_prompt(
            question=question,
            steps=steps,
            step_number=step_number,
            max_steps=max_steps,
        )
        response = self._call_llm(prompt)
        return self._parse_response(response, question)
