"""
LLM-based reranking for candidate relations.
"""

import json
from typing import List, Optional, Tuple
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from vector_graph_rag.config import Settings, get_settings
from vector_graph_rag.llm.cache import get_llm_cache, LLMCache


RERANK_EXAMPLE_1_INPUT = """I will provide you with a set of relationship descriptions from a knowledge graph. Select exactly 5 relationships most useful for answering this multi-hop question.

Return JSON with "thought_process" and "useful_relations" (list of 5 relation lines, most useful first).

Question:
When did Lothair Ii's mother die?

Relationship descriptions:
[53] bertha married to theobald of arles
[54] bertha married to adalbert ii of tuscany
[42] lothair ii son of ermengarde of tours
[43] lothair ii married to teutberga
[41] lothair ii son of emperor lothair i
[60] lothair ii husband of waldrada
[67] waldrada was mistress of lothair ii

"""

RERANK_EXAMPLE_1_OUTPUT = """{"thought_process": "2-hop question: First find Lothair II's mother (relation [42]: Ermengarde of Tours), then find death date. [41] gives father for family context.", "useful_relations": ["[42] lothair ii son of ermengarde of tours", "[41] lothair ii son of emperor lothair i", "[43] lothair ii married to teutberga", "[60] lothair ii husband of waldrada", "[67] waldrada was mistress of lothair ii"]}"""

RERANK_EXAMPLE_2_INPUT = """I will provide you with a set of relationship descriptions from a knowledge graph. Select exactly 5 relationships most useful for answering this multi-hop question.

Return JSON with "thought_process" and "useful_relations" (list of 5 relation lines, most useful first).

Question:
What country is the composer of "Erta Eterna" from?

Relationship descriptions:
[12] terra eterna composed by paulo flores
[15] paulo flores born in angola
[18] paulo flores genre is semba
[22] angola located in africa
[25] semba originated in angola
[30] paulo flores nationality angolan

"""

RERANK_EXAMPLE_2_OUTPUT = """{"thought_process": "2-hop question: First find composer of Terra Eterna ([12]: Paulo Flores), then find his country ([15] born in Angola or [30] nationality Angolan).", "useful_relations": ["[12] terra eterna composed by paulo flores", "[15] paulo flores born in angola", "[30] paulo flores nationality angolan", "[22] angola located in africa", "[25] semba originated in angola"]}"""

RERANK_EXAMPLE_3_INPUT = """I will provide you with a set of relationship descriptions from a knowledge graph. Select exactly 5 relationships most useful for answering this multi-hop question.

Return JSON with "thought_process" and "useful_relations" (list of 5 relation lines, most useful first).

Question:
Who is the director of the film that won the award also won by "The Hurt Locker"?

Relationship descriptions:
[5] the hurt locker won academy award best picture
[8] the hurt locker directed by kathryn bigelow
[12] moonlight won academy award best picture
[15] moonlight directed by barry jenkins
[20] la la land won golden globe best musical
[25] barry jenkins born in miami

"""

RERANK_EXAMPLE_3_OUTPUT = """{"thought_process": "3-hop question: (1) Find award won by The Hurt Locker ([5]: Academy Award Best Picture), (2) Find another film with same award ([12]: Moonlight), (3) Find director ([15]: Barry Jenkins).", "useful_relations": ["[5] the hurt locker won academy award best picture", "[12] moonlight won academy award best picture", "[15] moonlight directed by barry jenkins", "[8] the hurt locker directed by kathryn bigelow", "[25] barry jenkins born in miami"]}"""

RERANK_PROMPT_TEMPLATE = """Question:
{question}

Relationship descriptions:
{relation_descriptions}

"""


def _correct_line(
    predict_line: str, relation_des_texts: List[str], relation_des_ids: List[str]
) -> Optional[str]:
    """
    Correct a predicted line by matching text content.
    Similar to HippoRAG's _correct_line function.
    """
    predict_line_text = predict_line[predict_line.find("]") + 1 :].strip()
    for line_text, id_ in zip(relation_des_texts, relation_des_ids):
        if line_text.strip() == predict_line_text.strip():
            return id_
    return None


class LLMReranker:
    """
    LLM-based reranker for candidate relations.

    Uses GPT models with chain-of-thought prompting to select
    the most relevant relations for answering a query.

    Example:
        >>> reranker = LLMReranker()
        >>> selected_ids = reranker.rerank(
        ...     query="Who taught Einstein?",
        ...     relation_ids=[0, 1, 2],
        ...     relation_texts=["Einstein was born in Germany", "...", "..."],
        ... )
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model: Optional[str] = None,
        use_cache: Optional[bool] = None,
        cache: Optional[LLMCache] = None,
    ):
        """
        Initialize the reranker.

        Args:
            settings: Configuration settings.
            model: Override LLM model from settings.
            use_cache: Whether to use LLM response caching (default: from settings).
            cache: Custom cache instance.
        """
        self.settings = settings or get_settings()
        self.settings.validate_settings()

        self.model = model or self.settings.llm_model

        self.client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
        )

        # Use settings.use_llm_cache if use_cache not explicitly provided
        self.use_cache = use_cache if use_cache is not None else self.settings.use_llm_cache
        self.cache = cache or get_llm_cache() if self.use_cache else None

    def _format_relations(
        self,
        relation_ids: List[str],
        relation_texts: List[str],
    ) -> str:
        """Format relations for the prompt."""
        lines = []
        for rid, text in zip(relation_ids, relation_texts):
            lines.append(f"[{rid}] {text}")
        return "\n".join(lines)

    def _build_prompt(
        self, query: str, relation_descriptions: str
    ) -> str:
        """Build the full prompt for caching."""
        # Include all 3 few-shot examples in the cache key
        examples = (
            f"{RERANK_EXAMPLE_1_INPUT}\n{RERANK_EXAMPLE_1_OUTPUT}\n\n"
            f"{RERANK_EXAMPLE_2_INPUT}\n{RERANK_EXAMPLE_2_OUTPUT}\n\n"
            f"{RERANK_EXAMPLE_3_INPUT}\n{RERANK_EXAMPLE_3_OUTPUT}\n\n"
        )
        return f"{examples}{RERANK_PROMPT_TEMPLATE.format(question=query, relation_descriptions=relation_descriptions)}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _call_llm(self, query: str, relation_descriptions: str) -> str:
        """Call LLM API with retry logic and caching."""
        prompt = RERANK_PROMPT_TEMPLATE.format(
            question=query,
            relation_descriptions=relation_descriptions,
        )
        cache_key = self._build_prompt(query, relation_descriptions)

        # Check cache
        if self.cache:
            cached = self.cache.get(self.model, cache_key, temperature=0)
            if cached is not None:
                return cached

        # Use 3 diverse few-shot examples for better multi-hop reasoning
        messages = [
            {"role": "user", "content": RERANK_EXAMPLE_1_INPUT},
            {"role": "assistant", "content": RERANK_EXAMPLE_1_OUTPUT},
            {"role": "user", "content": RERANK_EXAMPLE_2_INPUT},
            {"role": "assistant", "content": RERANK_EXAMPLE_2_OUTPUT},
            {"role": "user", "content": RERANK_EXAMPLE_3_INPUT},
            {"role": "assistant", "content": RERANK_EXAMPLE_3_OUTPUT},
            {"role": "user", "content": prompt},
        ]

        # Build API call kwargs
        api_kwargs = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        # gpt-5 series doesn't support 'temperature' and 'stop' parameters
        if not self.model.startswith("gpt-5"):
            api_kwargs["temperature"] = 0
            api_kwargs["stop"] = ['\n\n']

        response = self.client.chat.completions.create(**api_kwargs)
        result = response.choices[0].message.content or "{}"

        # Store in cache
        if self.cache:
            self.cache.set(self.model, cache_key, result, temperature=0)

        return result

    def _parse_response(
        self,
        response: str,
        valid_ids: set,
        relation_ids: List[str],
        relation_texts: List[str],
    ) -> List[str]:
        """Parse LLM response to extract selected relation IDs."""
        try:
            data = json.loads(response)
            useful_relationships = data.get("useful_relations", [])

            selected_ids = []
            id_to_line = {}

            for line in useful_relationships:
                # Extract ID from format "[ID] text"
                if "[" in line and "]" in line:
                    start = line.find("[") + 1
                    end = line.find("]")
                    rel_id = line[start:end].strip()
                    id_to_line[rel_id] = line.strip()

                    if rel_id in valid_ids and rel_id not in selected_ids:
                        selected_ids.append(rel_id)
                    elif rel_id not in valid_ids:
                        # Try to correct the ID by matching text
                        corrected_id = _correct_line(
                            line, relation_texts, relation_ids
                        )
                        if (
                            corrected_id is not None
                            and corrected_id not in selected_ids
                        ):
                            selected_ids.append(corrected_id)

            return selected_ids
        except (json.JSONDecodeError, KeyError):
            return []

    def rerank(
        self,
        query: str,
        relation_ids: List[str],
        relation_texts: List[str],
        num_select: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Rerank candidate relations using LLM.

        Args:
            query: The query text.
            relation_ids: Candidate relation IDs (string UUIDs).
            relation_texts: Candidate relation texts.
            num_select: Number of relations to select (default: final_top_k).

        Returns:
            Tuple of (selected_relation_ids, selected_relation_texts).
        """
        if not relation_ids:
            return [], []

        num_select = 5

        # Format relations
        relation_descriptions = self._format_relations(relation_ids, relation_texts)

        # Call LLM
        response = self._call_llm(query, relation_descriptions)

        # Parse response
        valid_ids = set(relation_ids)
        selected_ids = self._parse_response(
            response, valid_ids, relation_ids, relation_texts
        )

        # No fallback - return whatever LLM selected (same as current project)

        # Get corresponding texts
        id_to_text = dict(zip(relation_ids, relation_texts))
        selected_texts = [id_to_text[rid] for rid in selected_ids if rid in id_to_text]

        return selected_ids, selected_texts


class AnswerGenerator:
    """
    Generate answers using retrieved context.
    """

    ANSWER_PROMPT = """Use the following pieces of retrieved context to answer the question. If there is not enough information in the retrieved context to answer the question, just say that you don't know.

Question: {question}

Context: {context}

Answer:"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model: Optional[str] = None,
        use_cache: Optional[bool] = None,
        cache: Optional[LLMCache] = None,
    ):
        """
        Initialize the answer generator.

        Args:
            settings: Configuration settings.
            model: Override LLM model from settings.
            use_cache: Whether to use LLM response caching (default: from settings).
            cache: Custom cache instance.
        """
        self.settings = settings or get_settings()
        self.settings.validate_settings()

        self.model = model or self.settings.llm_model

        self.client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
        )

        # Use settings.use_llm_cache if use_cache not explicitly provided
        self.use_cache = use_cache if use_cache is not None else self.settings.use_llm_cache
        self.cache = cache or get_llm_cache() if self.use_cache else None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate(self, query: str, passages: List[str]) -> str:
        """
        Generate an answer based on query and context.

        Args:
            query: The query text.
            passages: Retrieved passages for context.

        Returns:
            Generated answer string.
        """
        context = "\n\n".join(passages)
        prompt = self.ANSWER_PROMPT.format(question=query, context=context)

        # Check cache
        if self.cache:
            cached = self.cache.get(self.model, prompt, temperature=0)
            if cached is not None:
                return cached

        messages = [{"role": "user", "content": prompt}]

        # Build API call kwargs - gpt-5 series doesn't support temperature parameter
        api_kwargs = {
            "model": self.model,
            "messages": messages,
        }
        if not self.model.startswith("gpt-5"):
            api_kwargs["temperature"] = 0

        response = self.client.chat.completions.create(**api_kwargs)

        result = response.choices[0].message.content or "I don't know."

        # Store in cache
        if self.cache:
            self.cache.set(self.model, prompt, result, temperature=0)

        return result
