"""
LLM-based reranking for candidate relations.
"""

import json
from typing import List, Optional, Tuple
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from vector_graph_rag.config import Settings, get_settings
from vector_graph_rag.llm.cache import get_llm_cache, LLMCache


# One-shot example for reranking (similar to HippoRAG's prompt)
RERANK_EXAMPLE_INPUT = """I will provide you with a set of relationship descriptions in the knowledge graph, and you should select 5 relationships that may be useful to answer this question.
You just return me with a json, which contains a thought process description, and a list of the useful relation lines. The more useful the relationship is for answering the question, the higher rank it will be in the list.

Question:
When was the mother of the leader of the Third Crusade born?

Relationship descriptions:
[1] Eleanor was born in 1122.
[2] Eleanor married King Louis VII of France.
[3] Eleanor was the Duchess of Aquitaine.
[4] Eleanor participated in the Second Crusade.
[5] Eleanor had eight children.
[6] Eleanor was married to Henry II of England.
[7] Eleanor was the mother of Richard the Lionheart.
[8] Richard the Lionheart was the King of England.
[9] Henry II was the father of Richard the Lionheart.
[10] Henry II was the King of England.
[11] Richard the Lionheart led the Third Crusade.

"""

RERANK_EXAMPLE_OUTPUT = """{
    "thought_process": "To answer the question about the birth of the mother of the leader of the Third Crusade, I first need to identify who led the Third Crusade and then determine who his mother was. After identifying his mother, I can look for the relationship that mentions her birth.",
    "useful_relationships": [
        "[11] Richard the Lionheart led the Third Crusade",
        "[7] Eleanor was the mother of Richard the Lionheart",
        "[1] Eleanor was born in 1122",
        "[6] Eleanor was married to Henry II of England",
        "[9] Henry II was the father of Richard the Lionheart"
    ]
}"""

RERANK_PROMPT_TEMPLATE = """I will provide you with a set of relationship descriptions in the knowledge graph, and you should select {num_select} relationships that may be useful to answer this question.
You just return me with a json, which contains a thought process description, and a list of the useful relation lines. The more useful the relationship is for answering the question, the higher rank it will be in the list.

Question:
{question}

Relationship descriptions:
{relation_descriptions}

"""


def _correct_line(
    predict_line: str, relation_des_texts: List[str], relation_des_ids: List[int]
) -> Optional[int]:
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
        use_cache: bool = True,
        cache: Optional[LLMCache] = None,
    ):
        """
        Initialize the reranker.

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

        self.use_cache = use_cache
        self.cache = cache or get_llm_cache() if use_cache else None

    def _format_relations(
        self,
        relation_ids: List[int],
        relation_texts: List[str],
    ) -> str:
        """Format relations for the prompt."""
        lines = []
        for rid, text in zip(relation_ids, relation_texts):
            lines.append(f"[{rid}] {text}")
        return "\n".join(lines)

    def _build_prompt(
        self, query: str, relation_descriptions: str, num_select: int
    ) -> str:
        """Build the full prompt for caching."""
        return f"{RERANK_EXAMPLE_INPUT}\n{RERANK_EXAMPLE_OUTPUT}\n\n{RERANK_PROMPT_TEMPLATE.format(num_select=num_select, question=query, relation_descriptions=relation_descriptions)}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _call_llm(self, query: str, relation_descriptions: str, num_select: int) -> str:
        """Call LLM API with retry logic and caching."""
        prompt = RERANK_PROMPT_TEMPLATE.format(
            num_select=num_select,
            question=query,
            relation_descriptions=relation_descriptions,
        )
        cache_key = self._build_prompt(query, relation_descriptions, num_select)

        # Check cache
        if self.cache:
            cached = self.cache.get(self.model, cache_key, temperature=0)
            if cached is not None:
                return cached

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": RERANK_EXAMPLE_INPUT},
                {"role": "assistant", "content": RERANK_EXAMPLE_OUTPUT},
                {"role": "user", "content": prompt},
            ],
        )
        result = response.choices[0].message.content or "{}"

        # Store in cache
        if self.cache:
            self.cache.set(self.model, cache_key, result, temperature=0)

        return result

    def _parse_response(
        self,
        response: str,
        valid_ids: set,
        relation_ids: List[int],
        relation_texts: List[str],
    ) -> List[int]:
        """Parse LLM response to extract selected relation IDs."""
        try:
            data = json.loads(response)
            useful_relationships = data.get("useful_relationships", [])

            selected_ids = []
            id_to_line = {}

            for line in useful_relationships:
                # Extract ID from format "[ID] text"
                if "[" in line and "]" in line:
                    start = line.find("[") + 1
                    end = line.find("]")
                    try:
                        rel_id = int(line[start:end])
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
                    except ValueError:
                        continue

            return selected_ids
        except (json.JSONDecodeError, KeyError):
            return []

    def rerank(
        self,
        query: str,
        relation_ids: List[int],
        relation_texts: List[str],
        num_select: Optional[int] = None,
    ) -> Tuple[List[int], List[str]]:
        """
        Rerank candidate relations using LLM.

        Args:
            query: The query text.
            relation_ids: Candidate relation IDs.
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
        response = self._call_llm(query, relation_descriptions, num_select)

        # Parse response
        valid_ids = set(relation_ids)
        selected_ids = self._parse_response(
            response, valid_ids, relation_ids, relation_texts
        )

        # If parsing failed or not enough selected, fallback to first N
        if len(selected_ids) < num_select:
            for rid in relation_ids:
                if rid not in selected_ids:
                    selected_ids.append(rid)
                if len(selected_ids) >= num_select:
                    break

        # Get corresponding texts
        id_to_text = dict(zip(relation_ids, relation_texts))
        selected_texts = [id_to_text[rid] for rid in selected_ids if rid in id_to_text]

        return selected_ids[:num_select], selected_texts[:num_select]


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
        use_cache: bool = True,
        cache: Optional[LLMCache] = None,
    ):
        """
        Initialize the answer generator.

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

        self.use_cache = use_cache
        self.cache = cache or get_llm_cache() if use_cache else None

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

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        result = response.choices[0].message.content or "I don't know."

        # Store in cache
        if self.cache:
            self.cache.set(self.model, prompt, result, temperature=0)

        return result
