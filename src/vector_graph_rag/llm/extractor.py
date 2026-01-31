"""
Triplet extraction using LLM prompt engineering.
"""

import json
import re
from typing import List, Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from vector_graph_rag.config import Settings, get_settings
from vector_graph_rag.models import Document, Triplet
from vector_graph_rag.llm.cache import get_llm_cache, LLMCache


def processing_phrases(phrase: str) -> str:
    """
    Preprocess phrases for normalization.
    Same as HippoRAG's processing_phrases function.

    Replaces all non-alphanumeric characters (including apostrophes,
    commas, hyphens, accented characters) with spaces.
    """
    if not phrase:
        return ""
    # Replace all non-alphanumeric characters with space, convert to lowercase, strip
    return re.sub(r'[^A-Za-z0-9 ]', ' ', phrase.lower()).strip()


# System prompt for triplet extraction
EXTRACTION_SYSTEM_PROMPT = """You are an expert knowledge graph builder. Your task is to extract knowledge triplets from the given text.

A triplet consists of:
- Subject: An entity (person, place, thing, concept, etc.)
- Predicate: The relationship between subject and object
- Object: Another entity

Guidelines:
1. Extract all meaningful relationships from the text
2. Keep entities concise but complete (e.g., "Johann Bernoulli" not just "Johann")
3. Use clear, specific predicates (e.g., "was born in" instead of "relates to")
4. Extract both explicit and implicit relationships
5. Ensure triplets are factually accurate based on the text
6. Do not infer relationships not supported by the text

Return your response as a JSON object with a "triplets" array, where each triplet is an array of [subject, predicate, object].
"""

# One-shot example for better extraction
EXTRACTION_EXAMPLE_INPUT = """Text: Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of relativity, which revolutionized physics. Einstein worked at the Institute for Advanced Study in Princeton."""

EXTRACTION_EXAMPLE_OUTPUT = """{
    "triplets": [
        ["Albert Einstein", "was born in", "Ulm, Germany"],
        ["Albert Einstein", "was born in", "1879"],
        ["Albert Einstein", "developed", "the theory of relativity"],
        ["the theory of relativity", "revolutionized", "physics"],
        ["Albert Einstein", "worked at", "the Institute for Advanced Study"],
        ["the Institute for Advanced Study", "is located in", "Princeton"]
    ]
}"""


# NER extraction prompt (similar to HippoRAG)
NER_SYSTEM_PROMPT = "You're a very effective entity extraction system."

NER_ONE_SHOT_INPUT = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?

"""

NER_ONE_SHOT_OUTPUT = """{"named_entities": ["First for Women", "Arthur's Magazine"]}"""

NER_TEMPLATE = """
Question: {}

"""


class TripletExtractor:
    """
    Extract knowledge triplets from text using LLM.

    This class uses OpenAI's GPT models with carefully designed prompts
    to extract subject-predicate-object triplets from text passages.

    Example:
        >>> extractor = TripletExtractor()
        >>> triplets = extractor.extract("Einstein developed relativity.")
        >>> print(triplets[0].subject)
        "Einstein"
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        use_cache: bool = True,
        cache: Optional[LLMCache] = None,
    ):
        """
        Initialize the triplet extractor.

        Args:
            settings: Configuration settings. Uses default if not provided.
            model: Override LLM model from settings.
            temperature: Override temperature from settings.
            use_cache: Whether to use LLM response caching.
            cache: Custom cache instance. Uses global cache if not provided.
        """
        self.settings = settings or get_settings()
        self.settings.validate_settings()

        self.model = model or self.settings.llm_model
        self.temperature = (
            temperature if temperature is not None else self.settings.llm_temperature
        )

        self.client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
        )

        self.use_cache = use_cache
        self.cache = cache or get_llm_cache() if use_cache else None

    def _build_prompt(self, text: str) -> str:
        """Build the full prompt for caching."""
        return f"{EXTRACTION_SYSTEM_PROMPT}\n\n{EXTRACTION_EXAMPLE_INPUT}\n\n{EXTRACTION_EXAMPLE_OUTPUT}\n\nText: {text}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _call_llm(self, text: str) -> str:
        """Call LLM API with retry logic and caching."""
        prompt = self._build_prompt(text)

        # Check cache
        if self.cache:
            cached = self.cache.get(self.model, prompt, self.temperature)
            if cached is not None:
                return cached

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": EXTRACTION_EXAMPLE_INPUT},
                {"role": "assistant", "content": EXTRACTION_EXAMPLE_OUTPUT},
                {"role": "user", "content": f"Text: {text}"},
            ],
        )
        result = response.choices[0].message.content or "{}"

        # Store in cache
        if self.cache:
            self.cache.set(self.model, prompt, result, self.temperature)

        return result

    def _parse_response(self, response: str) -> List[Triplet]:
        """Parse LLM response into triplets."""
        try:
            data = json.loads(response)
            triplets = []

            raw_triplets = data.get("triplets", [])
            for raw in raw_triplets:
                if isinstance(raw, list) and len(raw) == 3:
                    subject, predicate, obj = raw
                    if subject and predicate and obj:
                        triplets.append(
                            Triplet(
                                subject=str(subject).strip(),
                                predicate=str(predicate).strip(),
                                object=str(obj).strip(),
                            )
                        )

            return triplets
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def extract(self, text: str) -> List[Triplet]:
        """
        Extract triplets from a single text passage.

        Args:
            text: The text to extract triplets from.

        Returns:
            List of extracted Triplet objects.
        """
        if not text or not text.strip():
            return []

        response = self._call_llm(text)
        return self._parse_response(response)

    def extract_from_documents(
        self,
        documents: List[Document],
        show_progress: bool = True,
    ) -> List[Document]:
        """
        Extract triplets from multiple documents.

        Args:
            documents: List of langchain Document objects to process.
            show_progress: Whether to show progress bar.

        Returns:
            Documents with extracted triplets stored in metadata["triplets"].
        """
        iterator = (
            tqdm(documents, desc="Extracting triplets") if show_progress else documents
        )

        for doc in iterator:
            triplets = self.extract(doc.page_content)
            # Store triplets as list of [subject, predicate, object] in metadata
            doc.metadata["triplets"] = [
                [t.subject, t.predicate, t.object] for t in triplets
            ]

        return documents


class EntityExtractor:
    """
    Extract named entities from text for query processing.

    This is used during query time to identify entities in the user's question,
    which are then used for entity-based retrieval.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model: Optional[str] = None,
        use_cache: bool = True,
        cache: Optional[LLMCache] = None,
        ner_cache_file: Optional[str] = None,
    ):
        """
        Initialize the entity extractor.

        Args:
            settings: Configuration settings.
            model: Override LLM model from settings.
            use_cache: Whether to use LLM response caching.
            cache: Custom cache instance.
            ner_cache_file: Path to TSV file with pre-computed NER results (HippoRAG format).
                If not provided, will try to auto-detect from settings.ner_cache_dir and
                settings.collection_prefix.
        """
        import os

        self.settings = settings or get_settings()
        self.settings.validate_settings()

        self.model = model or self.settings.llm_model

        self.client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
        )

        self.use_cache = use_cache
        self.cache = cache or get_llm_cache() if use_cache else None

        # Load TSV cache file if provided (HippoRAG format)
        self.ner_tsv_cache: dict = {}
        if ner_cache_file:
            self._load_tsv_cache(ner_cache_file)
        elif self.settings.ner_cache_dir and self.settings.collection_prefix:
            # Auto-detect cache file from settings
            # collection_prefix format: "ds_hotpotqa" -> dataset "hotpotqa"
            dataset = self.settings.collection_prefix
            if dataset.startswith("ds_"):
                dataset = dataset[3:]  # Remove "ds_" prefix
            cache_file = os.path.join(
                self.settings.ner_cache_dir,
                f"{dataset}_queries.named_entity_output.tsv"
            )
            if os.path.exists(cache_file):
                self._load_tsv_cache(cache_file)

    def _load_tsv_cache(self, cache_file: str) -> None:
        """Load NER results from TSV cache file (HippoRAG format)."""
        import pandas as pd
        try:
            df = pd.read_csv(cache_file, sep='\t')
            # The TSV has columns: query, triples (which contains JSON with named_entities)
            query_col = 'query' if 'query' in df.columns else 'question'
            for _, row in df.iterrows():
                query = row.get(query_col, '')
                triples_str = row.get('triples', '{}')
                try:
                    triples_data = eval(triples_str) if isinstance(triples_str, str) else triples_str
                    if isinstance(triples_data, dict) and 'named_entities' in triples_data:
                        self.ner_tsv_cache[query] = triples_data['named_entities']
                except:
                    pass
            print(f"Loaded {len(self.ner_tsv_cache)} NER entries from {cache_file}")
        except Exception as e:
            print(f"Warning: Could not load NER cache file {cache_file}: {e}")

    def _build_prompt(self, question: str) -> str:
        """Build prompt for caching."""
        return f"{NER_SYSTEM_PROMPT}\n\n{NER_ONE_SHOT_INPUT}\n{NER_ONE_SHOT_OUTPUT}\n\n{NER_TEMPLATE.format(question)}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def extract(self, question: str) -> List[str]:
        """
        Extract named entities from a question.

        Args:
            question: The question to extract entities from.

        Returns:
            List of entity strings.
        """
        # Check TSV cache first (HippoRAG format)
        if question in self.ner_tsv_cache:
            entities = self.ner_tsv_cache[question]
            return [processing_phrases(str(e)) for e in entities if e]

        prompt = self._build_prompt(question)

        # Check LLM cache
        if self.cache:
            cached = self.cache.get(self.model, prompt, temperature=0)
            if cached is not None:
                try:
                    data = json.loads(cached)
                    entities = data.get("named_entities", data.get("entities", []))
                    return [processing_phrases(str(e)) for e in entities if e]
                except (json.JSONDecodeError, KeyError):
                    pass

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": NER_SYSTEM_PROMPT},
                {"role": "user", "content": NER_ONE_SHOT_INPUT},
                {"role": "assistant", "content": NER_ONE_SHOT_OUTPUT},
                {"role": "user", "content": NER_TEMPLATE.format(question)},
            ],
        )

        content = response.choices[0].message.content or "{}"

        # Store in cache
        if self.cache:
            self.cache.set(self.model, prompt, content, temperature=0)

        try:
            data = json.loads(content)
            entities = data.get("named_entities", data.get("entities", []))
            return [processing_phrases(str(e)) for e in entities if e]
        except (json.JSONDecodeError, KeyError):
            return []
