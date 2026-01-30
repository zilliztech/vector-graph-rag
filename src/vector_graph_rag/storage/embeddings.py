"""
Embedding model wrapper for vector representations.

Supports both HuggingFace models (e.g., facebook/contriever) and OpenAI models.
Also supports instruction-based models like Qwen3-Embedding and BGE.
"""

from typing import List, Optional, Union, Literal
import numpy as np
import torch
from tqdm import tqdm

from vector_graph_rag.config import Settings, get_settings


# Predefined instruction templates for different models
INSTRUCTION_TEMPLATES = {
    # Qwen3 Embedding format
    "qwen3": {
        "query": "Instruct: {instruction}\nQuery: {text}",
        "document": "{text}",  # No instruction for documents
        "default_instruction": "Given a question, retrieve passages that contain the answer",
    },
    # BGE format - instruction only for queries
    "bge": {
        "query": "{instruction}: {text}",
        "document": "{text}",
        "default_instruction": "Represent this sentence for searching relevant passages",
    },
}


def _is_openai_model(model_name: str) -> bool:
    """Check if the model is an OpenAI embedding model."""
    openai_models = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]
    return model_name in openai_models or model_name.startswith("text-embedding")


def _get_model_family(model_name: str) -> Optional[str]:
    """Detect model family from model name for instruction templates."""
    model_lower = model_name.lower()
    if "qwen" in model_lower and "embed" in model_lower:
        return "qwen3"
    if "bge" in model_lower:
        return "bge"
    return None


def _mean_pooling(
    token_embeddings: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean pooling with attention mask."""
    token_embeddings = token_embeddings.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    sentence_embeddings = (
        token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    )
    return sentence_embeddings


class HuggingFaceEmbedding:
    """
    HuggingFace embedding model wrapper (e.g., facebook/contriever).
    Supports instruction-based models like Qwen3-Embedding and BGE.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        instruction: Optional[str] = None,
        instruction_template: Optional[str] = None,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()

        # Detect model family and set up instruction handling
        self.model_family = _get_model_family(model_name)
        self.instruction = instruction
        self.instruction_template = instruction_template

        # If instruction is provided but no template, use model family default
        if self.instruction and not self.instruction_template and self.model_family:
            self.instruction_template = self.model_family

    def _apply_instruction(
        self, texts: List[str], text_type: Literal["query", "document"] = "query"
    ) -> List[str]:
        """Apply instruction template to texts if configured."""
        if not self.instruction or not self.instruction_template:
            return texts

        template_config = INSTRUCTION_TEMPLATES.get(self.instruction_template)
        if not template_config:
            return texts

        template = template_config.get(text_type, "{text}")
        instruction = self.instruction or template_config.get("default_instruction", "")

        return [
            template.format(instruction=instruction, text=t) for t in texts
        ]

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        text_type: Literal["query", "document"] = "query",
    ) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: Text or list of texts to encode
            normalize: Whether to L2-normalize embeddings
            text_type: Type of text - "query" or "document" (affects instruction application)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Apply instruction if configured
        processed_texts = self._apply_instruction(texts, text_type)

        with torch.no_grad():
            inputs = self.tokenizer(
                processed_texts, padding=True, truncation=True, return_tensors="pt", max_length=512
            ).to(self.device)
            outputs = self.model(**inputs)
            embeddings = _mean_pooling(
                outputs.last_hidden_state, inputs["attention_mask"]
            )

            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings.cpu().numpy()


class OpenAIEmbedding:
    """
    OpenAI embedding model wrapper.
    """

    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        from openai import OpenAI
        from tenacity import retry, stop_after_attempt, wait_exponential

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._retry_decorator = retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
        )

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        text_type: Literal["query", "document"] = "query",
    ) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: Text or list of texts to encode
            normalize: Whether to L2-normalize embeddings
            text_type: Type of text (ignored for OpenAI, included for API consistency)
        """
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(model=self.model_name, input=texts)
        sorted_data = sorted(response.data, key=lambda x: x.index)
        embeddings = np.array([item.embedding for item in sorted_data])

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        return embeddings


class EmbeddingModel:
    """
    Unified embedding model wrapper.

    Supports both HuggingFace models (e.g., facebook/contriever) and OpenAI models.
    Also supports instruction-based models like Qwen3-Embedding and BGE.

    Example:
        >>> model = EmbeddingModel()  # Uses facebook/contriever by default
        >>> embedding = model.embed("Hello world")
        >>> print(len(embedding))
        768

        # With instruction for BGE
        >>> model = EmbeddingModel(model="BAAI/bge-base-en-v1.5", instruction="Represent this sentence for searching relevant passages")
        >>> query_emb = model.embed("What is AI?", text_type="query")
        >>> doc_emb = model.embed("AI is artificial intelligence.", text_type="document")
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model: Optional[str] = None,
        instruction: Optional[str] = None,
        instruction_template: Optional[str] = None,
    ):
        """
        Initialize the embedding model.

        Args:
            settings: Configuration settings.
            model: Override embedding model from settings.
            instruction: Custom instruction for query encoding (for models like Qwen3, BGE).
            instruction_template: Template style - "qwen3" or "bge". Auto-detected if not specified.
        """
        self.settings = settings or get_settings()
        self.model_name = model or self.settings.embedding_model
        self.instruction = instruction
        self.instruction_template = instruction_template

        if _is_openai_model(self.model_name):
            self.settings.validate_settings()  # Need API key for OpenAI
            self._backend = OpenAIEmbedding(
                model_name=self.model_name,
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
        else:
            # HuggingFace model
            self._backend = HuggingFaceEmbedding(
                model_name=self.model_name,
                instruction=instruction,
                instruction_template=instruction_template,
            )

        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """Get the embedding dimension by making a test call if needed."""
        if self._dimension is None:
            test_embedding = self.embed("test")
            self._dimension = len(test_embedding)
        return self._dimension

    def embed(
        self,
        text: str,
        text_type: Literal["query", "document"] = "query",
    ) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed.
            text_type: Type of text - "query" or "document" (affects instruction application).

        Returns:
            Embedding vector as a list of floats.
        """
        embeddings = self._backend.encode(text, text_type=text_type)
        return embeddings[0].tolist()

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        text_type: Literal["query", "document"] = "query",
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per batch.
            show_progress: Whether to show progress bar.
            text_type: Type of text - "query" or "document" (affects instruction application).

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        batch_size = batch_size or self.settings.batch_size
        all_embeddings = []

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        iterator = (
            tqdm(batches, desc="Generating embeddings") if show_progress else batches
        )

        for batch in iterator:
            embeddings = self._backend.encode(batch, text_type=text_type)
            all_embeddings.extend(embeddings.tolist())

        return all_embeddings
