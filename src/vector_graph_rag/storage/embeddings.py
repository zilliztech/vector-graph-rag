"""
Embedding model wrapper for vector representations.

Supports both HuggingFace models (e.g., facebook/contriever) and OpenAI models.
"""

from typing import List, Optional, Union
import numpy as np
import torch
from tqdm import tqdm

from vector_graph_rag.config import Settings, get_settings


def _is_openai_model(model_name: str) -> bool:
    """Check if the model is an OpenAI embedding model."""
    openai_models = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]
    return model_name in openai_models or model_name.startswith("text-embedding")


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
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def encode(
        self, texts: Union[str, List[str]], normalize: bool = True
    ) -> np.ndarray:
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            inputs = self.tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
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
        self, texts: Union[str, List[str]], normalize: bool = True
    ) -> np.ndarray:
        """Encode texts to embeddings."""
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

    Example:
        >>> model = EmbeddingModel()  # Uses facebook/contriever by default
        >>> embedding = model.embed("Hello world")
        >>> print(len(embedding))
        768
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the embedding model.

        Args:
            settings: Configuration settings.
            model: Override embedding model from settings.
        """
        self.settings = settings or get_settings()
        self.model_name = model or self.settings.embedding_model

        if _is_openai_model(self.model_name):
            self.settings.validate_settings()  # Need API key for OpenAI
            self._backend = OpenAIEmbedding(
                model_name=self.model_name,
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
        else:
            # HuggingFace model
            self._backend = HuggingFaceEmbedding(model_name=self.model_name)

        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """Get the embedding dimension by making a test call if needed."""
        if self._dimension is None:
            test_embedding = self.embed("test")
            self._dimension = len(test_embedding)
        return self._dimension

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        embeddings = self._backend.encode(text)
        return embeddings[0].tolist()

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per batch.
            show_progress: Whether to show progress bar.

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
            embeddings = self._backend.encode(batch)
            all_embeddings.extend(embeddings.tolist())

        return all_embeddings
