"""
Configuration management for Vector Graph RAG.
"""

import hashlib
import os
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """A model name with an optional task-specific OpenAI-compatible connection."""

    model: str = Field(min_length=1)
    base_url: Optional[str] = None
    api_key: Optional[SecretStr] = Field(default=None, repr=False)


ModelSpec = str | ModelConfig
RECOMMENDED_JEV_MODEL = "jev-1.13.0"


def is_jev(model: str) -> bool:
    return model == "jev" or model.startswith("jev-")


class Settings(BaseSettings):
    """
    Configuration settings for Vector Graph RAG.

    Settings can be configured via environment variables or passed directly.
    Environment variables should be prefixed with VGRAG_ (e.g., VGRAG_OPENAI_API_KEY).
    """

    # OpenAI Settings
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key for LLM and embeddings",
    )
    openai_base_url: Optional[str] = Field(default=None, description="Custom OpenAI API base URL")

    # Model Settings
    llm_model: ModelSpec = Field(
        default="gpt-6-luna",
        description="LLM model for triplet extraction and reranking",
    )
    llm_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "medium"
    extractor_model: Optional[ModelSpec] = None
    reranker_model: Optional[ModelSpec] = None
    answer_model: Optional[ModelSpec] = None

    @field_validator("llm_model", "extractor_model", "reranker_model", "answer_model")
    @classmethod
    def nonempty_model(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("Model names must not be empty.")
        return value

    @model_validator(mode="after")
    def validate_models(self):
        for name in ("llm_model", "extractor_model", "answer_model"):
            spec = getattr(self, name)
            model = spec.model if isinstance(spec, ModelConfig) else spec
            if model and is_jev(model):
                raise ValueError(f"{name} does not support Jev; use reranker_model for Jev.")
        if self.reranker_model is not None:
            spec = self.reranker_model
            name = spec.model if isinstance(spec, ModelConfig) else spec
            provider = "jev" if is_jev(name) else "llm"
            if "reranker_provider" in self.model_fields_set and self.reranker_provider != provider:
                raise ValueError("reranker_model conflicts with reranker_provider.")
            if provider == "jev" and "jev_model" in self.model_fields_set:
                if name != "jev" and name != self.jev_model:
                    raise ValueError("reranker_model conflicts with jev_model.")
        return self

    @property
    def uses_jev(self) -> bool:
        if self.reranker_model is None:
            return self.reranker_provider == "jev"
        spec = self.reranker_model
        return is_jev(spec.model if isinstance(spec, ModelConfig) else spec)

    def task_model(self, task: Literal["extractor", "reranker", "answer"]) -> ModelConfig:
        """Resolve task overrides without mixing Jev and generative credentials."""
        spec = getattr(self, task + "_model")
        if task == "reranker" and self.uses_jev:
            config = (
                spec if isinstance(spec, ModelConfig) else ModelConfig(model=spec or self.jev_model)
            )
            name = self.jev_model if config.model == "jev" else config.model
            return ModelConfig(
                model=name,
                base_url=config.base_url or "https://api.typesafe.ai/v1",
                api_key=config.api_key or self.jev_api_key,
            )
        default = self.llm_model
        base = default if isinstance(default, ModelConfig) else ModelConfig(model=default)
        config = spec if isinstance(spec, ModelConfig) else ModelConfig(model=spec or base.model)
        # An explicit endpoint must not silently inherit a credential for another service.
        endpoint = config.base_url or base.base_url or self.openai_base_url
        inherited_key = base.api_key or self.openai_api_key
        if base.base_url and base.base_url != self.openai_base_url and not base.api_key:
            inherited_key = None
        if config.base_url and config.base_url != (base.base_url or self.openai_base_url):
            inherited_key = None
        return ModelConfig(
            model=config.model, base_url=endpoint, api_key=config.api_key or inherited_key
        )

    def cache_model_name(self, model: str) -> str:
        if model.startswith("gpt-6"):
            model = f"{model}-responses-{self.llm_reasoning_effort}"
        if (
            not self.openai_base_url
            or self.openai_base_url.rstrip("/") == "https://api.openai.com/v1"
        ):
            return model
        suffix = hashlib.sha256(self.openai_base_url.rstrip("/").encode()).hexdigest()[:16]
        return f"{model}-{suffix}"

    def for_task(self, task: Literal["extractor", "reranker", "answer"]) -> "Settings":
        config = self.task_model(task)
        if not config.api_key:
            raise ValueError(f"Configure an API key for {task}_model.")
        return self.model_copy(
            update={
                "llm_model": config.model,
                "openai_base_url": config.base_url,
                "openai_api_key": config.api_key.get_secret_value(),
            }
        )

    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="Embedding model for vector representations (HuggingFace model name or OpenAI model name)",
    )
    embedding_provider: Optional[str] = Field(
        default=None,
        description=(
            "Embedding provider name. Supported providers include openai, huggingface, "
            "google, gemini, voyage, jina, mistral, ollama, local, and onnx. "
            "If omitted, a legacy model-name inference path is used for compatibility."
        ),
    )
    embedding_api_key: Optional[str] = Field(
        default=None,
        description="Optional API key override for embedding providers.",
    )
    embedding_base_url: Optional[str] = Field(
        default=None,
        description="Optional base URL override for embedding providers that support it.",
    )
    embedding_dimension: int = Field(
        default=3072,
        description="Dimension of embedding vectors (3072 for text-embedding-3-large, 1024 for bge-large)",
    )

    # Milvus Index Settings
    milvus_index_type: str = Field(
        default="AUTOINDEX",
        description="Milvus index type (e.g., AUTOINDEX, FLAT, IVF_FLAT, HNSW, etc.)",
    )
    milvus_index_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional index params (e.g., {'nlist': 128} for IVF_FLAT, {'M': 16, 'efConstruction': 200} for HNSW)",
    )
    milvus_consistency_level: str = Field(
        default="Bounded",
        description="Milvus consistency level (Strong, Bounded, Session, Eventually)",
    )

    # Milvus Settings
    milvus_uri: str = Field(
        default="./vector_graph_rag.db",
        description="Milvus connection URI (file path for Milvus Lite, or server URI)",
    )
    milvus_token: Optional[str] = Field(
        default=None, description="Milvus authentication token (for Zilliz Cloud)"
    )
    milvus_db: Optional[str] = Field(default=None, description="Milvus database name")

    # Collection prefix (useful for multiple datasets)
    collection_prefix: Optional[str] = Field(
        default=None, description="Prefix for collection names (e.g., dataset name)"
    )

    # Collection Names
    entity_collection: str = Field(
        default="vgrag_entities", description="Collection name for entities"
    )
    relation_collection: str = Field(
        default="vgrag_relations", description="Collection name for relations"
    )
    passage_collection: str = Field(
        default="vgrag_passages", description="Collection name for passages"
    )

    # Retrieval Settings
    entity_top_k: int = Field(default=20, description="Number of top entities to retrieve")
    relation_top_k: int = Field(default=20, description="Number of top relations to retrieve")
    entity_similarity_threshold: float = Field(
        default=0.9,
        description="Similarity threshold for entity retrieval (keep if score > threshold)",
    )
    relation_similarity_threshold: float = Field(
        default=-1.0,
        description="Similarity threshold for relation retrieval (keep if score > threshold, -1 keeps all)",
    )
    expansion_degree: int = Field(
        default=1, description="Degree of subgraph expansion (1 or 2 recommended)"
    )
    relation_number_threshold: int = Field(
        default=1000,
        description="Maximum number of expanded relations. If exceeded, use eviction strategy to filter by similarity.",
    )
    final_top_k: int = Field(default=3, description="Number of final passages to return")

    # LLM Settings
    llm_temperature: float = Field(default=0.0, description="Temperature for LLM generation")
    llm_max_retries: int = Field(default=3, description="Maximum retries for LLM API calls")
    use_llm_cache: bool = Field(default=True, description="Whether to use LLM response caching")

    # Optional Jev relation reranker; extraction and answer generation still use OpenAI.
    reranker_provider: Literal["llm", "jev"] = "llm"
    jev_api_key: Optional[SecretStr] = Field(
        default_factory=lambda: os.getenv("TYPESAFE_API_KEY"), repr=False
    )
    jev_model: str = RECOMMENDED_JEV_MODEL
    jev_threshold: float = Field(default=0.5, ge=0, le=1)
    jev_timeout: float = Field(default=60.0, gt=0)
    jev_max_concurrency: int = Field(default=3, ge=1, le=16)

    # Processing Settings
    batch_size: int = Field(default=32, description="Batch size for embedding and insertion")

    # NER Cache Settings
    ner_cache_dir: Optional[str] = Field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # -> vector-graph-rag/
            "evaluation",
            "data",
            "ner_cache",
        ),
        description="Directory containing NER cache TSV files (HippoRAG format). "
        "Files should be named {dataset}_queries.named_entity_output.tsv",
    )

    model_config = {
        "env_prefix": "VGRAG_",
        "env_file": ".env",
        "extra": "ignore",
    }

    def validate_settings(self) -> None:
        """Validate that required settings are configured."""
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY or VGRAG_OPENAI_API_KEY "
                "environment variable, or pass openai_api_key to Settings."
            )


# Global default settings instance
_default_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the default settings instance."""
    global _default_settings
    if _default_settings is None:
        _default_settings = Settings()
    return _default_settings


def set_settings(settings: Settings) -> None:
    """Set the default settings instance."""
    global _default_settings
    _default_settings = settings
