"""
Configuration management for Vector Graph RAG.
"""

import os
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field


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
    openai_base_url: Optional[str] = Field(
        default=None, description="Custom OpenAI API base URL"
    )

    # Model Settings
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for triplet extraction and reranking",
    )
    embedding_model: str = Field(
        default="facebook/contriever",
        description="Embedding model for vector representations (HuggingFace model name or OpenAI model name)",
    )
    embedding_dimension: int = Field(
        default=768,
        description="Dimension of embedding vectors (768 for contriever, 1536 for text-embedding-3-small)",
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
    entity_top_k: int = Field(
        default=10, description="Number of top entities to retrieve"
    )
    relation_top_k: int = Field(
        default=10, description="Number of top relations to retrieve"
    )
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
    final_top_k: int = Field(
        default=3, description="Number of final passages to return"
    )

    # LLM Settings
    llm_temperature: float = Field(
        default=0.0, description="Temperature for LLM generation"
    )
    llm_max_retries: int = Field(
        default=3, description="Maximum retries for LLM API calls"
    )
    use_llm_cache: bool = Field(
        default=True, description="Whether to use LLM response caching"
    )

    # Processing Settings
    batch_size: int = Field(
        default=100, description="Batch size for embedding and insertion"
    )

    # NER Cache Settings
    ner_cache_dir: Optional[str] = Field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # -> vector-graph-rag-reference/
            "data", "ner_cache"
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
