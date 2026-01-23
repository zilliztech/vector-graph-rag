"""
Pydantic schemas for API responses.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class EntitySchema(BaseModel):
    """Entity in the knowledge graph."""

    id: int
    name: str
    relation_ids: List[int] = []
    passage_ids: List[int] = []


class RelationSchema(BaseModel):
    """Relation in the knowledge graph."""

    id: int
    text: str
    subject: str
    predicate: str
    object: str
    entity_ids: List[int] = []
    passage_ids: List[int] = []


class PassageSchema(BaseModel):
    """Passage in the knowledge graph."""

    id: int
    text: str


class ExpansionStepSchema(BaseModel):
    """A single step in the expansion history."""

    step: int
    operation: str
    description: Optional[str] = None
    new_entity_ids: List[int] = []
    new_relation_ids: List[int] = []
    total_entities: int = 0
    total_relations: int = 0


class SubGraphSchema(BaseModel):
    """Subgraph data for visualization."""

    entity_ids: List[int]
    relation_ids: List[int]
    passage_ids: List[int]
    entities: List[EntitySchema]
    relations: List[RelationSchema]
    passages: List[PassageSchema]
    expansion_history: List[Dict[str, Any]]


class RetrievalDetailSchema(BaseModel):
    """Details of initial retrieval step (before expansion)."""

    entity_ids: List[int] = []
    entity_texts: List[str] = []
    entity_scores: List[float] = []
    relation_ids: List[int] = []
    relation_texts: List[str] = []
    relation_scores: List[float] = []


class RerankResultSchema(BaseModel):
    """Result of LLM reranking."""

    selected_relation_ids: List[int] = []
    selected_relation_texts: List[str] = []


class QueryRequest(BaseModel):
    """Query request."""

    question: str
    graph_name: Optional[str] = (
        None  # Graph/dataset to query (uses default if not specified)
    )
    entity_top_k: int = 10
    relation_top_k: int = 10
    entity_similarity_threshold: float = 0.9
    relation_similarity_threshold: float = -1.0
    expansion_degree: int = 1


class QueryResponse(BaseModel):
    """Query response with subgraph and answer."""

    question: str
    answer: str
    query_entities: List[str]
    subgraph: SubGraphSchema
    retrieved_passages: List[str]
    stats: Dict[str, Any]
    retrieval_detail: Optional[RetrievalDetailSchema] = None
    rerank_result: Optional[RerankResultSchema] = None


class GraphStatsResponse(BaseModel):
    """Graph statistics."""

    entities: int
    relations: int
    passages: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    message: str


class GraphInfo(BaseModel):
    """Information about an available graph."""

    name: str
    entity_collection: str
    relation_collection: str
    passage_collection: str
    has_all_collections: bool


class MilvusConfig(BaseModel):
    """Milvus connection configuration (read-only)."""

    uri: str
    database: Optional[str] = None
    has_token: bool = False  # Only indicate if token exists, don't expose it


class ListGraphsResponse(BaseModel):
    """Response for listing available graphs."""

    graphs: List[GraphInfo]
    milvus_config: MilvusConfig
