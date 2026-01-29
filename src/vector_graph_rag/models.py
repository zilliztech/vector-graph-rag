"""
Data models for Vector Graph RAG.
"""

from typing import Optional, List, Any
from pydantic import BaseModel, Field


class Triplet(BaseModel):
    """
    A knowledge graph triplet representing a relationship between entities.

    Attributes:
        subject: The subject entity of the triplet.
        predicate: The relationship/predicate connecting subject and object.
        object: The object entity of the triplet.
    """

    subject: str = Field(..., description="Subject entity")
    predicate: str = Field(..., description="Predicate/relationship")
    object: str = Field(..., description="Object entity")

    def to_relation_text(self) -> str:
        """Convert triplet to a relation text string."""
        return f"{self.subject} {self.predicate} {self.object}"

    def __hash__(self) -> int:
        return hash((self.subject.lower(), self.predicate.lower(), self.object.lower()))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Triplet):
            return False
        return (
            self.subject.lower() == other.subject.lower()
            and self.predicate.lower() == other.predicate.lower()
            and self.object.lower() == other.object.lower()
        )


class Entity(BaseModel):
    """
    An entity extracted from text.

    Attributes:
        id: Unique identifier for the entity (string, UUID or user-provided).
        name: The name/text of the entity.
        embedding: Optional embedding vector.
    """

    id: Optional[str] = Field(default=None, description="Entity ID")
    name: str = Field(..., description="Entity name")
    embedding: Optional[List[float]] = Field(
        default=None, description="Entity embedding"
    )

    def __hash__(self) -> int:
        return hash(self.name.lower())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.name.lower() == other.name.lower()


class Relation(BaseModel):
    """
    A relation (edge) in the knowledge graph.

    Attributes:
        id: Unique identifier for the relation (string, UUID or user-provided).
        text: The full relation text (subject + predicate + object).
        triplet: The original triplet.
        source_passage_ids: IDs of passages where this relation was extracted from.
        embedding: Optional embedding vector.
    """

    id: Optional[str] = Field(default=None, description="Relation ID")
    text: str = Field(..., description="Full relation text")
    triplet: Triplet = Field(..., description="Original triplet")
    source_passage_ids: List[str] = Field(
        default_factory=list, description="Source passage IDs"
    )
    embedding: Optional[List[float]] = Field(
        default=None, description="Relation embedding"
    )


class Document(BaseModel):
    """
    A document/passage to be processed.

    Attributes:
        id: Unique identifier for the document (string, UUID or user-provided).
        text: The document text content.
        metadata: Optional metadata dictionary.
        triplets: Extracted triplets from this document.
        embedding: Optional embedding vector.
    """

    id: Optional[str] = Field(default=None, description="Document ID")
    text: str = Field(..., description="Document text content")
    metadata: dict = Field(default_factory=dict, description="Optional metadata")
    triplets: List[Triplet] = Field(
        default_factory=list, description="Extracted triplets"
    )
    embedding: Optional[List[float]] = Field(
        default=None, description="Document embedding"
    )


class Passage(BaseModel):
    """
    A passage in the knowledge graph.

    Attributes:
        id: Unique identifier for the passage (string, UUID or user-provided).
        text: The passage text content.
        entity_ids: IDs of entities in this passage.
        relation_ids: IDs of relations from this passage.
        embedding: Optional embedding vector.
    """

    id: Optional[str] = Field(default=None, description="Passage ID")
    text: str = Field(..., description="Passage text content")
    entity_ids: List[str] = Field(default_factory=list, description="Entity IDs")
    relation_ids: List[str] = Field(default_factory=list, description="Relation IDs")
    embedding: Optional[List[float]] = Field(
        default=None, description="Passage embedding"
    )


class RetrievalDetail(BaseModel):
    """
    Details of initial retrieval step (before expansion).

    Attributes:
        entity_ids: IDs of entities retrieved via vector search.
        entity_texts: Names of retrieved entities.
        entity_scores: Similarity scores of retrieved entities.
        relation_ids: IDs of relations retrieved via vector search.
        relation_texts: Texts of retrieved relations.
        relation_scores: Similarity scores of retrieved relations.
    """

    entity_ids: List[str] = Field(default_factory=list)
    entity_texts: List[str] = Field(default_factory=list)
    entity_scores: List[float] = Field(default_factory=list)
    relation_ids: List[str] = Field(default_factory=list)
    relation_texts: List[str] = Field(default_factory=list)
    relation_scores: List[float] = Field(default_factory=list)


class RerankResult(BaseModel):
    """
    Result of LLM reranking.

    Attributes:
        selected_relation_ids: IDs of relations selected by LLM reranker.
        selected_relation_texts: Texts of selected relations.
    """

    selected_relation_ids: List[str] = Field(default_factory=list)
    selected_relation_texts: List[str] = Field(default_factory=list)


class QueryResult(BaseModel):
    """
    Result of a Graph RAG query.

    Attributes:
        query: The original query.
        answer: The generated answer.
        query_entities: Entities extracted from the query.
        retrieved_passages: List of retrieved passage texts.
        retrieved_relations: List of retrieved relation texts.
        expanded_relations: Relations after subgraph expansion.
        reranked_relations: Relations after LLM reranking.
        subgraph: The expanded subgraph (for visualization).
        passages: Final passages used for answer generation.
        retrieval_detail: Details of initial retrieval step.
        rerank_result: Result of LLM reranking.
    """

    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    query_entities: List[str] = Field(
        default_factory=list, description="Query entities"
    )
    retrieved_passages: List[str] = Field(
        default_factory=list, description="Retrieved passages"
    )
    retrieved_relations: List[str] = Field(
        default_factory=list, description="Retrieved relations"
    )
    expanded_relations: List[str] = Field(
        default_factory=list, description="Expanded relations"
    )
    reranked_relations: List[str] = Field(
        default_factory=list, description="Reranked relations"
    )
    subgraph: Optional[Any] = Field(
        default=None, description="Expanded subgraph for visualization"
    )
    passages: List[str] = Field(default_factory=list, description="Final passages")
    retrieval_detail: Optional[RetrievalDetail] = Field(
        default=None, description="Details of initial retrieval"
    )
    rerank_result: Optional[RerankResult] = Field(
        default=None, description="LLM rerank result"
    )


class ExtractionResult(BaseModel):
    """
    Result of triplet extraction from documents.

    Attributes:
        documents: Processed documents with extracted triplets.
        entities: All unique entities extracted.
        relations: All unique relations extracted.
        entity_to_relation_ids: Mapping from entity ID to relation IDs.
        relation_to_passage_ids: Mapping from relation ID to passage IDs.
    """

    documents: List[Document] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    entity_to_relation_ids: dict[str, List[str]] = Field(default_factory=dict)
    relation_to_passage_ids: dict[str, List[str]] = Field(default_factory=dict)
