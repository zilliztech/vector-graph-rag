"""
Graph builder for constructing knowledge graph structures from extracted triplets.
"""

from collections import defaultdict
from typing import List, Dict, Optional

from vector_graph_rag.models import (
    Document,
    Entity,
    Relation,
    Triplet,
    ExtractionResult,
)
from vector_graph_rag.config import Settings, get_settings


class GraphBuilder:
    """
    Build graph structures from extracted triplets.

    Extracts entities, relations, and adjacency mappings from documents.

    Example:
        >>> builder = GraphBuilder()
        >>> result = builder.build_from_documents(documents)
        >>> print(len(result.entities))
        10
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the graph builder.

        Args:
            settings: Configuration settings.
        """
        self.settings = settings or get_settings()

        # Graph data structures
        self.entities: List[str] = []
        self.relations: List[str] = []
        self.passages: List[str] = []

        # Mappings
        self.entity_to_id: Dict[str, int] = {}
        self.relation_to_id: Dict[str, int] = {}

        # Store original triplets for relations (to avoid parsing from text)
        self.relation_id_to_triplet: Dict[int, Triplet] = {}

        # Adjacency data
        self.entity_to_relation_ids: Dict[int, List[int]] = defaultdict(list)
        self.entity_to_passage_ids: Dict[int, List[int]] = defaultdict(list)
        self.relation_to_passage_ids: Dict[int, List[int]] = defaultdict(list)
        self.relation_to_entity_ids: Dict[int, List[int]] = defaultdict(list)
        self.passage_to_entity_ids: Dict[int, List[int]] = defaultdict(list)
        self.passage_to_relation_ids: Dict[int, List[int]] = defaultdict(list)

    def clear(self) -> None:
        """Clear all graph data structures."""
        self.entities = []
        self.relations = []
        self.passages = []
        self.entity_to_id = {}
        self.relation_to_id = {}
        self.relation_id_to_triplet = {}
        self.entity_to_relation_ids = defaultdict(list)
        self.entity_to_passage_ids = defaultdict(list)
        self.relation_to_passage_ids = defaultdict(list)
        self.relation_to_entity_ids = defaultdict(list)
        self.passage_to_entity_ids = defaultdict(list)
        self.passage_to_relation_ids = defaultdict(list)

    def _add_entity(self, entity_name: str, passage_id: int) -> int:
        """Add an entity and return its ID."""
        normalized = entity_name.strip()
        if normalized not in self.entity_to_id:
            entity_id = len(self.entities)
            self.entities.append(normalized)
            self.entity_to_id[normalized] = entity_id

        entity_id = self.entity_to_id[normalized]

        # Link entity to passage
        if passage_id not in self.entity_to_passage_ids[entity_id]:
            self.entity_to_passage_ids[entity_id].append(passage_id)

        # Link passage to entity
        if entity_id not in self.passage_to_entity_ids[passage_id]:
            self.passage_to_entity_ids[passage_id].append(entity_id)

        return entity_id

    def _add_relation(self, triplet: Triplet, passage_id: int) -> int:
        """Add a relation and return its ID."""
        relation_text = triplet.to_relation_text()

        if relation_text not in self.relation_to_id:
            relation_id = len(self.relations)
            self.relations.append(relation_text)
            self.relation_to_id[relation_text] = relation_id

            # Store the original triplet (to avoid re-parsing from text)
            self.relation_id_to_triplet[relation_id] = triplet

            # Link entities to this relation (also links entity to passage)
            subject_id = self._add_entity(triplet.subject, passage_id)
            object_id = self._add_entity(triplet.object, passage_id)

            self.entity_to_relation_ids[subject_id].append(relation_id)
            self.entity_to_relation_ids[object_id].append(relation_id)

            # Link relation to entities
            self.relation_to_entity_ids[relation_id] = [subject_id, object_id]

        relation_id = self.relation_to_id[relation_text]

        # Link relation to passage
        if passage_id not in self.relation_to_passage_ids[relation_id]:
            self.relation_to_passage_ids[relation_id].append(passage_id)

        # Link passage to relation
        if relation_id not in self.passage_to_relation_ids[passage_id]:
            self.passage_to_relation_ids[passage_id].append(relation_id)

        return relation_id

    def _process_documents(self, documents: List[Document]) -> None:
        """Process documents to extract graph structure."""
        self.clear()

        for doc in documents:
            passage_id = len(self.passages)
            self.passages.append(doc.text)

            for triplet in doc.triplets:
                self._add_relation(triplet, passage_id)

    def build_from_documents(self, documents: List[Document]) -> ExtractionResult:
        """
        Build graph structures from documents with extracted triplets.

        This method maintains backward compatibility with the old API.

        Args:
            documents: Documents with triplets already extracted.

        Returns:
            ExtractionResult containing all graph data.
        """
        self._process_documents(documents)

        # Build result
        entities = [Entity(id=i, name=name) for i, name in enumerate(self.entities)]

        relations = []
        for i, text in enumerate(self.relations):
            triplet = self.relation_id_to_triplet[i]
            relations.append(
                Relation(
                    id=i,
                    text=text,
                    triplet=triplet,
                    source_passage_ids=list(self.relation_to_passage_ids[i]),
                )
            )

        return ExtractionResult(
            documents=documents,
            entities=entities,
            relations=relations,
            entity_to_relation_ids=dict(self.entity_to_relation_ids),
            relation_to_passage_ids=dict(self.relation_to_passage_ids),
        )

    def get_entity_id(self, entity_name: str) -> Optional[int]:
        """Get entity ID by name."""
        return self.entity_to_id.get(entity_name.strip())

    def get_relation_id(self, relation_text: str) -> Optional[int]:
        """Get relation ID by text."""
        return self.relation_to_id.get(relation_text)
