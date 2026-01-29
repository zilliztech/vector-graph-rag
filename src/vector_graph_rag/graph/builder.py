"""
Graph builder for constructing knowledge graph structures from extracted triplets.
"""

import uuid
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
from vector_graph_rag.llm.extractor import processing_phrases


def generate_id() -> str:
    """Generate a unique ID using UUID4."""
    return str(uuid.uuid4())


class GraphBuilder:
    """
    Build graph structures from extracted triplets.

    Extracts entities, relations, and adjacency mappings from documents.
    All IDs are now strings (UUIDs or user-provided).

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
        self.clear()

    def clear(self) -> None:
        """Clear all graph data structures."""
        # Graph data structures - indexed by string IDs
        self.entities: Dict[str, str] = {}  # id -> name
        self.relations: Dict[str, str] = {}  # id -> text
        self.passages: Dict[str, str] = {}  # id -> text

        # For ordered iteration
        self.entity_ids: List[str] = []
        self.relation_ids: List[str] = []
        self.passage_ids: List[str] = []

        # Mappings for deduplication
        self.entity_name_to_id: Dict[str, str] = {}  # normalized name -> id
        self.relation_text_to_id: Dict[str, str] = {}  # relation text -> id

        # Store original triplets for relations (to avoid parsing from text)
        self.relation_id_to_triplet: Dict[str, Triplet] = {}

        # Adjacency data - all string IDs
        self.entity_to_relation_ids: Dict[str, List[str]] = defaultdict(list)
        self.entity_to_passage_ids: Dict[str, List[str]] = defaultdict(list)
        self.relation_to_passage_ids: Dict[str, List[str]] = defaultdict(list)
        self.relation_to_entity_ids: Dict[str, List[str]] = defaultdict(list)
        self.passage_to_entity_ids: Dict[str, List[str]] = defaultdict(list)
        self.passage_to_relation_ids: Dict[str, List[str]] = defaultdict(list)

    def _add_entity(self, entity_name: str, passage_id: str) -> str:
        """Add an entity and return its ID."""
        normalized = processing_phrases(entity_name)
        if normalized not in self.entity_name_to_id:
            entity_id = generate_id()
            self.entities[entity_id] = normalized
            self.entity_ids.append(entity_id)
            self.entity_name_to_id[normalized] = entity_id

        entity_id = self.entity_name_to_id[normalized]

        # Link entity to passage
        if passage_id not in self.entity_to_passage_ids[entity_id]:
            self.entity_to_passage_ids[entity_id].append(passage_id)

        # Link passage to entity
        if entity_id not in self.passage_to_entity_ids[passage_id]:
            self.passage_to_entity_ids[passage_id].append(entity_id)

        return entity_id

    def _add_relation(self, triplet: Triplet, passage_id: str) -> str:
        """Add a relation and return its ID."""
        # Normalize each part of the triplet and build relation text
        subject = processing_phrases(triplet.subject)
        predicate = processing_phrases(triplet.predicate)
        obj = processing_phrases(triplet.object)
        relation_text = f"{subject} {predicate} {obj}"

        if relation_text not in self.relation_text_to_id:
            relation_id = generate_id()
            self.relations[relation_id] = relation_text
            self.relation_ids.append(relation_id)
            self.relation_text_to_id[relation_text] = relation_id

            # Store the original triplet (to avoid re-parsing from text)
            self.relation_id_to_triplet[relation_id] = triplet

            # Link entities to this relation (also links entity to passage)
            subject_id = self._add_entity(triplet.subject, passage_id)
            object_id = self._add_entity(triplet.object, passage_id)

            self.entity_to_relation_ids[subject_id].append(relation_id)
            self.entity_to_relation_ids[object_id].append(relation_id)

            # Link relation to entities
            self.relation_to_entity_ids[relation_id] = [subject_id, object_id]

        relation_id = self.relation_text_to_id[relation_text]

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
            # Use document's ID if provided, otherwise generate one
            passage_id = doc.id if doc.id else generate_id()
            self.passages[passage_id] = doc.text
            self.passage_ids.append(passage_id)

            # Update the document's ID if it was generated
            if not doc.id:
                doc.id = passage_id

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
        entities = [
            Entity(id=eid, name=self.entities[eid])
            for eid in self.entity_ids
        ]

        relations = []
        for rid in self.relation_ids:
            triplet = self.relation_id_to_triplet[rid]
            relations.append(
                Relation(
                    id=rid,
                    text=self.relations[rid],
                    triplet=triplet,
                    source_passage_ids=list(self.relation_to_passage_ids[rid]),
                )
            )

        return ExtractionResult(
            documents=documents,
            entities=entities,
            relations=relations,
            entity_to_relation_ids=dict(self.entity_to_relation_ids),
            relation_to_passage_ids=dict(self.relation_to_passage_ids),
        )

    def get_entity_id(self, entity_name: str) -> Optional[str]:
        """Get entity ID by name."""
        return self.entity_name_to_id.get(processing_phrases(entity_name))

    def get_relation_id(self, relation_text: str) -> Optional[str]:
        """Get relation ID by text."""
        return self.relation_text_to_id.get(relation_text)

    def get_entity_texts(self) -> List[str]:
        """Get all entity texts in order."""
        return [self.entities[eid] for eid in self.entity_ids]

    def get_relation_texts(self) -> List[str]:
        """Get all relation texts in order."""
        return [self.relations[rid] for rid in self.relation_ids]

    def get_passage_texts(self) -> List[str]:
        """Get all passage texts in order."""
        return [self.passages[pid] for pid in self.passage_ids]
