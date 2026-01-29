"""
SubGraph - Lazy-loading subgraph abstraction for graph expansion.

Provides a clean graph interface for subgraph expansion and traversal,
with on-demand data fetching from Milvus storage.
"""

from __future__ import annotations
from typing import List, Dict, Set, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from vector_graph_rag.storage.milvus import MilvusStore


@dataclass
class GraphEntity:
    """
    An entity node in the subgraph.

    Attributes:
        id: Unique identifier for the entity (string).
        name: The name/text of the entity.
        relation_ids: IDs of relations connected to this entity.
        passage_ids: IDs of passages where this entity appears.
    """

    id: str
    name: str
    relation_ids: List[str] = field(default_factory=list)
    passage_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "relation_ids": self.relation_ids,
            "passage_ids": self.passage_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEntity":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            relation_ids=data.get("relation_ids", []),
            passage_ids=data.get("passage_ids", []),
        )


@dataclass
class GraphRelation:
    """
    A relation edge in the subgraph.

    Attributes:
        id: Unique identifier for the relation (string).
        text: The full relation text (subject + predicate + object).
        subject: Subject entity name.
        predicate: Predicate/relationship.
        object: Object entity name.
        entity_ids: IDs of connected entities [subject_id, object_id].
        passage_ids: IDs of source passages.
    """

    id: str
    text: str
    subject: str
    predicate: str
    object: str
    entity_ids: List[str] = field(default_factory=list)
    passage_ids: List[str] = field(default_factory=list)

    @property
    def head_entity_id(self) -> Optional[str]:
        """Get the subject entity ID."""
        return self.entity_ids[0] if self.entity_ids else None

    @property
    def tail_entity_id(self) -> Optional[str]:
        """Get the object entity ID."""
        return self.entity_ids[1] if len(self.entity_ids) > 1 else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "entity_ids": self.entity_ids,
            "passage_ids": self.passage_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphRelation":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            entity_ids=data.get("entity_ids", []),
            passage_ids=data.get("passage_ids", []),
        )


@dataclass
class GraphPassage:
    """
    A passage node in the subgraph.

    Attributes:
        id: Unique identifier for the passage (string).
        text: The passage text content.
        entity_ids: IDs of entities mentioned in this passage.
        relation_ids: IDs of relations extracted from this passage.
    """

    id: str
    text: str
    entity_ids: List[str] = field(default_factory=list)
    relation_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "entity_ids": self.entity_ids,
            "relation_ids": self.relation_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphPassage":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            entity_ids=data.get("entity_ids", []),
            relation_ids=data.get("relation_ids", []),
        )


class SubGraph:
    """
    Lazy-loading subgraph for graph expansion.

    Starts from seed entities/relations and expands by fetching
    neighbor data on-demand from Milvus storage.

    Features:
    - Lazy loading: only fetches data when needed during expansion
    - Caching: caches fetched data to avoid repeated queries
    - Expansion history: tracks each expansion step for debugging/visualization

    Example:
        >>> subgraph = SubGraph(store)
        >>> # Start from relation IDs obtained via vector search
        >>> subgraph.add_relations([10, 20, 30])
        >>> # Expand by 2 degrees
        >>> subgraph.expand(degree=2)
        >>> # Access the expanded data
        >>> print(subgraph.relation_texts)
        >>> print(subgraph.expansion_history)
    """

    def __init__(self, store: "MilvusStore"):
        """
        Initialize a subgraph.

        Args:
            store: MilvusStore for fetching data on-demand.
        """
        self._store = store

        # Node IDs in this subgraph (all string IDs)
        self._entity_ids: Set[str] = set()
        self._relation_ids: Set[str] = set()
        self._passage_ids: Set[str] = set()

        # Cached node data
        self._entities: Dict[str, GraphEntity] = {}
        self._relations: Dict[str, GraphRelation] = {}
        self._passages: Dict[str, GraphPassage] = {}

        # Expansion history for debugging/visualization
        self._expansion_history: List[Dict[str, Any]] = []

    # ==================== Add Initial Nodes ====================

    def add_entities(self, entity_ids: List[str]) -> "SubGraph":
        """
        Add initial entity IDs to the subgraph.

        Args:
            entity_ids: Entity IDs to add (e.g., from vector search).

        Returns:
            Self for chaining.
        """
        new_ids = set(entity_ids) - self._entity_ids
        self._entity_ids.update(new_ids)

        if new_ids:
            self._fetch_entities(list(new_ids))

            if not self._expansion_history:
                self._expansion_history.append(
                    {
                        "step": 0,
                        "operation": "init",
                        "added_entity_ids": list(new_ids),
                        "added_relation_ids": [],
                    }
                )
            else:
                self._expansion_history[0]["added_entity_ids"].extend(new_ids)

        return self

    def add_relations(self, relation_ids: List[str]) -> "SubGraph":
        """
        Add initial relation IDs to the subgraph.

        Args:
            relation_ids: Relation IDs to add (e.g., from vector search).

        Returns:
            Self for chaining.
        """
        new_ids = set(relation_ids) - self._relation_ids
        self._relation_ids.update(new_ids)

        if new_ids:
            self._fetch_relations(list(new_ids))

            if not self._expansion_history:
                self._expansion_history.append(
                    {
                        "step": 0,
                        "operation": "init",
                        "added_entity_ids": [],
                        "added_relation_ids": list(new_ids),
                    }
                )
            else:
                self._expansion_history[0]["added_relation_ids"].extend(new_ids)

        return self

    # ==================== Expansion ====================

    def expand(self, degree: int = 1) -> "SubGraph":
        """
        Expand the subgraph by the given degree.

        Expansion logic:
        1. First, from initial entities -> find connected relations, merge with initial relations
           This merged set becomes the starting state for expansion.
        2. For each degree:
           - From current relations -> find connected entities
           - From those entities -> find connected relations
           This is "one degree" = relations expanding to next-hop relations.

        Data is fetched lazily from Milvus during expansion.

        Args:
            degree: Number of hops to expand.

        Returns:
            Self for chaining.
        """
        # Step 0: From initial entities -> relations, merge with initial relations
        # This creates the starting state
        init_new_relations: Set[str] = set()
        for eid in list(self._entity_ids):
            entity = self._entities.get(eid)
            if entity:
                for rid in entity.relation_ids:
                    if rid not in self._relation_ids:
                        init_new_relations.add(rid)

        if init_new_relations:
            self._fetch_relations(list(init_new_relations))
            self._relation_ids.update(init_new_relations)

        # Record initialization step
        self._expansion_history.append(
            {
                "step": len(self._expansion_history),
                "operation": "init_merge",
                "description": "Merged relations from initial entities with initial relations",
                "new_relation_ids": list(init_new_relations),
                "total_entities": len(self._entity_ids),
                "total_relations": len(self._relation_ids),
            }
        )

        # For each degree: relations -> entities -> relations
        for step in range(degree):
            step_new_entities: Set[str] = set()
            step_new_relations: Set[str] = set()

            # From current relations -> entities
            for rid in list(self._relation_ids):
                relation = self._relations.get(rid)
                if relation:
                    for eid in relation.entity_ids:
                        if eid not in self._entity_ids:
                            step_new_entities.add(eid)

            # Fetch new entities
            if step_new_entities:
                self._fetch_entities(list(step_new_entities))
                self._entity_ids.update(step_new_entities)

            # From new entities -> relations (next hop)
            for eid in step_new_entities:
                entity = self._entities.get(eid)
                if entity:
                    for rid in entity.relation_ids:
                        if rid not in self._relation_ids:
                            step_new_relations.add(rid)

            # Fetch new relations
            if step_new_relations:
                self._fetch_relations(list(step_new_relations))
                self._relation_ids.update(step_new_relations)

            # Record expansion step
            self._expansion_history.append(
                {
                    "step": len(self._expansion_history),
                    "operation": f"expand_degree_{step + 1}",
                    "description": f"Relations -> entities -> relations (hop {step + 1})",
                    "new_entity_ids": list(step_new_entities),
                    "new_relation_ids": list(step_new_relations),
                    "total_entities": len(self._entity_ids),
                    "total_relations": len(self._relation_ids),
                }
            )

        # Collect passages from all relations
        for rid in self._relation_ids:
            relation = self._relations.get(rid)
            if relation:
                self._passage_ids.update(relation.passage_ids)

        # Fetch passages
        if self._passage_ids:
            self._fetch_passages(list(self._passage_ids))

        return self

    # ==================== Fetch from Milvus ====================

    def _fetch_entities(self, entity_ids: List[str]) -> None:
        """Fetch entities from Milvus and cache them."""
        if not entity_ids:
            return

        ids_to_fetch = [eid for eid in entity_ids if eid not in self._entities]
        if not ids_to_fetch:
            return

        # Format IDs as quoted strings for Milvus filter
        ids_str = ", ".join(f'"{eid}"' for eid in ids_to_fetch)
        filter_expr = f"id in [{ids_str}]"
        results = self._store.client.query(
            collection_name=self._store.entity_collection,
            filter=filter_expr,
            output_fields=["id", "text", "relation_ids", "passage_ids"],
        )

        for r in results:
            entity = GraphEntity(
                id=r["id"],
                name=r["text"],
                relation_ids=r.get("relation_ids", []),
                passage_ids=r.get("passage_ids", []),
            )
            self._entities[entity.id] = entity

    def _fetch_relations(self, relation_ids: List[str]) -> None:
        """Fetch relations from Milvus and cache them."""
        if not relation_ids:
            return

        ids_to_fetch = [rid for rid in relation_ids if rid not in self._relations]
        if not ids_to_fetch:
            return

        # Format IDs as quoted strings for Milvus filter
        ids_str = ", ".join(f'"{rid}"' for rid in ids_to_fetch)
        filter_expr = f"id in [{ids_str}]"
        results = self._store.client.query(
            collection_name=self._store.relation_collection,
            filter=filter_expr,
            output_fields=["id", "text", "entity_ids", "passage_ids", "subject", "predicate", "object"],
        )

        for r in results:
            text = r["text"]
            # Use stored triplet fields if available, otherwise parse from text
            subject = r.get("subject") or ""
            predicate = r.get("predicate") or ""
            obj = r.get("object") or ""

            # Fallback to parsing if fields are empty
            if not subject and not predicate and not obj:
                parts = text.split(" ", 2)
                subject = parts[0] if len(parts) > 0 else ""
                predicate = parts[1] if len(parts) > 1 else ""
                obj = parts[2] if len(parts) > 2 else ""

            relation = GraphRelation(
                id=r["id"],
                text=text,
                subject=subject,
                predicate=predicate,
                object=obj,
                entity_ids=r.get("entity_ids", []),
                passage_ids=r.get("passage_ids", []),
            )
            self._relations[relation.id] = relation

    def _fetch_passages(self, passage_ids: List[str]) -> None:
        """Fetch passages from Milvus and cache them."""
        if not passage_ids:
            return

        ids_to_fetch = [pid for pid in passage_ids if pid not in self._passages]
        if not ids_to_fetch:
            return

        # Format IDs as quoted strings for Milvus filter
        ids_str = ", ".join(f'"{pid}"' for pid in ids_to_fetch)
        filter_expr = f"id in [{ids_str}]"
        results = self._store.client.query(
            collection_name=self._store.passage_collection,
            filter=filter_expr,
            output_fields=["id", "text", "entity_ids", "relation_ids"],
        )

        for r in results:
            passage = GraphPassage(
                id=r["id"],
                text=r["text"],
                entity_ids=r.get("entity_ids", []),
                relation_ids=r.get("relation_ids", []),
            )
            self._passages[passage.id] = passage

    # ==================== Accessors ====================

    @property
    def entity_ids(self) -> Set[str]:
        """Get entity IDs in this subgraph."""
        return self._entity_ids

    @property
    def relation_ids(self) -> Set[str]:
        """Get relation IDs in this subgraph."""
        return self._relation_ids

    @property
    def passage_ids(self) -> Set[str]:
        """Get passage IDs in this subgraph."""
        return self._passage_ids

    @property
    def entities(self) -> List[GraphEntity]:
        """Get all entity objects in this subgraph."""
        return [
            self._entities[eid] for eid in self._entity_ids if eid in self._entities
        ]

    @property
    def relations(self) -> List[GraphRelation]:
        """Get all relation objects in this subgraph."""
        return [
            self._relations[rid] for rid in self._relation_ids if rid in self._relations
        ]

    @property
    def passages(self) -> List[GraphPassage]:
        """Get all passage objects in this subgraph."""
        return [
            self._passages[pid] for pid in self._passage_ids if pid in self._passages
        ]

    @property
    def entity_names(self) -> List[str]:
        """Get entity names in this subgraph."""
        return [e.name for e in self.entities]

    @property
    def relation_texts(self) -> List[str]:
        """Get relation texts in this subgraph."""
        return [r.text for r in self.relations]

    @property
    def passage_texts(self) -> List[str]:
        """Get passage texts in this subgraph."""
        return [p.text for p in self.passages]

    @property
    def expansion_history(self) -> List[Dict[str, Any]]:
        """Get expansion history for debugging/visualization."""
        return self._expansion_history

    def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)

    def get_relation(self, relation_id: str) -> Optional[GraphRelation]:
        """Get relation by ID."""
        return self._relations.get(relation_id)

    def get_passage(self, passage_id: str) -> Optional[GraphPassage]:
        """Get passage by ID."""
        return self._passages.get(passage_id)

    # ==================== Serialization ====================

    def to_dict(self) -> Dict[str, Any]:
        """Convert subgraph to dictionary for serialization/debugging."""
        return {
            "entity_ids": list(self._entity_ids),
            "relation_ids": list(self._relation_ids),
            "passage_ids": list(self._passage_ids),
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "passages": [p.to_dict() for p in self.passages],
            "expansion_history": self._expansion_history,
        }

    def stats(self) -> Dict[str, int]:
        """Get subgraph statistics."""
        return {
            "entities": len(self._entity_ids),
            "relations": len(self._relation_ids),
            "passages": len(self._passage_ids),
        }

    def __repr__(self) -> str:
        return (
            f"SubGraph(entities={len(self._entity_ids)}, "
            f"relations={len(self._relation_ids)}, "
            f"passages={len(self._passage_ids)})"
        )
