"""
Graph abstraction layer for user-facing CRUD operations.

This module provides the Graph class as the main user interface for
graph operations. It shields users from the Milvus layer and provides
logical graph CRUD operations.

Design principles:
- Entity and Relation CRUD methods are private (prefixed with _)
- Users mainly interact through Passage interface
- Internal operations automatically handle Entity/Relation creation and linking
"""

from typing import List, Optional, Dict, Any

from vector_graph_rag.config import Settings, get_settings
from vector_graph_rag.storage.milvus import MilvusStore, generate_id
from vector_graph_rag.storage.embeddings import EmbeddingModel
from vector_graph_rag.models import Entity, Relation, Passage, Triplet
from vector_graph_rag.graph.knowledge_graph import SubGraph
from vector_graph_rag.llm.extractor import processing_phrases


class Graph:
    """
    User-facing graph operations interface.

    Shields users from Milvus details and provides logical-level graph CRUD operations.

    Design:
    - Entity and Relation CRUD methods use _ prefix as private methods
    - Users mainly use Passage interface (since users have passages)
    - Internal operations automatically handle Entity/Relation creation and linking

    Example:
        >>> graph = Graph()
        >>>
        >>> # Create passage (auto-generate UUID if no ID provided)
        >>> passage_id = graph.create_passage("Zhang San is an engineer at Alibaba")
        >>>
        >>> # Create passage with custom ID (for evaluation datasets)
        >>> passage_id = graph.create_passage(
        ...     text="Zhang San is an engineer at Alibaba",
        ...     id="doc_001"
        ... )
        >>>
        >>> # Create passage with triplets (auto-create entity and relation)
        >>> passage_id = graph.create_passage(
        ...     text="Zhang San is an engineer at Alibaba",
        ...     id="doc_002",
        ...     triplets=[Triplet(subject="Zhang San", predicate="works at", object="Alibaba")]
        ... )
        >>>
        >>> # Get passage
        >>> passage = graph.get_passage("doc_001")
        >>>
        >>> # Search passages by vector similarity
        >>> passages = graph.search_passages("Alibaba employees", top_k=5)
        >>>
        >>> # Update passage
        >>> graph.update_passage("doc_001", text="Zhang San is a senior engineer at Alibaba")
        >>>
        >>> # Delete passage (auto cascade update related entity and relation)
        >>> graph.delete_passage("doc_001")
        >>>
        >>> # Create subgraph for lazy loading retrieval
        >>> subgraph = graph.create_subgraph()
        >>> subgraph.add_entities(["entity_001", "entity_002"]).expand(degree=1)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        store: Optional[MilvusStore] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        """
        Initialize the Graph.

        Args:
            settings: Configuration settings.
            store: Optional MilvusStore instance (for sharing with other components).
            embedding_model: Optional embedding model instance.
        """
        self.settings = settings or get_settings()
        self._embedding_model = embedding_model or EmbeddingModel(settings=self.settings)
        self._store = store or MilvusStore(
            settings=self.settings,
            embedding_model=self._embedding_model,
        )

        # Internal mappings for deduplication
        self._entity_name_to_id: Dict[str, str] = {}  # normalized name -> id
        self._relation_text_to_id: Dict[str, str] = {}  # relation text -> id

    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for deduplication."""
        return processing_phrases(name)

    # ==================== Private: Entity CRUD ====================

    def _create_entity(
        self,
        name: str,
        id: Optional[str] = None,
        relation_ids: Optional[List[str]] = None,
        passage_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Create an entity.

        Args:
            name: Entity name.
            id: Optional ID. If not provided, UUID is generated.
            relation_ids: Optional list of connected relation IDs.
            passage_ids: Optional list of source passage IDs.

        Returns:
            Entity ID (provided or generated).
        """
        normalized_name = self._normalize_entity_name(name)

        # Check if entity already exists
        if normalized_name in self._entity_name_to_id:
            existing_id = self._entity_name_to_id[normalized_name]
            # Update relations and passages if provided
            if relation_ids or passage_ids:
                existing = self._store._get_entities_by_ids([existing_id])
                if existing:
                    current = existing[0]
                    new_relation_ids = list(set(current.get("relation_ids", []) + (relation_ids or [])))
                    new_passage_ids = list(set(current.get("passage_ids", []) + (passage_ids or [])))
                    self._store._update_entity(
                        existing_id,
                        relation_ids=new_relation_ids,
                        passage_ids=new_passage_ids,
                    )
            return existing_id

        # Generate ID if not provided
        entity_id = id or generate_id()

        # Generate embedding
        embedding = self._embedding_model.embed(normalized_name)

        # Build metadata
        metadata = {}
        if relation_ids:
            metadata["relation_ids"] = relation_ids
        if passage_ids:
            metadata["passage_ids"] = passage_ids

        # Insert into Milvus
        self._store._insert_entities(
            [normalized_name],
            ids=[entity_id],
            embeddings=[embedding],
            metadatas=[metadata] if metadata else None,
        )

        # Update internal mapping
        self._entity_name_to_id[normalized_name] = entity_id

        return entity_id

    def _create_entities(
        self,
        names: List[str],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Create multiple entities.

        Args:
            names: List of entity names.
            ids: Optional list of IDs. If not provided, UUIDs are generated.

        Returns:
            List of entity IDs.
        """
        if not names:
            return []

        result_ids = []
        for i, name in enumerate(names):
            entity_id = ids[i] if ids and i < len(ids) else None
            result_ids.append(self._create_entity(name, id=entity_id))

        return result_ids

    def _get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by ID.

        Args:
            entity_id: Entity ID.

        Returns:
            Entity object or None if not found.
        """
        results = self._store._get_entities_by_ids([entity_id])
        if not results:
            return None

        data = results[0]
        return Entity(
            id=data["id"],
            name=data["text"],
        )

    def _update_entity(
        self,
        entity_id: str,
        name: Optional[str] = None,
        relation_ids: Optional[List[str]] = None,
        passage_ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Update an entity.

        Args:
            entity_id: Entity ID to update.
            name: New name (optional).
            relation_ids: New relation IDs (optional).
            passage_ids: New passage IDs (optional).

        Returns:
            True if update succeeded, False if entity not found.
        """
        return self._store._update_entity(
            entity_id,
            text=self._normalize_entity_name(name) if name else None,
            relation_ids=relation_ids,
            passage_ids=passage_ids,
        )

    def _delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity with cascade updates.

        Updates related relations and passages to remove references to this entity.

        Args:
            entity_id: Entity ID to delete.

        Returns:
            True if deletion succeeded, False if entity not found.
        """
        # Get entity data for cascade updates
        entities = self._store._get_entities_by_ids([entity_id])
        if not entities:
            return False

        entity_data = entities[0]
        relation_ids = entity_data.get("relation_ids", [])
        passage_ids = entity_data.get("passage_ids", [])

        # Update related relations (remove this entity from entity_ids)
        for rid in relation_ids:
            relations = self._store._get_relations_by_ids([rid])
            if relations:
                rel_data = relations[0]
                new_entity_ids = [eid for eid in rel_data.get("entity_ids", []) if eid != entity_id]
                self._store._update_relation(rid, entity_ids=new_entity_ids)

        # Update related passages (remove this entity from entity_ids)
        for pid in passage_ids:
            passages = self._store.get_passages_by_ids([pid])
            if passages:
                p_data = passages[0]
                new_entity_ids = [eid for eid in p_data.get("entity_ids", []) if eid != entity_id]
                self._store.update_passage(pid, entity_ids=new_entity_ids)

        # Delete the entity
        result = self._store._delete_entity(entity_id)

        # Update internal mapping
        entity_name = entity_data.get("text", "")
        if entity_name in self._entity_name_to_id:
            del self._entity_name_to_id[entity_name]

        return result

    # ==================== Private: Relation CRUD ====================

    def _create_relation(
        self,
        subject: str,
        predicate: str,
        object_: str,
        id: Optional[str] = None,
        passage_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Create a relation from a triplet.

        Args:
            subject: Subject entity name.
            predicate: Predicate/relationship.
            object_: Object entity name.
            id: Optional ID. If not provided, UUID is generated.
            passage_ids: Optional list of source passage IDs.

        Returns:
            Relation ID (provided or generated).
        """
        # Normalize and build relation text
        norm_subject = self._normalize_entity_name(subject)
        norm_predicate = processing_phrases(predicate)
        norm_object = self._normalize_entity_name(object_)
        relation_text = f"{norm_subject} {norm_predicate} {norm_object}"

        # Check if relation already exists
        if relation_text in self._relation_text_to_id:
            existing_id = self._relation_text_to_id[relation_text]
            # Update passage_ids if provided
            if passage_ids:
                existing = self._store._get_relations_by_ids([existing_id])
                if existing:
                    current = existing[0]
                    new_passage_ids = list(set(current.get("passage_ids", []) + passage_ids))
                    self._store._update_relation(existing_id, passage_ids=new_passage_ids)
            return existing_id

        # Generate ID if not provided
        relation_id = id or generate_id()

        # Create entities (if not exist) and get their IDs
        subject_id = self._create_entity(subject, relation_ids=[relation_id], passage_ids=passage_ids)
        object_id = self._create_entity(object_, relation_ids=[relation_id], passage_ids=passage_ids)

        # Generate embedding
        embedding = self._embedding_model.embed(relation_text)

        # Build metadata
        metadata = {
            "entity_ids": [subject_id, object_id],
            "subject": norm_subject,
            "predicate": norm_predicate,
            "object": norm_object,
        }
        if passage_ids:
            metadata["passage_ids"] = passage_ids

        # Insert into Milvus
        self._store._insert_relations(
            [relation_text],
            ids=[relation_id],
            embeddings=[embedding],
            metadatas=[metadata],
        )

        # Update internal mapping
        self._relation_text_to_id[relation_text] = relation_id

        return relation_id

    def _get_relation(self, relation_id: str) -> Optional[Relation]:
        """
        Get a relation by ID.

        Args:
            relation_id: Relation ID.

        Returns:
            Relation object or None if not found.
        """
        results = self._store._get_relations_by_ids([relation_id])
        if not results:
            return None

        data = results[0]
        triplet = Triplet(
            subject=data.get("subject", ""),
            predicate=data.get("predicate", ""),
            object=data.get("object", ""),
        )
        return Relation(
            id=data["id"],
            text=data["text"],
            triplet=triplet,
            source_passage_ids=data.get("passage_ids", []),
        )

    def _update_relation(
        self,
        relation_id: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_: Optional[str] = None,
        passage_ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Update a relation.

        Args:
            relation_id: Relation ID to update.
            subject: New subject (optional).
            predicate: New predicate (optional).
            object_: New object (optional).
            passage_ids: New passage IDs (optional).

        Returns:
            True if update succeeded, False if relation not found.
        """
        # Get existing relation
        existing = self._store._get_relations_by_ids([relation_id])
        if not existing:
            return False

        data = existing[0]

        # Build new text if any triplet field changed
        new_text = None
        if subject or predicate or object_:
            new_subject = self._normalize_entity_name(subject) if subject else data.get("subject", "")
            new_predicate = processing_phrases(predicate) if predicate else data.get("predicate", "")
            new_object = self._normalize_entity_name(object_) if object_ else data.get("object", "")
            new_text = f"{new_subject} {new_predicate} {new_object}"

        return self._store._update_relation(
            relation_id,
            text=new_text,
            passage_ids=passage_ids,
            subject=self._normalize_entity_name(subject) if subject else None,
            predicate=processing_phrases(predicate) if predicate else None,
            object_=self._normalize_entity_name(object_) if object_ else None,
        )

    def _delete_relation(self, relation_id: str) -> bool:
        """
        Delete a relation with cascade updates.

        Updates related entities and passages to remove references to this relation.

        Args:
            relation_id: Relation ID to delete.

        Returns:
            True if deletion succeeded, False if relation not found.
        """
        # Get relation data for cascade updates
        relations = self._store._get_relations_by_ids([relation_id])
        if not relations:
            return False

        relation_data = relations[0]
        entity_ids = relation_data.get("entity_ids", [])
        passage_ids = relation_data.get("passage_ids", [])

        # Update related entities (remove this relation from relation_ids)
        for eid in entity_ids:
            entities = self._store._get_entities_by_ids([eid])
            if entities:
                e_data = entities[0]
                new_relation_ids = [rid for rid in e_data.get("relation_ids", []) if rid != relation_id]
                self._store._update_entity(eid, relation_ids=new_relation_ids)

        # Update related passages (remove this relation from relation_ids)
        for pid in passage_ids:
            passages = self._store.get_passages_by_ids([pid])
            if passages:
                p_data = passages[0]
                new_relation_ids = [rid for rid in p_data.get("relation_ids", []) if rid != relation_id]
                self._store.update_passage(pid, relation_ids=new_relation_ids)

        # Delete the relation
        result = self._store._delete_relation(relation_id)

        # Update internal mapping
        relation_text = relation_data.get("text", "")
        if relation_text in self._relation_text_to_id:
            del self._relation_text_to_id[relation_text]

        return result

    # ==================== Public: Passage CRUD ====================

    def create_passage(
        self,
        text: str,
        id: Optional[str] = None,
        triplets: Optional[List[Triplet]] = None,
    ) -> str:
        """
        Create a passage.

        Args:
            text: Passage text content.
            id: Optional ID. If not provided, UUID is generated.
            triplets: Optional list of triplets. If provided, auto-creates entities and relations.

        Returns:
            Passage ID (provided or generated).

        Example:
            >>> # Simple passage
            >>> passage_id = graph.create_passage("Some text here")
            >>>
            >>> # With custom ID
            >>> passage_id = graph.create_passage("Some text", id="doc_001")
            >>>
            >>> # With triplets (auto-create entities and relations)
            >>> passage_id = graph.create_passage(
            ...     text="Einstein developed relativity",
            ...     triplets=[Triplet(subject="Einstein", predicate="developed", object="relativity")]
            ... )
        """
        # Generate ID if not provided
        passage_id = id or generate_id()

        # Generate embedding
        embedding = self._embedding_model.embed(text)

        entity_ids: List[str] = []
        relation_ids: List[str] = []

        # Create entities and relations from triplets
        if triplets:
            for triplet in triplets:
                # Create relation (which also creates entities)
                relation_id = self._create_relation(
                    subject=triplet.subject,
                    predicate=triplet.predicate,
                    object_=triplet.object,
                    passage_ids=[passage_id],
                )
                relation_ids.append(relation_id)

                # Get entity IDs for this triplet
                subject_id = self._entity_name_to_id.get(self._normalize_entity_name(triplet.subject))
                object_id = self._entity_name_to_id.get(self._normalize_entity_name(triplet.object))

                if subject_id and subject_id not in entity_ids:
                    entity_ids.append(subject_id)
                if object_id and object_id not in entity_ids:
                    entity_ids.append(object_id)

        # Build metadata
        metadata = {}
        if entity_ids:
            metadata["entity_ids"] = entity_ids
        if relation_ids:
            metadata["relation_ids"] = relation_ids

        # Insert passage
        self._store.insert_passages(
            [text],
            ids=[passage_id],
            embeddings=[embedding],
            metadatas=[metadata] if metadata else None,
        )

        return passage_id

    def get_passage(self, passage_id: str) -> Optional[Passage]:
        """
        Get a passage by ID.

        Args:
            passage_id: Passage ID.

        Returns:
            Passage object or None if not found.
        """
        results = self._store.get_passages_by_ids([passage_id])
        if not results:
            return None

        data = results[0]
        return Passage(
            id=data["id"],
            text=data["text"],
            entity_ids=data.get("entity_ids", []),
            relation_ids=data.get("relation_ids", []),
        )

    def search_passages(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Passage]:
        """
        Search passages by vector similarity.

        Args:
            query: Query text.
            top_k: Number of results to return.

        Returns:
            List of Passage objects sorted by similarity.
        """
        query_embedding = self._embedding_model.embed(query)
        results = self._store.search_passages(query_embedding, top_k=top_k)

        passages = []
        for r in results:
            data = r["entity"]
            passages.append(Passage(
                id=data["id"],
                text=data["text"],
                entity_ids=data.get("entity_ids", []),
                relation_ids=data.get("relation_ids", []),
            ))

        return passages

    def update_passage(
        self,
        passage_id: str,
        text: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
        relation_ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Update a passage.

        Args:
            passage_id: Passage ID to update.
            text: New text (optional).
            entity_ids: New entity IDs (optional).
            relation_ids: New relation IDs (optional).

        Returns:
            True if update succeeded, False if passage not found.
        """
        return self._store.update_passage(
            passage_id,
            text=text,
            entity_ids=entity_ids,
            relation_ids=relation_ids,
        )

    def delete_passage(self, passage_id: str) -> bool:
        """
        Delete a passage with cascade updates.

        Updates related entities and relations to remove references to this passage.

        Args:
            passage_id: Passage ID to delete.

        Returns:
            True if deletion succeeded, False if passage not found.
        """
        # Get passage data for cascade updates
        passages = self._store.get_passages_by_ids([passage_id])
        if not passages:
            return False

        passage_data = passages[0]
        entity_ids = passage_data.get("entity_ids", [])
        relation_ids = passage_data.get("relation_ids", [])

        # Update related entities (remove this passage from passage_ids)
        for eid in entity_ids:
            entities = self._store._get_entities_by_ids([eid])
            if entities:
                e_data = entities[0]
                new_passage_ids = [pid for pid in e_data.get("passage_ids", []) if pid != passage_id]
                self._store._update_entity(eid, passage_ids=new_passage_ids)

        # Update related relations (remove this passage from passage_ids)
        for rid in relation_ids:
            relations = self._store._get_relations_by_ids([rid])
            if relations:
                r_data = relations[0]
                new_passage_ids = [pid for pid in r_data.get("passage_ids", []) if pid != passage_id]
                self._store._update_relation(rid, passage_ids=new_passage_ids)

        # Delete the passage
        return self._store.delete_passage(passage_id)

    # ==================== Public: Entity and Relation Read ====================

    def get_entity(self, entity_id: str) -> Optional[Any]:
        """
        Get an entity by ID (public method for API).

        Args:
            entity_id: Entity ID.

        Returns:
            Entity-like object with id, name, relation_ids, passage_ids.
        """
        results = self._store._get_entities_by_ids([entity_id])
        if not results:
            return None

        data = results[0]

        class EntityData:
            def __init__(self, d):
                self.id = d["id"]
                self.name = d["text"]
                self.relation_ids = d.get("relation_ids", [])
                self.passage_ids = d.get("passage_ids", [])

        return EntityData(data)

    def get_relations_for_entity(self, entity_id: str, limit: int = 20) -> List[Any]:
        """
        Get relations connected to an entity.

        Args:
            entity_id: Entity ID.
            limit: Maximum number of relations to return.

        Returns:
            List of relation-like objects.
        """
        # First get the entity to find its relation_ids
        entity = self._store._get_entities_by_ids([entity_id])
        if not entity:
            return []

        relation_ids = entity[0].get("relation_ids", [])[:limit]
        if not relation_ids:
            return []

        # Get relation details
        relations_data = self._store._get_relations_by_ids(relation_ids)

        class RelationData:
            def __init__(self, d):
                self.id = d["id"]
                self.text = d["text"]
                self.subject = d.get("subject", "")
                self.predicate = d.get("predicate", "")
                self.object = d.get("object", "")
                self.entity_ids = d.get("entity_ids", [])
                self.passage_ids = d.get("passage_ids", [])

        return [RelationData(r) for r in relations_data]

    def get_stats(self) -> Dict[str, int]:
        """
        Get graph statistics.

        Returns:
            Dict with entity_count, relation_count, passage_count.
        """
        return self._store.get_collection_stats()

    # ==================== SubGraph Creation ====================

    def create_subgraph(self) -> SubGraph:
        """
        Create a SubGraph for lazy loading retrieval.

        Returns:
            Empty SubGraph instance connected to this graph's store.

        Example:
            >>> subgraph = graph.create_subgraph()
            >>> subgraph.add_entities(["entity_001", "entity_002"])
            >>> subgraph.expand(degree=1)
            >>> print(subgraph.relation_texts)
        """
        return SubGraph(self._store)

    # ==================== Collection Management ====================

    def create_collections(self, drop_existing: bool = False) -> None:
        """
        Create required collections in Milvus.

        Args:
            drop_existing: Whether to drop existing collections.
        """
        self._store.create_collections(drop_existing=drop_existing)

    def drop_collections(self) -> None:
        """Drop all collections."""
        self._store.drop_collections()
        self._entity_name_to_id.clear()
        self._relation_text_to_id.clear()

    def reset(self) -> None:
        """Reset the graph, removing all data."""
        self.drop_collections()
        self.create_collections(drop_existing=True)
