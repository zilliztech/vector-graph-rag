"""
Milvus vector store for entities, relations, and passages.

Stores adjacency information in metadata:
- Entity metadata: relation_ids (directly connected relation IDs)
- Relation metadata: entity_ids (head and tail entity IDs), passage_ids (source passage IDs)
- Passage metadata: entity_ids, relation_ids (for reference)

Note: Entity and Relation methods are private (prefixed with _) as they are internal.
Users should interact with passages through the Graph abstraction layer.
"""

import logging
import re
import uuid
from typing import Any, Dict, Iterator, List, Optional

from pymilvus import DataType, MilvusClient
from tqdm import tqdm

from vector_graph_rag.config import Settings, get_settings
from vector_graph_rag.observability import start_span
from vector_graph_rag.storage.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def generate_id() -> str:
    """Generate a unique ID using UUID4."""
    return str(uuid.uuid4())


class MilvusStore:
    """
    Milvus vector store for Graph RAG data.

    Manages three collections:
    - Entity collection: Stores entities with their embeddings
    - Relation collection: Stores relations with their embeddings
    - Passage collection: Stores original passages with their embeddings

    Example:
        >>> store = MilvusStore()
        >>> store.create_collections()
        >>> store.insert_entities(["Einstein", "Relativity"], [[0.1]*1536, [0.2]*1536])
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        """
        Initialize the Milvus store.

        Args:
            settings: Configuration settings.
            embedding_model: Embedding model for generating vectors.
        """
        self.settings = settings or get_settings()
        self.embedding_model = embedding_model or EmbeddingModel(settings=self.settings)

        # Initialize Milvus client
        client_kwargs: Dict[str, Any] = {"uri": self.settings.milvus_uri}
        if self.settings.milvus_token:
            client_kwargs["token"] = self.settings.milvus_token
        if self.settings.milvus_db:
            client_kwargs["db_name"] = self.settings.milvus_db

        self.client = MilvusClient(**client_kwargs)

        # Collection names with optional prefix
        prefix = f"{self.settings.collection_prefix}_" if self.settings.collection_prefix else ""
        self.entity_collection = f"{prefix}{self.settings.entity_collection}"
        self.relation_collection = f"{prefix}{self.settings.relation_collection}"
        self.passage_collection = f"{prefix}{self.settings.passage_collection}"

    @staticmethod
    def _combine_filters(*filters: Optional[str]) -> str:
        """Combine non-empty Milvus filter expressions with AND."""
        parts = [f.strip() for f in filters if f and f.strip()]
        return " and ".join(f"({part})" for part in parts)

    @staticmethod
    def _quote_string(value: str) -> str:
        """Quote a string value for Milvus filter expressions."""
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    @staticmethod
    def _validate_field_name(field_name: str) -> str:
        """Validate a metadata field name for safe filter construction."""
        if not isinstance(field_name, str) or not _FIELD_NAME_RE.fullmatch(field_name):
            raise ValueError(
                "source_field must be a simple metadata field name containing only "
                "letters, numbers, and underscores, and it must not start with a number."
            )
        return field_name

    def _batch_items(
        self,
        items: List[Any],
        batch_size: Optional[int] = None,
    ) -> Iterator[List[Any]]:
        """Yield items in batches using the configured batch size."""
        effective_batch_size = batch_size or self.settings.batch_size
        for start_idx in range(0, len(items), effective_batch_size):
            yield items[start_idx : start_idx + effective_batch_size]

    @staticmethod
    def _unique_preserve_order(values: List[str]) -> List[str]:
        """Return unique strings without changing first-seen order."""
        seen = set()
        unique_values = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            unique_values.append(value)
        return unique_values

    def _query_by_texts(
        self,
        collection_name: str,
        texts: List[str],
        output_fields: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Query records by exact text in batches."""
        records_by_text: Dict[str, Dict[str, Any]] = {}
        unique_texts = self._unique_preserve_order(texts)
        if not unique_texts:
            return records_by_text

        with start_span(
            "vgrag.milvus.query_by_texts",
            {
                "db.system": "milvus",
                "db.collection.name": collection_name,
                "vgrag.record_count": len(texts),
                "vgrag.unique_record_count": len(unique_texts),
                "vgrag.batch_size": self.settings.batch_size,
            },
        ):
            for batch_texts in self._batch_items(unique_texts):
                quoted_texts = ", ".join(self._quote_string(text) for text in batch_texts)
                results = self.client.query(
                    collection_name=collection_name,
                    filter=f"text in [{quoted_texts}]",
                    output_fields=output_fields,
                )
                for record in results:
                    text = record.get("text")
                    if isinstance(text, str) and text not in records_by_text:
                        records_by_text[text] = record

        return records_by_text

    def _upsert_records(
        self,
        collection_name: str,
        records: List[Dict[str, Any]],
    ) -> None:
        """Upsert records in batches using the configured batch size."""
        if not records:
            return

        with start_span(
            "vgrag.milvus.upsert",
            {
                "db.system": "milvus",
                "db.collection.name": collection_name,
                "vgrag.record_count": len(records),
                "vgrag.batch_size": self.settings.batch_size,
            },
        ):
            for batch_records in self._batch_items(records):
                self.client.upsert(
                    collection_name=collection_name,
                    data=batch_records,
                )

    def _upsert_entity_records(self, records: List[Dict[str, Any]]) -> None:
        """Upsert fully materialized entity records."""
        self._upsert_records(self.entity_collection, records)

    def _upsert_relation_records(self, records: List[Dict[str, Any]]) -> None:
        """Upsert fully materialized relation records."""
        self._upsert_records(self.relation_collection, records)

    def _create_collection(
        self,
        collection_name: str,
        dimension: Optional[int] = None,
        drop_existing: bool = False,
    ) -> None:
        """Create a single collection with standard schema."""

        if self.client.has_collection(collection_name):
            if drop_existing:
                self.client.drop_collection(collection_name)
            else:
                return

        dim = dimension or self.embedding_model.dimension
        index_type = self.settings.milvus_index_type
        index_params_config = self.settings.milvus_index_params

        # Build schema using MilvusClient API (required for index_type to take effect)
        # Use VARCHAR for ID to support user-provided IDs or UUIDs
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

        # Build index params
        index_params = self.client.prepare_index_params()
        add_index_kwargs = {
            "field_name": "vector",
            "index_type": index_type,
            "metric_type": "IP",
        }
        if index_params_config:
            add_index_kwargs["params"] = index_params_config
        index_params.add_index(**add_index_kwargs)

        with start_span(
            "vgrag.milvus.create_collection",
            {
                "db.system": "milvus",
                "db.collection.name": collection_name,
                "vgrag.embedding_dimension": dim,
                "vgrag.milvus_index_type": index_type,
                "vgrag.milvus_consistency_level": self.settings.milvus_consistency_level,
            },
        ):
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
                # consistency_level="Strong",  # Strong waits for all loads to complete
                consistency_level=self.settings.milvus_consistency_level,
            )

    def create_collections(
        self,
        dimension: Optional[int] = None,
        drop_existing: bool = False,
    ) -> None:
        """
        Create all required collections.

        Args:
            dimension: Embedding dimension. Auto-detected if not provided.
            drop_existing: Whether to drop existing collections.
        """
        for collection_name in [
            self.entity_collection,
            self.relation_collection,
            self.passage_collection,
        ]:
            self._create_collection(
                collection_name,
                dimension=dimension,
                drop_existing=drop_existing,
            )

    def drop_collections(self) -> None:
        """Drop all collections."""
        with start_span(
            "vgrag.milvus.drop_collections",
            {
                "db.system": "milvus",
                "vgrag.collection_count": 3,
            },
        ):
            for collection_name in [
                self.entity_collection,
                self.relation_collection,
                self.passage_collection,
            ]:
                if self.client.has_collection(collection_name):
                    self.client.drop_collection(collection_name)

    def _insert_data(
        self,
        collection_name: str,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> None:
        """Insert data into a collection with optional metadata.

        Args:
            collection_name: Name of the collection.
            ids: List of string IDs (user-provided or UUIDs).
            texts: List of text content.
            embeddings: List of embedding vectors.
            metadatas: Optional list of metadata dicts.
            batch_size: Batch size for insertion.
            show_progress: Whether to show progress bar.
        """
        batch_size = batch_size or self.settings.batch_size

        total_batches = (len(ids) + batch_size - 1) // batch_size
        iterator = range(0, len(ids), batch_size)

        if show_progress:
            iterator = tqdm(iterator, total=total_batches, desc=f"Inserting to {collection_name}")

        with start_span(
            "vgrag.milvus.insert",
            {
                "db.system": "milvus",
                "db.collection.name": collection_name,
                "vgrag.record_count": len(ids),
                "vgrag.batch_size": batch_size,
                "vgrag.batch_count": total_batches,
            },
        ):
            for start_idx in iterator:
                end_idx = min(start_idx + batch_size, len(ids))
                batch_data = [
                    {
                        "id": ids[i],
                        "text": texts[i],
                        "vector": embeddings[i],
                    }
                    for i in range(start_idx, end_idx)
                ]
                # Add metadata if provided
                if metadatas:
                    for i, data in enumerate(batch_data):
                        idx = start_idx + i
                        if idx < len(metadatas):
                            data.update(metadatas[idx])

                self.client.insert(collection_name=collection_name, data=batch_data)

    def _insert_entities(
        self,
        entity_texts: List[str],
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = False,
    ) -> List[str]:
        """
        Insert entities into the entity collection.

        This is a private method. Users should use the Graph abstraction layer.

        Args:
            entity_texts: List of entity names/texts.
            ids: Optional list of string IDs. If not provided, UUIDs are generated.
            embeddings: Pre-computed embeddings. Generated if not provided.
            metadatas: List of metadata dicts, each containing:
                - relation_ids: List of directly connected relation IDs
                - passage_ids: List of source passage IDs (optional)
            show_progress: Whether to show progress bar.

        Returns:
            List of IDs (provided or generated).
        """
        if not entity_texts:
            return []

        if embeddings is None:
            embeddings = self.embedding_model.embed_batch(
                entity_texts, show_progress=show_progress, text_type="document"
            )

        # Generate UUIDs if IDs not provided
        if ids is None:
            ids = [generate_id() for _ in entity_texts]

        self._insert_data(
            self.entity_collection,
            ids,
            entity_texts,
            embeddings,
            metadatas=metadatas,
            show_progress=show_progress,
        )
        return ids

    def _insert_relations(
        self,
        relation_texts: List[str],
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = False,
    ) -> List[str]:
        """
        Insert relations into the relation collection.

        This is a private method. Users should use the Graph abstraction layer.

        Args:
            relation_texts: List of relation texts.
            ids: Optional list of string IDs. If not provided, UUIDs are generated.
            embeddings: Pre-computed embeddings. Generated if not provided.
            metadatas: List of metadata dicts, each containing:
                - entity_ids: List of connected entity IDs (head and tail)
                - passage_ids: List of source passage IDs
                - subject: Subject entity text (optional, for structured triplet)
                - predicate: Predicate text (optional, for structured triplet)
                - object: Object entity text (optional, for structured triplet)
            show_progress: Whether to show progress bar.

        Returns:
            List of IDs (provided or generated).
        """
        if not relation_texts:
            return []

        if embeddings is None:
            embeddings = self.embedding_model.embed_batch(
                relation_texts, show_progress=show_progress, text_type="document"
            )

        # Generate UUIDs if IDs not provided
        if ids is None:
            ids = [generate_id() for _ in relation_texts]

        self._insert_data(
            self.relation_collection,
            ids,
            relation_texts,
            embeddings,
            metadatas=metadatas,
            show_progress=show_progress,
        )
        return ids

    def insert_passages(
        self,
        passage_texts: List[str],
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = False,
    ) -> List[str]:
        """
        Insert passages into the passage collection.

        Args:
            passage_texts: List of passage texts.
            ids: Optional list of string IDs. If not provided, UUIDs are generated.
            embeddings: Pre-computed embeddings. Generated if not provided.
            metadatas: List of metadata dicts, each containing:
                - entity_ids: List of entity IDs in this passage
                - relation_ids: List of relation IDs in this passage
            show_progress: Whether to show progress bar.

        Returns:
            List of IDs (provided or generated).
        """
        if not passage_texts:
            return []

        if embeddings is None:
            embeddings = self.embedding_model.embed_batch(
                passage_texts, show_progress=show_progress, text_type="document"
            )

        # Generate UUIDs if IDs not provided
        if ids is None:
            ids = [generate_id() for _ in passage_texts]

        self._insert_data(
            self.passage_collection,
            ids,
            passage_texts,
            embeddings,
            metadatas=metadatas,
            show_progress=show_progress,
        )
        return ids

    def _search_entities(
        self,
        query_embeddings: List[List[float]],
        top_k: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for similar entities.

        This is a private method. Users should use the Graph abstraction layer.

        Args:
            query_embeddings: Query embedding vectors.
            top_k: Number of results to return per query.

        Returns:
            List of search results per query, each result includes:
            - id (str), text, distance
            - relation_ids, passage_ids (if stored)
        """
        top_k = top_k or self.settings.entity_top_k

        with start_span(
            "vgrag.milvus.search",
            {
                "db.system": "milvus",
                "db.collection.name": self.entity_collection,
                "vgrag.query_vector_count": len(query_embeddings),
                "vgrag.top_k": top_k,
                "vgrag.record_type": "entity",
            },
        ):
            results = self.client.search(
                collection_name=self.entity_collection,
                data=query_embeddings,
                limit=top_k,
                output_fields=["id", "text", "relation_ids", "passage_ids"],
            )
        return results

    def _search_relations(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar relations.

        This is a private method. Users should use the Graph abstraction layer.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of search results, each includes:
            - id (str), text, distance
            - entity_ids, passage_ids (if stored)
        """
        top_k = top_k or self.settings.relation_top_k

        with start_span(
            "vgrag.milvus.search",
            {
                "db.system": "milvus",
                "db.collection.name": self.relation_collection,
                "vgrag.query_vector_count": 1,
                "vgrag.top_k": top_k,
                "vgrag.record_type": "relation",
            },
        ):
            results = self.client.search(
                collection_name=self.relation_collection,
                data=[query_embedding],
                limit=top_k,
                output_fields=[
                    "id",
                    "text",
                    "entity_ids",
                    "passage_ids",
                    "subject",
                    "predicate",
                    "object",
                ],
            )
        return results[0] if results else []

    def search_passages(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar passages (naive RAG).

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            filter: Optional Milvus filter expression for passage metadata.

        Returns:
            List of search results with id (str) and text.
        """
        top_k = top_k or self.settings.final_top_k

        with start_span(
            "vgrag.milvus.search",
            {
                "db.system": "milvus",
                "db.collection.name": self.passage_collection,
                "vgrag.query_vector_count": 1,
                "vgrag.top_k": top_k,
                "vgrag.record_type": "passage",
                "vgrag.has_filter": bool(filter),
            },
        ):
            results = self.client.search(
                collection_name=self.passage_collection,
                data=[query_embedding],
                limit=top_k,
                filter=filter,
                output_fields=["id", "text", "entity_ids", "relation_ids"],
            )
        return results[0] if results else []

    def _get_entities_by_ids(
        self,
        entity_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Get entities by their IDs.

        This is a private method. Users should use the Graph abstraction layer.

        Args:
            entity_ids: List of entity IDs (strings).

        Returns:
            List of entity data with id, text, relation_ids, and passage_ids.
        """
        if not entity_ids:
            return []

        # Format IDs as quoted strings for Milvus filter
        ids_str = ", ".join(f'"{eid}"' for eid in entity_ids)
        with start_span(
            "vgrag.milvus.query",
            {
                "db.system": "milvus",
                "db.collection.name": self.entity_collection,
                "vgrag.record_type": "entity",
                "vgrag.id_count": len(entity_ids),
            },
        ):
            results = self.client.query(
                collection_name=self.entity_collection,
                filter=f"id in [{ids_str}]",
                output_fields=["id", "text", "relation_ids", "passage_ids"],
            )
        return results

    def _get_entities_by_texts(
        self,
        entity_texts: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get entities by exact text.

        This is a private helper for incremental graph updates.

        Args:
            entity_texts: Entity texts to look up.

        Returns:
            Mapping from entity text to the first matching entity record.
        """
        return self._query_by_texts(
            self.entity_collection,
            entity_texts,
            output_fields=["id", "text", "relation_ids", "passage_ids"],
        )

    def _get_relations_by_ids(
        self,
        relation_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Get relations by their IDs.

        This is a private method. Users should use the Graph abstraction layer.

        Args:
            relation_ids: List of relation IDs (strings).

        Returns:
            List of relation data with id, text, entity_ids, passage_ids,
            and structured triplet fields (subject, predicate, object).
        """
        if not relation_ids:
            return []

        # Format IDs as quoted strings for Milvus filter
        ids_str = ", ".join(f'"{rid}"' for rid in relation_ids)
        with start_span(
            "vgrag.milvus.query",
            {
                "db.system": "milvus",
                "db.collection.name": self.relation_collection,
                "vgrag.record_type": "relation",
                "vgrag.id_count": len(relation_ids),
            },
        ):
            results = self.client.query(
                collection_name=self.relation_collection,
                filter=f"id in [{ids_str}]",
                output_fields=[
                    "id",
                    "text",
                    "entity_ids",
                    "passage_ids",
                    "subject",
                    "predicate",
                    "object",
                ],
            )
        return results

    def _get_relations_by_texts(
        self,
        relation_texts: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get relations by exact text.

        This is a private helper for incremental graph updates.

        Args:
            relation_texts: Relation texts to look up.

        Returns:
            Mapping from relation text to the first matching relation record.
        """
        return self._query_by_texts(
            self.relation_collection,
            relation_texts,
            output_fields=[
                "id",
                "text",
                "entity_ids",
                "passage_ids",
                "subject",
                "predicate",
                "object",
            ],
        )

    def get_passages_by_ids(
        self,
        passage_ids: List[str],
        filter: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get passages by their IDs.

        Args:
            passage_ids: List of passage IDs (strings).
            filter: Optional Milvus filter expression for passage metadata.
            output_fields: Optional output fields. Defaults to graph adjacency fields.

        Returns:
            List of passage data with id, text, entity_ids, and relation_ids.
        """
        if not passage_ids:
            return []

        # Format IDs as quoted strings for Milvus filter
        ids_str = ", ".join(f'"{pid}"' for pid in passage_ids)
        filter_expr = self._combine_filters(f"id in [{ids_str}]", filter)
        with start_span(
            "vgrag.milvus.query",
            {
                "db.system": "milvus",
                "db.collection.name": self.passage_collection,
                "vgrag.record_type": "passage",
                "vgrag.id_count": len(passage_ids),
                "vgrag.has_filter": bool(filter),
            },
        ):
            results = self.client.query(
                collection_name=self.passage_collection,
                filter=filter_expr,
                output_fields=output_fields or ["id", "text", "entity_ids", "relation_ids"],
            )
        return list(results)

    def get_passages_by_document_id(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get passages that belong to a legacy document_id metadata value.

        This is retained for compatibility with data written by older
        incremental APIs. New source-level updates use get_passages_by_source().

        Args:
            document_id: Legacy source document ID.

        Returns:
            Matching passage records with graph adjacency metadata.
        """
        with start_span(
            "vgrag.milvus.query",
            {
                "db.system": "milvus",
                "db.collection.name": self.passage_collection,
                "vgrag.record_type": "passage",
                "vgrag.lookup_field": "document_id",
            },
        ):
            results = self.client.query(
                collection_name=self.passage_collection,
                filter=f"document_id == {self._quote_string(document_id)}",
                output_fields=["id", "text", "entity_ids", "relation_ids", "document_id"],
            )
        return list(results)

    def get_passages_by_source(
        self,
        source: str,
        source_field: str = "source",
    ) -> List[Dict[str, Any]]:
        """
        Get passages that belong to a source metadata value.

        Args:
            source: Stable source value stored in passage metadata.
            source_field: Metadata field used to group chunks by source.

        Returns:
            Matching passage records with graph adjacency metadata.
        """
        source_field = self._validate_field_name(source_field)
        with start_span(
            "vgrag.milvus.query",
            {
                "db.system": "milvus",
                "db.collection.name": self.passage_collection,
                "vgrag.record_type": "passage",
                "vgrag.lookup_field": source_field,
            },
        ):
            results = self.client.query(
                collection_name=self.passage_collection,
                filter=f"{source_field} == {self._quote_string(source)}",
                output_fields=["id", "text", "entity_ids", "relation_ids", source_field],
            )
        return list(results)

    def query_passage_ids(self, filter: str) -> List[str]:
        """
        Query passage IDs that match a Milvus filter expression.

        Args:
            filter: Milvus filter expression for passage metadata.

        Returns:
            Matching passage IDs.
        """
        if not filter or not filter.strip():
            return []

        with start_span(
            "vgrag.milvus.query",
            {
                "db.system": "milvus",
                "db.collection.name": self.passage_collection,
                "vgrag.record_type": "passage",
                "vgrag.has_filter": True,
            },
        ):
            results = self.client.query(
                collection_name=self.passage_collection,
                filter=filter,
                output_fields=["id"],
            )
        return [r["id"] for r in results]

    # ==================== Update Operations ====================

    def _update_entity(
        self,
        entity_id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        relation_ids: Optional[List[str]] = None,
        passage_ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Update an entity by ID using upsert.

        This is a private method. Users should use the Graph abstraction layer.

        Args:
            entity_id: The entity ID to update.
            text: New text (optional).
            embedding: New embedding (optional, auto-generated if text changes).
            relation_ids: New relation IDs (optional).
            passage_ids: New passage IDs (optional).

        Returns:
            True if update succeeded, False if entity not found.
        """
        # First get the existing entity
        existing = self._get_entities_by_ids([entity_id])
        if not existing:
            return False

        entity_data = existing[0]

        # Build update data
        update_data = {"id": entity_id}

        if text is not None:
            update_data["text"] = text
            # Re-compute embedding if text changed
            if embedding is None:
                embedding = self.embedding_model.embed(text)
        else:
            update_data["text"] = entity_data["text"]

        if embedding is not None:
            update_data["vector"] = embedding
        else:
            # Need to fetch the vector - query doesn't return it by default
            # For now, re-compute from text
            update_data["vector"] = self.embedding_model.embed(update_data["text"])

        if relation_ids is not None:
            update_data["relation_ids"] = relation_ids
        elif "relation_ids" in entity_data:
            update_data["relation_ids"] = entity_data["relation_ids"]

        if passage_ids is not None:
            update_data["passage_ids"] = passage_ids
        elif "passage_ids" in entity_data:
            update_data["passage_ids"] = entity_data["passage_ids"]

        # Upsert the entity
        with start_span(
            "vgrag.milvus.upsert",
            {
                "db.system": "milvus",
                "db.collection.name": self.entity_collection,
                "vgrag.record_type": "entity",
                "vgrag.record_count": 1,
            },
        ):
            self.client.upsert(
                collection_name=self.entity_collection,
                data=[update_data],
            )
        return True

    def _update_relation(
        self,
        relation_id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        entity_ids: Optional[List[str]] = None,
        passage_ids: Optional[List[str]] = None,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_: Optional[str] = None,
    ) -> bool:
        """
        Update a relation by ID using upsert.

        This is a private method. Users should use the Graph abstraction layer.

        Args:
            relation_id: The relation ID to update.
            text: New text (optional).
            embedding: New embedding (optional, auto-generated if text changes).
            entity_ids: New entity IDs (optional).
            passage_ids: New passage IDs (optional).
            subject: New subject (optional).
            predicate: New predicate (optional).
            object_: New object (optional).

        Returns:
            True if update succeeded, False if relation not found.
        """
        existing = self._get_relations_by_ids([relation_id])
        if not existing:
            return False

        relation_data = existing[0]

        # Build update data
        update_data = {"id": relation_id}

        if text is not None:
            update_data["text"] = text
            if embedding is None:
                embedding = self.embedding_model.embed(text)
        else:
            update_data["text"] = relation_data["text"]

        if embedding is not None:
            update_data["vector"] = embedding
        else:
            update_data["vector"] = self.embedding_model.embed(update_data["text"])

        if entity_ids is not None:
            update_data["entity_ids"] = entity_ids
        elif "entity_ids" in relation_data:
            update_data["entity_ids"] = relation_data["entity_ids"]

        if passage_ids is not None:
            update_data["passage_ids"] = passage_ids
        elif "passage_ids" in relation_data:
            update_data["passage_ids"] = relation_data["passage_ids"]

        if subject is not None:
            update_data["subject"] = subject
        elif "subject" in relation_data:
            update_data["subject"] = relation_data["subject"]

        if predicate is not None:
            update_data["predicate"] = predicate
        elif "predicate" in relation_data:
            update_data["predicate"] = relation_data["predicate"]

        if object_ is not None:
            update_data["object"] = object_
        elif "object" in relation_data:
            update_data["object"] = relation_data["object"]

        with start_span(
            "vgrag.milvus.upsert",
            {
                "db.system": "milvus",
                "db.collection.name": self.relation_collection,
                "vgrag.record_type": "relation",
                "vgrag.record_count": 1,
            },
        ):
            self.client.upsert(
                collection_name=self.relation_collection,
                data=[update_data],
            )
        return True

    def update_passage(
        self,
        passage_id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        entity_ids: Optional[List[str]] = None,
        relation_ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Update a passage by ID using upsert.

        Args:
            passage_id: The passage ID to update.
            text: New text (optional).
            embedding: New embedding (optional, auto-generated if text changes).
            entity_ids: New entity IDs (optional).
            relation_ids: New relation IDs (optional).

        Returns:
            True if update succeeded, False if passage not found.
        """
        existing = self.get_passages_by_ids([passage_id])
        if not existing:
            return False

        passage_data = existing[0]

        # Build update data
        update_data = {"id": passage_id}

        if text is not None:
            update_data["text"] = text
            if embedding is None:
                embedding = self.embedding_model.embed(text)
        else:
            update_data["text"] = passage_data["text"]

        if embedding is not None:
            update_data["vector"] = embedding
        else:
            update_data["vector"] = self.embedding_model.embed(update_data["text"])

        if entity_ids is not None:
            update_data["entity_ids"] = entity_ids
        elif "entity_ids" in passage_data:
            update_data["entity_ids"] = passage_data["entity_ids"]

        if relation_ids is not None:
            update_data["relation_ids"] = relation_ids
        elif "relation_ids" in passage_data:
            update_data["relation_ids"] = passage_data["relation_ids"]

        with start_span(
            "vgrag.milvus.upsert",
            {
                "db.system": "milvus",
                "db.collection.name": self.passage_collection,
                "vgrag.record_type": "passage",
                "vgrag.record_count": 1,
            },
        ):
            self.client.upsert(
                collection_name=self.passage_collection,
                data=[update_data],
            )
        return True

    # ==================== Delete Operations ====================

    def _delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity by ID.

        This is a private method. Users should use the Graph abstraction layer.

        Args:
            entity_id: The entity ID to delete.

        Returns:
            True if deletion succeeded, False if entity not found.
        """
        existing = self._get_entities_by_ids([entity_id])
        if not existing:
            return False

        with start_span(
            "vgrag.milvus.delete",
            {
                "db.system": "milvus",
                "db.collection.name": self.entity_collection,
                "vgrag.record_type": "entity",
                "vgrag.record_count": 1,
            },
        ):
            self.client.delete(
                collection_name=self.entity_collection,
                filter=f'id == "{entity_id}"',
            )
        return True

    def _delete_relation(self, relation_id: str) -> bool:
        """
        Delete a relation by ID.

        This is a private method. Users should use the Graph abstraction layer.

        Args:
            relation_id: The relation ID to delete.

        Returns:
            True if deletion succeeded, False if relation not found.
        """
        existing = self._get_relations_by_ids([relation_id])
        if not existing:
            return False

        with start_span(
            "vgrag.milvus.delete",
            {
                "db.system": "milvus",
                "db.collection.name": self.relation_collection,
                "vgrag.record_type": "relation",
                "vgrag.record_count": 1,
            },
        ):
            self.client.delete(
                collection_name=self.relation_collection,
                filter=f'id == "{relation_id}"',
            )
        return True

    def delete_passage(self, passage_id: str) -> bool:
        """
        Delete a passage by ID.

        Args:
            passage_id: The passage ID to delete.

        Returns:
            True if deletion succeeded, False if passage not found.
        """
        existing = self.get_passages_by_ids([passage_id])
        if not existing:
            return False

        with start_span(
            "vgrag.milvus.delete",
            {
                "db.system": "milvus",
                "db.collection.name": self.passage_collection,
                "vgrag.record_type": "passage",
                "vgrag.record_count": 1,
            },
        ):
            self.client.delete(
                collection_name=self.passage_collection,
                filter=f'id == "{passage_id}"',
            )
        return True

    def _delete_entities(self, entity_ids: List[str]) -> int:
        """
        Delete multiple entities by IDs.

        This is a private method. Users should use the Graph abstraction layer.

        Args:
            entity_ids: List of entity IDs to delete.

        Returns:
            Number of entities deleted.
        """
        if not entity_ids:
            return 0

        ids_str = ", ".join(f'"{eid}"' for eid in entity_ids)
        with start_span(
            "vgrag.milvus.delete",
            {
                "db.system": "milvus",
                "db.collection.name": self.entity_collection,
                "vgrag.record_type": "entity",
                "vgrag.record_count": len(entity_ids),
            },
        ):
            self.client.delete(
                collection_name=self.entity_collection,
                filter=f"id in [{ids_str}]",
            )
        return len(entity_ids)

    def _delete_relations(self, relation_ids: List[str]) -> int:
        """
        Delete multiple relations by IDs.

        This is a private method. Users should use the Graph abstraction layer.

        Args:
            relation_ids: List of relation IDs to delete.

        Returns:
            Number of relations deleted.
        """
        if not relation_ids:
            return 0

        ids_str = ", ".join(f'"{rid}"' for rid in relation_ids)
        with start_span(
            "vgrag.milvus.delete",
            {
                "db.system": "milvus",
                "db.collection.name": self.relation_collection,
                "vgrag.record_type": "relation",
                "vgrag.record_count": len(relation_ids),
            },
        ):
            self.client.delete(
                collection_name=self.relation_collection,
                filter=f"id in [{ids_str}]",
            )
        return len(relation_ids)

    def delete_passages(self, passage_ids: List[str]) -> int:
        """
        Delete multiple passages by IDs.

        Args:
            passage_ids: List of passage IDs to delete.

        Returns:
            Number of passages deleted.
        """
        if not passage_ids:
            return 0

        ids_str = ", ".join(f'"{pid}"' for pid in passage_ids)
        with start_span(
            "vgrag.milvus.delete",
            {
                "db.system": "milvus",
                "db.collection.name": self.passage_collection,
                "vgrag.record_type": "passage",
                "vgrag.record_count": len(passage_ids),
            },
        ):
            self.client.delete(
                collection_name=self.passage_collection,
                filter=f"id in [{ids_str}]",
            )
        return len(passage_ids)

    # ==================== Utility Methods ====================

    @classmethod
    def list_graphs(
        cls,
        milvus_uri: str = "./vector_graph_rag.db",
        milvus_token: Optional[str] = None,
        milvus_db: Optional[str] = None,
        entity_suffix: str = "vgrag_entities",
        relation_suffix: str = "vgrag_relations",
        passage_suffix: str = "vgrag_passages",
    ) -> List[Dict[str, Any]]:
        """
        List all available graphs (datasets) in Milvus.

        Discovers graphs by finding collections that match the pattern:
        - {prefix}_{entity_suffix} (e.g., ds_2wikimultihopqa_vgrag_entities)
        - {prefix}_{relation_suffix}
        - {prefix}_{passage_suffix}

        The graph name is extracted from the prefix (e.g., "ds_2wikimultihopqa").

        Args:
            milvus_uri: Milvus connection URI.
            milvus_token: Milvus authentication token.
            milvus_db: Milvus database name.
            entity_suffix: Suffix for entity collections.
            relation_suffix: Suffix for relation collections.
            passage_suffix: Suffix for passage collections.

        Returns:
            List of graph info dicts with keys:
            - name: Graph name (collection prefix)
            - entity_collection: Entity collection name
            - relation_collection: Relation collection name
            - passage_collection: Passage collection name
            - has_all_collections: Whether all 3 collections exist
        """
        # Create a temporary client
        client_kwargs: Dict[str, Any] = {"uri": milvus_uri}
        if milvus_token:
            client_kwargs["token"] = milvus_token
        if milvus_db:
            client_kwargs["db_name"] = milvus_db

        client = MilvusClient(**client_kwargs)

        # List all collections
        with start_span(
            "vgrag.milvus.list_collections",
            {
                "db.system": "milvus",
            },
        ):
            all_collections = client.list_collections()

        # Find entity collections and extract prefixes
        graphs: List[Dict[str, Any]] = []
        entity_suffix_pattern = f"_{entity_suffix}"

        for collection in all_collections:
            if collection.endswith(entity_suffix_pattern):
                # Extract prefix (graph name)
                prefix = collection[: -len(entity_suffix_pattern)]

                # Check for corresponding relation and passage collections
                relation_col = f"{prefix}_{relation_suffix}"
                passage_col = f"{prefix}_{passage_suffix}"

                has_relations = relation_col in all_collections
                has_passages = passage_col in all_collections

                graphs.append(
                    {
                        "name": prefix,
                        "entity_collection": collection,
                        "relation_collection": relation_col,
                        "passage_collection": passage_col,
                        "has_all_collections": has_relations and has_passages,
                    }
                )

        # Sort by name
        graphs.sort(key=lambda g: g["name"])

        return graphs

    @classmethod
    def delete_graph(
        cls,
        graph_name: str,
        milvus_uri: str = "./vector_graph_rag.db",
        milvus_token: Optional[str] = None,
        milvus_db: Optional[str] = None,
        entity_suffix: str = "vgrag_entities",
        relation_suffix: str = "vgrag_relations",
        passage_suffix: str = "vgrag_passages",
    ) -> bool:
        """
        Delete a graph and all its collections.

        Args:
            graph_name: Name of the graph to delete (collection prefix).
            milvus_uri: Milvus connection URI.
            milvus_token: Milvus authentication token.
            milvus_db: Milvus database name.
            entity_suffix: Suffix for entity collections.
            relation_suffix: Suffix for relation collections.
            passage_suffix: Suffix for passage collections.

        Returns:
            True if at least one collection was deleted, False otherwise.
        """
        # Create a temporary client
        client_kwargs: Dict[str, Any] = {"uri": milvus_uri}
        if milvus_token:
            client_kwargs["token"] = milvus_token
        if milvus_db:
            client_kwargs["db_name"] = milvus_db

        client = MilvusClient(**client_kwargs)

        # Collection names for this graph
        entity_col = f"{graph_name}_{entity_suffix}"
        relation_col = f"{graph_name}_{relation_suffix}"
        passage_col = f"{graph_name}_{passage_suffix}"

        deleted = False

        # Delete each collection if it exists
        with start_span(
            "vgrag.milvus.delete_graph",
            {
                "db.system": "milvus",
                "vgrag.graph_name": graph_name,
                "vgrag.collection_count": 3,
            },
        ):
            for collection_name in [entity_col, relation_col, passage_col]:
                try:
                    if client.has_collection(collection_name):
                        client.drop_collection(collection_name)
                        deleted = True
                except Exception as e:
                    # Log error but continue with other collections
                    logger.error("Error deleting collection %s: %s", collection_name, e)

        return deleted

    def get_collection_stats(self) -> Dict[str, int]:
        """
        Count current records in each collection using strong consistency.

        Missing collections contribute zero. Errors querying existing collections
        propagate to the caller. Counts are read separately, so concurrent writes
        can be observed at different stages across the three collections.

        Returns:
            Dict with entity_count, relation_count, passage_count.
        """
        stats = {
            "entity_count": 0,
            "relation_count": 0,
            "passage_count": 0,
        }

        with start_span(
            "vgrag.milvus.collection_stats",
            {
                "db.system": "milvus",
                "vgrag.collection_count": 3,
            },
        ):
            for key, collection_name in (
                ("entity_count", self.entity_collection),
                ("relation_count", self.relation_collection),
                ("passage_count", self.passage_collection),
            ):
                if not self.client.has_collection(collection_name):
                    continue
                result = self.client.query(
                    collection_name=collection_name,
                    filter="",
                    output_fields=["count(*)"],
                    consistency_level="Strong",
                )
                stats[key] = int(result[0]["count(*)"])

        return stats
