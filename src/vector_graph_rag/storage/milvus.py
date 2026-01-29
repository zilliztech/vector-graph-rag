"""
Milvus vector store for entities, relations, and passages.

Stores adjacency information in metadata:
- Entity metadata: relation_ids (directly connected relation IDs)
- Relation metadata: entity_ids (head and tail entity IDs), passage_ids (source passage IDs)
- Passage metadata: entity_ids, relation_ids (for reference)
"""

from typing import List, Optional, Dict, Any
from pymilvus import MilvusClient, DataType
from tqdm import tqdm

from vector_graph_rag.config import Settings, get_settings
from vector_graph_rag.storage.embeddings import EmbeddingModel


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
        prefix = (
            f"{self.settings.collection_prefix}_"
            if self.settings.collection_prefix
            else ""
        )
        self.entity_collection = f"{prefix}{self.settings.entity_collection}"
        self.relation_collection = f"{prefix}{self.settings.relation_collection}"
        self.passage_collection = f"{prefix}{self.settings.passage_collection}"

    def _create_collection(
        self,
        collection_name: str,
        dimension: Optional[int] = None,
        drop_existing: bool = False,
    ) -> None:
        """Create a single collection with standard schema."""
        from pymilvus import DataType

        if self.client.has_collection(collection_name):
            if drop_existing:
                self.client.drop_collection(collection_name)
            else:
                return

        dim = dimension or self.embedding_model.dimension
        index_type = self.settings.milvus_index_type
        index_params_config = self.settings.milvus_index_params

        # Build schema using MilvusClient API (required for index_type to take effect)
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
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
        ids: List[int],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> None:
        """Insert data into a collection with optional metadata."""
        batch_size = batch_size or self.settings.batch_size

        total_batches = (len(ids) + batch_size - 1) // batch_size
        iterator = range(0, len(ids), batch_size)

        if show_progress:
            iterator = tqdm(
                iterator, total=total_batches, desc=f"Inserting to {collection_name}"
            )

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

    def insert_entities(
        self,
        entity_texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = False,
    ) -> None:
        """
        Insert entities into the entity collection.

        Args:
            entity_texts: List of entity names/texts.
            embeddings: Pre-computed embeddings. Generated if not provided.
            metadatas: List of metadata dicts, each containing:
                - relation_ids: List of directly connected relation IDs
                - passage_ids: List of source passage IDs (optional)
            show_progress: Whether to show progress bar.
        """
        if not entity_texts:
            return

        if embeddings is None:
            embeddings = self.embedding_model.embed_batch(
                entity_texts, show_progress=show_progress
            )

        ids = list(range(len(entity_texts)))
        self._insert_data(
            self.entity_collection,
            ids,
            entity_texts,
            embeddings,
            metadatas=metadatas,
            show_progress=show_progress,
        )

    def insert_relations(
        self,
        relation_texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = False,
    ) -> None:
        """
        Insert relations into the relation collection.

        Args:
            relation_texts: List of relation texts.
            embeddings: Pre-computed embeddings. Generated if not provided.
            metadatas: List of metadata dicts, each containing:
                - entity_ids: List of connected entity IDs (head and tail)
                - passage_ids: List of source passage IDs
            show_progress: Whether to show progress bar.
        """
        if not relation_texts:
            return

        if embeddings is None:
            embeddings = self.embedding_model.embed_batch(
                relation_texts, show_progress=show_progress
            )

        ids = list(range(len(relation_texts)))
        self._insert_data(
            self.relation_collection,
            ids,
            relation_texts,
            embeddings,
            metadatas=metadatas,
            show_progress=show_progress,
        )

    def insert_passages(
        self,
        passage_texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = False,
    ) -> None:
        """
        Insert passages into the passage collection.

        Args:
            passage_texts: List of passage texts.
            embeddings: Pre-computed embeddings. Generated if not provided.
            metadatas: List of metadata dicts, each containing:
                - entity_ids: List of entity IDs in this passage
                - relation_ids: List of relation IDs in this passage
            show_progress: Whether to show progress bar.
        """
        if not passage_texts:
            return

        if embeddings is None:
            embeddings = self.embedding_model.embed_batch(
                passage_texts, show_progress=show_progress
            )

        ids = list(range(len(passage_texts)))
        self._insert_data(
            self.passage_collection,
            ids,
            passage_texts,
            embeddings,
            metadatas=metadatas,
            show_progress=show_progress,
        )

    def search_entities(
        self,
        query_embeddings: List[List[float]],
        top_k: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for similar entities.

        Args:
            query_embeddings: Query embedding vectors.
            top_k: Number of results to return per query.

        Returns:
            List of search results per query, each result includes:
            - id, text, distance
            - relation_ids, passage_ids (if stored)
        """
        top_k = top_k or self.settings.entity_top_k

        results = self.client.search(
            collection_name=self.entity_collection,
            data=query_embeddings,
            limit=top_k,
            output_fields=["id", "text", "relation_ids", "passage_ids"],
        )
        return results

    def search_relations(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar relations.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of search results, each includes:
            - id, text, distance
            - entity_ids, passage_ids (if stored)
        """
        top_k = top_k or self.settings.relation_top_k

        results = self.client.search(
            collection_name=self.relation_collection,
            data=[query_embedding],
            limit=top_k,
            output_fields=["id", "text", "entity_ids", "passage_ids"],
        )
        return results[0] if results else []

    def search_passages(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar passages (naive RAG).

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of search results.
        """
        top_k = top_k or self.settings.final_top_k

        results = self.client.search(
            collection_name=self.passage_collection,
            data=[query_embedding],
            limit=top_k,
            output_fields=["id", "text"],
        )
        return results[0] if results else []

    def get_entities_by_ids(
        self,
        entity_ids: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Get entities by their IDs.

        Args:
            entity_ids: List of entity IDs.

        Returns:
            List of entity data with id, text, relation_ids, and passage_ids.
        """
        if not entity_ids:
            return []

        results = self.client.query(
            collection_name=self.entity_collection,
            filter=f"id in {entity_ids}",
            output_fields=["id", "text", "relation_ids", "passage_ids"],
        )
        return results

    def get_relations_by_ids(
        self,
        relation_ids: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Get relations by their IDs.

        Args:
            relation_ids: List of relation IDs.

        Returns:
            List of relation data with id, text, entity_ids, passage_ids.
        """
        if not relation_ids:
            return []

        results = self.client.query(
            collection_name=self.relation_collection,
            filter=f"id in {relation_ids}",
            output_fields=["id", "text", "entity_ids", "passage_ids"],
        )
        return results

    def get_passages_by_ids(
        self,
        passage_ids: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Get passages by their IDs.

        Args:
            passage_ids: List of passage IDs.

        Returns:
            List of passage data.
        """
        if not passage_ids:
            return []

        results = self.client.query(
            collection_name=self.passage_collection,
            filter=f"id in {passage_ids}",
            output_fields=["id", "text"],
        )
        return results

    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics for all collections."""
        stats = {}
        for name in [
            self.entity_collection,
            self.relation_collection,
            self.passage_collection,
        ]:
            if self.client.has_collection(name):
                # Use describe_collection to get entity count
                info = self.client.describe_collection(name)
                stats[name] = (
                    self.client.query(
                        collection_name=name,
                        filter="",
                        output_fields=["count(*)"],
                    )[0].get("count(*)", 0)
                    if self.client.has_collection(name)
                    else 0
                )
            else:
                stats[name] = 0
        return stats

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
