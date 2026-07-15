"""
Main Vector Graph RAG class with user-friendly API.
"""

import hashlib
import logging
import uuid
import warnings
from typing import Any, Dict, List, Optional, Tuple

from vector_graph_rag.config import Settings
from vector_graph_rag.graph.builder import GraphBuilder
from vector_graph_rag.graph.knowledge_graph import SubGraph
from vector_graph_rag.graph.retriever import GraphRetriever
from vector_graph_rag.llm.extractor import TripletExtractor
from vector_graph_rag.llm.react import ReActPlanner
from vector_graph_rag.llm.reranker import AnswerGenerator, LLMReranker
from vector_graph_rag.models import (
    Document,
    Entity,
    EvictionResult,
    ExtractionResult,
    QueryResult,
    ReActResult,
    ReActStep,
    Relation,
    RerankResult,
    RetrievalDetail,
)
from vector_graph_rag.storage.embeddings import EmbeddingModel
from vector_graph_rag.storage.milvus import MilvusStore

logger = logging.getLogger(__name__)


class VectorGraphRAG:
    """
    Vector Graph RAG - Graph RAG using pure vector search with Milvus.

    This class provides a simple, user-friendly API for building and querying
    a Graph RAG system. It uses knowledge graph triplets extracted from documents
    and performs multi-way retrieval with subgraph expansion.

    Example:
        >>> # Initialize
        >>> rag = VectorGraphRAG()
        >>>
        >>> # Rebuild from texts
        >>> texts = [
        ...     "Einstein was a physicist who developed relativity.",
        ...     "Relativity revolutionized our understanding of space and time.",
        ... ]
        >>> rag.rebuild_texts(texts)
        >>>
        >>> # Query
        >>> result = rag.query("What did Einstein develop?")
        >>> print(result.answer)
        >>>
        >>> # Access the subgraph for debugging
        >>> subgraph = result.subgraph
        >>> print(subgraph.expansion_history)

    Quick Start:
        >>> from vector_graph_rag import VectorGraphRAG
        >>> rag = VectorGraphRAG()
        >>> rag.rebuild_texts(["Your text here..."])
        >>> answer = rag.query("Your question?").answer
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        milvus_uri: Optional[str] = None,
        milvus_token: Optional[str] = None,
        milvus_db: Optional[str] = None,
        collection_prefix: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
    ):
        """
        Initialize Vector Graph RAG.

        Args:
            settings: Full settings object (overrides other parameters).
            milvus_uri: Milvus connection URI. Defaults to local file.
            milvus_token: Milvus authentication token (for Zilliz Cloud).
            milvus_db: Milvus database name. Defaults to None (use server default).
            collection_prefix: Prefix for collection names (e.g., graph/dataset name).
            openai_api_key: OpenAI API key. Uses environment variable if not provided.
            llm_model: LLM model name. Defaults to "gpt-4o-mini".
            embedding_provider: Embedding provider name (e.g., "openai", "huggingface",
                "ollama", "jina").
            embedding_model: Embedding model name.
            embedding_api_key: Optional API key override for embedding providers.
            embedding_base_url: Optional base URL override for embedding providers.

        Example:
            >>> # Use defaults (reads OPENAI_API_KEY from environment)
            >>> rag = VectorGraphRAG()
            >>>
            >>> # Custom configuration
            >>> rag = VectorGraphRAG(
            ...     milvus_uri="./my_data.db",
            ...     llm_model="gpt-4o",
            ... )
            >>>
            >>> # Zilliz Cloud
            >>> rag = VectorGraphRAG(
            ...     milvus_uri="https://in03-xxx.api.gcp-us-west1.zillizcloud.com",
            ...     milvus_token="your-api-key",
            ... )
            >>>
            >>> # Separate graphs with collection prefix
            >>> rag = VectorGraphRAG(
            ...     milvus_uri="http://localhost:19530",
            ...     milvus_db="my_database",
            ...     collection_prefix="paper_a",
            ... )
        """
        # Build settings
        if settings:
            self.settings = settings
        else:
            settings_kwargs = {}
            if milvus_uri:
                settings_kwargs["milvus_uri"] = milvus_uri
            if milvus_token:
                settings_kwargs["milvus_token"] = milvus_token
            if milvus_db:
                settings_kwargs["milvus_db"] = milvus_db
            if collection_prefix:
                settings_kwargs["collection_prefix"] = collection_prefix
            if openai_api_key:
                settings_kwargs["openai_api_key"] = openai_api_key
            if llm_model:
                settings_kwargs["llm_model"] = llm_model
            if embedding_provider:
                settings_kwargs["embedding_provider"] = embedding_provider
            if embedding_model:
                settings_kwargs["embedding_model"] = embedding_model
            if embedding_api_key:
                settings_kwargs["embedding_api_key"] = embedding_api_key
            if embedding_base_url:
                settings_kwargs["embedding_base_url"] = embedding_base_url

            self.settings = Settings(**settings_kwargs)

        # Validate settings
        self.settings.validate_settings()

        # Initialize components
        self._embedding_model = EmbeddingModel(settings=self.settings)
        self._store = MilvusStore(
            settings=self.settings,
            embedding_model=self._embedding_model,
        )
        self._graph_builder = GraphBuilder(settings=self.settings)
        self._triplet_extractor = TripletExtractor(settings=self.settings)
        self._reranker = LLMReranker(settings=self.settings)
        self._answer_generator = AnswerGenerator(settings=self.settings)
        self._react_planner = ReActPlanner(settings=self.settings)

        # Retriever is initialized after documents are added
        self._retriever: Optional[GraphRetriever] = None

        # Track extraction result
        self._extraction_result: Optional[ExtractionResult] = None

        # Create collections
        self._store.create_collections(drop_existing=False)

    def _ensure_retriever(self) -> GraphRetriever:
        """Ensure retriever is initialized."""
        if self._retriever is None:
            self._retriever = GraphRetriever(
                store=self._store,
                graph_builder=self._graph_builder,
                settings=self.settings,
                embedding_model=self._embedding_model,
            )
        return self._retriever

    def _get_passages_from_subgraph(self, subgraph: SubGraph) -> List[str]:
        """
        Get passages from a subgraph.

        The subgraph lazily loads passage data from Milvus.

        Args:
            subgraph: The subgraph containing passage IDs.

        Returns:
            List of passage texts.
        """
        return subgraph.passage_texts

    @staticmethod
    def _append_unique(values: List[str], additions: List[str]) -> None:
        """Append unique values in-place while preserving order."""
        seen = set(values)
        for value in additions:
            if value not in seen:
                seen.add(value)
                values.append(value)

    @staticmethod
    def _format_react_observation(
        passages: List[str],
        relations: List[str],
        max_items: int = 5,
    ) -> str:
        """Format retrieved context as a compact ReAct observation."""
        lines: List[str] = []
        if passages:
            lines.append("Passages:")
            for index, passage in enumerate(passages[:max_items], start=1):
                lines.append(f"{index}. {passage}")
        else:
            lines.append("Passages: none")

        if relations:
            lines.append("Relations:")
            for index, relation in enumerate(relations[:max_items], start=1):
                lines.append(f"{index}. {relation}")
        else:
            lines.append("Relations: none")

        return "\n".join(lines)

    @staticmethod
    def _get_user_passage_metadata(doc: Document) -> Dict[str, Any]:
        """Return user metadata that can be stored on passage records."""
        metadata = dict(doc.metadata or {})
        metadata.pop("triplets", None)
        return metadata

    @staticmethod
    def _merge_passage_metadata(
        user_metadata: Dict[str, Any],
        entity_ids: List[str],
        relation_ids: List[str],
    ) -> Dict[str, Any]:
        """Merge user metadata with internal graph adjacency fields."""
        metadata = dict(user_metadata)
        metadata["entity_ids"] = entity_ids
        metadata["relation_ids"] = relation_ids
        return metadata

    @staticmethod
    def _merge_unique(existing: List[str], additions: List[str]) -> List[str]:
        """Merge lists while preserving order and removing duplicates."""
        merged: List[str] = []
        seen = set()
        for item in [*existing, *additions]:
            if item not in seen:
                seen.add(item)
                merged.append(item)
        return merged

    @staticmethod
    def _remove_many(values: List[str], removed: set[str]) -> List[str]:
        """Remove a set of values from a list while preserving order."""
        return [value for value in values if value not in removed]

    @staticmethod
    def _normalize_source_value(value: Any, field_name: str) -> str:
        """Validate and normalize a source metadata value."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")
        return value.strip()

    @staticmethod
    def _make_passage_id(source: str, chunk_index: int) -> str:
        """Create a stable Milvus-safe passage ID for a document chunk."""
        digest = hashlib.sha256(f"{source}:{chunk_index}".encode("utf-8")).hexdigest()
        return f"doc_{digest[:60]}"

    def _resolve_upsert_source(
        self,
        documents: List[Document],
        source: Optional[str],
        source_field: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Resolve the single source value for a by-source document update."""
        self._store._validate_field_name(source_field)
        if not documents:
            raise ValueError("documents must not be empty.")

        explicit_source = (
            self._normalize_source_value(source, "source") if source is not None else None
        )
        observed_sources: List[str] = []

        if metadata and source_field in metadata and metadata[source_field] is not None:
            observed_sources.append(
                self._normalize_source_value(metadata[source_field], source_field)
            )

        for doc in documents:
            doc_metadata = dict(doc.metadata or {})
            if source_field in doc_metadata and doc_metadata[source_field] is not None:
                observed_sources.append(
                    self._normalize_source_value(doc_metadata[source_field], source_field)
                )

        unique_sources = set(observed_sources)
        if explicit_source is not None:
            conflicting_sources = unique_sources - {explicit_source}
            if conflicting_sources:
                conflicts = ", ".join(sorted(conflicting_sources))
                raise ValueError(
                    f"documents contain {source_field} values that differ from source "
                    f'"{explicit_source}": {conflicts}'
                )
            return explicit_source

        if not unique_sources:
            raise ValueError(
                f'documents must include metadata["{source_field}"] or source must be provided.'
            )

        if len(unique_sources) > 1:
            sources = ", ".join(sorted(unique_sources))
            raise ValueError(
                f"upsert_documents_by_source() expects one source per call, but found "
                f"multiple {source_field} values: {sources}"
            )

        return next(iter(unique_sources))

    def _prepare_upsert_documents(
        self,
        source: str,
        documents: List[Document],
        source_field: str = "source",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Copy documents and attach source metadata for by-source upsert."""
        if not source:
            raise ValueError("source must not be empty.")
        if not documents:
            raise ValueError("documents must not be empty.")

        prepared: List[Document] = []
        base_metadata = dict(metadata or {})

        for chunk_index, doc in enumerate(documents):
            passage_id = doc.id or self._make_passage_id(source, chunk_index)
            chunk_metadata = {
                **base_metadata,
                **dict(doc.metadata or {}),
                source_field: source,
                "chunk_index": chunk_index,
            }
            prepared.append(
                Document(
                    page_content=doc.page_content,
                    metadata=chunk_metadata,
                    id=passage_id,
                )
            )

        return prepared

    def _remap_extraction_result(
        self,
        result: ExtractionResult,
        entity_id_map: Dict[str, str],
        relation_id_map: Dict[str, str],
    ) -> ExtractionResult:
        """Return an extraction result with stored entity/relation IDs."""
        entities = [
            Entity(id=entity_id_map.get(entity.id or "", entity.id), name=entity.name)
            for entity in result.entities
        ]
        relations = [
            Relation(
                id=relation_id_map.get(relation.id or "", relation.id),
                text=relation.text,
                triplet=relation.triplet,
                source_passage_ids=relation.source_passage_ids,
                embedding=relation.embedding,
            )
            for relation in result.relations
        ]
        entity_to_relation_ids = {
            entity_id_map.get(entity_id, entity_id): [
                relation_id_map.get(relation_id, relation_id) for relation_id in relation_ids
            ]
            for entity_id, relation_ids in result.entity_to_relation_ids.items()
        }
        relation_to_passage_ids = {
            relation_id_map.get(relation_id, relation_id): passage_ids
            for relation_id, passage_ids in result.relation_to_passage_ids.items()
        }
        return ExtractionResult(
            documents=result.documents,
            entities=entities,
            relations=relations,
            entity_to_relation_ids=entity_to_relation_ids,
            relation_to_passage_ids=relation_to_passage_ids,
        )

    def _build_graph_records(
        self,
        builder: GraphBuilder,
        documents: List[Document],
        show_progress: bool,
    ) -> Tuple[
        ExtractionResult,
        List[List[float]],
        List[List[float]],
        List[List[float]],
        Dict[str, Dict[str, Any]],
    ]:
        """Build graph records and embeddings for a document batch."""
        result = builder.build_from_documents(documents)

        if show_progress:
            logger.info("Generating embeddings...")

        entity_texts = builder.get_entity_texts()
        relation_texts = builder.get_relation_texts()
        passage_texts = builder.get_passage_texts()

        entity_embeddings = (
            self._embedding_model.embed_batch(entity_texts, show_progress=show_progress)
            if entity_texts
            else []
        )

        relation_embeddings = (
            self._embedding_model.embed_batch(relation_texts, show_progress=show_progress)
            if relation_texts
            else []
        )

        passage_embeddings = (
            self._embedding_model.embed_batch(passage_texts, show_progress=show_progress)
            if passage_texts
            else []
        )

        passage_user_metadatas = {
            doc.id: self._get_user_passage_metadata(doc) for doc in documents if doc.id is not None
        }

        return (
            result,
            entity_embeddings,
            relation_embeddings,
            passage_embeddings,
            passage_user_metadatas,
        )

    def _insert_rebuilt_graph(
        self,
        builder: GraphBuilder,
        passage_user_metadatas: Dict[str, Dict[str, Any]],
        entity_embeddings: List[List[float]],
        relation_embeddings: List[List[float]],
        passage_embeddings: List[List[float]],
        show_progress: bool,
    ) -> None:
        """Insert a full graph after collections have been recreated."""
        entity_metadatas = []
        for eid in builder.entity_ids:
            entity_metadatas.append(
                {
                    "relation_ids": builder.entity_to_relation_ids.get(eid, []),
                    "passage_ids": builder.entity_to_passage_ids.get(eid, []),
                }
            )

        relation_metadatas = []
        for rid in builder.relation_ids:
            triplet = builder.relation_id_to_triplet.get(rid)
            metadata = {
                "entity_ids": builder.relation_to_entity_ids.get(rid, []),
                "passage_ids": builder.relation_to_passage_ids.get(rid, []),
            }
            if triplet:
                metadata["subject"] = triplet.subject
                metadata["predicate"] = triplet.predicate
                metadata["object"] = triplet.object
            relation_metadatas.append(metadata)

        passage_metadatas = []
        for pid in builder.passage_ids:
            passage_metadatas.append(
                self._merge_passage_metadata(
                    passage_user_metadatas.get(pid, {}),
                    builder.passage_to_entity_ids.get(pid, []),
                    builder.passage_to_relation_ids.get(pid, []),
                )
            )

        if show_progress:
            logger.info("Inserting into Milvus...")

        self._store._insert_entities(
            builder.get_entity_texts(),
            ids=builder.entity_ids,
            embeddings=entity_embeddings,
            metadatas=entity_metadatas,
            show_progress=show_progress,
        )
        self._store._insert_relations(
            builder.get_relation_texts(),
            ids=builder.relation_ids,
            embeddings=relation_embeddings,
            metadatas=relation_metadatas,
            show_progress=show_progress,
        )
        self._store.insert_passages(
            builder.get_passage_texts(),
            ids=builder.passage_ids,
            embeddings=passage_embeddings,
            metadatas=passage_metadatas,
            show_progress=show_progress,
        )

    def _insert_incremental_graph(
        self,
        builder: GraphBuilder,
        passage_user_metadatas: Dict[str, Dict[str, Any]],
        entity_embeddings: List[List[float]],
        relation_embeddings: List[List[float]],
        passage_embeddings: List[List[float]],
        source: str,
        source_field: str,
        show_progress: bool,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Insert graph records for one source, reusing existing nodes."""
        entity_texts = builder.get_entity_texts()
        relation_texts = builder.get_relation_texts()
        existing_entities = self._store._get_entities_by_texts(entity_texts)
        existing_relations = self._store._get_relations_by_texts(relation_texts)

        entity_id_map = {
            eid: existing_entities.get(builder.entities[eid], {}).get("id", eid)
            for eid in builder.entity_ids
        }
        relation_id_map = {
            rid: existing_relations.get(builder.relations[rid], {}).get("id", rid)
            for rid in builder.relation_ids
        }

        existing_passages = self._store.get_passages_by_ids(
            builder.passage_ids,
            output_fields=["id", "text", source_field],
        )
        for passage in existing_passages:
            if passage.get(source_field) != source:
                raise ValueError(f'Passage ID "{passage["id"]}" already belongs to another source.')

        if show_progress:
            logger.info("Upserting incremental graph into Milvus...")

        entity_text_by_id = {eid: builder.entities[eid] for eid in builder.entity_ids}
        relation_text_by_id = {rid: builder.relations[rid] for rid in builder.relation_ids}

        entity_insert_ids: List[str] = []
        entity_insert_texts: List[str] = []
        entity_insert_embeddings: List[List[float]] = []
        entity_insert_metadatas: List[Dict[str, Any]] = []
        entity_update_records: List[Dict[str, Any]] = []

        for index, eid in enumerate(builder.entity_ids):
            stored_id = entity_id_map[eid]
            relation_ids = [
                relation_id_map[relation_id]
                for relation_id in builder.entity_to_relation_ids.get(eid, [])
            ]
            passage_ids = builder.entity_to_passage_ids.get(eid, [])
            metadata = {
                "relation_ids": relation_ids,
                "passage_ids": passage_ids,
            }
            existing = existing_entities.get(entity_text_by_id[eid])
            if existing:
                entity_update_records.append(
                    {
                        "id": stored_id,
                        "text": entity_text_by_id[eid],
                        "vector": entity_embeddings[index],
                        "relation_ids": self._merge_unique(
                            existing.get("relation_ids", []),
                            relation_ids,
                        ),
                        "passage_ids": self._merge_unique(
                            existing.get("passage_ids", []),
                            passage_ids,
                        ),
                    }
                )
            else:
                entity_insert_ids.append(stored_id)
                entity_insert_texts.append(entity_text_by_id[eid])
                entity_insert_embeddings.append(entity_embeddings[index])
                entity_insert_metadatas.append(metadata)

        if entity_update_records:
            self._store._upsert_entity_records(entity_update_records)

        if entity_insert_ids:
            self._store._insert_entities(
                entity_insert_texts,
                ids=entity_insert_ids,
                embeddings=entity_insert_embeddings,
                metadatas=entity_insert_metadatas,
                show_progress=show_progress,
            )

        relation_insert_ids: List[str] = []
        relation_insert_texts: List[str] = []
        relation_insert_embeddings: List[List[float]] = []
        relation_insert_metadatas: List[Dict[str, Any]] = []
        relation_update_records: List[Dict[str, Any]] = []

        for index, rid in enumerate(builder.relation_ids):
            stored_id = relation_id_map[rid]
            triplet = builder.relation_id_to_triplet.get(rid)
            entity_ids = [
                entity_id_map[entity_id]
                for entity_id in builder.relation_to_entity_ids.get(rid, [])
            ]
            passage_ids = builder.relation_to_passage_ids.get(rid, [])
            metadata = {
                "entity_ids": entity_ids,
                "passage_ids": passage_ids,
            }
            if triplet:
                metadata["subject"] = triplet.subject
                metadata["predicate"] = triplet.predicate
                metadata["object"] = triplet.object

            existing = existing_relations.get(relation_text_by_id[rid])
            if existing:
                update_record = {
                    "id": stored_id,
                    "text": relation_text_by_id[rid],
                    "vector": relation_embeddings[index],
                    "entity_ids": entity_ids,
                    "passage_ids": self._merge_unique(
                        existing.get("passage_ids", []),
                        passage_ids,
                    ),
                }
                for field in ["subject", "predicate", "object"]:
                    if metadata.get(field) is not None:
                        update_record[field] = metadata[field]
                    elif field in existing:
                        update_record[field] = existing[field]
                relation_update_records.append(update_record)
            else:
                relation_insert_ids.append(stored_id)
                relation_insert_texts.append(relation_text_by_id[rid])
                relation_insert_embeddings.append(relation_embeddings[index])
                relation_insert_metadatas.append(metadata)

        if relation_update_records:
            self._store._upsert_relation_records(relation_update_records)

        if relation_insert_ids:
            self._store._insert_relations(
                relation_insert_texts,
                ids=relation_insert_ids,
                embeddings=relation_insert_embeddings,
                metadatas=relation_insert_metadatas,
                show_progress=show_progress,
            )

        passage_metadatas = []
        for pid in builder.passage_ids:
            passage_metadatas.append(
                self._merge_passage_metadata(
                    passage_user_metadatas.get(pid, {}),
                    [
                        entity_id_map[entity_id]
                        for entity_id in builder.passage_to_entity_ids.get(pid, [])
                    ],
                    [
                        relation_id_map[relation_id]
                        for relation_id in builder.passage_to_relation_ids.get(pid, [])
                    ],
                )
            )

        self._store.insert_passages(
            builder.get_passage_texts(),
            ids=builder.passage_ids,
            embeddings=passage_embeddings,
            metadatas=passage_metadatas,
            show_progress=show_progress,
        )

        return entity_id_map, relation_id_map

    def _get_passages_from_relations(
        self,
        relation_ids: List[str],
        filter: Optional[str] = None,
    ) -> tuple[List[str], List[str]]:
        """
        Get passages associated with given relations.

        Args:
            relation_ids: List of relation IDs (strings).
            filter: Optional Milvus filter expression for passage metadata.

        Returns:
            Tuple of (passage_ids, passage_texts).
        """
        if not relation_ids:
            return [], []

        # Query Milvus for relation data (using private method)
        relation_data = self._store._get_relations_by_ids(relation_ids)

        passage_ids: List[str] = []
        seen_ids: set = set()
        for rel in relation_data:
            for pid in rel.get("passage_ids", []):
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    passage_ids.append(pid)

        if not passage_ids:
            return [], []

        passage_data = self._store.get_passages_by_ids(passage_ids, filter=filter)
        id_to_text = {p["id"]: p["text"] for p in passage_data}

        passages = [id_to_text[pid] for pid in passage_ids if pid in id_to_text]
        passage_ids = [pid for pid in passage_ids if pid in id_to_text]
        return passage_ids, passages

    def add_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
        extract_triplets: bool = True,
        show_progress: bool = True,
    ) -> ExtractionResult:
        """
        Add text strings to the knowledge base.

        Warning:
            This legacy convenience method rebuilds the full knowledge base and
            is planned for removal in v1.0.0. Use rebuild_texts() for full
            refreshes, or upsert_documents_by_source() for source-level
            incremental create/update.

        This is a convenience method that converts texts to Document objects
        and calls rebuild_texts().

        Args:
            texts: List of text strings.
            ids: Optional list of IDs. If not provided, UUIDs are generated.
            metadatas: Optional list of metadata dicts.
            extract_triplets: Whether to extract triplets using LLM.
            show_progress: Whether to show progress bars.

        Returns:
            ExtractionResult with graph statistics.

        Example:
            >>> rag.add_texts([
            ...     "Albert Einstein developed the theory of relativity.",
            ...     "The theory of relativity changed physics forever.",
            ... ])
            >>>
            >>> # With custom IDs
            >>> rag.add_texts(
            ...     ["Document 1", "Document 2"],
            ...     ids=["doc_001", "doc_002"]
            ... )
        """
        warnings.warn(
            "add_texts() is deprecated and will be removed in v1.0.0. "
            "Use rebuild_texts() for full refreshes, or upsert_documents_by_source() "
            "for source-level incremental create/update.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.rebuild_texts(
            texts,
            ids=ids,
            metadatas=metadatas,
            extract_triplets=extract_triplets,
            show_progress=show_progress,
        )

    def rebuild_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
        extract_triplets: bool = True,
        show_progress: bool = True,
    ) -> ExtractionResult:
        """
        Rebuild the knowledge base from text strings.

        This is a convenience method that converts texts to Document objects
        and calls rebuild_documents().

        Args:
            texts: List of text strings.
            ids: Optional list of IDs. If not provided, UUIDs are generated.
            metadatas: Optional list of metadata dicts.
            extract_triplets: Whether to extract triplets using LLM.
            show_progress: Whether to show progress bars.

        Returns:
            ExtractionResult with graph statistics for the rebuilt graph.

        Example:
            >>> rag.rebuild_texts([
            ...     "Albert Einstein developed the theory of relativity.",
            ...     "The theory of relativity changed physics forever.",
            ... ])
        """
        documents = []
        for i, text in enumerate(texts):
            doc_id = ids[i] if ids and i < len(ids) else str(uuid.uuid4())
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            documents.append(Document(page_content=text, metadata=metadata, id=doc_id))

        return self.rebuild_documents(
            documents, extract_triplets=extract_triplets, show_progress=show_progress
        )

    def add_documents(
        self,
        documents: List[Document],
        extract_triplets: bool = True,
        show_progress: bool = True,
    ) -> ExtractionResult:
        """
        Add Document objects by rebuilding the knowledge base.

        Warning:
            This method keeps its original full-rebuild behavior for backward
            compatibility. It drops and recreates the Milvus collections before
            indexing the provided documents. Use upsert_documents_by_source()
            for source-level incremental create/update. This legacy method is
            planned for removal in v1.0.0; use rebuild_documents() for full
            refreshes.

        This method:
        1. Extracts triplets from documents (if enabled)
        2. Builds the knowledge graph structure
        3. Drops and recreates the Milvus collections
        4. Indexes entities, relations, and passages in Milvus

        Args:
            documents: List of langchain_core Document objects.
                       Each Document has page_content (text) and metadata.
                       If Document.id is None, a UUID will be generated.
                       Pre-extracted triplets can be stored in metadata["triplets"].
            extract_triplets: Whether to extract triplets using LLM.
            show_progress: Whether to show progress bars.

        Returns:
            ExtractionResult with graph statistics.

        Example:
            >>> from langchain_core.documents import Document
            >>> rag.add_documents([
            ...     Document(page_content="Einstein developed relativity."),
            ...     Document(page_content="Relativity changed physics.", id="doc_002"),
            ... ])
        """
        warnings.warn(
            "add_documents() is deprecated and will be removed in v1.0.0. "
            "Use rebuild_documents() for full refreshes, or upsert_documents_by_source() "
            "for source-level incremental create/update.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.rebuild_documents(
            documents,
            extract_triplets=extract_triplets,
            show_progress=show_progress,
        )

    def rebuild_documents(
        self,
        documents: List[Document],
        extract_triplets: bool = True,
        show_progress: bool = True,
    ) -> ExtractionResult:
        """
        Rebuild the knowledge base from the provided documents.

        This is the explicit full-refresh API. It removes all existing graph
        records from this collection prefix and then indexes the given
        documents.

        Args:
            documents: List of langchain_core Document objects.
                       Each Document has page_content (text) and metadata.
                       If Document.id is None, a UUID will be generated.
                       Pre-extracted triplets can be stored in metadata["triplets"].
            extract_triplets: Whether to extract triplets using LLM.
            show_progress: Whether to show progress bars.

        Returns:
            ExtractionResult with graph statistics for the rebuilt graph.
        """
        # Ensure all documents have IDs
        for doc in documents:
            if not doc.id:
                doc.id = str(uuid.uuid4())

        # Extract triplets if needed
        if extract_triplets:
            documents = self._triplet_extractor.extract_from_documents(
                documents, show_progress=show_progress
            )

        (
            self._extraction_result,
            entity_embeddings,
            relation_embeddings,
            passage_embeddings,
            passage_user_metadatas,
        ) = self._build_graph_records(
            self._graph_builder,
            documents,
            show_progress=show_progress,
        )

        # Drop and recreate collections for fresh data
        self._store.drop_collections()
        self._store.create_collections(drop_existing=True)

        self._insert_rebuilt_graph(
            self._graph_builder,
            passage_user_metadatas,
            entity_embeddings,
            relation_embeddings,
            passage_embeddings,
            show_progress=show_progress,
        )

        # Reset retriever to pick up new knowledge graph
        self._retriever = None

        return self._extraction_result

    def upsert_documents_by_source(
        self,
        documents: List[Document],
        source: Optional[str] = None,
        source_field: str = "source",
        metadata: Optional[Dict[str, Any]] = None,
        extract_triplets: bool = True,
        show_progress: bool = True,
    ) -> ExtractionResult:
        """
        Incrementally create or replace documents that belong to one source.

        In this project, a LangChain Document is a passage/chunk. This method
        uses a stable source metadata value to replace all chunks that belong to
        one source object, such as a file, URL, message, or business record.

        Args:
            documents: Parsed chunks/passages for one source.
            source: Optional stable source value. If omitted, it is inferred
                    from ``Document.metadata[source_field]``.
            source_field: Metadata field used to group chunks by source.
            metadata: Optional source-level metadata merged into each chunk.
            extract_triplets: Whether to extract triplets using LLM.
            show_progress: Whether to show progress bars.

        Returns:
            ExtractionResult with graph statistics for this document update.

        Example:
            >>> rag.upsert_documents_by_source(
            ...     documents=[
            ...         Document(
            ...             page_content="Einstein developed relativity.",
            ...             metadata={"source": "sharepoint:file-123"},
            ...         )
            ...     ],
            ... )
        """
        resolved_source = self._resolve_upsert_source(
            documents,
            source=source,
            source_field=source_field,
            metadata=metadata,
        )
        prepared_documents = self._prepare_upsert_documents(
            resolved_source,
            documents,
            source_field=source_field,
            metadata=metadata,
        )

        if extract_triplets:
            prepared_documents = self._triplet_extractor.extract_from_documents(
                prepared_documents,
                show_progress=show_progress,
            )

        builder = GraphBuilder(settings=self.settings)
        (
            result,
            entity_embeddings,
            relation_embeddings,
            passage_embeddings,
            passage_user_metadatas,
        ) = self._build_graph_records(
            builder,
            prepared_documents,
            show_progress=show_progress,
        )

        self.delete_documents_by_source(resolved_source, source_field=source_field)
        entity_id_map, relation_id_map = self._insert_incremental_graph(
            builder,
            passage_user_metadatas,
            entity_embeddings,
            relation_embeddings,
            passage_embeddings,
            source=resolved_source,
            source_field=source_field,
            show_progress=show_progress,
        )

        self._extraction_result = self._remap_extraction_result(
            result,
            entity_id_map=entity_id_map,
            relation_id_map=relation_id_map,
        )
        self._retriever = None
        return self._extraction_result

    def upsert_documents(self, *args: Any, **kwargs: Any) -> ExtractionResult:
        """
        Removed incremental API.

        Use upsert_documents_by_source() instead.
        """
        raise RuntimeError(
            "upsert_documents(document_id=..., documents=...) was removed because "
            "Document represents a chunk/passage in vector-graph-rag. Use "
            "upsert_documents_by_source(documents=..., source=...) or set a stable "
            'Document.metadata["source"] value and call upsert_documents_by_source(documents=...).'
        )

    def delete_documents_by_source(
        self,
        source: str,
        source_field: str = "source",
    ) -> bool:
        """
        Incrementally delete all documents that belong to one source.

        Args:
            source: Stable source value previously used with
                    upsert_documents_by_source().
            source_field: Metadata field used to group chunks by source.

        Returns:
            True when at least one passage was deleted, otherwise False.
        """
        source = self._normalize_source_value(source, "source")
        self._store._validate_field_name(source_field)

        passages = self._store.get_passages_by_source(source, source_field=source_field)
        if not passages:
            return False

        passage_ids = [passage["id"] for passage in passages]
        removed_passage_ids = set(passage_ids)

        relation_ids = sorted(
            {relation_id for passage in passages for relation_id in passage.get("relation_ids", [])}
        )
        entity_ids = sorted(
            {entity_id for passage in passages for entity_id in passage.get("entity_ids", [])}
        )

        relations = self._store._get_relations_by_ids(relation_ids)
        relation_update_records: List[Dict[str, Any]] = []
        relation_delete_ids: List[str] = []
        deleted_relation_ids: set[str] = set()
        for relation in relations:
            relation_id = relation["id"]
            new_passage_ids = self._remove_many(
                relation.get("passage_ids", []),
                removed_passage_ids,
            )
            if new_passage_ids:
                update_record = {
                    "id": relation_id,
                    "text": relation["text"],
                    "vector": self._embedding_model.embed(relation["text"]),
                    "entity_ids": relation.get("entity_ids", []),
                    "passage_ids": new_passage_ids,
                }
                for field in ["subject", "predicate", "object"]:
                    if field in relation:
                        update_record[field] = relation[field]
                relation_update_records.append(update_record)
            else:
                relation_delete_ids.append(relation_id)
                deleted_relation_ids.add(relation_id)

        if relation_update_records:
            self._store._upsert_relation_records(relation_update_records)
        if relation_delete_ids:
            self._store._delete_relations(relation_delete_ids)

        entities = self._store._get_entities_by_ids(entity_ids)
        entity_update_records: List[Dict[str, Any]] = []
        entity_delete_ids: List[str] = []
        for entity in entities:
            entity_id = entity["id"]
            new_passage_ids = self._remove_many(
                entity.get("passage_ids", []),
                removed_passage_ids,
            )
            new_relation_ids = self._remove_many(
                entity.get("relation_ids", []),
                deleted_relation_ids,
            )

            if new_passage_ids or new_relation_ids:
                entity_update_records.append(
                    {
                        "id": entity_id,
                        "text": entity["text"],
                        "vector": self._embedding_model.embed(entity["text"]),
                        "passage_ids": new_passage_ids,
                        "relation_ids": new_relation_ids,
                    }
                )
            else:
                entity_delete_ids.append(entity_id)

        if entity_update_records:
            self._store._upsert_entity_records(entity_update_records)
        if entity_delete_ids:
            self._store._delete_entities(entity_delete_ids)

        self._store.delete_passages(passage_ids)
        self._retriever = None
        return True

    def delete_documents(self, *args: Any, **kwargs: Any) -> bool:
        """
        Removed incremental API.

        Use delete_documents_by_source() instead.
        """
        raise RuntimeError(
            "delete_documents(document_id) was removed because Document represents "
            "a chunk/passage in vector-graph-rag. Use "
            "delete_documents_by_source(source=...) instead."
        )

    def add_documents_with_triplets(
        self,
        documents: List[dict],
        show_progress: bool = True,
    ) -> ExtractionResult:
        """
        Add documents with pre-extracted triplets.

        Warning:
            This legacy convenience method rebuilds the full knowledge base and
            is planned for removal in v1.0.0. Use
            rebuild_documents_with_triplets() for full refreshes. For
            incremental updates with pre-extracted triplets, store triplets in
            each chunk's metadata["triplets"] and call
            upsert_documents_by_source(..., extract_triplets=False).

        Use this method if you already have triplets extracted,
        to avoid the LLM triplet extraction step.

        Args:
            documents: List of dicts with "passage" and "triplets" keys.
                       Optionally include "id" for custom document ID.
                       Each triplet is [subject, predicate, object].
            show_progress: Whether to show progress bars.

        Returns:
            ExtractionResult with graph statistics.

        Example:
            >>> rag.add_documents_with_triplets([
            ...     {
            ...         "id": "doc_001",  # optional
            ...         "passage": "Einstein developed relativity.",
            ...         "triplets": [
            ...             ["Einstein", "developed", "relativity"],
            ...         ],
            ...     },
            ... ])
        """
        warnings.warn(
            "add_documents_with_triplets() is deprecated and will be removed in "
            "v1.0.0. Use rebuild_documents_with_triplets() for full refreshes, "
            "or upsert_documents_by_source(..., extract_triplets=False) for "
            "source-level incremental create/update.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.rebuild_documents_with_triplets(documents, show_progress=show_progress)

    def rebuild_documents_with_triplets(
        self,
        documents: List[dict],
        show_progress: bool = True,
    ) -> ExtractionResult:
        """
        Rebuild the knowledge base from documents with pre-extracted triplets.

        Use this method if you already have triplets extracted,
        to avoid the LLM triplet extraction step.

        Args:
            documents: List of dicts with "passage" and "triplets" keys.
                       Optionally include "id" for custom document ID.
                       Each triplet is [subject, predicate, object].
            show_progress: Whether to show progress bars.

        Returns:
            ExtractionResult with graph statistics for the rebuilt graph.

        Example:
            >>> rag.rebuild_documents_with_triplets([
            ...     {
            ...         "id": "doc_001",  # optional
            ...         "passage": "Einstein developed relativity.",
            ...         "triplets": [
            ...             ["Einstein", "developed", "relativity"],
            ...         ],
            ...     },
            ... ])
        """
        docs = []
        for doc_data in documents:
            passage = doc_data.get("passage", doc_data.get("text"))
            if passage is None:
                raise ValueError('Each document must include "passage" or "text".')
            doc_id = doc_data.get("id") or str(uuid.uuid4())
            # Store triplets in metadata as list of [subject, predicate, object]
            triplets = doc_data.get("triplets", [])
            metadata = dict(doc_data.get("metadata") or {})
            metadata["triplets"] = triplets
            docs.append(
                Document(
                    page_content=passage,
                    metadata=metadata,
                    id=doc_id,
                )
            )

        return self.rebuild_documents(docs, extract_triplets=False, show_progress=show_progress)

    def query(
        self,
        question: str,
        use_reranking: bool = True,
        compare_naive: bool = False,
        entity_top_k: Optional[int] = None,
        relation_top_k: Optional[int] = None,
        entity_similarity_threshold: Optional[float] = None,
        relation_similarity_threshold: Optional[float] = None,
        expansion_degree: Optional[int] = None,
        filter: Optional[str] = None,
    ) -> QueryResult:
        """
        Query the knowledge base.

        This method performs Graph RAG retrieval:
        1. Extracts entities from the question
        2. Retrieves similar entities and relations (with similarity threshold filtering)
        3. Expands the subgraph
        4. Reranks candidate relations (optional)
        5. Retrieves final passages
        6. Generates an answer

        Args:
            question: The question to answer.
            use_reranking: Whether to use LLM reranking.
            compare_naive: If True, also runs naive RAG for comparison.
            entity_top_k: Override entity retrieval top_k.
            relation_top_k: Override relation retrieval top_k.
            entity_similarity_threshold: Override entity similarity threshold.
            relation_similarity_threshold: Override relation similarity threshold.
            expansion_degree: Override expansion degree.
            filter: Optional Milvus filter expression for passage metadata.

        Returns:
            QueryResult with answer, retrieval details, and subgraph for visualization.

        Example:
            >>> result = rag.query("What did Einstein develop?")
            >>> print(result.answer)
            >>> print(result.subgraph.expansion_history)  # Visualize expansion
        """
        retriever = self._ensure_retriever()

        # Retrieve with custom parameters
        retrieval_result = retriever.retrieve(
            question,
            entity_top_k=entity_top_k,
            relation_top_k=relation_top_k,
            entity_similarity_threshold=entity_similarity_threshold,
            relation_similarity_threshold=relation_similarity_threshold,
            expansion_degree=expansion_degree,
            filter=filter,
        )

        # Build retrieval detail for visualization
        retrieval_detail = RetrievalDetail(
            entity_ids=retrieval_result.entity_ids,
            entity_texts=retrieval_result.entity_texts,
            entity_scores=retrieval_result.entity_scores,
            relation_ids=retrieval_result.relation_ids,
            relation_texts=retrieval_result.relation_texts,
            relation_scores=retrieval_result.relation_scores,
        )

        # Get candidate relations from subgraph
        candidate_ids = retrieval_result.expanded_relation_ids
        candidate_texts = retrieval_result.expanded_relation_texts

        # Rerank if enabled
        rerank_result = None
        if use_reranking and candidate_ids:
            reranked_ids, reranked_texts = self._reranker.rerank(
                question, candidate_ids, candidate_texts
            )
            rerank_result = RerankResult(
                selected_relation_ids=reranked_ids,
                selected_relation_texts=reranked_texts,
            )
        else:
            reranked_ids = candidate_ids[: self.settings.final_top_k]
            reranked_texts = candidate_texts[: self.settings.final_top_k]

        # Get passages from reranked relations
        passage_ids, passages = self._get_passages_from_relations(reranked_ids, filter=filter)
        final_passages = passages[: self.settings.final_top_k]

        # Generate answer
        answer = self._answer_generator.generate(question, final_passages)

        # Build eviction result
        eviction_result = EvictionResult(
            occurred=retrieval_result.eviction_occurred,
            before_count=retrieval_result.eviction_before_count,
            after_count=retrieval_result.eviction_after_count,
        )

        return QueryResult(
            query=question,
            answer=answer,
            query_entities=retrieval_result.query_entities,
            retrieved_passages=final_passages,
            retrieved_relations=retrieval_result.relation_texts,
            expanded_relations=candidate_texts,
            reranked_relations=reranked_texts,
            subgraph=retrieval_result.subgraph,
            passages=final_passages,
            retrieval_detail=retrieval_detail,
            rerank_result=rerank_result,
            eviction_result=eviction_result,
        )

    def query_simple(self, question: str, filter: Optional[str] = None) -> str:
        """
        Simple query that returns just the answer.

        Args:
            question: The question to answer.
            filter: Optional Milvus filter expression for passage metadata.

        Returns:
            The answer string.

        Example:
            >>> answer = rag.query_simple("What did Einstein develop?")
            >>> print(answer)
        """
        return self.query(question, filter=filter).answer

    def query_naive(self, question: str, filter: Optional[str] = None) -> QueryResult:
        """
        Query using naive RAG (direct passage retrieval).

        Useful for comparing Graph RAG performance against baseline.

        Args:
            question: The question to answer.
            filter: Optional Milvus filter expression for passage metadata.

        Returns:
            QueryResult with answer and retrieved passages.
        """
        retriever = self._ensure_retriever()
        passages = retriever.retrieve_passages_naive(question, filter=filter)
        answer = self._answer_generator.generate(question, passages)

        return QueryResult(
            query=question,
            answer=answer,
            retrieved_passages=passages,
            retrieved_relations=[],
            expanded_relations=[],
            reranked_relations=[],
        )

    def query_react(
        self,
        question: str,
        max_steps: int = 3,
        use_reranking: bool = True,
        top_k: Optional[int] = None,
        filter: Optional[str] = None,
    ) -> ReActResult:
        """
        Query with a basic ReAct loop over Graph RAG retrieval.

        The planner can search multiple times before returning a final answer.
        Each search action uses the existing Graph RAG retrieval path. If the
        planner does not finish within max_steps, an answer is generated from
        the passages observed across search steps.

        Args:
            question: The question to answer.
            max_steps: Maximum planner steps.
            use_reranking: Whether each search uses LLM reranking.
            top_k: Number of passages to retrieve per search step.
            filter: Optional Milvus filter expression for passage metadata.

        Returns:
            ReActResult with final answer and step trace.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")

        steps: List[ReActStep] = []
        observed_passages: List[str] = []
        observed_relations: List[str] = []

        for step_number in range(1, max_steps + 1):
            action = self._react_planner.plan(
                question=question,
                steps=steps,
                step_number=step_number,
                max_steps=max_steps,
            )

            if action.action == "finish":
                answer = action.answer or ""
                steps.append(
                    ReActStep(
                        step=step_number,
                        thought=action.thought,
                        action="finish",
                        answer=answer,
                    )
                )
                return ReActResult(
                    query=question,
                    answer=answer,
                    steps=steps,
                    passages=observed_passages,
                    relations=observed_relations,
                    finished=True,
                )

            search_query = action.query or question
            retrieval_result = self.retrieve(
                search_query,
                use_reranking=use_reranking,
                top_k=top_k,
                filter=filter,
            )
            passages = retrieval_result.passages or retrieval_result.retrieved_passages
            relations = retrieval_result.reranked_relations or retrieval_result.retrieved_relations
            self._append_unique(observed_passages, passages)
            self._append_unique(observed_relations, relations)
            observation = self._format_react_observation(passages, relations)

            steps.append(
                ReActStep(
                    step=step_number,
                    thought=action.thought,
                    action="search",
                    query=search_query,
                    observation=observation,
                    retrieved_passages=passages,
                    retrieved_relations=relations,
                )
            )

        answer = self._answer_generator.generate(question, observed_passages)
        return ReActResult(
            query=question,
            answer=answer,
            steps=steps,
            passages=observed_passages,
            relations=observed_relations,
            finished=False,
        )

    def retrieve(
        self,
        question: str,
        use_reranking: bool = True,
        top_k: Optional[int] = None,
        filter: Optional[str] = None,
    ) -> QueryResult:
        """
        Retrieve passages using Graph RAG without generating an answer.

        This method performs all retrieval steps but skips LLM answer generation,
        useful for evaluation or when only retrieved documents are needed.

        Args:
            question: The question to answer.
            use_reranking: Whether to use LLM reranking.
            top_k: Number of passages to retrieve. Uses settings.final_top_k if not provided.
            filter: Optional Milvus filter expression for passage metadata.

        Returns:
            QueryResult with retrieved passages (answer will be empty).
        """
        retriever = self._ensure_retriever()
        top_k = top_k or self.settings.final_top_k

        # Retrieve
        retrieval_result = retriever.retrieve(question, filter=filter)

        # Get candidate relations
        candidate_ids = retrieval_result.expanded_relation_ids
        candidate_texts = retrieval_result.expanded_relation_texts

        # Rerank if enabled
        if use_reranking and candidate_ids:
            reranked_ids, reranked_texts = self._reranker.rerank(
                question, candidate_ids, candidate_texts
            )
        else:
            reranked_ids = candidate_ids[:top_k]
            reranked_texts = candidate_texts[:top_k]

        # Get passages from reranked relations
        passage_ids, passages = self._get_passages_from_relations(reranked_ids, filter=filter)

        if len(passages) < top_k:
            additional_passages = retriever.retrieve_passages_naive(
                question,
                top_k=top_k,
                filter=filter,
            )
            for passage in additional_passages:
                if passage not in passages:
                    passages.append(passage)
                    if len(passages) >= top_k:
                        break
        final_passages = passages[:top_k]

        return QueryResult(
            query=question,
            answer="",  # No answer generation
            retrieved_passages=final_passages,
            retrieved_relations=retrieval_result.relation_texts,
            expanded_relations=candidate_texts,
            reranked_relations=reranked_texts,
        )

    def retrieve_naive(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter: Optional[str] = None,
    ) -> QueryResult:
        """
        Retrieve passages using naive RAG without generating an answer.

        This method performs direct passage retrieval but skips LLM answer generation,
        useful for evaluation or when only retrieved documents are needed.

        Args:
            question: The question to answer.
            top_k: Number of passages to retrieve. Uses settings.final_top_k if not provided.
            filter: Optional Milvus filter expression for passage metadata.

        Returns:
            QueryResult with retrieved passages (answer will be empty).
        """
        retriever = self._ensure_retriever()
        top_k = top_k or self.settings.final_top_k
        passages = retriever.retrieve_passages_naive(question, top_k=top_k, filter=filter)

        return QueryResult(
            query=question,
            answer="",  # No answer generation
            retrieved_passages=passages,
            retrieved_relations=[],
            expanded_relations=[],
            reranked_relations=[],
        )

    def get_stats(self) -> dict:
        """
        Get statistics about the knowledge base.

        Returns:
            Dictionary with counts of entities, relations, passages.
        """
        if self._extraction_result is None:
            return {
                "entities": 0,
                "relations": 0,
                "passages": 0,
            }

        return {
            "entities": len(self._extraction_result.entities),
            "relations": len(self._extraction_result.relations),
            "passages": len(self._extraction_result.documents),
        }

    def reset(self) -> None:
        """
        Reset the knowledge base, removing all data.
        """
        self._store.drop_collections()
        self._store.create_collections(drop_existing=True)
        self._extraction_result = None
        self._retriever = None

        # Reset graph builder state
        self._graph_builder = GraphBuilder(settings=self.settings)


def create_rag(
    milvus_uri: Optional[str] = None,
    milvus_token: Optional[str] = None,
    milvus_db: Optional[str] = None,
    collection_prefix: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
) -> VectorGraphRAG:
    """
    Factory function to create a VectorGraphRAG instance.

    A convenient way to create a RAG instance with common configurations.

    Args:
        milvus_uri: Milvus connection URI. Defaults to local file.
        milvus_token: Milvus authentication token (for Zilliz Cloud).
        milvus_db: Milvus database name. Defaults to None (use server default).
        collection_prefix: Prefix for collection names (e.g., graph/dataset name).
        openai_api_key: OpenAI API key. Uses environment variable if not provided.
        llm_model: LLM model name.
        embedding_provider: Embedding provider name (e.g., "openai", "huggingface",
            "ollama", "jina").
        embedding_model: Embedding model name. If omitted with no provider, uses
            "text-embedding-3-small". If omitted with a provider, uses that
            provider's default model.
        embedding_api_key: Optional API key override for embedding providers.
        embedding_base_url: Optional base URL override for embedding providers.

    Returns:
        Configured VectorGraphRAG instance.

    Example:
        >>> from vector_graph_rag import create_rag
        >>> rag = create_rag()
        >>> rag.rebuild_texts(["Your documents here..."])
    """
    effective_embedding_model = embedding_model
    if effective_embedding_model is None and embedding_provider is None:
        effective_embedding_model = "text-embedding-3-small"

    return VectorGraphRAG(
        milvus_uri=milvus_uri,
        milvus_token=milvus_token,
        milvus_db=milvus_db,
        collection_prefix=collection_prefix,
        openai_api_key=openai_api_key,
        llm_model=llm_model,
        embedding_provider=embedding_provider,
        embedding_model=effective_embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
    )
