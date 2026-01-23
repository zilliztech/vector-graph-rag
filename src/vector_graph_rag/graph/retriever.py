"""
Graph-based retriever using vector similarity search.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from vector_graph_rag.config import Settings, get_settings
from vector_graph_rag.storage.embeddings import EmbeddingModel
from vector_graph_rag.storage.milvus import MilvusStore
from vector_graph_rag.graph.builder import GraphBuilder
from vector_graph_rag.graph.knowledge_graph import SubGraph
from vector_graph_rag.llm.extractor import EntityExtractor


@dataclass
class RetrievalResult:
    """Result of graph-based retrieval."""

    # Entity retrieval
    entity_ids: List[int]
    entity_texts: List[str]
    entity_scores: List[float]

    # Relation retrieval
    relation_ids: List[int]
    relation_texts: List[str]
    relation_scores: List[float]

    # Expanded subgraph
    subgraph: Optional[SubGraph] = None

    # Expanded relations (for backward compatibility)
    expanded_relation_ids: List[int] = field(default_factory=list)
    expanded_relation_texts: List[str] = field(default_factory=list)

    # Query info
    query: str = ""
    query_entities: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Populate expanded relation fields from subgraph if available."""
        if self.subgraph and not self.expanded_relation_ids:
            self.expanded_relation_ids = list(self.subgraph.relation_ids)
            self.expanded_relation_texts = self.subgraph.relation_texts


class GraphRetriever:
    """
    Graph-based retriever using multi-way vector search.

    Performs both entity-based and relation-based retrieval,
    then expands the subgraph using lazy loading from Milvus.

    The subgraph expansion fetches data on-demand, avoiding
    loading the entire graph into memory.

    Example:
        >>> retriever = GraphRetriever(store)
        >>> result = retriever.retrieve("Who was Einstein's teacher?")
        >>> print(result.subgraph.expansion_history)  # Debug expansion steps
        >>> print(result.expanded_relation_texts)
    """

    def __init__(
        self,
        store: MilvusStore,
        graph_builder: Optional[GraphBuilder] = None,
        settings: Optional[Settings] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        entity_extractor: Optional[EntityExtractor] = None,
    ):
        """
        Initialize the graph retriever.

        Args:
            store: Milvus store containing indexed data.
            graph_builder: Graph builder (kept for backward compatibility).
            settings: Configuration settings.
            embedding_model: Embedding model for queries.
            entity_extractor: Entity extractor for query entities.
        """
        self.settings = settings or get_settings()
        self.store = store
        self.graph_builder = graph_builder

        self.embedding_model = embedding_model or EmbeddingModel(settings=self.settings)
        self.entity_extractor = entity_extractor or EntityExtractor(
            settings=self.settings
        )

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract named entities from the query."""
        return self.entity_extractor.extract(query)

    def _retrieve_entities(
        self,
        query_entities: List[str],
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> Tuple[List[int], List[str], List[float]]:
        """
        Retrieve similar entities based on query entities.

        Args:
            query_entities: Entities extracted from the query.
            top_k: Number of entities to retrieve per query entity.
            similarity_threshold: Minimum similarity score to keep (default from settings).

        Returns:
            Tuple of (entity_ids, entity_texts, scores).
        """
        if not query_entities:
            return [], [], []

        top_k = top_k or self.settings.entity_top_k
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.settings.entity_similarity_threshold
        )

        # Embed query entities
        query_embeddings = self.embedding_model.embed_batch(query_entities)

        # Search for similar entities
        search_results = self.store.search_entities(query_embeddings, top_k=top_k)

        # Aggregate results with threshold filtering
        entity_ids = []
        entity_texts = []
        scores = []
        seen_ids = set()

        for result_list in search_results:
            for result in result_list:
                score = result["distance"]
                # Filter by similarity threshold
                if score <= threshold:
                    continue

                entity_id = result["entity"]["id"]
                if entity_id not in seen_ids:
                    seen_ids.add(entity_id)
                    entity_ids.append(entity_id)
                    entity_texts.append(result["entity"]["text"])
                    scores.append(score)

        return entity_ids, entity_texts, scores

    def _retrieve_relations(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> Tuple[List[int], List[str], List[float]]:
        """
        Retrieve similar relations based on query.

        Args:
            query: The query text.
            top_k: Number of relations to retrieve.
            similarity_threshold: Minimum similarity score to keep (default from settings).

        Returns:
            Tuple of (relation_ids, relation_texts, scores).
        """
        top_k = top_k or self.settings.relation_top_k
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.settings.relation_similarity_threshold
        )

        # Embed query
        query_embedding = self.embedding_model.embed(query)

        # Search for similar relations
        results = self.store.search_relations(query_embedding, top_k=top_k)

        # Filter by similarity threshold
        relation_ids = []
        relation_texts = []
        scores = []

        for r in results:
            score = r["distance"]
            if score > threshold:
                relation_ids.append(r["entity"]["id"])
                relation_texts.append(r["entity"]["text"])
                scores.append(score)

        return relation_ids, relation_texts, scores

    def _expand_subgraph(
        self,
        entity_ids: List[int],
        relation_ids: List[int],
        degree: Optional[int] = None,
    ) -> SubGraph:
        """
        Expand subgraph from retrieved entities and relations.

        Creates a SubGraph and expands it by fetching neighbor data
        on-demand from Milvus.

        Args:
            entity_ids: Retrieved entity IDs.
            relation_ids: Retrieved relation IDs.
            degree: Expansion degree.

        Returns:
            Expanded SubGraph with fetched data.
        """
        degree = degree or self.settings.expansion_degree

        # Create subgraph with initial nodes
        subgraph = SubGraph(self.store)
        subgraph.add_entities(entity_ids)
        subgraph.add_relations(relation_ids)

        # Expand by given degree
        subgraph.expand(degree=degree)

        return subgraph

    def retrieve(
        self,
        query: str,
        entity_top_k: Optional[int] = None,
        relation_top_k: Optional[int] = None,
        entity_similarity_threshold: Optional[float] = None,
        relation_similarity_threshold: Optional[float] = None,
        expansion_degree: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Perform graph-based retrieval for a query.

        This method:
        1. Extracts entities from the query
        2. Retrieves similar entities via vector search (with threshold filtering)
        3. Retrieves similar relations via vector search (with threshold filtering)
        4. Expands the subgraph (lazy loading from Milvus)

        Args:
            query: The query text.
            entity_top_k: Override entity retrieval top_k.
            relation_top_k: Override relation retrieval top_k.
            entity_similarity_threshold: Override entity similarity threshold.
            relation_similarity_threshold: Override relation similarity threshold.
            expansion_degree: Override expansion degree.

        Returns:
            RetrievalResult with all retrieval information, including
            the SubGraph for debugging/visualization.
        """
        # Extract query entities
        query_entities = self._extract_query_entities(query)

        # Retrieve entities (with threshold filtering)
        entity_ids, entity_texts, entity_scores = self._retrieve_entities(
            query_entities,
            top_k=entity_top_k,
            similarity_threshold=entity_similarity_threshold,
        )

        # Retrieve relations (with threshold filtering)
        relation_ids, relation_texts, relation_scores = self._retrieve_relations(
            query,
            top_k=relation_top_k,
            similarity_threshold=relation_similarity_threshold,
        )

        # Expand subgraph
        subgraph = self._expand_subgraph(
            entity_ids, relation_ids, degree=expansion_degree
        )

        return RetrievalResult(
            entity_ids=entity_ids,
            entity_texts=entity_texts,
            entity_scores=entity_scores,
            relation_ids=relation_ids,
            relation_texts=relation_texts,
            relation_scores=relation_scores,
            subgraph=subgraph,
            expanded_relation_ids=list(subgraph.relation_ids),
            expanded_relation_texts=subgraph.relation_texts,
            query=query,
            query_entities=query_entities,
        )

    def retrieve_passages_naive(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """
        Naive RAG passage retrieval for comparison.

        Args:
            query: The query text.
            top_k: Number of passages to return.

        Returns:
            List of passage texts.
        """
        top_k = top_k or self.settings.final_top_k
        query_embedding = self.embedding_model.embed(query)
        results = self.store.search_passages(query_embedding, top_k=top_k)
        return [r["entity"]["text"] for r in results]
