"""
Main Vector Graph RAG class with user-friendly API.
"""

import uuid
from typing import List, Optional
from tqdm import tqdm

from vector_graph_rag.config import Settings, get_settings
from vector_graph_rag.models import Document, Triplet, QueryResult, ExtractionResult, RetrievalDetail, RerankResult
from vector_graph_rag.llm.extractor import TripletExtractor
from vector_graph_rag.llm.reranker import LLMReranker, AnswerGenerator
from vector_graph_rag.storage.embeddings import EmbeddingModel
from vector_graph_rag.storage.milvus import MilvusStore
from vector_graph_rag.graph.builder import GraphBuilder
from vector_graph_rag.graph.retriever import GraphRetriever, RetrievalResult
from vector_graph_rag.graph.knowledge_graph import SubGraph


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
        >>> # Add documents
        >>> documents = [
        ...     "Einstein was a physicist who developed relativity.",
        ...     "Relativity revolutionized our understanding of space and time.",
        ... ]
        >>> rag.add_documents(documents)
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
        >>> rag.add_documents(["Your text here..."])
        >>> answer = rag.query("Your question?").answer
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        milvus_uri: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize Vector Graph RAG.

        Args:
            settings: Full settings object (overrides other parameters).
            milvus_uri: Milvus connection URI. Defaults to local file.
            openai_api_key: OpenAI API key. Uses environment variable if not provided.
            llm_model: LLM model name. Defaults to "gpt-4o-mini".
            embedding_model: Embedding model name. Defaults to "text-embedding-3-small".

        Example:
            >>> # Use defaults (reads OPENAI_API_KEY from environment)
            >>> rag = VectorGraphRAG()
            >>>
            >>> # Custom configuration
            >>> rag = VectorGraphRAG(
            ...     milvus_uri="./my_data.db",
            ...     llm_model="gpt-4o",
            ... )
        """
        # Build settings
        if settings:
            self.settings = settings
        else:
            settings_kwargs = {}
            if milvus_uri:
                settings_kwargs["milvus_uri"] = milvus_uri
            if openai_api_key:
                settings_kwargs["openai_api_key"] = openai_api_key
            if llm_model:
                settings_kwargs["llm_model"] = llm_model
            if embedding_model:
                settings_kwargs["embedding_model"] = embedding_model

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

    def _get_passages_from_relations(
        self, relation_ids: List[str]
    ) -> tuple[List[str], List[str]]:
        """
        Get passages associated with given relations.

        Args:
            relation_ids: List of relation IDs (strings).

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

        passage_data = self._store.get_passages_by_ids(passage_ids)
        id_to_text = {p["id"]: p["text"] for p in passage_data}

        passages = [id_to_text[pid] for pid in passage_ids if pid in id_to_text]
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

        This is a convenience method that converts texts to Document objects
        and calls add_documents().

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
        documents = []
        for i, text in enumerate(texts):
            doc_id = ids[i] if ids and i < len(ids) else str(uuid.uuid4())
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            documents.append(Document(page_content=text, metadata=metadata, id=doc_id))

        return self.add_documents(
            documents, extract_triplets=extract_triplets, show_progress=show_progress
        )

    def add_documents(
        self,
        documents: List[Document],
        extract_triplets: bool = True,
        show_progress: bool = True,
    ) -> ExtractionResult:
        """
        Add Document objects to the knowledge base.

        This method:
        1. Extracts triplets from documents (if enabled)
        2. Builds the knowledge graph structure
        3. Indexes entities, relations, and passages in Milvus

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
        # Ensure all documents have IDs
        for doc in documents:
            if not doc.id:
                doc.id = str(uuid.uuid4())

        # Extract triplets if needed
        if extract_triplets:
            documents = self._triplet_extractor.extract_from_documents(
                documents, show_progress=show_progress
            )

        # Build ExtractionResult
        self._extraction_result = self._graph_builder.build_from_documents(documents)

        # Generate embeddings
        if show_progress:
            print("Generating embeddings...")

        entity_texts = self._graph_builder.get_entity_texts()
        relation_texts = self._graph_builder.get_relation_texts()
        passage_texts = self._graph_builder.get_passage_texts()

        entity_embeddings = (
            self._embedding_model.embed_batch(entity_texts, show_progress=show_progress)
            if entity_texts
            else []
        )

        relation_embeddings = (
            self._embedding_model.embed_batch(
                relation_texts, show_progress=show_progress
            )
            if relation_texts
            else []
        )

        passage_embeddings = (
            self._embedding_model.embed_batch(
                passage_texts, show_progress=show_progress
            )
            if passage_texts
            else []
        )

        # Build metadata for adjacency information
        # Entity metadata: relation_ids (directly connected relations), passage_ids
        entity_metadatas = []
        for eid in self._graph_builder.entity_ids:
            entity_metadatas.append({
                "relation_ids": self._graph_builder.entity_to_relation_ids.get(eid, []),
                "passage_ids": self._graph_builder.entity_to_passage_ids.get(eid, []),
            })

        # Relation metadata: entity_ids (head and tail), passage_ids, triplet fields
        relation_metadatas = []
        for rid in self._graph_builder.relation_ids:
            triplet = self._graph_builder.relation_id_to_triplet.get(rid)
            entity_ids = self._graph_builder.relation_to_entity_ids.get(rid, [])
            passage_ids = self._graph_builder.relation_to_passage_ids.get(rid, [])

            metadata = {
                "entity_ids": entity_ids,
                "passage_ids": passage_ids,
            }
            # Add structured triplet fields
            if triplet:
                metadata["subject"] = triplet.subject
                metadata["predicate"] = triplet.predicate
                metadata["object"] = triplet.object

            relation_metadatas.append(metadata)

        # Passage metadata
        passage_metadatas = []
        for pid in self._graph_builder.passage_ids:
            passage_metadatas.append({
                "entity_ids": self._graph_builder.passage_to_entity_ids.get(pid, []),
                "relation_ids": self._graph_builder.passage_to_relation_ids.get(pid, []),
            })

        # Drop and recreate collections for fresh data
        self._store.drop_collections()
        self._store.create_collections(drop_existing=True)

        # Insert into Milvus
        if show_progress:
            print("Inserting into Milvus...")

        # Use private methods for entities and relations
        self._store._insert_entities(
            entity_texts,
            ids=self._graph_builder.entity_ids,
            embeddings=entity_embeddings,
            metadatas=entity_metadatas,
            show_progress=show_progress,
        )
        self._store._insert_relations(
            relation_texts,
            ids=self._graph_builder.relation_ids,
            embeddings=relation_embeddings,
            metadatas=relation_metadatas,
            show_progress=show_progress,
        )
        self._store.insert_passages(
            passage_texts,
            ids=self._graph_builder.passage_ids,
            embeddings=passage_embeddings,
            metadatas=passage_metadatas,
            show_progress=show_progress,
        )

        # Reset retriever to pick up new knowledge graph
        self._retriever = None

        return self._extraction_result

    def add_documents_with_triplets(
        self,
        documents: List[dict],
        show_progress: bool = True,
    ) -> ExtractionResult:
        """
        Add documents with pre-extracted triplets.

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
        docs = []
        for doc_data in documents:
            passage = doc_data["passage"]
            doc_id = doc_data.get("id") or str(uuid.uuid4())
            # Store triplets in metadata as list of [subject, predicate, object]
            triplets = doc_data.get("triplets", [])
            docs.append(Document(
                page_content=passage,
                metadata={"triplets": triplets},
                id=doc_id,
            ))

        return self.add_documents(
            docs, extract_triplets=False, show_progress=show_progress
        )

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
        passage_ids, passages = self._get_passages_from_relations(reranked_ids)
        final_passages = passages[: self.settings.final_top_k]

        # Generate answer
        answer = self._answer_generator.generate(question, final_passages)

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
        )

    def query_simple(self, question: str) -> str:
        """
        Simple query that returns just the answer.

        Args:
            question: The question to answer.

        Returns:
            The answer string.

        Example:
            >>> answer = rag.query_simple("What did Einstein develop?")
            >>> print(answer)
        """
        return self.query(question).answer

    def query_naive(self, question: str) -> QueryResult:
        """
        Query using naive RAG (direct passage retrieval).

        Useful for comparing Graph RAG performance against baseline.

        Args:
            question: The question to answer.

        Returns:
            QueryResult with answer and retrieved passages.
        """
        retriever = self._ensure_retriever()
        passages = retriever.retrieve_passages_naive(question)
        answer = self._answer_generator.generate(question, passages)

        return QueryResult(
            query=question,
            answer=answer,
            retrieved_passages=passages,
            retrieved_relations=[],
            expanded_relations=[],
            reranked_relations=[],
        )

    def retrieve(
        self,
        question: str,
        use_reranking: bool = True,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        """
        Retrieve passages using Graph RAG without generating an answer.

        This method performs all retrieval steps but skips LLM answer generation,
        useful for evaluation or when only retrieved documents are needed.

        Args:
            question: The question to answer.
            use_reranking: Whether to use LLM reranking.
            top_k: Number of passages to retrieve. Uses settings.final_top_k if not provided.

        Returns:
            QueryResult with retrieved passages (answer will be empty).
        """
        retriever = self._ensure_retriever()
        top_k = top_k or self.settings.final_top_k

        # Retrieve
        retrieval_result = retriever.retrieve(question)

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
        passage_ids, passages = self._get_passages_from_relations(reranked_ids)

        if len(passages) < top_k:
            additional_passages = retriever.retrieve_passages_naive(
                question, top_k=top_k
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
    ) -> QueryResult:
        """
        Retrieve passages using naive RAG without generating an answer.

        This method performs direct passage retrieval but skips LLM answer generation,
        useful for evaluation or when only retrieved documents are needed.

        Args:
            question: The question to answer.
            top_k: Number of passages to retrieve. Uses settings.final_top_k if not provided.

        Returns:
            QueryResult with retrieved passages (answer will be empty).
        """
        retriever = self._ensure_retriever()
        top_k = top_k or self.settings.final_top_k
        passages = retriever.retrieve_passages_naive(question, top_k=top_k)

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
    openai_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
) -> VectorGraphRAG:
    """
    Factory function to create a VectorGraphRAG instance.

    A convenient way to create a RAG instance with common configurations.

    Args:
        milvus_uri: Milvus connection URI. Defaults to local file.
        openai_api_key: OpenAI API key. Uses environment variable if not provided.
        llm_model: LLM model name.
        embedding_model: Embedding model name.

    Returns:
        Configured VectorGraphRAG instance.

    Example:
        >>> from vector_graph_rag import create_rag
        >>> rag = create_rag()
        >>> rag.add_documents(["Your documents here..."])
    """
    return VectorGraphRAG(
        milvus_uri=milvus_uri,
        openai_api_key=openai_api_key,
        llm_model=llm_model,
        embedding_model=embedding_model,
    )
