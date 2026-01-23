"""
Vector Graph RAG - A Graph RAG implementation using pure vector search with Milvus.

This package provides a simple yet powerful approach to implement Graph RAG
using only vector similarity search, without requiring a separate graph database.
"""

from vector_graph_rag.config import Settings
from vector_graph_rag.models import Document, Triplet, Entity, Relation
from vector_graph_rag.llm.extractor import TripletExtractor
from vector_graph_rag.storage.embeddings import EmbeddingModel
from vector_graph_rag.storage.milvus import MilvusStore
from vector_graph_rag.graph.builder import GraphBuilder
from vector_graph_rag.graph.retriever import GraphRetriever
from vector_graph_rag.graph.knowledge_graph import SubGraph
from vector_graph_rag.llm.reranker import LLMReranker
from vector_graph_rag.rag import VectorGraphRAG, create_rag
from vector_graph_rag.llm.cache import LLMCache, get_llm_cache

__version__ = "0.1.0"

__all__ = [
    "Settings",
    "Document",
    "Triplet",
    "Entity",
    "Relation",
    "TripletExtractor",
    "EmbeddingModel",
    "MilvusStore",
    "GraphBuilder",
    "GraphRetriever",
    "SubGraph",
    "LLMReranker",
    "VectorGraphRAG",
    "create_rag",
    "LLMCache",
    "get_llm_cache",
]
