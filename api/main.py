"""
FastAPI application for Vector Graph RAG.

Provides REST API for querying and visualizing the knowledge graph.
"""

import os
import sys
from typing import Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vector_graph_rag import VectorGraphRAG, Settings
from vector_graph_rag.storage.milvus import MilvusStore
from api.schemas import (
    QueryRequest,
    QueryResponse,
    SubGraphSchema,
    EntitySchema,
    RelationSchema,
    PassageSchema,
    GraphStatsResponse,
    HealthResponse,
    GraphInfo,
    ListGraphsResponse,
    MilvusConfig,
    RetrievalDetailSchema,
    RerankResultSchema,
)


# Global RAG instances cache (keyed by graph_name)
rag_instances: Dict[str, VectorGraphRAG] = {}

# Default Milvus settings from environment
DEFAULT_MILVUS_URI = os.getenv("MILVUS_URI", "./vector_graph_rag.db")
DEFAULT_MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
DEFAULT_MILVUS_DB = os.getenv("MILVUS_DB")


def get_rag(graph_name: Optional[str] = None) -> VectorGraphRAG:
    """
    Get or create a RAG instance for the specified graph.

    Args:
        graph_name: Graph/dataset name (collection prefix). If None, uses default settings.

    Returns:
        VectorGraphRAG instance configured for the specified graph.
    """
    cache_key = graph_name or "__default__"

    if cache_key not in rag_instances:
        # Configure from environment variables or defaults
        settings_kwargs = {
            "milvus_uri": DEFAULT_MILVUS_URI,
        }
        if DEFAULT_MILVUS_TOKEN:
            settings_kwargs["milvus_token"] = DEFAULT_MILVUS_TOKEN
        if DEFAULT_MILVUS_DB:
            settings_kwargs["milvus_db"] = DEFAULT_MILVUS_DB

        # Set collection prefix if graph_name is specified
        if graph_name:
            settings_kwargs["collection_prefix"] = graph_name

        settings = Settings(**settings_kwargs)
        rag_instances[cache_key] = VectorGraphRAG(settings=settings)

    return rag_instances[cache_key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup: initialize RAG
    print("Initializing Vector Graph RAG...")
    get_rag()
    print("RAG initialized.")
    yield
    # Shutdown: cleanup if needed
    print("Shutting down...")


app = FastAPI(
    title="Vector Graph RAG API",
    description="API for querying and visualizing knowledge graph-based RAG",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", message="Vector Graph RAG API is running")


@app.get("/graphs", response_model=ListGraphsResponse)
async def list_graphs():
    """
    List all available graphs (datasets) in Milvus.

    Discovers graphs by finding collections that match the pattern:
    - {prefix}_vgrag_entities
    - {prefix}_vgrag_relations
    - {prefix}_vgrag_passages

    The graph name is extracted from the prefix.

    Note: Milvus configuration is read-only, set via environment variables at startup.
    """
    try:
        graphs_data = MilvusStore.list_graphs(
            milvus_uri=DEFAULT_MILVUS_URI,
            milvus_token=DEFAULT_MILVUS_TOKEN,
            milvus_db=DEFAULT_MILVUS_DB,
        )

        graphs = [GraphInfo(**g) for g in graphs_data]

        # Return config info (read-only, don't expose token value)
        milvus_config = MilvusConfig(
            uri=DEFAULT_MILVUS_URI,
            database=DEFAULT_MILVUS_DB,
            has_token=bool(DEFAULT_MILVUS_TOKEN),
        )

        return ListGraphsResponse(
            graphs=graphs,
            milvus_config=milvus_config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=GraphStatsResponse)
async def get_stats(graph_name: Optional[str] = None):
    """Get knowledge graph statistics for a specific graph."""
    rag = get_rag(graph_name)
    stats = rag.get_stats()
    return GraphStatsResponse(**stats)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the knowledge graph and return subgraph with expansion history.

    The response includes:
    - answer: Generated answer
    - subgraph: Full subgraph data including expansion_history for visualization
    - retrieved_passages: Passages used for answer generation

    Pass graph_name to query a specific graph/dataset.
    """
    try:
        rag = get_rag(request.graph_name)

        # Perform query with custom parameters
        result = rag.query(
            question=request.question,
            entity_top_k=request.entity_top_k,
            relation_top_k=request.relation_top_k,
            entity_similarity_threshold=request.entity_similarity_threshold,
            relation_similarity_threshold=request.relation_similarity_threshold,
            expansion_degree=request.expansion_degree,
        )

        # Build subgraph response
        subgraph = result.subgraph
        subgraph_data = SubGraphSchema(
            entity_ids=list(subgraph.entity_ids),
            relation_ids=list(subgraph.relation_ids),
            passage_ids=list(subgraph.passage_ids),
            entities=[
                EntitySchema(
                    id=e.id,
                    name=e.name,
                    relation_ids=e.relation_ids,
                    passage_ids=e.passage_ids,
                )
                for e in subgraph.entities
            ],
            relations=[
                RelationSchema(
                    id=r.id,
                    text=r.text,
                    subject=r.subject,
                    predicate=r.predicate,
                    object=r.object,
                    entity_ids=r.entity_ids,
                    passage_ids=r.passage_ids,
                )
                for r in subgraph.relations
            ],
            passages=[PassageSchema(id=p.id, text=p.text) for p in subgraph.passages],
            expansion_history=subgraph.expansion_history,
        )

        # Build retrieval detail if available
        retrieval_detail_data = None
        if result.retrieval_detail:
            retrieval_detail_data = RetrievalDetailSchema(
                entity_ids=result.retrieval_detail.entity_ids,
                entity_texts=result.retrieval_detail.entity_texts,
                entity_scores=result.retrieval_detail.entity_scores,
                relation_ids=result.retrieval_detail.relation_ids,
                relation_texts=result.retrieval_detail.relation_texts,
                relation_scores=result.retrieval_detail.relation_scores,
            )

        # Build rerank result if available
        rerank_result_data = None
        if result.rerank_result:
            rerank_result_data = RerankResultSchema(
                selected_relation_ids=result.rerank_result.selected_relation_ids,
                selected_relation_texts=result.rerank_result.selected_relation_texts,
            )

        return QueryResponse(
            question=request.question,
            answer=result.answer,
            query_entities=result.query_entities,
            subgraph=subgraph_data,
            retrieved_passages=result.passages,
            stats=subgraph.stats(),
            retrieval_detail=retrieval_detail_data,
            rerank_result=rerank_result_data,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_documents")
async def add_documents(documents: list[str]):
    """Add documents to the knowledge graph."""
    try:
        rag = get_rag()
        rag.add_documents(documents)
        return {"status": "ok", "message": f"Added {len(documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
