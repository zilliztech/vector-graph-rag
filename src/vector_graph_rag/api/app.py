"""
FastAPI application for Vector Graph RAG.

Provides RESTful API endpoints for:
- Health check
- Listing available graphs
- Adding and querying documents
- Document CRUD operations
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from vector_graph_rag import VectorGraphRAG
from vector_graph_rag.config import Settings, get_settings
from vector_graph_rag.storage.milvus import MilvusStore
from vector_graph_rag.graph.graph import Graph
from vector_graph_rag.models import Triplet


# ==================== Request/Response Schemas ====================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="ok", description="Service status")
    version: str = Field(default="0.1.0", description="API version")


class GraphInfo(BaseModel):
    """Information about a graph."""
    name: str = Field(..., description="Graph name (collection prefix)")
    entity_collection: str = Field(..., description="Entity collection name")
    relation_collection: str = Field(..., description="Relation collection name")
    passage_collection: str = Field(..., description="Passage collection name")
    has_all_collections: bool = Field(..., description="Whether all 3 collections exist")


class ListGraphsResponse(BaseModel):
    """Response for listing graphs."""
    graphs: List[GraphInfo] = Field(default_factory=list, description="Available graphs")


class TripletInput(BaseModel):
    """Input for a triplet."""
    subject: str = Field(..., description="Subject entity")
    predicate: str = Field(..., description="Predicate/relationship")
    object: str = Field(..., description="Object entity")


class AddDocumentsRequest(BaseModel):
    """Request to add documents."""
    documents: List[str] = Field(..., description="List of document texts")
    ids: Optional[List[str]] = Field(default=None, description="Optional list of document IDs")
    extract_triplets: bool = Field(default=True, description="Whether to extract triplets using LLM")
    triplets: Optional[List[List[TripletInput]]] = Field(
        default=None,
        description="Optional pre-extracted triplets for each document"
    )


class AddDocumentsResponse(BaseModel):
    """Response after adding documents."""
    num_documents: int = Field(..., description="Number of documents added")
    num_entities: int = Field(..., description="Number of entities created")
    num_relations: int = Field(..., description="Number of relations created")
    document_ids: List[str] = Field(default_factory=list, description="IDs of added documents")


class QueryRequest(BaseModel):
    """Request to query the knowledge base."""
    question: str = Field(..., description="Question to answer")
    use_reranking: bool = Field(default=True, description="Whether to use LLM reranking")
    entity_top_k: Optional[int] = Field(default=None, description="Number of entities to retrieve")
    relation_top_k: Optional[int] = Field(default=None, description="Number of relations to retrieve")
    expansion_degree: Optional[int] = Field(default=None, description="Subgraph expansion degree")


class QueryResponse(BaseModel):
    """Response for a query."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    query_entities: List[str] = Field(default_factory=list, description="Entities extracted from query")
    retrieved_passages: List[str] = Field(default_factory=list, description="Retrieved passages")
    retrieved_relations: List[str] = Field(default_factory=list, description="Retrieved relations")
    expanded_relations: List[str] = Field(default_factory=list, description="Expanded relations")
    reranked_relations: List[str] = Field(default_factory=list, description="Reranked relations")


class DocumentResponse(BaseModel):
    """Response for a single document."""
    id: str = Field(..., description="Document ID")
    text: str = Field(..., description="Document text")
    entity_ids: List[str] = Field(default_factory=list, description="Entity IDs in this document")
    relation_ids: List[str] = Field(default_factory=list, description="Relation IDs from this document")


class ListDocumentsResponse(BaseModel):
    """Response for listing documents."""
    documents: List[DocumentResponse] = Field(default_factory=list, description="List of documents")
    total: int = Field(..., description="Total number of documents returned")


class UpdateDocumentRequest(BaseModel):
    """Request to update a document."""
    text: Optional[str] = Field(default=None, description="New document text")


class DeleteResponse(BaseModel):
    """Response for delete operations."""
    success: bool = Field(..., description="Whether deletion succeeded")
    message: str = Field(default="", description="Additional message")


# ==================== Application Factory ====================

def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        settings: Optional settings override.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="Vector Graph RAG API",
        description="Graph RAG using pure vector search with Milvus",
        version="0.1.0",
    )

    # Store settings and RAG instances
    app.state.settings = settings or get_settings()
    app.state.rag_instances: dict = {}  # graph_name -> VectorGraphRAG
    app.state.graph_instances: dict = {}  # graph_name -> Graph

    def get_rag(graph_name: Optional[str] = None) -> VectorGraphRAG:
        """Get or create RAG instance for a graph."""
        key = graph_name or "default"
        if key not in app.state.rag_instances:
            settings_copy = app.state.settings.model_copy()
            if graph_name:
                settings_copy.collection_prefix = graph_name
            app.state.rag_instances[key] = VectorGraphRAG(settings=settings_copy)
        return app.state.rag_instances[key]

    def get_graph(graph_name: Optional[str] = None) -> Graph:
        """Get or create Graph instance."""
        key = graph_name or "default"
        if key not in app.state.graph_instances:
            settings_copy = app.state.settings.model_copy()
            if graph_name:
                settings_copy.collection_prefix = graph_name
            app.state.graph_instances[key] = Graph(settings=settings_copy)
        return app.state.graph_instances[key]

    # ==================== Endpoints ====================

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Check if the service is running."""
        return HealthResponse(status="ok", version="0.1.0")

    @app.get("/graphs", response_model=ListGraphsResponse, tags=["System"])
    async def list_graphs():
        """
        List all available graphs (datasets) in Milvus.

        Discovers graphs by finding collections that match the expected pattern.
        """
        graphs = MilvusStore.list_graphs(
            milvus_uri=app.state.settings.milvus_uri,
            milvus_token=app.state.settings.milvus_token,
            milvus_db=app.state.settings.milvus_db,
        )
        return ListGraphsResponse(
            graphs=[GraphInfo(**g) for g in graphs]
        )

    @app.post("/add_documents", response_model=AddDocumentsResponse, tags=["Documents"])
    async def add_documents(
        request: AddDocumentsRequest,
        graph_name: Optional[str] = Query(default=None, description="Graph name to use"),
    ):
        """
        Add documents to the knowledge base.

        If triplets are provided, they will be used directly.
        Otherwise, triplets will be extracted using LLM if extract_triplets is True.
        """
        rag = get_rag(graph_name)

        if request.triplets:
            # Use pre-extracted triplets
            documents_with_triplets = []
            for i, text in enumerate(request.documents):
                doc_data = {
                    "passage": text,
                    "triplets": [
                        [t.subject, t.predicate, t.object]
                        for t in request.triplets[i]
                    ] if i < len(request.triplets) else [],
                }
                if request.ids and i < len(request.ids):
                    doc_data["id"] = request.ids[i]
                documents_with_triplets.append(doc_data)

            result = rag.add_documents_with_triplets(documents_with_triplets, show_progress=False)
        else:
            result = rag.add_documents(
                request.documents,
                ids=request.ids,
                extract_triplets=request.extract_triplets,
                show_progress=False,
            )

        return AddDocumentsResponse(
            num_documents=len(result.documents),
            num_entities=len(result.entities),
            num_relations=len(result.relations),
            document_ids=[doc.id for doc in result.documents if doc.id],
        )

    @app.post("/query", response_model=QueryResponse, tags=["Query"])
    async def query(
        request: QueryRequest,
        graph_name: Optional[str] = Query(default=None, description="Graph name to use"),
    ):
        """
        Query the knowledge base using Graph RAG.

        Performs entity extraction, multi-way retrieval, subgraph expansion,
        optional reranking, and answer generation.
        """
        rag = get_rag(graph_name)

        result = rag.query(
            question=request.question,
            use_reranking=request.use_reranking,
            entity_top_k=request.entity_top_k,
            relation_top_k=request.relation_top_k,
            expansion_degree=request.expansion_degree,
        )

        return QueryResponse(
            query=result.query,
            answer=result.answer,
            query_entities=result.query_entities,
            retrieved_passages=result.retrieved_passages,
            retrieved_relations=result.retrieved_relations,
            expanded_relations=result.expanded_relations,
            reranked_relations=result.reranked_relations,
        )

    # ==================== Document CRUD ====================

    @app.get("/documents/{document_id}", response_model=DocumentResponse, tags=["Documents"])
    async def get_document(
        document_id: str,
        graph_name: Optional[str] = Query(default=None, description="Graph name to use"),
    ):
        """
        Get a single document by ID.
        """
        graph = get_graph(graph_name)
        passage = graph.get_passage(document_id)

        if not passage:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

        return DocumentResponse(
            id=passage.id,
            text=passage.text,
            entity_ids=passage.entity_ids,
            relation_ids=passage.relation_ids,
        )

    @app.get("/documents", response_model=ListDocumentsResponse, tags=["Documents"])
    async def list_documents(
        graph_name: Optional[str] = Query(default=None, description="Graph name to use"),
        query: Optional[str] = Query(default=None, description="Optional search query for vector similarity"),
        top_k: int = Query(default=10, description="Number of documents to return"),
    ):
        """
        List or search documents.

        If query is provided, performs vector similarity search.
        Otherwise, this endpoint is limited as Milvus doesn't support listing all documents efficiently.
        """
        graph = get_graph(graph_name)

        if query:
            passages = graph.search_passages(query, top_k=top_k)
            docs = [
                DocumentResponse(
                    id=p.id,
                    text=p.text,
                    entity_ids=p.entity_ids,
                    relation_ids=p.relation_ids,
                )
                for p in passages
            ]
            return ListDocumentsResponse(documents=docs, total=len(docs))
        else:
            # Without a query, we can't efficiently list all documents in Milvus
            # Return empty list with a note
            return ListDocumentsResponse(
                documents=[],
                total=0,
            )

    @app.put("/documents/{document_id}", response_model=DocumentResponse, tags=["Documents"])
    async def update_document(
        document_id: str,
        request: UpdateDocumentRequest,
        graph_name: Optional[str] = Query(default=None, description="Graph name to use"),
    ):
        """
        Update a document by ID.
        """
        graph = get_graph(graph_name)

        # Check if document exists
        existing = graph.get_passage(document_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

        # Update
        success = graph.update_passage(
            document_id,
            text=request.text,
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update document")

        # Return updated document
        updated = graph.get_passage(document_id)
        return DocumentResponse(
            id=updated.id,
            text=updated.text,
            entity_ids=updated.entity_ids,
            relation_ids=updated.relation_ids,
        )

    @app.delete("/documents/{document_id}", response_model=DeleteResponse, tags=["Documents"])
    async def delete_document(
        document_id: str,
        graph_name: Optional[str] = Query(default=None, description="Graph name to use"),
    ):
        """
        Delete a document by ID.

        This performs cascade updates to remove references from related entities and relations.
        """
        graph = get_graph(graph_name)

        success = graph.delete_passage(document_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

        return DeleteResponse(
            success=True,
            message=f"Document {document_id} deleted successfully",
        )

    return app


# Create default app instance
app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the API server.

    Args:
        host: Host to bind to.
        port: Port to listen on.
        reload: Whether to enable auto-reload.
    """
    import uvicorn
    uvicorn.run("vector_graph_rag.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()
