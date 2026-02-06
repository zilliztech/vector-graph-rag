"""
FastAPI application for Vector Graph RAG.

Provides RESTful API endpoints for:
- Health check
- Listing available graphs
- Adding and querying documents
- Document CRUD operations
- Graph exploration (stats, neighbors)
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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


class ImportRequest(BaseModel):
    """Request to import documents from various sources."""
    sources: List[str] = Field(
        ...,
        description="List of file paths or URLs to import",
        examples=["/path/to/doc.pdf", "https://example.com/article"]
    )
    chunk_documents: bool = Field(True, description="Whether to chunk large documents")
    chunk_size: int = Field(1000, description="Max characters per chunk")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    extract_triplets: bool = Field(True, description="Extract triplets using LLM")
    graph_name: Optional[str] = Field(None, description="Target graph name")


class ImportResponse(BaseModel):
    """Response from document import."""
    success: bool
    num_sources: int = Field(..., description="Number of input sources")
    num_documents: int = Field(..., description="Number of documents after conversion")
    num_chunks: int = Field(..., description="Number of chunks after splitting")
    num_entities: int = Field(..., description="Number of entities extracted")
    num_relations: int = Field(..., description="Number of relations extracted")
    errors: List[str] = Field(default_factory=list, description="Any errors during import")


class QueryRequest(BaseModel):
    """Request to query the knowledge base."""
    question: str = Field(..., description="Question to answer")
    use_reranking: bool = Field(default=True, description="Whether to use LLM reranking")
    entity_top_k: Optional[int] = Field(default=None, description="Number of entities to retrieve")
    relation_top_k: Optional[int] = Field(default=None, description="Number of relations to retrieve")
    expansion_degree: Optional[int] = Field(default=None, description="Subgraph expansion degree")


class EntitySchema(BaseModel):
    """Schema for an entity in the subgraph."""
    id: str = Field(..., description="Entity ID")
    name: str = Field(..., description="Entity name")
    relation_ids: List[str] = Field(default_factory=list, description="Connected relation IDs")
    passage_ids: List[str] = Field(default_factory=list, description="Source passage IDs")


class RelationSchema(BaseModel):
    """Schema for a relation in the subgraph."""
    id: str = Field(..., description="Relation ID")
    text: str = Field(..., description="Full relation text")
    subject: str = Field(..., description="Subject entity name")
    predicate: str = Field(..., description="Predicate/relationship")
    object: str = Field(..., description="Object entity name")
    entity_ids: List[str] = Field(default_factory=list, description="Connected entity IDs [subject_id, object_id]")
    passage_ids: List[str] = Field(default_factory=list, description="Source passage IDs")


class PassageSchema(BaseModel):
    """Schema for a passage in the subgraph."""
    id: str = Field(..., description="Passage ID")
    text: str = Field(..., description="Passage text")


class ExpansionStepSchema(BaseModel):
    """Schema for an expansion step in the history."""
    step: int = Field(..., description="Step number")
    operation: str = Field(..., description="Operation type")
    description: Optional[str] = Field(default=None, description="Operation description")
    added_entity_ids: List[str] = Field(default_factory=list, description="Entity IDs added in this step")
    added_relation_ids: List[str] = Field(default_factory=list, description="Relation IDs added in this step")
    total_entities: int = Field(default=0, description="Total entities after this step")
    total_relations: int = Field(default=0, description="Total relations after this step")


class SubGraphSchema(BaseModel):
    """Schema for the expanded subgraph."""
    entity_ids: List[str] = Field(default_factory=list, description="All entity IDs in subgraph")
    relation_ids: List[str] = Field(default_factory=list, description="All relation IDs in subgraph")
    passage_ids: List[str] = Field(default_factory=list, description="All passage IDs in subgraph")
    entities: List[EntitySchema] = Field(default_factory=list, description="Entity details")
    relations: List[RelationSchema] = Field(default_factory=list, description="Relation details")
    passages: List[PassageSchema] = Field(default_factory=list, description="Passage details")
    expansion_history: List[ExpansionStepSchema] = Field(default_factory=list, description="Expansion history")


class RetrievalDetailSchema(BaseModel):
    """Schema for retrieval details."""
    entity_ids: List[str] = Field(default_factory=list, description="Retrieved entity IDs")
    entity_texts: List[str] = Field(default_factory=list, description="Retrieved entity names")
    entity_scores: List[float] = Field(default_factory=list, description="Entity similarity scores")
    relation_ids: List[str] = Field(default_factory=list, description="Retrieved relation IDs")
    relation_texts: List[str] = Field(default_factory=list, description="Retrieved relation texts")
    relation_scores: List[float] = Field(default_factory=list, description="Relation similarity scores")


class RerankResultSchema(BaseModel):
    """Schema for rerank results."""
    selected_relation_ids: List[str] = Field(default_factory=list, description="Selected relation IDs")
    selected_relation_texts: List[str] = Field(default_factory=list, description="Selected relation texts")


class EvictionResultSchema(BaseModel):
    """Schema for eviction results (when relations exceed threshold)."""
    occurred: bool = Field(default=False, description="Whether eviction occurred")
    before_count: int = Field(default=0, description="Number of relations before eviction")
    after_count: int = Field(default=0, description="Number of relations after eviction")


class QueryResponse(BaseModel):
    """Response for a query."""
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    query_entities: List[str] = Field(default_factory=list, description="Entities extracted from query")
    subgraph: Optional[SubGraphSchema] = Field(default=None, description="Expanded subgraph for visualization")
    retrieved_passages: List[str] = Field(default_factory=list, description="Retrieved passages")
    stats: Dict[str, Any] = Field(default_factory=dict, description="Query statistics")
    retrieval_detail: Optional[RetrievalDetailSchema] = Field(default=None, description="Initial retrieval details")
    rerank_result: Optional[RerankResultSchema] = Field(default=None, description="LLM rerank results")
    eviction_result: Optional[EvictionResultSchema] = Field(default=None, description="Eviction results if relations exceeded threshold")


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


class GraphStatsResponse(BaseModel):
    """Response for graph statistics."""
    graph_name: str = Field(..., description="Graph name")
    entity_count: int = Field(default=0, description="Number of entities")
    relation_count: int = Field(default=0, description="Number of relations")
    passage_count: int = Field(default=0, description="Number of passages")


class NeighborResponse(BaseModel):
    """Response for entity neighbors."""
    entity_id: str = Field(..., description="Central entity ID")
    neighbors: List[EntitySchema] = Field(default_factory=list, description="Neighbor entities")
    relations: List[RelationSchema] = Field(default_factory=list, description="Connecting relations")


class SettingsResponse(BaseModel):
    """Response for system settings."""
    llm_model: str = Field(..., description="LLM model name")
    embedding_model: str = Field(..., description="Embedding model name")
    embedding_dimension: int = Field(..., description="Embedding dimension")
    milvus_uri: str = Field(..., description="Milvus connection URI")
    milvus_db: Optional[str] = Field(None, description="Milvus database name")
    openai_api_key_set: bool = Field(..., description="Whether OpenAI API key is configured")
    openai_base_url: Optional[str] = Field(None, description="Custom OpenAI base URL")


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

    # Add CORS middleware for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify your frontend URL
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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

    @app.get("/settings", response_model=SettingsResponse, tags=["System"])
    async def get_system_settings():
        """
        Get current system settings.

        Returns configuration information including models, Milvus connection, etc.
        """
        settings = app.state.settings
        return SettingsResponse(
            llm_model=settings.llm_model,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
            milvus_uri=settings.milvus_uri,
            milvus_db=settings.milvus_db,
            openai_api_key_set=bool(settings.openai_api_key),
            openai_base_url=settings.openai_base_url,
        )

    @app.delete("/graph/{graph_name}", response_model=DeleteResponse, tags=["System"])
    async def delete_graph(graph_name: str):
        """
        Delete a graph (knowledge base) and all its collections.

        This will permanently delete all entities, relations, and passages
        associated with the graph.
        """
        try:
            # Delete all collections for this graph
            deleted = MilvusStore.delete_graph(
                graph_name=graph_name,
                milvus_uri=app.state.settings.milvus_uri,
                milvus_token=app.state.settings.milvus_token,
                milvus_db=app.state.settings.milvus_db,
            )

            # Clear cached instances for this graph
            if graph_name in app.state.rag_instances:
                del app.state.rag_instances[graph_name]
            if graph_name in app.state.graph_instances:
                del app.state.graph_instances[graph_name]

            if deleted:
                return DeleteResponse(
                    success=True,
                    message=f"Successfully deleted graph '{graph_name}' and all its collections"
                )
            else:
                return DeleteResponse(
                    success=False,
                    message=f"Graph '{graph_name}' not found or already deleted"
                )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete graph: {str(e)}"
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
            result = rag.add_texts(
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

    @app.post("/import", response_model=ImportResponse, tags=["Documents"])
    async def import_documents(request: ImportRequest):
        """
        Import text documents from files or URLs.

        Supported formats:
        - Documents: PDF, DOCX
        - Web: URLs (webpage content)
        - Text: TXT, MD, HTML
        """
        from vector_graph_rag.loaders import DocumentImporter

        # Initialize importer
        importer = DocumentImporter(
            chunk_documents=request.chunk_documents,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        # Load and convert documents
        load_result = importer.import_sources(sources=request.sources)

        if not load_result.documents:
            return ImportResponse(
                success=False,
                num_sources=len(request.sources),
                num_documents=0,
                num_chunks=0,
                num_entities=0,
                num_relations=0,
                errors=load_result.errors or ["No documents loaded"],
            )

        # Get or create RAG instance
        rag = get_rag(request.graph_name)

        # Add documents to graph
        result = rag.add_documents(
            documents=load_result.documents,
            extract_triplets=request.extract_triplets,
            show_progress=False,
        )

        return ImportResponse(
            success=True,
            num_sources=len(request.sources),
            num_documents=len(load_result.documents),
            num_chunks=len(load_result.documents),  # After chunking
            num_entities=len(result.entities),
            num_relations=len(result.relations),
            errors=load_result.errors,
        )

    @app.post("/upload", response_model=ImportResponse, tags=["Documents"])
    async def upload_files(
        files: List[UploadFile] = File(...),
        chunk_documents: bool = Form(True),
        chunk_size: int = Form(1000),
        chunk_overlap: int = Form(200),
        extract_triplets: bool = Form(True),
        graph_name: Optional[str] = Form(None),
    ):
        """
        Upload and import text documents directly.

        Accepts multipart form upload of files.
        Supported formats: PDF, DOCX, TXT, MD, HTML
        """
        from vector_graph_rag.loaders import DocumentImporter

        temp_paths = []
        try:
            # Save uploaded files to temp directory
            for file in files:
                suffix = Path(file.filename).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    shutil.copyfileobj(file.file, tmp)
                    temp_paths.append(tmp.name)

            # Import using the standard pipeline
            importer = DocumentImporter(
                chunk_documents=chunk_documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            load_result = importer.import_sources(sources=temp_paths)

            if not load_result.documents:
                return ImportResponse(
                    success=False,
                    num_sources=len(files),
                    num_documents=0,
                    num_chunks=0,
                    num_entities=0,
                    num_relations=0,
                    errors=load_result.errors or ["No documents loaded"],
                )

            # Get or create RAG instance
            rag = get_rag(graph_name)

            # Add documents to graph
            result = rag.add_documents(
                documents=load_result.documents,
                extract_triplets=extract_triplets,
                show_progress=False,
            )

            return ImportResponse(
                success=True,
                num_sources=len(files),
                num_documents=len(load_result.documents),
                num_chunks=len(load_result.documents),
                num_entities=len(result.entities),
                num_relations=len(result.relations),
                errors=load_result.errors,
            )

        finally:
            # Cleanup temp files
            for path in temp_paths:
                try:
                    Path(path).unlink()
                except Exception:
                    pass

    @app.post("/query", response_model=QueryResponse, tags=["Query"])
    async def query(
        request: QueryRequest,
        graph_name: Optional[str] = Query(default=None, description="Graph name to use"),
    ):
        """
        Query the knowledge base using Graph RAG.

        Performs entity extraction, multi-way retrieval, subgraph expansion,
        optional reranking, and answer generation.

        Returns detailed information including the expanded subgraph for visualization.
        """
        rag = get_rag(graph_name)

        result = rag.query(
            question=request.question,
            use_reranking=request.use_reranking,
            entity_top_k=request.entity_top_k,
            relation_top_k=request.relation_top_k,
            expansion_degree=request.expansion_degree,
        )

        # Convert subgraph to schema if available
        subgraph_schema = None
        if result.subgraph:
            sg = result.subgraph
            subgraph_schema = SubGraphSchema(
                entity_ids=list(sg.entity_ids) if hasattr(sg, 'entity_ids') else [],
                relation_ids=list(sg.relation_ids) if hasattr(sg, 'relation_ids') else [],
                passage_ids=list(sg.passage_ids) if hasattr(sg, 'passage_ids') else [],
                entities=[
                    EntitySchema(
                        id=e.id,
                        name=e.name,
                        relation_ids=e.relation_ids,
                        passage_ids=e.passage_ids,
                    )
                    for e in (sg.entities if hasattr(sg, 'entities') else [])
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
                    for r in (sg.relations if hasattr(sg, 'relations') else [])
                ],
                passages=[
                    PassageSchema(id=p.id, text=p.text)
                    for p in (sg.passages if hasattr(sg, 'passages') else [])
                ],
                expansion_history=[
                    ExpansionStepSchema(
                        step=h.get('step', i),
                        operation=h.get('operation', 'unknown'),
                        description=h.get('description'),
                        added_entity_ids=h.get('added_entity_ids', h.get('new_entity_ids', [])),
                        added_relation_ids=h.get('added_relation_ids', h.get('new_relation_ids', [])),
                        total_entities=h.get('total_entities', 0),
                        total_relations=h.get('total_relations', 0),
                    )
                    for i, h in enumerate(sg.expansion_history if hasattr(sg, 'expansion_history') else [])
                ],
            )

        # Convert retrieval_detail to schema
        retrieval_detail_schema = None
        if result.retrieval_detail:
            rd = result.retrieval_detail
            retrieval_detail_schema = RetrievalDetailSchema(
                entity_ids=rd.entity_ids,
                entity_texts=rd.entity_texts,
                entity_scores=rd.entity_scores,
                relation_ids=rd.relation_ids,
                relation_texts=rd.relation_texts,
                relation_scores=rd.relation_scores,
            )

        # Convert rerank_result to schema
        rerank_result_schema = None
        if result.rerank_result:
            rr = result.rerank_result
            rerank_result_schema = RerankResultSchema(
                selected_relation_ids=rr.selected_relation_ids,
                selected_relation_texts=rr.selected_relation_texts,
            )

        # Convert eviction_result to schema
        eviction_result_schema = None
        if result.eviction_result:
            er = result.eviction_result
            eviction_result_schema = EvictionResultSchema(
                occurred=er.occurred,
                before_count=er.before_count,
                after_count=er.after_count,
            )

        return QueryResponse(
            question=result.query,
            answer=result.answer,
            query_entities=result.query_entities,
            subgraph=subgraph_schema,
            retrieved_passages=result.retrieved_passages,
            stats={
                "retrieved_relations": len(result.retrieved_relations),
                "expanded_relations": len(result.expanded_relations),
                "reranked_relations": len(result.reranked_relations),
                "passages_used": len(result.passages),
            },
            retrieval_detail=retrieval_detail_schema,
            rerank_result=rerank_result_schema,
            eviction_result=eviction_result_schema,
        )

    # ==================== Graph Exploration ====================

    @app.get("/graph/{graph_name}/stats", response_model=GraphStatsResponse, tags=["Graph"])
    async def get_graph_stats(graph_name: str):
        """
        Get statistics for a graph.

        Returns counts of entities, relations, and passages.
        """
        graph = get_graph(graph_name)

        try:
            stats = graph.get_stats()
            return GraphStatsResponse(
                graph_name=graph_name,
                entity_count=stats.get('entity_count', 0),
                relation_count=stats.get('relation_count', 0),
                passage_count=stats.get('passage_count', 0),
            )
        except Exception as e:
            # If stats not available, return zeros
            return GraphStatsResponse(
                graph_name=graph_name,
                entity_count=0,
                relation_count=0,
                passage_count=0,
            )

    @app.get("/graph/{graph_name}/neighbors/{entity_id}", response_model=NeighborResponse, tags=["Graph"])
    async def get_entity_neighbors(
        graph_name: str,
        entity_id: str,
        limit: int = Query(default=20, description="Maximum number of neighbors to return"),
    ):
        """
        Get neighbors of an entity.

        Returns entities connected to the given entity and the relations between them.
        Used for lazy-loading graph expansion in the frontend.
        """
        graph = get_graph(graph_name)

        try:
            # Get the entity
            entity = graph.get_entity(entity_id)
            if not entity:
                raise HTTPException(status_code=404, detail=f"Entity not found: {entity_id}")

            # Get connected relations
            relations = graph.get_relations_for_entity(entity_id, limit=limit)

            # Collect neighbor entity IDs
            neighbor_ids = set()
            for rel in relations:
                for eid in rel.entity_ids:
                    if eid != entity_id:
                        neighbor_ids.add(eid)

            # Get neighbor entities
            neighbors = []
            for nid in list(neighbor_ids)[:limit]:
                neighbor = graph.get_entity(nid)
                if neighbor:
                    neighbors.append(EntitySchema(
                        id=neighbor.id,
                        name=neighbor.name,
                        relation_ids=neighbor.relation_ids,
                        passage_ids=neighbor.passage_ids,
                    ))

            # Convert relations to schema
            relation_schemas = [
                RelationSchema(
                    id=r.id,
                    text=r.text,
                    subject=r.subject,
                    predicate=r.predicate,
                    object=r.object,
                    entity_ids=r.entity_ids,
                    passage_ids=r.passage_ids,
                )
                for r in relations
            ]

            return NeighborResponse(
                entity_id=entity_id,
                neighbors=neighbors,
                relations=relation_schemas,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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

    # ==================== Static Files (Frontend) ====================

    # Serve static files if the static directory exists (for Docker deployment)
    static_dir = Path(__file__).parent.parent.parent.parent / "static"
    if not static_dir.exists():
        # Also check relative to current working directory
        static_dir = Path("static")

    if static_dir.exists():
        # Mount static assets (js, css, etc.)
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        # Serve index.html for SPA routing
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            """Serve the SPA for any unmatched routes."""
            # Don't intercept API routes
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")

            file_path = static_dir / full_path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)

            # Return index.html for SPA routing
            index_path = static_dir / "index.html"
            if index_path.exists():
                return FileResponse(index_path)

            raise HTTPException(status_code=404, detail="Not found")

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
