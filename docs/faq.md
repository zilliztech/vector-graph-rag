# FAQ

Frequently asked questions about Vector Graph RAG — covering when to use it, configuration, performance, and deployment.

---

??? note "When should I use Vector Graph RAG vs regular RAG?"
    Use Vector Graph RAG when your questions require connecting facts across multiple documents or passages — for example, multi-hop reasoning like "Who collaborated with X at organization Y?" Regular (naive) RAG retrieves the top-k most similar chunks independently, which works well when a single passage contains the full answer. Vector Graph RAG shines in knowledge-intensive domains such as legal, financial, medical, and academic literature where entity relationships are dense and answers depend on traversing those relationships. If your use case involves simple factual lookups or FAQ-style questions, naive RAG is simpler and faster. As a rule of thumb: if you find naive RAG missing answers because the relevant facts are scattered across documents, Vector Graph RAG is worth trying.

??? note "What vector databases are supported?"
    Vector Graph RAG uses [Milvus](https://milvus.io/) as its vector database. For local development and testing, it uses **Milvus Lite**, which runs embedded as a local file — no server setup required. For production deployments, you can point to a remote Milvus instance. Configuration is straightforward:

    ```python
    # Local file (Milvus Lite) — no setup needed
    rag = VectorGraphRAG(milvus_uri="./my_graph.db")

    # Remote Milvus server
    rag = VectorGraphRAG(milvus_uri="http://localhost:19530")

    # With database and collection isolation
    rag = VectorGraphRAG(
        milvus_uri="http://localhost:19530",
        milvus_db="production",
        collection_prefix="legal_v2",
    )
    ```

    Other vector databases (Pinecone, Weaviate, Qdrant, etc.) are not currently supported, as the system relies on Milvus-specific features for storing and querying the graph topology.

??? note "How does it handle large documents?"
    Large documents should be split into smaller chunks before indexing. The built-in `DocumentImporter` handles this automatically with configurable chunk size and overlap. Each chunk is processed independently for triplet extraction, and the resulting entities and relations are merged into a unified knowledge graph. Duplicate entities across chunks are resolved during indexing.

    ```python
    from vector_graph_rag.loaders import DocumentImporter

    importer = DocumentImporter(chunk_size=1000, chunk_overlap=200)
    result = importer.import_sources(["/path/to/large_document.pdf"])

    rag = VectorGraphRAG(milvus_uri="./my_graph.db")
    rag.add_documents(result.documents, extract_triplets=True)
    ```

    For very large corpora, consider using a remote Milvus instance rather than Milvus Lite for better performance and scalability.

??? note "Can I use local/open-source LLMs?"
    Yes. Vector Graph RAG uses the OpenAI-compatible API format, so any LLM that exposes an OpenAI-compatible endpoint will work. This includes local models served via [Ollama](https://ollama.com/), [vLLM](https://github.com/vllm-project/vllm), [LM Studio](https://lmstudio.ai/), or any other OpenAI-compatible server. You can configure the base URL and model name when initializing the RAG instance. Keep in mind that triplet extraction and reranking quality depend heavily on the LLM's capability — weaker models may produce incomplete or inaccurate triplets, which directly affects retrieval quality. For best results, use a model with strong instruction-following and reasoning abilities.

??? note "How does subgraph expansion work?"
    After vector search identifies the initial seed entities and relations, subgraph expansion traverses the knowledge graph outward from those seeds. For each seed entity, the system looks up all relations that reference that entity (via the `entity_ids` field stored in Milvus), then discovers the entities on the other end of those relations. This process can be configured to run for one or more hops. The result is a candidate subgraph that captures the local neighborhood around the query-relevant entities. This expanded set of relations is then passed to the LLM reranking step, which selects only the most relevant ones for answer generation.

??? note "What's the performance overhead compared to naive RAG?"
    Vector Graph RAG adds two main sources of overhead compared to naive RAG: (1) an additional LLM call for reranking the candidate relations, and (2) extra vector searches for subgraph expansion. In practice, the reranking call is fast because it processes a structured list of relations rather than long passages. The total latency is typically 2-4x that of naive RAG, depending on the graph size and LLM speed. The indexing pipeline is slower than naive RAG because triplet extraction requires an LLM call per chunk. For most knowledge-intensive applications, the improved answer quality justifies the overhead.

??? note "How to tune retrieval parameters?"
    The key parameters that affect retrieval quality are the number of seed results from vector search, the depth of subgraph expansion, and the number of relations selected during reranking. Start with the defaults and adjust based on your use case. If answers are missing relevant context, increase the seed count or expansion depth. If answers include too much irrelevant information, reduce them or rely more heavily on the LLM reranking step to filter.

    ```python
    rag = VectorGraphRAG(
        milvus_uri="./my_graph.db",
        llm_model="gpt-4o",
        embedding_model="text-embedding-3-large",
    )

    # Query with custom parameters
    result = rag.query(
        "What did Einstein develop?",
        top_k=10,          # Number of seed results from vector search
        top_k_rerank=5,    # Number of relations to keep after LLM reranking
    )
    ```

??? note "Can I use my own embeddings?"
    Vector Graph RAG uses OpenAI embedding models by default (`text-embedding-3-large`), but you can configure the embedding model via the `embedding_model` parameter. Any model accessible through the OpenAI-compatible API will work. If you are using a local or custom embedding endpoint, set the appropriate base URL and model name. The embedding dimensionality is detected automatically. Note that all entities, relations, and passages in a single graph must use the same embedding model — mixing models within one collection prefix is not supported.

??? note "How do I deploy to production?"
    For production deployments, use a remote Milvus instance instead of Milvus Lite for better performance, scalability, and persistence. Run the FastAPI backend behind a reverse proxy (e.g., Nginx) with appropriate rate limiting and authentication. The frontend can be built as static files and served from any CDN or static file server.

    ```bash
    # Build the frontend
    cd frontend && npm run build

    # Run the API server
    uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
    ```

    Set `VGRAG_API_PORT` and your LLM API keys as environment variables. For high availability, deploy Milvus in cluster mode and run multiple API server instances behind a load balancer. Note that Milvus does not support transactions, so avoid concurrent write operations to the same collection prefix.

??? note "What file formats can I import?"
    The `DocumentImporter` (available via the `loaders` extra) supports a variety of file formats including PDF, DOCX, plain text, Markdown, and HTML. It also supports importing directly from URLs, automatically fetching and parsing web pages. Install the loaders extra to enable this functionality:

    ```bash
    pip install "vector-graph-rag[loaders]"
    ```

    ```python
    from vector_graph_rag.loaders import DocumentImporter

    importer = DocumentImporter(chunk_size=1000, chunk_overlap=200)
    result = importer.import_sources([
        "https://en.wikipedia.org/wiki/Albert_Einstein",  # URL
        "/path/to/document.pdf",                           # PDF
        "/path/to/report.docx",                            # DOCX
        "/path/to/notes.md",                               # Markdown
        "/path/to/page.html",                              # HTML
    ])
    ```

    All imported documents are automatically chunked with configurable size and overlap before being passed to the triplet extraction pipeline.
