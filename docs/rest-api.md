# REST API Reference

Vector Graph RAG provides a REST API powered by FastAPI. Start the server with:

```bash
uv run uvicorn vector_graph_rag.api.app:app --host 0.0.0.0 --port 8000
```

Interactive Swagger documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

!!! tip "Base URL"
    All endpoints below are relative to `http://localhost:8000`. Adjust the host and port to match your deployment.

---

## System

### Health Check

<span class="api-method api-get">GET</span> `/health`

Returns the service health status and version.

**Response Body**

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

**curl Example**

```bash
curl http://localhost:8000/health
```

---

### List Graphs

<span class="api-method api-get">GET</span> `/graphs`

Lists all available graphs and their collection metadata.

**Response Body**

```json
{
  "graphs": [
    {
      "name": "my_graph",
      "entity_collection": "my_graph_entities",
      "relation_collection": "my_graph_relations",
      "passage_collection": "my_graph_passages",
      "has_all_collections": true
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `name` | `string` | The graph name (derived from the collection prefix). |
| `entity_collection` | `string` | Name of the entity collection in Milvus. |
| `relation_collection` | `string` | Name of the relation collection in Milvus. |
| `passage_collection` | `string` | Name of the passage collection in Milvus. |
| `has_all_collections` | `boolean` | Whether all three required collections exist. |

**curl Example**

```bash
curl http://localhost:8000/graphs
```

---

### Get Settings

<span class="api-method api-get">GET</span> `/settings`

Returns the current server configuration. Sensitive values (such as the API key) are indicated as set or not, but never exposed.

**Response Body**

```json
{
  "llm_model": "gpt-4o",
  "embedding_model": "text-embedding-3-small",
  "embedding_dimension": 1536,
  "milvus_uri": "./vector_graph_rag.db",
  "milvus_db": "default",
  "openai_api_key_set": true,
  "openai_base_url": "https://api.openai.com/v1"
}
```

| Field | Type | Description |
|---|---|---|
| `llm_model` | `string` | The LLM model used for generation. |
| `embedding_model` | `string` | The embedding model used for vector search. |
| `embedding_dimension` | `integer` | Dimensionality of the embedding vectors. |
| `milvus_uri` | `string` | Milvus connection URI. |
| `milvus_db` | `string` | Milvus database name. |
| `openai_api_key_set` | `boolean` | Whether the OpenAI API key is configured. |
| `openai_base_url` | `string` | The base URL for the OpenAI-compatible API. |

**curl Example**

```bash
curl http://localhost:8000/settings
```

---

### Delete Graph

<span class="api-method api-delete">DELETE</span> `/graph/{graph_name}`

Deletes a graph and all of its associated collections (entities, relations, passages).

**Path Parameters**

| Parameter | Type | Description |
|---|---|---|
| `graph_name` | `string` | Name of the graph to delete. |

**Response Body**

```json
{
  "success": true,
  "message": "Graph 'my_graph' deleted successfully"
}
```

**curl Example**

```bash
curl -X DELETE http://localhost:8000/graph/my_graph
```

!!! warning
    This operation is irreversible. All entities, relations, and passages belonging to the graph will be permanently removed.

---

## Documents

### Add Documents

<span class="api-method api-post">POST</span> `/add_documents?graph_name={graph_name}`

Adds documents to a graph. Optionally extracts knowledge graph triplets from the text using the configured LLM.

**Query Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `graph_name` | `string` | — | Target graph name. |

**Request Body**

```json
{
  "documents": [
    "Albert Einstein was born in Ulm, Germany in 1879.",
    "Einstein developed the theory of relativity."
  ],
  "ids": ["doc_1", "doc_2"],
  "extract_triplets": true,
  "triplets": null
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `documents` | `string[]` | Yes | List of document texts to add. |
| `ids` | `string[]` | No | Custom IDs for the documents. Auto-generated if omitted. |
| `extract_triplets` | `boolean` | No | Whether to extract triplets via LLM. Defaults to `true`. |
| `triplets` | `string[][][]` | No | Pre-extracted triplets per document. Each triplet is `["subject", "predicate", "object"]`. |

!!! note "Providing Triplets Directly"
    If you already have structured knowledge, you can pass `triplets` directly and set `extract_triplets` to `false` to skip LLM extraction. The `triplets` array must align by index with the `documents` array — `triplets[i]` contains the list of triplets for `documents[i]`.

**Response Body**

```json
{
  "num_documents": 2,
  "num_entities": 3,
  "num_relations": 2,
  "document_ids": ["doc_1", "doc_2"]
}
```

**curl Example**

```bash
curl -X POST "http://localhost:8000/add_documents?graph_name=my_graph" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["Albert Einstein was born in Ulm, Germany in 1879."],
    "extract_triplets": true
  }'
```

---

### Import from URLs or Paths

<span class="api-method api-post">POST</span> `/import`

Imports documents from URLs or local file paths. Supports chunking for large documents.

**Request Body**

```json
{
  "sources": [
    "https://en.wikipedia.org/wiki/Albert_Einstein",
    "/path/to/local/file.txt"
  ],
  "chunk_documents": true,
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "extract_triplets": true,
  "graph_name": "my_graph"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `sources` | `string[]` | Yes | — | List of URLs or local file paths to import. |
| `chunk_documents` | `boolean` | No | `true` | Whether to split documents into chunks. |
| `chunk_size` | `integer` | No | `1000` | Maximum number of characters per chunk. |
| `chunk_overlap` | `integer` | No | `200` | Number of overlapping characters between adjacent chunks. |
| `extract_triplets` | `boolean` | No | `true` | Whether to extract triplets via LLM. |
| `graph_name` | `string` | No | `""` | Target graph name. Uses the default graph if empty. |

**Response Body**

```json
{
  "success": true,
  "num_sources": 2,
  "num_documents": 2,
  "num_chunks": 15,
  "num_entities": 42,
  "num_relations": 38,
  "errors": []
}
```

| Field | Type | Description |
|---|---|---|
| `success` | `boolean` | Whether the import completed without fatal errors. |
| `num_sources` | `integer` | Number of sources processed. |
| `num_documents` | `integer` | Number of raw documents loaded. |
| `num_chunks` | `integer` | Number of chunks produced after splitting. |
| `num_entities` | `integer` | Number of entities extracted. |
| `num_relations` | `integer` | Number of relations extracted. |
| `errors` | `string[]` | List of non-fatal errors encountered during import. |

**curl Example**

```bash
curl -X POST http://localhost:8000/import \
  -H "Content-Type: application/json" \
  -d '{
    "sources": ["https://en.wikipedia.org/wiki/Albert_Einstein"],
    "chunk_documents": true,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "extract_triplets": true,
    "graph_name": "my_graph"
  }'
```

---

### Upload Files

<span class="api-method api-post">POST</span> `/upload`

Uploads files via multipart form data. Accepts the same chunking and extraction options as the import endpoint.

**Form Fields**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `files` | `file(s)` | Yes | — | One or more files to upload. |
| `chunk_documents` | `boolean` | No | `true` | Whether to split documents into chunks. |
| `chunk_size` | `integer` | No | `1000` | Maximum number of characters per chunk. |
| `chunk_overlap` | `integer` | No | `200` | Number of overlapping characters between adjacent chunks. |
| `extract_triplets` | `boolean` | No | `true` | Whether to extract triplets via LLM. |
| `graph_name` | `string` | No | `""` | Target graph name. |

**Response Body**

Same as the [Import](#import-from-urls-or-paths) endpoint (`ImportResponse`).

```json
{
  "success": true,
  "num_sources": 1,
  "num_documents": 1,
  "num_chunks": 8,
  "num_entities": 20,
  "num_relations": 17,
  "errors": []
}
```

**curl Example**

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@document.txt" \
  -F "chunk_documents=true" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=200" \
  -F "extract_triplets=true" \
  -F "graph_name=my_graph"
```

!!! tip "Multiple Files"
    You can upload multiple files by repeating the `-F "files=@..."` flag:
    ```bash
    curl -X POST http://localhost:8000/upload \
      -F "files=@file1.txt" \
      -F "files=@file2.pdf" \
      -F "graph_name=my_graph"
    ```

---

### Search Documents

<span class="api-method api-get">GET</span> `/documents?graph_name={graph_name}&query={query}&top_k={top_k}`

Searches for documents in a graph using vector similarity.

**Query Parameters**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `graph_name` | `string` | Yes | — | Graph to search in. |
| `query` | `string` | No | `""` | Search query text. If empty, returns documents without filtering. |
| `top_k` | `integer` | No | `10` | Maximum number of documents to return. |

**Response Body**

```json
{
  "documents": [
    {
      "id": "doc_1",
      "text": "Albert Einstein was born in Ulm, Germany in 1879.",
      "entity_ids": ["entity_1", "entity_2"],
      "relation_ids": ["relation_1"]
    }
  ],
  "total": 1
}
```

**curl Example**

```bash
curl "http://localhost:8000/documents?graph_name=my_graph&query=Einstein&top_k=5"
```

---

### Get Document by ID

<span class="api-method api-get">GET</span> `/documents/{document_id}?graph_name={graph_name}`

Retrieves a single document by its ID.

**Path Parameters**

| Parameter | Type | Description |
|---|---|---|
| `document_id` | `string` | The document ID. |

**Query Parameters**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `graph_name` | `string` | Yes | Graph the document belongs to. |

**Response Body**

```json
{
  "id": "doc_1",
  "text": "Albert Einstein was born in Ulm, Germany in 1879.",
  "entity_ids": ["entity_1", "entity_2"],
  "relation_ids": ["relation_1"]
}
```

**curl Example**

```bash
curl "http://localhost:8000/documents/doc_1?graph_name=my_graph"
```

---

## Query

### Ask a Question

<span class="api-method api-post">POST</span> `/query?graph_name={graph_name}`

Performs retrieval-augmented generation over the knowledge graph. The system retrieves relevant entities, relations, and passages, optionally reranks them, and generates an answer using the configured LLM.

**Query Parameters**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `graph_name` | `string` | Yes | Graph to query against. |

**Request Body**

```json
{
  "question": "Where was Albert Einstein born?",
  "use_reranking": true,
  "entity_top_k": 20,
  "relation_top_k": 20,
  "expansion_degree": 1
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `question` | `string` | Yes | — | The natural-language question to answer. |
| `use_reranking` | `boolean` | No | `true` | Whether to apply LLM-based reranking to retrieved results. |
| `entity_top_k` | `integer` | No | `20` | Number of top entities to retrieve via vector search. |
| `relation_top_k` | `integer` | No | `20` | Number of top relations to retrieve via vector search. |
| `expansion_degree` | `integer` | No | `1` | Number of hops for graph neighborhood expansion. |

!!! info "Expansion Degree"
    The `expansion_degree` parameter controls how many hops outward from the initially retrieved entities the system will traverse. A value of `1` means direct neighbors are included; `2` means neighbors-of-neighbors, and so on. Higher values retrieve more context but increase latency.

**Response Body**

```json
{
  "question": "Where was Albert Einstein born?",
  "answer": "Albert Einstein was born in Ulm, Germany.",
  "query_entities": [
    {
      "id": "entity_1",
      "name": "Albert Einstein",
      "description": "Theoretical physicist",
      "score": 0.95
    }
  ],
  "subgraph": {
    "entities": [
      {
        "id": "entity_1",
        "name": "Albert Einstein",
        "description": "Theoretical physicist"
      },
      {
        "id": "entity_2",
        "name": "Ulm",
        "description": "City in Germany"
      }
    ],
    "relations": [
      {
        "id": "relation_1",
        "source": "Albert Einstein",
        "target": "Ulm",
        "description": "born in"
      }
    ],
    "passages": [
      {
        "id": "doc_1",
        "text": "Albert Einstein was born in Ulm, Germany in 1879.",
        "score": 0.92
      }
    ],
    "expansion_history": [
      {
        "step": 0,
        "entity_ids": ["entity_1"],
        "description": "Initial retrieval"
      },
      {
        "step": 1,
        "entity_ids": ["entity_2"],
        "description": "1-hop expansion"
      }
    ]
  },
  "retrieved_passages": [
    {
      "id": "doc_1",
      "text": "Albert Einstein was born in Ulm, Germany in 1879.",
      "score": 0.92
    }
  ],
  "stats": {
    "retrieval_time_ms": 120,
    "generation_time_ms": 850,
    "total_time_ms": 970,
    "num_entities_retrieved": 2,
    "num_relations_retrieved": 1,
    "num_passages_retrieved": 1
  },
  "retrieval_detail": {
    "entity_search_results": [...],
    "relation_search_results": [...],
    "expanded_entities": [...]
  },
  "rerank_result": {
    "reranked_entities": [...],
    "reranked_relations": [...],
    "reranked_passages": [...]
  }
}
```

| Field | Type | Description |
|---|---|---|
| `question` | `string` | The original question. |
| `answer` | `string` | The generated answer. |
| `query_entities` | `object[]` | Entities identified from the question. |
| `subgraph` | `object` | The retrieved subgraph containing entities, relations, passages, and expansion history. |
| `retrieved_passages` | `object[]` | Passages used for answer generation. |
| `stats` | `object` | Performance statistics (timing, counts). |
| `retrieval_detail` | `object` | Detailed retrieval results before reranking. |
| `rerank_result` | `object` | Results after LLM-based reranking (present only when `use_reranking` is `true`). |

**curl Example**

```bash
curl -X POST "http://localhost:8000/query?graph_name=my_graph" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Where was Albert Einstein born?",
    "use_reranking": true,
    "entity_top_k": 20,
    "relation_top_k": 20,
    "expansion_degree": 1
  }'
```

---

## Graph Exploration

### Get Graph Statistics

<span class="api-method api-get">GET</span> `/graph/{graph_name}/stats`

Returns aggregate statistics for a graph.

**Path Parameters**

| Parameter | Type | Description |
|---|---|---|
| `graph_name` | `string` | Name of the graph. |

**Response Body**

```json
{
  "graph_name": "my_graph",
  "entity_count": 150,
  "relation_count": 200,
  "passage_count": 50
}
```

**curl Example**

```bash
curl http://localhost:8000/graph/my_graph/stats
```

---

### Get Entity Neighbors

<span class="api-method api-get">GET</span> `/graph/{graph_name}/neighbors/{entity_id}?limit={limit}`

Retrieves the immediate neighbors of a given entity and the relations connecting them.

**Path Parameters**

| Parameter | Type | Description |
|---|---|---|
| `graph_name` | `string` | Name of the graph. |
| `entity_id` | `string` | ID of the entity to explore. |

**Query Parameters**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `limit` | `integer` | No | `20` | Maximum number of neighbors to return. |

**Response Body**

```json
{
  "entity_id": "entity_1",
  "neighbors": [
    {
      "id": "entity_2",
      "name": "Ulm",
      "description": "City in Germany"
    },
    {
      "id": "entity_3",
      "name": "Theory of Relativity",
      "description": "Scientific theory developed by Einstein"
    }
  ],
  "relations": [
    {
      "id": "relation_1",
      "source": "Albert Einstein",
      "target": "Ulm",
      "description": "born in"
    },
    {
      "id": "relation_2",
      "source": "Albert Einstein",
      "target": "Theory of Relativity",
      "description": "developed"
    }
  ]
}
```

**curl Example**

```bash
curl "http://localhost:8000/graph/my_graph/neighbors/entity_1?limit=20"
```

---

## Error Handling

All endpoints return standard HTTP status codes. Error responses follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

| Status Code | Meaning |
|---|---|
| `200` | Success. |
| `400` | Bad request — invalid parameters or request body. |
| `404` | Resource not found — graph, document, or entity does not exist. |
| `422` | Validation error — request body failed schema validation. |
| `500` | Internal server error. |

!!! tip "Debugging Validation Errors"
    FastAPI returns detailed `422` responses that include which field failed validation and why. Check the `detail` array in the response body for specifics.
