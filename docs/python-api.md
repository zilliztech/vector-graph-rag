# Python API Reference

This page provides comprehensive documentation for the Vector Graph RAG Python API. All classes and functions described here are available from the top-level `vector_graph_rag` package unless otherwise noted.

---

## Overview

The Vector Graph RAG library exposes a small, focused surface area:

| Component | Purpose |
|---|---|
| [`Settings`](#settings) | Global configuration via environment variables or direct assignment |
| [`VectorGraphRAG`](#vectorgraphrag) | Core RAG engine — ingest documents, query with graph-augmented retrieval |
| [`QueryResult`](#queryresult) | Structured output from queries |
| [`ExtractionResult`](#extractionresult) | Structured output from document ingestion |
| [`DocumentImporter`](#documentimporter) | Load and chunk files (PDF, DOCX, TXT, MD, HTML, URLs) |
| [`create_rag()`](#create_rag) | Convenience factory for quick setup |

---

## Settings

`Settings` is a [Pydantic BaseSettings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) class that centralises all configuration. Values can be set via constructor arguments, environment variables (with the `VGRAG_` prefix), or a `.env` file.

```python
from vector_graph_rag.config import Settings

settings = Settings(
    llm_model="gpt-4o",
    entity_top_k=30,
)
```

!!! tip "Environment variable mapping"
    Every field is read from an environment variable named `VGRAG_<FIELD_NAME>` (upper-case).
    For example, `llm_model` maps to `VGRAG_LLM_MODEL`. The special field `openai_api_key`
    is also read from the standard `OPENAI_API_KEY` variable.

### Fields

#### LLM & Embedding

| Field | Type | Default | Description |
|---|---|---|---|
| `openai_api_key` | `Optional[str]` | `None` | OpenAI API key. Also read from `OPENAI_API_KEY` env var. |
| `openai_base_url` | `Optional[str]` | `None` | Custom OpenAI-compatible API base URL (e.g. for Azure or local proxies). |
| `llm_model` | `str` | `"gpt-4o-mini"` | Model name used for LLM calls (extraction, answering, reranking). |
| `embedding_model` | `str` | `"text-embedding-3-large"` | Model name used for embedding generation. |
| `embedding_dimension` | `int` | `3072` | Dimensionality of the embedding vectors. |
| `llm_temperature` | `float` | `0.0` | Temperature for LLM generation. |
| `use_llm_cache` | `bool` | `True` | Whether to cache LLM responses to avoid redundant API calls. |

#### Milvus / Zilliz

| Field | Type | Default | Description |
|---|---|---|---|
| `milvus_uri` | `str` | `"./vector_graph_rag.db"` | Milvus connection URI. Default uses **Milvus Lite** (embedded, file-based). |
| `milvus_token` | `Optional[str]` | `None` | Authentication token, required for **Zilliz Cloud**. |
| `milvus_db` | `Optional[str]` | `None` | Database name within Milvus. |
| `collection_prefix` | `Optional[str]` | `None` | Prefix prepended to all collection names for multi-dataset isolation. |
| `entity_collection` | `str` | `"vgrag_entities"` | Name of the entity collection. |
| `relation_collection` | `str` | `"vgrag_relations"` | Name of the relation collection. |
| `passage_collection` | `str` | `"vgrag_passages"` | Name of the passage collection. |

#### Retrieval

| Field | Type | Default | Description |
|---|---|---|---|
| `entity_top_k` | `int` | `20` | Number of top entities to retrieve during vector search. |
| `relation_top_k` | `int` | `20` | Number of top relations to retrieve during vector search. |
| `entity_similarity_threshold` | `float` | `0.9` | Minimum similarity score to keep an entity match. |
| `relation_similarity_threshold` | `float` | `-1.0` | Minimum similarity score to keep a relation match. `-1.0` effectively disables filtering. |
| `expansion_degree` | `int` | `1` | Number of hops to expand in the knowledge graph from matched entities. |
| `final_top_k` | `int` | `3` | Number of final passages returned after reranking. |

#### Processing

| Field | Type | Default | Description |
|---|---|---|---|
| `batch_size` | `int` | `32` | Batch size for embedding and extraction operations. |

### Example: configuring via `.env`

```dotenv
# .env
OPENAI_API_KEY=sk-...
VGRAG_LLM_MODEL=gpt-4o
VGRAG_MILVUS_URI=http://localhost:19530
VGRAG_ENTITY_TOP_K=30
VGRAG_EXPANSION_DEGREE=2
```

```python
from vector_graph_rag.config import Settings

# Automatically picks up .env values
settings = Settings()
```

---

## VectorGraphRAG

The central class that orchestrates document ingestion, knowledge-graph construction, and graph-augmented retrieval.

### Constructor

```python
from vector_graph_rag import VectorGraphRAG

rag = VectorGraphRAG(
    settings=None,              # Optional[Settings] — pre-built Settings object
    milvus_uri=None,            # Optional[str] — override milvus_uri
    milvus_db=None,             # Optional[str] — override milvus_db
    collection_prefix=None,     # Optional[str] — override collection_prefix
    openai_api_key=None,        # Optional[str] — override openai_api_key
    llm_model=None,             # Optional[str] — override llm_model
    embedding_model=None,       # Optional[str] — override embedding_model
)
```

!!! info "Parameter precedence"
    Keyword arguments passed directly to the constructor (e.g. `milvus_uri`) take
    precedence over values in the `settings` object, which in turn take precedence
    over environment variables and `.env` file values.

### Methods

#### `add_texts`

Ingest plain text strings. Each string is stored as a passage; optionally, knowledge-graph triplets are extracted automatically.

```python
def add_texts(
    texts: List[str],
    ids: Optional[List[str]] = None,
    metadatas: Optional[List[dict]] = None,
    extract_triplets: bool = True,
    show_progress: bool = True,
) -> ExtractionResult
```

| Parameter | Description |
|---|---|
| `texts` | List of text strings to ingest. |
| `ids` | Optional list of unique IDs (one per text). Auto-generated if omitted. |
| `metadatas` | Optional list of metadata dicts attached to each text. |
| `extract_triplets` | If `True`, the LLM extracts entity–relation–entity triplets from each text. |
| `show_progress` | Show a progress bar during processing. |

**Returns:** [`ExtractionResult`](#extractionresult)

```python
result = rag.add_texts([
    "Albert Einstein developed the theory of general relativity.",
    "Einstein was born in Ulm, Germany in 1879.",
])

print(f"Extracted {len(result.entities)} entities, {len(result.relations)} relations")
```

---

#### `add_documents`

Ingest [LangChain `Document`](https://python.langchain.com/docs/modules/data_connection/document_loaders/) objects directly.

```python
def add_documents(
    documents: List[Document],
    extract_triplets: bool = True,
    show_progress: bool = True,
) -> ExtractionResult
```

| Parameter | Description |
|---|---|
| `documents` | List of LangChain `Document` objects (each has `.page_content` and `.metadata`). |
| `extract_triplets` | If `True`, extract knowledge-graph triplets from each document. |
| `show_progress` | Show a progress bar. |

**Returns:** [`ExtractionResult`](#extractionresult)

```python
from langchain_core.documents import Document

docs = [
    Document(page_content="Marie Curie discovered radium.", metadata={"source": "wiki"}),
    Document(page_content="She was awarded two Nobel Prizes.", metadata={"source": "wiki"}),
]

result = rag.add_documents(docs)
```

---

#### `add_documents_with_triplets`

Ingest documents where triplets have already been extracted externally. Use this when you have your own extraction pipeline or pre-annotated data.

```python
def add_documents_with_triplets(
    documents: List[dict],
    show_progress: bool = True,
) -> ExtractionResult
```

Each dict in `documents` should contain the document text and its pre-extracted triplets.

**Returns:** [`ExtractionResult`](#extractionresult)

!!! example "Pre-extracted triplets"
    ```python
    docs_with_triplets = [
        {
            "text": "Albert Einstein developed the theory of general relativity.",
            "triplets": [
                ("Albert Einstein", "developed", "theory of general relativity"),
            ],
        },
    ]

    result = rag.add_documents_with_triplets(docs_with_triplets)
    ```

---

#### `query`

Full-featured query with graph-augmented retrieval, optional reranking, and optional naive-RAG comparison.

```python
def query(
    question: str,
    use_reranking: bool = True,
    compare_naive: bool = False,
    entity_top_k: Optional[int] = None,
    relation_top_k: Optional[int] = None,
    entity_similarity_threshold: Optional[float] = None,
    relation_similarity_threshold: Optional[float] = None,
    expansion_degree: Optional[int] = None,
) -> QueryResult
```

| Parameter | Description |
|---|---|
| `question` | The natural-language question to answer. |
| `use_reranking` | Whether to apply LLM-based reranking on retrieved passages. |
| `compare_naive` | If `True`, also run a naive vector-only retrieval for comparison. |
| `entity_top_k` | Override `Settings.entity_top_k` for this query. |
| `relation_top_k` | Override `Settings.relation_top_k` for this query. |
| `entity_similarity_threshold` | Override `Settings.entity_similarity_threshold` for this query. |
| `relation_similarity_threshold` | Override `Settings.relation_similarity_threshold` for this query. |
| `expansion_degree` | Override `Settings.expansion_degree` for this query. |

**Returns:** [`QueryResult`](#queryresult)

```python
result = rag.query("What did Einstein contribute to physics?")

print(result.answer)
print(f"Found {len(result.passages)} relevant passages")
```

!!! tip "Per-query tuning"
    You can override retrieval parameters on a per-query basis without modifying global
    settings. This is useful for experimentation or for queries that need different
    sensitivity levels.

    ```python
    # Broader search with more hops
    result = rag.query(
        "How are quantum mechanics and relativity connected?",
        entity_top_k=40,
        expansion_degree=2,
    )
    ```

---

#### `query_simple`

A convenience method that returns just the answer string — no metadata, no retrieval details.

```python
def query_simple(question: str) -> str
```

**Returns:** `str` — the generated answer.

```python
answer = rag.query_simple("When was Einstein born?")
print(answer)
# "Albert Einstein was born on March 14, 1879, in Ulm, Germany."
```

---

#### `query_naive`

Run a **naive vector-only** retrieval (no graph expansion or reranking). Useful as a baseline for comparison.

```python
def query_naive(question: str) -> QueryResult
```

**Returns:** [`QueryResult`](#queryresult)

```python
naive_result = rag.query_naive("What did Einstein contribute to physics?")
graph_result = rag.query("What did Einstein contribute to physics?")

print("Naive:", naive_result.answer)
print("Graph:", graph_result.answer)
```

---

#### `retrieve`

Retrieve relevant passages **without** generating an answer. Useful when you want to feed the passages into your own downstream pipeline.

```python
def retrieve(
    question: str,
    use_reranking: bool = True,
    top_k: Optional[int] = None,
) -> QueryResult
```

| Parameter | Description |
|---|---|
| `question` | The natural-language question. |
| `use_reranking` | Whether to apply LLM-based reranking. |
| `top_k` | Number of passages to return (overrides `Settings.final_top_k`). |

**Returns:** [`QueryResult`](#queryresult) (with `answer` field empty or `None`).

```python
result = rag.retrieve("Tell me about general relativity", top_k=5)

for passage in result.passages:
    print(passage)
```

---

#### `get_stats`

Return collection statistics.

```python
def get_stats() -> dict
```

**Returns:** A dictionary with entity, relation, and passage counts.

```python
stats = rag.get_stats()
print(stats)
# {"entities": 142, "relations": 87, "passages": 50}
```

---

#### `reset`

Delete all data from all collections (entities, relations, passages). **This is destructive and irreversible.**

```python
def reset() -> None
```

!!! warning "Destructive operation"
    `reset()` drops all ingested data. There is no confirmation prompt. Use with caution
    in production environments.

```python
rag.reset()
print(rag.get_stats())
# {"entities": 0, "relations": 0, "passages": 0}
```

---

## QueryResult

A data class returned by `query()`, `query_naive()`, and `retrieve()`. It contains the generated answer along with full retrieval diagnostics.

```python
from vector_graph_rag import QueryResult
```

### Fields

| Field | Type | Description |
|---|---|---|
| `query` | `str` | The original question. |
| `answer` | `str` | The generated answer (empty for `retrieve()`). |
| `query_entities` | `list` | Entities extracted from the query. |
| `passages` | `list` | Final list of passages used for answer generation. |
| `retrieved_passages` | `list` | Passages retrieved via initial vector search. |
| `retrieved_relations` | `list` | Relations retrieved via vector search. |
| `expanded_relations` | `list` | Relations discovered through graph expansion. |
| `reranked_relations` | `list` | Relations after LLM-based reranking. |
| `subgraph` | `dict` | The local subgraph explored during retrieval. |
| `retrieval_detail` | `dict` | Detailed retrieval metrics and intermediate results. |
| `rerank_result` | `object` | Raw reranking output. |
| `eviction_result` | `object` | Details of any evicted (filtered-out) passages. |

```python
result = rag.query("What is general relativity?")

# Access the answer
print(result.answer)

# Inspect retrieval diagnostics
print(f"Query entities: {result.query_entities}")
print(f"Retrieved {len(result.retrieved_relations)} relations")
print(f"Expanded to {len(result.expanded_relations)} relations via graph")
print(f"Final passages: {len(result.passages)}")
```

---

## ExtractionResult

A data class returned by all `add_*` methods. It summarises what was ingested and extracted.

```python
from vector_graph_rag import ExtractionResult
```

### Fields

| Field | Type | Description |
|---|---|---|
| `documents` | `list` | The ingested documents. |
| `entities` | `list` | All extracted entities. |
| `relations` | `list` | All extracted relations (triplets). |
| `entity_to_relation_ids` | `dict` | Mapping from entity IDs to their related relation IDs. |
| `relation_to_passage_ids` | `dict` | Mapping from relation IDs to source passage IDs. |

```python
result = rag.add_texts(["Marie Curie discovered radium and polonium."])

print(f"Documents: {len(result.documents)}")
print(f"Entities:  {len(result.entities)}")
print(f"Relations: {len(result.relations)}")

# Explore the knowledge graph mappings
for entity_id, relation_ids in result.entity_to_relation_ids.items():
    print(f"Entity {entity_id} -> Relations {relation_ids}")
```

---

## DocumentImporter

A utility class for loading and chunking documents from various file formats and URLs.

```python
from vector_graph_rag.loaders import DocumentImporter
```

### Constructor

```python
importer = DocumentImporter(
    chunk_documents=True,    # Whether to split documents into chunks
    chunk_size=1000,         # Maximum characters per chunk
    chunk_overlap=200,       # Character overlap between consecutive chunks
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chunk_documents` | `bool` | `True` | If `True`, loaded documents are split into chunks. |
| `chunk_size` | `int` | `1000` | Maximum number of characters per chunk. |
| `chunk_overlap` | `int` | `200` | Number of overlapping characters between adjacent chunks. |

### Supported formats

| Format | Extensions / Patterns |
|---|---|
| PDF | `.pdf` |
| Word | `.docx` |
| Plain text | `.txt` |
| Markdown | `.md` |
| HTML | `.html`, `.htm` |
| URLs | `http://...`, `https://...` |

### Methods

#### `import_sources`

Load documents from a list of file paths and/or URLs.

```python
def import_sources(sources: List[str]) -> LoaderResult
```

```python
importer = DocumentImporter(chunk_size=500, chunk_overlap=100)

result = importer.import_sources([
    "/path/to/report.pdf",
    "/path/to/notes.md",
    "https://example.com/article",
])

print(f"Loaded {len(result.documents)} chunks")
```

#### `import_text`

Load a raw text string as a document.

```python
def import_text(text: str, source: str = "text_input") -> LoaderResult
```

| Parameter | Description |
|---|---|
| `text` | The raw text content. |
| `source` | A label for the source (used in metadata). |

```python
result = importer.import_text(
    "Einstein published four groundbreaking papers in 1905...",
    source="annus_mirabilis",
)
```

---

## `create_rag`

A convenience factory function for quickly creating a `VectorGraphRAG` instance with common defaults.

```python
from vector_graph_rag import create_rag

rag = create_rag(
    milvus_uri=None,                        # Optional[str]
    milvus_db=None,                         # Optional[str]
    collection_prefix=None,                 # Optional[str]
    openai_api_key=None,                    # Optional[str]
    llm_model="gpt-4o-mini",               # str
    embedding_model="text-embedding-3-small",  # str
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `milvus_uri` | `Optional[str]` | `None` | Milvus connection URI. Falls back to `Settings` default. |
| `milvus_db` | `Optional[str]` | `None` | Milvus database name. |
| `collection_prefix` | `Optional[str]` | `None` | Prefix for collection names (multi-dataset isolation). |
| `openai_api_key` | `Optional[str]` | `None` | OpenAI API key. Falls back to environment variable. |
| `llm_model` | `str` | `"gpt-4o-mini"` | LLM model to use. |
| `embedding_model` | `str` | `"text-embedding-3-small"` | Embedding model to use. |

**Returns:** [`VectorGraphRAG`](#vectorgraphrag)

!!! note "Default embedding model"
    `create_rag()` defaults to `text-embedding-3-small` (not `text-embedding-3-large`
    as in `Settings`). This is a deliberate choice for quick-start scenarios where lower
    cost and faster embedding are preferred.

```python
from vector_graph_rag import create_rag

# Minimal setup — just needs OPENAI_API_KEY in the environment
rag = create_rag()

rag.add_texts(["The mitochondria is the powerhouse of the cell."])
answer = rag.query_simple("What is the powerhouse of the cell?")
print(answer)
```

---

## Complete Examples

### End-to-end ingestion and query

```python
from vector_graph_rag import create_rag
from vector_graph_rag.loaders import DocumentImporter

# 1. Create the RAG instance
rag = create_rag(
    collection_prefix="my_project",
    llm_model="gpt-4o",
)

# 2. Load and chunk documents
importer = DocumentImporter(chunk_size=800, chunk_overlap=150)
loader_result = importer.import_sources([
    "research_paper.pdf",
    "https://en.wikipedia.org/wiki/General_relativity",
])

# 3. Ingest into the vector graph
extraction = rag.add_documents(loader_result.documents)
print(f"Ingested {len(extraction.entities)} entities and {len(extraction.relations)} relations")

# 4. Query
result = rag.query("What experimental evidence supports general relativity?")
print(result.answer)

# 5. Check stats
print(rag.get_stats())
```

### Multi-dataset isolation

```python
from vector_graph_rag import create_rag

# Two separate knowledge bases sharing the same Milvus instance
physics_rag = create_rag(collection_prefix="physics")
biology_rag = create_rag(collection_prefix="biology")

physics_rag.add_texts(["E=mc² is the mass-energy equivalence formula."])
biology_rag.add_texts(["DNA carries genetic information in living organisms."])

# Each RAG instance only searches its own collections
print(physics_rag.query_simple("What is E=mc²?"))
print(biology_rag.query_simple("What carries genetic information?"))
```

### Using Zilliz Cloud

```python
from vector_graph_rag import VectorGraphRAG
from vector_graph_rag.config import Settings

settings = Settings(
    milvus_uri="https://your-instance.zillizcloud.com",
    milvus_token="your-api-key",
    milvus_db="my_database",
)

rag = VectorGraphRAG(settings=settings)
```

### Retrieval-only pipeline

```python
from vector_graph_rag import create_rag

rag = create_rag()

# Retrieve passages without generating an answer
result = rag.retrieve("What causes earthquakes?", top_k=10)

# Feed into your own generation pipeline
for i, passage in enumerate(result.passages):
    print(f"[{i+1}] {passage}")
```
