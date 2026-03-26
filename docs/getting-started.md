# Getting Started

## Installation

=== "pip"

    ```bash
    pip install vector-graph-rag
    ```

=== "uv"

    ```bash
    uv add vector-graph-rag
    ```

=== "With document loaders"

    ```bash
    pip install "vector-graph-rag[loaders]"
    ```

!!! note "Prerequisites"
    - Python 3.9+
    - An OpenAI API key (set `OPENAI_API_KEY` environment variable)

## Quick Start

```python
from vector_graph_rag import VectorGraphRAG

rag = VectorGraphRAG()  # reads OPENAI_API_KEY from environment

rag.add_texts([
    "Albert Einstein developed the theory of relativity.",
    "The theory of relativity revolutionized our understanding of space and time.",
])

result = rag.query("What did Einstein develop?")
print(result.answer)
```

!!! tip "Zero configuration"
    By default, Vector Graph RAG uses Milvus Lite — a local file-based vector database that requires no server setup. Your data is stored in `./vector_graph_rag.db`.

## Configuration

### Basic Configuration

```python
rag = VectorGraphRAG(
    milvus_uri="./my_data.db",                    # local file (Milvus Lite)
    llm_model="gpt-4o",                           # LLM for extraction and reranking
    embedding_model="text-embedding-3-large",      # embedding model
)
```

### With Remote Milvus

```python
rag = VectorGraphRAG(
    milvus_uri="http://localhost:19530",
    milvus_db="my_database",           # optional: specify database
    collection_prefix="my_project",    # optional: isolate collections
)
```

### With Zilliz Cloud

```python
rag = VectorGraphRAG(
    milvus_uri="https://in03-xxx.api.gcp-us-west1.zillizcloud.com",
    milvus_token="your-api-key",
)
```

!!! tip "Free tier available"
    [Sign up for Zilliz Cloud](https://cloud.zilliz.com/signup?utm_source=github&utm_medium=referral&utm_campaign=vector-graph-rag-docs) to get a free cluster. Use the cluster endpoint as `milvus_uri` and your API key as `milvus_token`.

### Milvus Deployment Modes

| Mode | `milvus_uri` | `milvus_token` | Best for |
|------|-------------|----------------|----------|
| **Milvus Lite** (default) | `./vector_graph_rag.db` | — | Personal use, dev — zero config |
| **Milvus Server** | `http://localhost:19530` | Optional | Multi-dataset, team environments |
| ⭐ **Zilliz Cloud** | `https://in03-xxx.api.gcp-us-west1.zillizcloud.com` | API key | Production, fully managed |

!!! info "Collection naming"
    With `collection_prefix="my_project"`, collections are named `my_project_vgrag_entities`, `my_project_vgrag_relations`, `my_project_vgrag_passages`.

### Environment Variables

All settings can be configured via environment variables with the `VGRAG_` prefix:

```bash
export OPENAI_API_KEY="sk-..."
export VGRAG_MILVUS_URI="http://localhost:19530"
export VGRAG_LLM_MODEL="gpt-4o"
export VGRAG_EMBEDDING_MODEL="text-embedding-3-large"
export VGRAG_COLLECTION_PREFIX="my_project"
```

Or use a `.env` file in your project root:

```ini
OPENAI_API_KEY=sk-...
VGRAG_MILVUS_URI=http://localhost:19530
VGRAG_LLM_MODEL=gpt-4o
```

### Multiple Knowledge Bases

Use `collection_prefix` to maintain separate graphs in the same Milvus instance:

```python
# Different documents → different prefixes
legal_rag = VectorGraphRAG(milvus_uri="./data.db", collection_prefix="legal")
finance_rag = VectorGraphRAG(milvus_uri="./data.db", collection_prefix="finance")
```

## Adding Documents

### From Text Strings

```python
rag.add_texts([
    "Einstein developed relativity at Princeton.",
    "The theory changed modern physics.",
])
```

### With Pre-extracted Triplets

Skip LLM extraction if you already have knowledge graph triplets:

```python
rag.add_documents_with_triplets([
    {
        "passage": "Einstein developed relativity at Princeton.",
        "triplets": [
            ["Einstein", "developed", "relativity"],
            ["Einstein", "worked at", "Princeton"],
        ],
    },
])
```

### From URLs and Files

Import web pages, PDFs, and other documents with automatic chunking:

```python
from vector_graph_rag.loaders import DocumentImporter

importer = DocumentImporter(chunk_size=1000, chunk_overlap=200)
result = importer.import_sources([
    "https://en.wikipedia.org/wiki/Albert_Einstein",
    "/path/to/document.pdf",
    "/path/to/report.docx",
])

rag = VectorGraphRAG(milvus_uri="./my_graph.db")
rag.add_documents(result.documents, extract_triplets=True)
```

!!! warning "Loader dependencies"
    Install with `pip install "vector-graph-rag[loaders]"` to enable URL fetching and document conversion.

## Querying

### Full Query Result

```python
result = rag.query("What did Einstein develop?")
print(result.answer)           # Generated answer
print(result.query_entities)   # Extracted entities from question
print(result.passages)         # Retrieved passages used for answer
```

### Simple String Answer

```python
answer = rag.query_simple("What did Einstein develop?")
print(answer)  # Just the answer string
```

### Custom Retrieval Parameters

```python
result = rag.query(
    "What did Einstein develop?",
    entity_top_k=30,           # more entity candidates
    relation_top_k=30,         # more relation candidates
    expansion_degree=2,        # 2-hop expansion
)
```

## REST API

Run the API server for programmatic access or to power the frontend:

```bash
uv run uvicorn vector_graph_rag.api.app:app --host 0.0.0.0 --port 8000
```

Interactive docs available at `http://localhost:8000/docs`.

See the [REST API Reference](rest-api.md) for full endpoint documentation.

## Next Steps

- **[How It Works](how-it-works.md)** — Understand the indexing and query pipelines
- **[Design Philosophy](design-philosophy.md)** — Why pure vector search instead of graph databases
- **[Python API](python-api.md)** — Complete API reference with all parameters
- **[Use Cases](use-cases.md)** — Domain-specific examples and best practices
- **[Frontend](frontend.md)** — Interactive graph visualization
