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

!!! note
    Set the `OPENAI_API_KEY` environment variable before running.

## Custom Configuration

```python
rag = VectorGraphRAG(
    milvus_uri="./my_data.db",
    llm_model="gpt-4o",
    embedding_model="text-embedding-3-large",
)
```

## With Pre-extracted Triplets

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

## Import from URLs and Files

Import web pages, PDFs, and other documents:

```bash
pip install "vector-graph-rag[loaders]"
```

```python
from vector_graph_rag import VectorGraphRAG
from vector_graph_rag.loaders import DocumentImporter

# Import from URLs, PDFs, DOCX, etc. (with automatic chunking)
importer = DocumentImporter(chunk_size=1000, chunk_overlap=200)
result = importer.import_sources([
    "https://en.wikipedia.org/wiki/Albert_Einstein",
    "/path/to/document.pdf",
    "/path/to/report.docx",
])

rag = VectorGraphRAG(milvus_uri="./my_graph.db")
rag.add_documents(result.documents, extract_triplets=True)

# Query
result = rag.query("What did Einstein discover?")
print(result.answer)
```

## REST API

Run the API server for programmatic access:

```bash
uv sync --extra api
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/graphs` | GET | List available graphs |
| `/stats` | GET | Get graph statistics |
| `/query` | POST | Query the knowledge graph |
| `/add_documents` | POST | Add documents |

API docs are available at `http://localhost:8000/docs` after starting the server.
