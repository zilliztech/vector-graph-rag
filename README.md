# Vector Graph RAG

🔗 A Graph RAG implementation using pure vector search with Milvus.

## ✨ Features

- **🚀 No Graph Database Required** - Pure vector search approach, no need for Neo4j or other graph databases
- **📦 Zero Configuration** - Uses Milvus Lite by default, works out of the box with a single file
- **🎯 High Accuracy** - LLM-based reranking for precise relation filtering
- **🔍 Multi-hop Reasoning** - Subgraph expansion enables complex multi-hop question answering
- **📊 State-of-the-Art Performance** - Outperforms HippoRAG on multi-hop QA benchmarks (87.8% avg Recall@5)
- **🛠️ Simple API** - Just 3 lines of code to get started

## 📦 Installation

```bash
pip install vector-graph-rag
# or
uv add vector-graph-rag
```

## 🚀 Quick Start

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

### Custom Configuration

```python
rag = VectorGraphRAG(
    milvus_uri="./my_data.db",
    llm_model="gpt-4o",
    embedding_model="text-embedding-3-large",
)
```

> **Note:** Set `OPENAI_API_KEY` environment variable before running.

### Import from URLs and Files

Import web pages, PDFs, and other documents:

```bash
pip install "vector-graph-rag[loaders]"
```

```python
from vector_graph_rag import VectorGraphRAG
from vector_graph_rag.loaders import URLFetcher, DocumentImporter

# Import from URLs
fetcher = URLFetcher()
result = fetcher.fetch_batch([
    "https://en.wikipedia.org/wiki/Albert_Einstein",
    "https://example.com/article.html",
])

rag = VectorGraphRAG(milvus_uri="./my_graph.db")
rag.add_documents(result.documents, extract_triplets=True)

# Or import from files (PDF, DOCX, etc.) with chunking
importer = DocumentImporter(chunk_size=1000, chunk_overlap=200)
docs = importer.import_sources([
    "/path/to/document.pdf",
    "/path/to/report.docx",
    "https://example.com/page",  # URLs also supported
])
rag.add_documents(docs.documents, extract_triplets=True)

# Query
result = rag.query("What did Einstein discover?")
print(result.answer)
```

## 🔬 How It Works

### Indexing Pipeline

```
Documents → Triplet Extraction (LLM) → Entities + Relations → Embedding → Milvus
```

### Query Pipeline

```
Question → Entity Extraction → Vector Search → Subgraph Expansion → LLM Reranking → Answer
```

### Example

**Indexing:** *"Einstein developed the theory of relativity at Princeton."*
- Entities: `Einstein`, `theory of relativity`, `Princeton`
- Relations: `(Einstein, developed, theory of relativity)`, `(Einstein, worked at, Princeton)`

**Query:** *"What did Einstein develop?"*
1. Extract entity: `Einstein`
2. Vector search finds similar entities and relations
3. Subgraph expansion collects candidate relations
4. **LLM reranking** selects `(Einstein, developed, theory of relativity)`
5. Generate answer: *"Einstein developed the theory of relativity."*

## 📊 Evaluation Results

Evaluated on three multi-hop QA datasets:

| Method | MuSiQue | HotpotQA | 2WikiMultiHopQA | Average |
|--------|---------|----------|-----------------|---------|
| Naive RAG | 55.6% | 90.8% | 73.7% | 73.4% |
| IRCoT + HippoRAG¹ | 57.6% | 83.0% | 93.9% | 78.2% |
| HippoRAG 2² | **74.7%** | **96.3%** | 90.4% | 87.1% |
| **Vector Graph RAG** | 73.0% | **96.3%** | **94.1%** | **87.8%** |

¹ [HippoRAG (NeurIPS 2024)](https://arxiv.org/abs/2405.14831) ² [HippoRAG 2 (2025)](https://arxiv.org/abs/2502.14802)

See [evaluation/README.md](evaluation/README.md) for reproduction steps.

## 🛠️ Development

### Running the Demo (Frontend + API)

```bash
# Backend
uv sync --extra api
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

### REST API

The API server provides endpoints for integration:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/graphs` | GET | List available graphs |
| `/stats` | GET | Get graph statistics |
| `/query` | POST | Query the knowledge graph |
| `/add_documents` | POST | Add documents |

See API docs at `http://localhost:8000/docs` after starting the server.

## 📄 License

MIT
