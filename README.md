<h1 align="center">
  Vector Graph RAG
</h1>

<p align="center">
  <strong>Graph RAG with pure vector search — no graph database needed.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/vector-graph-rag/"><img src="https://img.shields.io/pypi/v/vector-graph-rag?style=flat-square&color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/vector-graph-rag/"><img src="https://img.shields.io/badge/python-%3E%3D3.10-blue?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/zilliztech/vector-graph-rag/blob/main/LICENSE"><img src="https://img.shields.io/github/license/zilliztech/vector-graph-rag?style=flat-square" alt="License"></a>
  <a href="https://zilliztech.github.io/vector-graph-rag/"><img src="https://img.shields.io/badge/docs-vector--graph--rag-blue?style=flat-square" alt="Docs"></a>
  <a href="https://github.com/zilliztech/vector-graph-rag/stargazers"><img src="https://img.shields.io/github/stars/zilliztech/vector-graph-rag?style=flat-square" alt="Stars"></a>
  <a href="https://discord.com/invite/FG6hMJStWu"><img src="https://img.shields.io/badge/Discord-chat-7289da?style=flat-square&logo=discord&logoColor=white" alt="Discord"></a>
</p>

> 💡 Encode entities and relations as vectors in [Milvus](https://milvus.io/), replace iterative LLM agents with a single reranking pass — achieve state-of-the-art multi-hop retrieval at a fraction of the operational and computational cost.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1185b651-ed72-4408-9dcd-25a74b12835b" alt="Vector Graph RAG Demo" width="800">
</p>

## ✨ Features

- **No Graph Database Required** — Pure vector search with Milvus, no Neo4j or other graph databases needed
- **Single-Pass LLM Reranking** — One LLM call to rerank, no iterative agent loops (unlike IRCoT or multi-step reflection)
- **Knowledge-Intensive Friendly** — Optimized for domains with dense factual content: legal, finance, medical, literature, etc.
- **Zero Configuration** — Uses Milvus Lite by default, works out of the box with a single file
- **Multi-hop Reasoning** — Subgraph expansion enables complex multi-hop question answering
- **State-of-the-Art Performance** — 87.8% avg Recall@5 on multi-hop QA benchmarks, outperforming HippoRAG

## 📦 Installation

```bash
pip install vector-graph-rag
# or
uv add vector-graph-rag
```

<details>
<summary><b>With document loaders (PDF, DOCX, web pages)</b></summary>

```bash
pip install "vector-graph-rag[loaders]"
# or
uv add "vector-graph-rag[loaders]"
```

</details>

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

> **Note:** Set `OPENAI_API_KEY` environment variable before running.

<details>
<summary>📄 <b>With pre-extracted triplets</b> — click to expand</summary>

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

</details>

<details>
<summary>🌐 <b>Import from URLs and files</b> — click to expand</summary>

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

result = rag.query("What did Einstein discover?")
print(result.answer)
```

</details>

<details>
<summary>⚙️ <b>Custom configuration</b> — click to expand</summary>

```python
rag = VectorGraphRAG(
    milvus_uri="./my_data.db",          # or remote Milvus / Zilliz Cloud
    llm_model="gpt-4o",
    embedding_model="text-embedding-3-large",
    collection_prefix="my_project",     # isolate multiple datasets
)
```

All settings can also be configured via environment variables with `VGRAG_` prefix or a `.env` file:

```bash
VGRAG_LLM_MODEL=gpt-4o
VGRAG_EMBEDDING_MODEL=text-embedding-3-large
VGRAG_MILVUS_URI=http://localhost:19530
```

</details>

> 📖 Full Python API reference → [Python API docs](https://zilliztech.github.io/vector-graph-rag/python-api/)

## 🔬 How It Works

**Indexing:**

```
Documents → Triplet Extraction (LLM) → Entities + Relations → Embedding → Milvus
```

**Query:**

```
Question → Entity Extraction → Vector Search → Subgraph Expansion → LLM Reranking → Answer
```

**Example:** *"What did Einstein develop?"*

1. Extract entity: `Einstein`
2. Vector search finds similar entities and relations in Milvus
3. Subgraph expansion collects neighboring relations
4. **Single-pass LLM reranking** selects the most relevant passages
5. Generate answer from selected passages

> 📖 Detailed pipeline walkthrough with diagrams → [How It Works](https://zilliztech.github.io/vector-graph-rag/how-it-works/) · [Design Philosophy](https://zilliztech.github.io/vector-graph-rag/design-philosophy/)

## 📊 Evaluation Results

Evaluated on three multi-hop QA benchmarks (Recall@5):

| Method | MuSiQue | HotpotQA | 2WikiMultiHopQA | Average |
|--------|---------|----------|-----------------|---------|
| Naive RAG | 55.6% | 90.8% | 73.7% | 73.4% |
| IRCoT + HippoRAG¹ | 57.6% | 83.0% | 93.9% | 78.2% |
| HippoRAG 2² | **74.7%** | **96.3%** | 90.4% | 87.1% |
| **Vector Graph RAG** | 73.0% | **96.3%** | **94.1%** | **87.8%** |

¹ [HippoRAG (NeurIPS 2024)](https://arxiv.org/abs/2405.14831)  ² [HippoRAG 2 (2025)](https://arxiv.org/abs/2502.14802)

> 📖 Detailed analysis and reproduction steps → [Evaluation](https://zilliztech.github.io/vector-graph-rag/evaluation/)

## 🗄️ Milvus Backend

Vector Graph RAG supports three Milvus deployment modes — just change `milvus_uri`:

| Mode | `milvus_uri` | Best for |
|------|-------------|----------|
| **Milvus Lite** (default) | `./vector_graph_rag.db` | Personal use, dev — zero config |
| **Milvus Server** | `http://localhost:19530` | Multi-dataset, team environments |
| ⭐ **Zilliz Cloud** | `https://in03-xxx.api.gcp-us-west1.zillizcloud.com` | Production, fully managed — [free tier available](https://cloud.zilliz.com/signup?utm_source=github&utm_medium=referral&utm_campaign=vector-graph-rag-readme) |

> **Recommended:** [Zilliz Cloud](https://cloud.zilliz.com/signup?utm_source=github&utm_medium=referral&utm_campaign=vector-graph-rag-readme) gives you zero-config, zero-ops Milvus with concurrent access and real-time indexing — no Docker needed.

<details>
<summary>Sign up for a free Zilliz Cloud cluster 👈</summary>

You can [sign up](https://cloud.zilliz.com/signup?utm_source=github&utm_medium=referral&utm_campaign=vector-graph-rag-readme) on Zilliz Cloud to get a free cluster and API key.

![Sign up and get API key](https://raw.githubusercontent.com/zilliztech/CodeIndexer/master/assets/signup_and_get_apikey.png)

Use your endpoint as `milvus_uri` and your API key as `milvus_token`:

```python
rag = VectorGraphRAG(
    milvus_uri="https://in03-xxx.api.gcp-us-west1.zillizcloud.com",
    milvus_token="your-api-key",
)
```

</details>

## 🖥️ Frontend & REST API

Vector Graph RAG includes a React-based frontend for interactive graph visualization and a FastAPI backend.

```bash
# Backend
uv sync --extra api
uv run uvicorn vector_graph_rag.api.app:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/graphs` | GET | List available graphs |
| `/api/graph/{name}/stats` | GET | Get graph statistics |
| `/api/query` | POST | Query the knowledge graph |
| `/api/documents` | POST | Add documents |
| `/api/import` | POST | Import from URLs/paths |
| `/api/upload` | POST | Upload files |

See API docs at `http://localhost:8000/docs` after starting the server.

> 📖 Full endpoint reference → [REST API docs](https://zilliztech.github.io/vector-graph-rag/rest-api/) · [Frontend guide](https://zilliztech.github.io/vector-graph-rag/frontend/)

## 📚 Links

- [Documentation](https://zilliztech.github.io/vector-graph-rag/) — full guides, API reference, and architecture details
- [How It Works](https://zilliztech.github.io/vector-graph-rag/how-it-works/) — pipeline walkthrough with diagrams
- [Design Philosophy](https://zilliztech.github.io/vector-graph-rag/design-philosophy/) — why pure vector search, no graph DB
- [Milvus](https://milvus.io/) — the vector database powering Vector Graph RAG
- [FAQ](https://zilliztech.github.io/vector-graph-rag/faq/) — common questions and troubleshooting

## Contributing

Bug reports, feature requests, and pull requests are welcome! For questions and discussions, join us on [Discord](https://discord.com/invite/FG6hMJStWu).

## 📄 License

[MIT](LICENSE)
