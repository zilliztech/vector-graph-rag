# Vector Graph RAG

Graph RAG with pure vector search — no graph database needed, single-pass LLM reranking, optimized for knowledge-intensive domains.

## Why Vector Graph RAG?

Most Graph RAG systems require a dedicated graph database (Neo4j, etc.) and complex multi-step retrieval with iterative LLM calls. Vector Graph RAG takes a fundamentally different approach:

- **No graph database** — The entire knowledge graph lives in Milvus as vectors. No extra infrastructure, no schema management, no graph query language.
- **Single-pass reranking** — Unlike agentic approaches (IRCoT, multi-step reflection), we call the LLM just once to rerank candidate relations. This is simpler, faster, and cheaper.
- **Knowledge-intensive friendly** — Designed for domains where dense factual knowledge matters: legal documents, financial reports, medical literature, novels, and more.

## Features

| | |
|---|---|
| **No Graph Database** | Pure vector search with Milvus — no Neo4j, no ArangoDB, no extra infra |
| **Single-Pass Reranking** | One LLM call, no iterative agent loops like IRCoT |
| **Knowledge-Intensive** | Optimized for legal, finance, medical, literature domains |
| **Zero Configuration** | Milvus Lite by default, works out of the box |
| **Multi-hop Reasoning** | Subgraph expansion for complex multi-hop QA |
| **State-of-the-Art** | 87.8% avg Recall@5 on standard benchmarks |

## Quick Example

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

## Performance

| Method | MuSiQue | HotpotQA | 2WikiMultiHopQA | Average |
|--------|---------|----------|-----------------|---------|
| Naive RAG | 55.6% | 90.8% | 73.7% | 73.4% |
| IRCoT + HippoRAG | 57.6% | 83.0% | 93.9% | 78.2% |
| HippoRAG 2 | **74.7%** | **96.3%** | 90.4% | 87.1% |
| **Vector Graph RAG** | 73.0% | **96.3%** | **94.1%** | **87.8%** |

See [Evaluation](evaluation.md) for details.
