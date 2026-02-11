# Vector Graph RAG

A Graph RAG implementation using pure vector search with [Milvus](https://milvus.io/).

## Features

- **No Graph Database Required** — Pure vector search approach, no need for Neo4j or other graph databases
- **Zero Configuration** — Uses Milvus Lite by default, works out of the box with a single file
- **High Accuracy** — LLM-based reranking for precise relation filtering
- **Multi-hop Reasoning** — Subgraph expansion enables complex multi-hop question answering
- **State-of-the-Art Performance** — Outperforms HippoRAG on multi-hop QA benchmarks (87.8% avg Recall@5)
- **Simple API** — Just 3 lines of code to get started

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

## Performance at a Glance

| Method | MuSiQue | HotpotQA | 2WikiMultiHopQA | Average |
|--------|---------|----------|-----------------|---------|
| Naive RAG | 55.6% | 90.8% | 73.7% | 73.4% |
| IRCoT + HippoRAG | 57.6% | 83.0% | 93.9% | 78.2% |
| HippoRAG 2 | **74.7%** | **96.3%** | 90.4% | 87.1% |
| **Vector Graph RAG** | 73.0% | **96.3%** | **94.1%** | **87.8%** |

See [Evaluation](evaluation.md) for details.
