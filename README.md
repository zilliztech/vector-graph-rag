# Vector Graph RAG

A Graph RAG implementation using pure vector search with Milvus.

## Features

- Knowledge graph construction from documents
- Vector-based entity and relation retrieval
- Subgraph expansion with configurable degree
- LLM-based reranking
- Interactive web UI for visualizing search process

## Quick Start

### Using Docker

```bash
# Build with CPU-only PyTorch (smaller image, ~2GB)
docker build -t vector-graph-rag:cpu --build-arg TORCH_BACKEND=cpu .

# Build with CUDA 12.4 support (~7GB)
docker build -t vector-graph-rag:cu124 --build-arg TORCH_BACKEND=cu124 .

# Build with CUDA 12.8 support
docker build -t vector-graph-rag:cu128 --build-arg TORCH_BACKEND=cu128 .

# Run
docker run -p 8000:8000 vector-graph-rag:cpu
```

### Development

Backend:
```bash
uv run uvicorn vector_graph_rag.api.app:app --reload --port 8000
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```

## License

MIT
