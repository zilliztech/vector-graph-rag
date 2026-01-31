# Vector Graph RAG

A Graph RAG implementation using pure vector search with Milvus.

## How It Works

### Indexing Pipeline

```
┌──────────────┐    ┌─────────────────────┐    ┌─────────────────────────────┐
│  Documents   │───▶│  Triplet Extraction │───▶│   Entities  +  Relations    │
│              │    │       (LLM)         │    │  (Einstein)   (developed)   │
└──────────────┘    └─────────────────────┘    └──────────────┬──────────────┘
                                                              │
                    ┌─────────────────────┐                   │ Embedding
                    │       Milvus        │◀──────────────────┘
                    │  ┌───────────────┐  │
                    │  │ Entity Vectors│  │
                    │  │Relation Vectors│ │
                    │  │Passage Vectors │ │
                    │  └───────────────┘  │
                    └─────────────────────┘
```

### Query Pipeline

```
┌──────────────┐    ┌─────────────────────┐    ┌─────────────────────────────┐
│   Question   │───▶│  Entity Extraction  │───▶│      Vector Search          │
│              │    │       (NER)         │    │  (Entities + Relations)     │
└──────────────┘    └─────────────────────┘    └──────────────┬──────────────┘
                                                              │
                                                              ▼
┌──────────────┐    ┌─────────────────────┐    ┌─────────────────────────────┐
│    Answer    │◀───│  Answer Generation  │◀───│    Subgraph Expansion       │
│              │    │       (LLM)         │    │  (Graph Traversal)          │
└──────────────┘    └─────────────────────┘    └──────────────┬──────────────┘
                              ▲                               │
                              │                               ▼
                    ┌─────────┴───────────┐    ┌─────────────────────────────┐
                    │  Passage Retrieval  │◀───│      LLM Reranking          │
                    │                     │    │  (Filter relevant relations)│
                    └─────────────────────┘    └─────────────────────────────┘
```

### Example

**Indexing:** Given document *"Einstein developed the theory of relativity at Princeton."*
- Extracted entities: `Einstein`, `theory of relativity`, `Princeton`
- Extracted relations: `(Einstein, developed, theory of relativity)`, `(Einstein, worked at, Princeton)`

**Query:** *"What did Einstein develop?"*
1. Extract query entity: `Einstein`
2. Vector search finds similar entities and relations from Milvus
3. Subgraph expansion traverses the graph to collect candidate relations
4. **LLM reranking** (key step): ranks all candidate relations by relevance to the question, selects `(Einstein, developed, theory of relativity)` over `(Einstein, worked at, Princeton)`
5. Retrieve source passage → Generate answer: *"Einstein developed the theory of relativity."*

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Installation

```bash
# Install Python dependencies
uv sync --extra api

# Install frontend dependencies
cd frontend && npm install
```

### Running the Application

#### Backend API

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

#### Frontend

```bash
cd frontend
npm run dev -- --host 0.0.0.0
```

The frontend will be available at `http://localhost:5173` (or next available port).

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MILVUS_URI` | Milvus connection URI | `./vector_graph_rag.db` |
| `MILVUS_TOKEN` | Milvus authentication token | None |
| `MILVUS_DB` | Milvus database name | None |
| `OPENAI_API_KEY` | OpenAI API key for LLM and embeddings | Required |

## API Reference

Base URL: `http://localhost:8000`

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "message": "Vector Graph RAG API is running"
}
```

### List Graphs

```
GET /graphs
```

List all available graphs (datasets) in Milvus.

**Response:**
```json
{
  "graphs": [
    {
      "name": "my_dataset",
      "entity_collection": "my_dataset_vgrag_entities",
      "relation_collection": "my_dataset_vgrag_relations",
      "passage_collection": "my_dataset_vgrag_passages",
      "has_all_collections": true
    }
  ],
  "milvus_config": {
    "uri": "./vector_graph_rag.db",
    "database": null,
    "has_token": false
  }
}
```

### Get Graph Statistics

```
GET /stats?graph_name={graph_name}
```

**Query Parameters:**
- `graph_name` (optional): Name of the graph to query

**Response:**
```json
{
  "entities": 100,
  "relations": 250,
  "passages": 50
}
```

### Query Knowledge Graph

```
POST /query
```

Query the knowledge graph and get an answer with the retrieved subgraph.

**Request Body:**
```json
{
  "question": "What is the relationship between A and B?",
  "graph_name": "my_dataset",
  "entity_top_k": 10,
  "relation_top_k": 10,
  "entity_similarity_threshold": 0.9,
  "relation_similarity_threshold": -1.0,
  "expansion_degree": 1
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | Required | The question to ask |
| `graph_name` | string | null | Graph/dataset to query |
| `entity_top_k` | int | 10 | Max entities to retrieve |
| `relation_top_k` | int | 10 | Max relations to retrieve |
| `entity_similarity_threshold` | float | 0.9 | Min similarity for entities |
| `relation_similarity_threshold` | float | -1.0 | Min similarity for relations |
| `expansion_degree` | int | 1 | Graph expansion hops |

**Response:**
```json
{
  "question": "What is the relationship between A and B?",
  "answer": "A and B are connected through...",
  "query_entities": ["A", "B"],
  "subgraph": {
    "entity_ids": ["e1", "e2"],
    "relation_ids": ["r1"],
    "passage_ids": ["p1"],
    "entities": [
      {
        "id": "e1",
        "name": "Entity A",
        "relation_ids": ["r1"],
        "passage_ids": ["p1"]
      }
    ],
    "relations": [
      {
        "id": "r1",
        "text": "A is related to B",
        "subject": "A",
        "predicate": "related_to",
        "object": "B",
        "entity_ids": ["e1", "e2"],
        "passage_ids": ["p1"]
      }
    ],
    "passages": [
      {
        "id": "p1",
        "text": "The passage text..."
      }
    ],
    "expansion_history": []
  },
  "retrieved_passages": ["The passage text..."],
  "stats": {
    "entities": 2,
    "relations": 1,
    "passages": 1
  },
  "retrieval_detail": {
    "entity_ids": ["e1", "e2"],
    "entity_texts": ["Entity A", "Entity B"],
    "entity_scores": [0.95, 0.92],
    "relation_ids": ["r1"],
    "relation_texts": ["A is related to B"],
    "relation_scores": [0.88]
  },
  "rerank_result": {
    "selected_relation_ids": ["r1"],
    "selected_relation_texts": ["A is related to B"]
  }
}
```

### Add Documents

```
POST /add_documents
```

Add documents to the knowledge graph.

**Request Body:**
```json
["Document 1 text...", "Document 2 text..."]
```

**Response:**
```json
{
  "status": "ok",
  "message": "Added 2 documents"
}
```

## CLI Usage

```bash
# Build knowledge graph from documents
uv run vector-graph-rag build --input docs.txt --dataset my_dataset

# Query the knowledge graph
uv run vector-graph-rag query --question "What is X?" --dataset my_dataset
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│  FastAPI    │────▶│   Milvus    │
│   (React)   │     │  Backend    │     │  (Vector DB)│
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   OpenAI    │
                    │  (LLM/Emb)  │
                    └─────────────┘
```

## Evaluation Results

We evaluated Vector Graph RAG on three multi-hop QA datasets, comparing against Naive RAG baseline and state-of-the-art methods from HippoRAG papers.

### Recall@5 Comparison

| Method | MuSiQue | HotpotQA | 2WikiMultiHopQA | Average |
|--------|---------|----------|-----------------|---------|
| Naive RAG | 55.6% | 90.8% | 73.7% | 73.4% |
| IRCoT + HippoRAG¹ | 57.6% | 83.0% | <u>93.9%</u> | 78.2% |
| NV-Embed-v2² | 69.7% | <u>94.5%</u> | 76.5% | 80.2% |
| HippoRAG 2² | **74.7%** | **96.3%** | 90.4% | <u>87.1%</u> |
| Vector Graph RAG | <u>73.0%</u> | **96.3%** | **94.1%** | **87.8%** |

¹ From [HippoRAG (NeurIPS 2024)](https://arxiv.org/abs/2405.14831) - best result with IRCoT + ColBERTv2
² From [HippoRAG 2 (2025)](https://arxiv.org/abs/2502.14802)

See [evaluation/README.md](evaluation/README.md) for detailed reproduction steps

## License

MIT
