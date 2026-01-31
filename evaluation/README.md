# Evaluation

This directory contains the evaluation framework for Vector Graph RAG, comparing it against Naive RAG baseline on multi-hop question answering datasets.

## Background

### What is Multi-hop QA?

Multi-hop question answering requires reasoning over multiple pieces of evidence to arrive at the correct answer. For example:

> Q: "What is the capital of the country where the Eiffel Tower is located?"
>
> This requires: (1) Eiffel Tower → France, (2) France → Paris

Traditional RAG systems often struggle with multi-hop questions because they retrieve passages independently based on surface-level similarity to the query. Graph RAG addresses this by modeling relationships between entities, enabling traversal across connected facts.

### Datasets

We evaluate on three standard multi-hop QA benchmarks:

| Dataset | Description | Hop Count |
|---------|-------------|-----------|
| **MuSiQue** | Multi-hop questions requiring 2-4 reasoning steps | 2-4 hops |
| **HotpotQA** | Wikipedia-based multi-hop QA | 2 hops |
| **2WikiMultiHopQA** | Cross-document reasoning over Wikipedia | 2 hops |

### Evaluation Metric

We use **Recall@K** as the primary metric, measuring whether the ground-truth supporting passages are retrieved within the top-K results. Higher recall indicates better retrieval quality.

## Results

### Recall@5 Comparison with Baselines

| Method | MuSiQue | HotpotQA | 2WikiMultiHopQA | Average |
|--------|---------|----------|-----------------|---------|
| Naive RAG | 55.6% | 90.8% | 73.7% | 73.4% |
| Vector Graph RAG | **73.0%** | **96.3%** | **94.1%** | **87.8%** |
| Improvement | +31.4% | +6.1% | +27.7% | +19.6% |

### Comparison with State-of-the-Art

We compare against methods from HippoRAG papers:

| Method | MuSiQue | HotpotQA | 2WikiMultiHopQA | Average |
|--------|---------|----------|-----------------|---------|
| HippoRAG (ColBERTv2)¹ | 51.9% | 77.7% | 89.1% | 72.9% |
| IRCoT + HippoRAG¹ | 57.6% | 83.0% | <u>93.9%</u> | 78.2% |
| NV-Embed-v2² | 69.7% | <u>94.5%</u> | 76.5% | 80.2% |
| HippoRAG 2² | **74.7%** | **96.3%** | 90.4% | <u>87.1%</u> |
| Vector Graph RAG | <u>73.0%</u> | **96.3%** | **94.1%** | **87.8%** |

¹ [HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs (NeurIPS 2024)](https://arxiv.org/abs/2405.14831)
² [From RAG to Memory: Non-Parametric Continual Learning for LLMs (2025)](https://arxiv.org/abs/2502.14802)

## Reproduction Steps

### Prerequisites

1. Install dependencies:
```bash
uv sync
```

2. Set environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
```

3. Ensure data files are in place:
```
evaluation/data/
├── musique.json
├── musique_corpus.json
├── hotpotqa.json
├── hotpotqa_corpus.json
├── 2wikimultihopqa.json
├── 2wikimultihopqa_corpus.json
└── openie_*_results_*.json  # Pre-extracted triplets
```

### Running Evaluations

#### Graph RAG Evaluation

```bash
# MuSiQue
uv run python evaluation/evaluate.py \
    --dataset musique \
    --method graph \
    --data-dir evaluation/data \
    --milvus-uri ./eval_musique_graph.db \
    --max-samples 300 \
    --force-reindex \
    --entity-top-k 20 \
    --relation-top-k 20 \
    --llm-model gpt-5-mini \
    --embedding-model BAAI/bge-large-en-v1.5 \
    --no-llm-cache

# HotpotQA
uv run python evaluation/evaluate.py \
    --dataset hotpotqa \
    --method graph \
    --data-dir evaluation/data \
    --milvus-uri ./eval_hotpotqa_graph.db \
    --max-samples 300 \
    --force-reindex \
    --entity-top-k 20 \
    --relation-top-k 20 \
    --llm-model gpt-5-mini \
    --embedding-model BAAI/bge-large-en-v1.5 \
    --no-llm-cache

# 2WikiMultiHopQA
uv run python evaluation/evaluate.py \
    --dataset 2wikimultihopqa \
    --method graph \
    --data-dir evaluation/data \
    --milvus-uri ./eval_2wikimultihopqa_graph.db \
    --max-samples 300 \
    --force-reindex \
    --entity-top-k 20 \
    --relation-top-k 20 \
    --llm-model gpt-5-mini \
    --embedding-model BAAI/bge-large-en-v1.5 \
    --no-llm-cache
```

#### Naive RAG Baseline

```bash
# MuSiQue
uv run python evaluation/evaluate.py \
    --dataset musique \
    --method naive \
    --data-dir evaluation/data \
    --milvus-uri ./eval_musique_naive.db \
    --max-samples 300 \
    --force-reindex \
    --llm-model gpt-5-mini \
    --embedding-model BAAI/bge-large-en-v1.5 \
    --no-llm-cache

# HotpotQA
uv run python evaluation/evaluate.py \
    --dataset hotpotqa \
    --method naive \
    --data-dir evaluation/data \
    --milvus-uri ./eval_hotpotqa_naive.db \
    --max-samples 300 \
    --force-reindex \
    --llm-model gpt-5-mini \
    --embedding-model BAAI/bge-large-en-v1.5 \
    --no-llm-cache

# 2WikiMultiHopQA
uv run python evaluation/evaluate.py \
    --dataset 2wikimultihopqa \
    --method naive \
    --data-dir evaluation/data \
    --milvus-uri ./eval_2wikimultihopqa_naive.db \
    --max-samples 300 \
    --force-reindex \
    --llm-model gpt-5-mini \
    --embedding-model BAAI/bge-large-en-v1.5 \
    --no-llm-cache
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name (musique, hotpotqa, 2wikimultihopqa) | Required |
| `--method` | Retrieval method (graph, naive) | graph |
| `--max-samples` | Number of questions to evaluate | 300 |
| `--entity-top-k` | Top-K entities to retrieve | 20 |
| `--relation-top-k` | Top-K relations to retrieve | 20 |
| `--embedding-model` | Embedding model for vector search | BAAI/bge-large-en-v1.5 |
| `--llm-model` | LLM for reranking | gpt-5-mini |
| `--force-reindex` | Rebuild index from scratch | False |
| `--milvus-uri` | Milvus database URI (use .db for local) | Required |

### Output

Results are saved to:
- `output/eval_results_{dataset}.json` - Detailed metrics
- `logs/{dataset}_{timestamp}.log` - Evaluation logs

## File Structure

```
evaluation/
├── README.md           # This file
├── evaluate.py         # Main evaluation script
└── data/
    ├── {dataset}.json           # Questions with ground truth
    ├── {dataset}_corpus.json    # Corpus passages
    ├── openie_*_results_*.json  # Pre-extracted triplets for Graph RAG
    └── ner_cache/               # NER cache files (HippoRAG format)
```
