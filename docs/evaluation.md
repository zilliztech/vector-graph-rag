# Evaluation

Vector Graph RAG is evaluated on three standard multi-hop QA benchmarks.

## Datasets

| Dataset | Description | Hop Count |
|---------|-------------|-----------|
| **MuSiQue** | Multi-hop questions requiring 2–4 reasoning steps | 2–4 hops |
| **HotpotQA** | Wikipedia-based multi-hop QA | 2 hops |
| **2WikiMultiHopQA** | Cross-document reasoning over Wikipedia | 2 hops |

**Metric:** Recall@5 — whether the ground-truth supporting passages appear within the top-5 retrieved results.

## Results

### Recall@5 vs. Naive RAG

| Method | MuSiQue | HotpotQA | 2WikiMultiHopQA | Average |
|--------|---------|----------|-----------------|---------|
| Naive RAG | 55.6% | 90.8% | 73.7% | 73.4% |
| **Vector Graph RAG** | **73.0%** | **96.3%** | **94.1%** | **87.8%** |
| Improvement | +31.4% | +6.1% | +27.7% | +19.6% |

### Comparison with State-of-the-Art

| Method | MuSiQue | HotpotQA | 2WikiMultiHopQA | Average |
|--------|---------|----------|-----------------|---------|
| HippoRAG (ColBERTv2)¹ | 51.9% | 77.7% | 89.1% | 72.9% |
| IRCoT + HippoRAG¹ | 57.6% | 83.0% | 93.9% | 78.2% |
| NV-Embed-v2² | 69.7% | 94.5% | 76.5% | 80.2% |
| HippoRAG 2² | **74.7%** | **96.3%** | 90.4% | 87.1% |
| **Vector Graph RAG** | 73.0% | **96.3%** | **94.1%** | **87.8%** |

¹ [HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs (NeurIPS 2024)](https://arxiv.org/abs/2405.14831)
² [From RAG to Memory: Non-Parametric Continual Learning for LLMs (2025)](https://arxiv.org/abs/2502.14802)

## Methodology

For fair comparison with HippoRAG, we use **the same pre-extracted triplets** from HippoRAG's repository rather than re-extracting them. This ensures the evaluation isolates the **retrieval algorithm improvements** without interference from triplet extraction quality differences.

## Reproduction

See [`evaluation/README.md`](https://github.com/zilliztech/vector-graph-rag/blob/main/evaluation/README.md) for full reproduction steps.
