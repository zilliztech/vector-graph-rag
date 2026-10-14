# Jev: frozen recipe and score reproduction

Jev evaluation began in September 2026; the full two-stage results were added in October 2026. This directory contains the Jev and direct-retrieval evidence used in the [evaluation](../current/README.md).

## Frozen recipe

- Model: `jev-1.13.0`; exact prompts are in [`llm/jev.py`](../../../src/vector_graph_rag/llm/jev.py).
- Score the full candidate relation table with shared context; retain the top 64, breaking ties by original order.
- Expand relations to their source passages round-robin, preserving each relation's passage order.
- Alternate graph passages with the dense retrieval top 10; deduplicate full text and cap the union at 16.
- Score complete source passages in shared context, sort scores stably and evaluate the final top 5/10. No RRF, score threshold or text truncation. A source passage is not an entire article.

The frozen graph relation candidates use historical Contriever retrieval; the direct-passage branch uses BGE-large-en-v1.5. This is a hybrid retrieval evaluation, not a controlled comparison of embedding models. The general application uses its configured embedding provider for both branches; rebuilding an index can change the candidate pool and scores.

## Comparison basis

MuSiQue and 2Wiki question IDs, question order and corpus contents were checked against the corresponding HippoRAG 2 reproduction data. Recall counts distinct recovered evidence: full passages for MuSiQue and titles for 2Wiki. Earlier exploration used some of these queries; this is an expanded public-benchmark evaluation rather than an untouched holdout.

External aggregates in the evaluation report come from [HippoRAG Tables 2–3](https://arxiv.org/abs/2405.14831) and [HippoRAG 2 Table 3](https://arxiv.org/abs/2502.14802); those systems were not rerun. HotpotQA is excluded from this comparison because its published comparison uses a resampled question set.

## Recompute without API calls

```bash
uv run --no-project python evaluation/jev/two-stage/summarize.py
```

`results.json.gz` holds 6,000 method/query records for naive retrieval, relation-only Jev and two-stage Jev, including ordered SHA-256 evidence fingerprints and metrics. `manifest.json` identifies the 2,000 queries and dataset file hashes. The script independently checks saved Recall@5/10 against evidence fingerprints and verifies the expected query set for each method. Fingerprints do not reconstruct candidate generation or model calls; raw model caches and local databases are not included.

The [evaluation report](../current/README.md#saved-data-and-chart-reproduction) documents chart reproduction and the saved aggregates for Luna and Flash.

## Run against your index

```bash
uv sync --extra hf
export TYPESAFE_API_KEY=your-key
export OPENAI_API_KEY=your-generation-key
uv run python evaluation/evaluate.py --dataset musique --data-dir evaluation/data \
  --sample-manifest evaluation/jev/two-stage/manifest.json \
  --reranker-model jev-1.13.0 --method graph --top-k 10 --max-samples 10
```

Repeat with `--dataset 2wikimultihopqa`. Remove `--max-samples` for all 1,000 questions. This exercises the shipped two-stage implementation on the configured index; it does not reconstruct the frozen hybrid candidate pools. Indexing, embeddings and model calls can incur charges. Errors must be inspected in evaluation logs.
