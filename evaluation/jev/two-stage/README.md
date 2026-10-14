> Historical October 8, 2026 snapshot. For the current Luna/Flash comparison and figure, see [current results](../current/README.md). The older model scores and timing estimates below are retained for traceability.

# Two-stage Jev: full MuSiQue and 2Wiki evaluation

October 2026 — expanding on the initial Jev evaluation from September 2026.

Jev first ranks candidate relations, then ranks the passages reached through those relations together with direct vector-search candidates. This evaluation covers all 1,000 questions in each dataset. Recall@5 is reported in percent; the average weights each dataset equally.

| Method | MuSiQue | 2Wiki | Average |
|---|---:|---:|---:|
| Naive RAG · BGE-large-en-v1.5 | 58.03 | 72.73 | 65.38 |
| Naive RAG · ColBERTv2 | 49.20 | 68.20 | 58.70 |
| Naive RAG · NV-Embed-v2 | 69.70 | 76.50 | 73.10 |
| HippoRAG · ColBERTv2 | 51.90 | 89.10 | 70.50 |
| IRCoT + HippoRAG | 57.60 | 93.90 | 75.75 |
| HippoRAG 2 | 74.70 | 90.40 | 82.55 |
| Vector Graph RAG + GPT-4o-mini | 64.41 | 91.35 | 77.88 |
| Vector Graph RAG + GPT-5-mini | 73.32 | 93.80 | 83.56 |
| Vector Graph RAG + Jev · relations only | 69.30 | 90.78 | 80.04 |
| **Vector Graph RAG + Jev · two-stage** | **76.78** | **95.35** | **86.07** |

The original figure is available in [Git history](https://github.com/zilliztech/vector-graph-rag/blob/e32c077/docs/assets/evaluation/quality-latency.png).

## Reaching the quality–latency Pareto frontier

**Two-stage Jev delivers the highest Recall@5 in this comparison on both datasets.** Its 86.07% average improves on relation-only Jev by **6.03 percentage points**, Vector Graph RAG + GPT-5-mini by **2.51 points**, and HippoRAG 2 by **3.52 points**.

The extra passage-scoring stage raises recorded mean model-call time from **2.28 to 3.13 seconds**—about **0.85 seconds** for the recall gain. Relation selection finds candidate evidence through the graph; passage reranking then evaluates that evidence in its full source context, together with direct-search candidates. This separates finding relevant relations from deciding which passages belong in the final result.

Under the figure's reference latency estimates, two-stage Jev is the **highest-quality point on the Pareto frontier**. No plotted alternative matches or exceeds its recall while taking no more additional model-call time, with a strict improvement in one dimension. This does not mean every application should choose it: direct retrieval adds no judgment calls, and relation-only Jev spends less time at lower recall.

The frontier compares additional model-call latency, not end-to-end search time. Jev values are recorded means and generative-model values are estimates; the [latency section](#latency-and-cost) specifies their scope and assumptions.

## Frozen recipe

- Model: `jev-1.13.0`; exact prompts are in [`llm/jev.py`](../../../src/vector_graph_rag/llm/jev.py).
- Score the full candidate relation table with shared context; retain the top 64, breaking ties by original order.
- Expand relations to their source passages round-robin, preserving each relation's passage order.
- Alternate graph passages with the dense retrieval top 10; deduplicate full text and cap the union at 16.
- Score complete source passages in shared context, sort scores stably and evaluate the final top 5/10. No RRF, score threshold or text truncation. A source passage is not an entire article.

The frozen graph relation candidates use historical Contriever retrieval; the direct-passage branch uses BGE-large-en-v1.5. This is a hybrid retrieval evaluation, not a controlled comparison of embedding models. The general application uses its configured embedding provider for both branches; rebuilding an index can change the candidate pool and scores.

## Comparison basis

MuSiQue and 2Wiki question IDs, question order and corpus contents were checked against the corresponding HippoRAG 2 reproduction data. HotpotQA is excluded here because its published comparison uses a resampled question set. The original three-dataset results remain [historical results](../../../docs/evaluation.md#historical-results).

GPT results replay saved selections with corrected relation-to-passage ordering. MuSiQue matches all 1,000 original prompts; 2Wiki matches 999. For index 685, both GPT baselines retain the historical saved result (Recall@5 = 1), rather than claiming a corrected replay for that row. The difference cannot reverse the reported ranking. No new GPT calls were made.

External aggregates come from [HippoRAG Tables 2–3](https://arxiv.org/abs/2405.14831) and [HippoRAG 2 Table 3](https://arxiv.org/abs/2502.14802); these systems were not rerun. The BGE baseline and both Jev methods have full per-query outputs. Earlier exploration used some of these queries, so this is an expanded public-benchmark evaluation, not a previously untouched holdout.

Recall counts distinct recovered supporting evidence: full passages for MuSiQue, titles for 2Wiki. Duplicate titles do not receive repeated credit. The regular evaluator now uses this same distinct-evidence definition; historical tables are not retroactively rewritten.

## Latency and cost

The figure shows **additional model-call latency**, excluding embeddings, retrieval and final answer generation. Naive RAG makes no additional decision call and is at zero on this axis, not zero end-to-end time. HippoRAG and IRCoT lack comparable latency estimates and remain in the quality table only.

Jev's 3.13 seconds is the equal-weight mean of recorded per-query request-duration sums from 500 MuSiQue and 900 2Wiki queries. Relation batches in this run were sequential within each query; three queries ran concurrently. Cached reads are never timed as requests. The library retains configurable request concurrency, so production wall time may differ. The old relation-stage mean is 2.28 seconds on that same timing basis.

Generative model points are illustrative range midpoints, not measured averages: GPT-4o-mini 7 s (4–10), GPT-5-mini 20 s (10–30), HippoRAG 2 with Llama-3.3-70B 8.25 s (1.5–15). The plotted frontier is conditional on these reference times. See the [historical assumptions and sources](../api-cost-latency.md); no controlled end-to-end speed claim is made.

At the recorded $0.042/M input-token assumption, two-stage Jev averages approximately $2.43 per 1,000 queries across the two timing subsets, equally weighted. This is not the incremental bill for completing the experiment. Revised GPT input counts average 3,871.631 tokens: using the historical price assumptions and 500 output tokens gives about $0.88/1,000 for GPT-4o-mini; 500–2,000 output tokens gives $1.97–4.97 for GPT-5-mini. These estimates do not establish that Jev is always cheaper. The old 20,000-input-token scenario is superseded for this comparison.

## Recompute without API calls

```bash
uv run python evaluation/jev/two-stage/summarize.py
uv run --extra evaluation python evaluation/jev/two-stage/plot.py
```

`results.json.gz` holds 10,000 method/query records with ordered SHA-256 evidence fingerprints and metrics. `manifest.json` identifies all queries and dataset file hashes. Scoring can be independently recomputed; fingerprints do not reconstruct candidate generation or model calls. Raw model caches, local databases and exploratory variants are not included.

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
