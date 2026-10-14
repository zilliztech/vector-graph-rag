# Two-stage Jev reaches the quality–latency Pareto frontier

Jev evaluation began in September 2026; the full two-stage results below were added in October 2026. The Luna/Flash comparison was updated on October 14, 2026. MuSiQue and 2Wiki each contain 1,000 questions; averages weight the two datasets equally. The generative rerankers are GPT-6 Luna and official DeepSeek V4.1 Flash.

![Quality and additional model-call latency](../../../docs/assets/evaluation/quality-latency.png)

| Method | MuSiQue R@5 | 2Wiki R@5 | Average R@5 | Mean model-call time |
|---|---:|---:|---:|---:|
| Naive RAG · BGE-large-en-v1.5 | 58.03% | 72.73% | 65.38% | 0 s |
| Naive RAG · ColBERTv2 | 49.20% | 68.20% | 58.70% | 0 s |
| Naive RAG · NV-Embed-v2 | 69.70% | 76.50% | 73.10% | 0 s |
| HippoRAG 2 | 74.70% | 90.40% | 82.55% | ≈8.25 s (reference) |
| Vector Graph RAG + GPT-6 Luna | 72.18% | 93.23% | 82.70% | 5.35 s |
| Vector Graph RAG + DeepSeek V4.1 Flash | 73.20% | 93.48% | 83.34% | 10.97 s |
| Vector Graph RAG + Jev · relations only | 69.30% | 90.78% | 80.04% | 2.28 s |
| **Vector Graph RAG + Jev · two-stage** | **76.78%** | **95.35%** | **86.07%** | **3.13 s** |

Luna uses **medium** reasoning: xhigh had similar recall (82.53% versus 82.70%) but higher mean latency (8.86 s versus 5.35 s); Flash uses **low** because high produced very long reasoning and exhausted the output limit on some questions.

Two-stage Jev has the highest recall among these methods: 6.03 percentage points above relation-only Jev, 3.36 above Luna, and 2.73 above Flash. It is the highest-quality non-dominated point using the displayed timings. The chart computes that frontier from the data; this is not a controlled end-to-end speed comparison.

## Measurement scope

The horizontal axis excludes embedding, retrieval and answer generation. Zero means no extra model call. Jev timings use the recorded timing subsets: equal-weight means of per-query request-duration sums from 500 MuSiQue and 900 2Wiki queries, with three concurrent queries. Luna and Flash use measured full-evaluation API means, not the older generative-model timing estimates. Network, provider queueing, concurrency and measurement dates differ between runs. HippoRAG 2 retains an approximate reference time; its scores and the ColBERTv2/NV-Embed-v2 points are published aggregates rather than new runs. See the [comparison provenance](#comparison-basis).

The frozen candidates, input prompts, relation order and historical scoring are retained for the new generative runs. Their selected relations expand to source passages in order, followed by naive retrieval to fill ten passages. Only message output text and the `useful_relations` JSON field are scored; reasoning text and field aliases are not used. MuSiQue matches distinct full passages and 2Wiki distinct titles. Jev's two-stage recipe is described in the [frozen recipe](#frozen-recipe).

## Flash audit

All 2,000 requests completed with validated payload hashes and unique indices 0–999 per dataset: no retries, truncation or refusals. Seven responses lacked `useful_relations` and used the historical empty-selection/naive fallback without being resent. Forty-one exact-payload successful sample responses were reused; the remaining 1,959 were newly requested. High results are excluded.

Requests used the official Responses API, `model=deepseek-flash`, `reasoning.effort=low`, `max_output_tokens=393216`, and four concurrent requests. This output ceiling includes reasoning and differs from the older proxy experiment's 4,096-token cap. Actual output usage, not the ceiling, determines estimated charges.

| Dataset | R@10 | Mean | P50 | P90 | Maximum |
|---|---:|---:|---:|---:|---:|
| MuSiQue | 79.41% | 14.57 s | 9.24 s | 29.15 s | 648.71 s |
| 2Wiki | 94.73% | 7.37 s | 5.59 s | 12.08 s | 407.18 s |
| Average / pooled timing | 87.07% | 10.97 s | 7.05 s | 20.69 s | 648.71 s |

Low still has long-tail requests. The runner waits for each batch of four, so a slow request can delay the next batch; individual API durations retain that delay rather than treating cache reads as calls. The 41-question sample took 301.42 seconds; the remaining run took 10,647.41 seconds. Including the 567.86-second gap between stages, first request to completion was 11,516.69 seconds.

Reported usage was 8,092,904 input tokens (769,664 cached) and 4,181,868 output tokens, including 3,774,026 reasoning tokens. The fixed off-peak assumptions of CNY 1/M uncached input, 0.02/M cached input and 4/M output give an estimated CNY 24.07, not an account invoice. Reasoning is not charged twice.

## Saved data and chart reproduction

`comparison.csv` contains per-dataset Recall@5/10 aggregates; `flash-summary.json` contains the Flash audit, usage and timing statistics; `chart-data.json` freezes the plotted coordinates. These aggregates reproduce the figure, not the underlying model calls or independent per-query scoring. Raw responses and credentials are not included in the repository. `results.json.gz` and `summarize.py` provide frozen evidence fingerprints and score replay for the naive and both Jev methods; `summary.json` contains their recomputed scores.

```bash
uv run --offline --no-project --python 3.12 --with matplotlib evaluation/jev/two-stage/plot.py
```

The offline command requires matplotlib to be cached locally. The figure omits reasoning-strength labels for readability; the configuration note above identifies the measurements.

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

Earlier model comparisons and their source artifacts are preserved in the [evaluation snapshot](https://github.com/zilliztech/vector-graph-rag/tree/e32c077ccde8c577f9335b7966ebf22d46e18334/evaluation/jev).
