# Jev relation-reranker evaluation

This evaluation compares Jev (`jev-1.13.0`, shared graph context, zero-shot, score ≥ 0.5) with cached GPT-4o-mini and GPT-5-mini relation selections and a shared BGE-large-en-v1.5 passage-retrieval baseline. Each method is evaluated on the same 500 rows per dataset. The default reranker remains unchanged; see the [Jev usage guide](../../docs/guides/reranking.md).

## Final results

Scores are macro-averaged evidence Recall@k, in percent. MuSiQue matches full supporting passage text; HotpotQA matches supporting titles, using the existing evaluator's normalization and recall formula.

| Method | MuSiQue R@5 | MuSiQue R@10 | HotpotQA R@5 | HotpotQA R@10 |
|---|---:|---:|---:|---:|
| Naive RAG · BGE-large-en-v1.5 | 58.05 | 67.60 | 88.60 | 93.20 |
| Vector Graph RAG + GPT-4o-mini | 64.08 | 74.00 | 90.90 | 96.30 |
| Vector Graph RAG + GPT-5-mini | 73.00 | 79.00 | 94.50 | 97.30 |
| Vector Graph RAG + Jev | 68.87 | 76.43 | 93.50 | 97.20 |

Jev is between the two model baselines on Recall@5: 4.13 percentage points below GPT-5-mini on MuSiQue and 1.00 point below it on HotpotQA. Its gap to GPT-5-mini at Recall@10 is 2.57 and 0.10 points, respectively. This supports Jev as an optional tradeoff, not a replacement that dominates retrieval quality.

![Retrieval comparison](../../docs/assets/evaluation/retrieval-comparison.png)

The circles compare our same-row evaluations. Diamonds are **published 1,000-question aggregates**, not reruns on our 500 rows. They provide context; differences against them cannot isolate the reranker effect. Their embeddings, graph construction and sample sets differ. In particular, Naive ColBERTv2 and Naive BGE are different retrieval systems. Published sources: [HippoRAG](https://arxiv.org/html/2405.14831v2) and [HippoRAG 2](https://arxiv.org/html/2502.14802v1).

Intervals are 95% percentile intervals from 10,000 query-ID cluster bootstrap resamples, seed 20260921. They describe sample uncertainty, not model-run variability, per-query score ranges, or uncertainty in published points. MuSiQue has 500 distinct IDs; HotpotQA has 490 distinct IDs in 500 rows, so repeated IDs are resampled together while the point estimate remains row-weighted.

## Query categories

These are the datasets' native hop/type categories, not inferred difficulty labels. Scores below are Recall@5 (%); `summary.json` also contains Recall@10 for every category.

| Dataset / category | Rows | Naive BGE | GPT-4o-mini | GPT-5-mini | Jev |
|---|---:|---:|---:|---:|---:|
| MuSiQue · 2-hop | 259 | 67.95 | 75.87 | 82.24 | 79.15 |
| MuSiQue · 3-hop | 158 | 55.70 | 59.28 | 72.15 | 66.67 |
| MuSiQue · 4-hop | 83 | 31.63 | 36.45 | 45.78 | 40.96 |
| HotpotQA · bridge | 398 | 85.80 | 89.20 | 93.09 | 92.34 |
| HotpotQA · comparison | 102 | 99.51 | 97.55 | 100.00 | 98.04 |

## Protocol and compatibility with older tables

The sample selection retained the earlier 100-row evaluation subsets and filled each dataset to 500 with a seeded, category-stratified selection from the existing 1,000-row files. `manifest.json` records exact row indices, IDs, questions and categories; an ID alone is not a unique HotpotQA row. MuSiQue includes queries used during prompt exploration, so this is **not an untouched holdout**. Only the final frozen recipe and results are included here.

The comparisons freeze the historical Contriever-derived relation candidate pools and reuse hash-matched GPT-4o-mini/GPT-5-mini responses. Jev sees the same candidate relations. All graph methods then use the corrected relation-to-passage expansion and the same copied BGE passage index for naive fallback. This is a **reranker replay with a shared downstream pipeline**, not a newly rebuilt, single-embedding end-to-end benchmark. No new GPT calls were made for these tables. Candidate counts average 374.7 for MuSiQue and 305.2 for HotpotQA, reaching the configured cap of 1,000 on 90 and 47 rows respectively.

The ordering correction restores the selected relation order after Milvus's unordered ID lookup, before expanding and deduplicating passage IDs. The first encountered occurrence of a passage wins. Short graph results are filled with vector-search passages; the final result is truncated to ten passages for computing R@5 and R@10. A change in relation order changes Recall@k only when it changes the evidence inside that k boundary. Identical scores before/after a fix do not prove identical rankings.

The existing historical three-dataset tables remain unchanged and are explicitly labeled historical. They were produced before this ordering correction; their reported retrieval settings and sample basis must not be silently replaced with this replay's settings. Future runs use the corrected ordering for both rerankers. The default model's selection prompt itself is unchanged. The current Jev release and recipe were checked against all 1,000 frozen query inputs; `validation.json` records payload and selection equivalence and source artifact hashes.

## Reproduce the tables and figures without API calls

From the repository root:

```bash
uv run evaluation/jev/summarize.py
uv run evaluation/jev/plot_retrieval.py
uv run evaluation/jev/plot_api_comparison.py
```

`results.json` contains 4,000 method/query rows. Gold and retrieved evidence are represented by SHA-256 hashes of the evaluator-normalized strings (full passages for MuSiQue, titles for HotpotQA), preserving order and exact equality without republishing corpus text. The summarizer recomputes R@5/R@10 from these fingerprints, validates the sample manifest and regenerates aggregate/category scores and confidence intervals. This reproduces scoring and chart construction; fingerprints do not reproduce embeddings or independently verify corpus normalization. Raw candidate caches, private API response caches, databases and exploratory scripts are not distributed in this directory.

## Run a new evaluation

Use the existing dataset/index setup in [Evaluation](../README.md), then select Jev through the regular evaluator. Run from the repository root:

```bash
uv sync --extra jev --extra hf
export TYPESAFE_API_KEY='your-typesafe-key'
export OPENAI_API_KEY='your-openai-key'
uv run python evaluation/evaluate.py \
  --dataset musique --data-dir evaluation/data \
  --sample-manifest evaluation/jev/manifest.json \
  --reranker-provider jev --jev-model jev-1.13.0 --jev-threshold 0.5 \
  --method both --top-k 10 --output output/jev-musique.json
```

Use `--dataset hotpotqa` for the other manifest subset. `--max-samples 10` is available for a small smoke run. The evaluator validates row IDs and questions against the manifest before retrieval. To compare another reranker on the same index and rows, use `--reranker-provider llm --llm-model gpt-4o-mini` or `gpt-5-mini` with the same remaining arguments.

This entry point runs fresh retrieval on the configured index; it **does not reconstruct the frozen historical candidate pools** used in the published replay, and identical scores are not promised. Index building, extraction, embeddings and model calls can incur additional charges. Existing evaluator behavior records failed retrievals as empty results and prints the error; inspect logs when running new experiments.

## API cost and latency

![API cost and latency scenarios](../../docs/assets/evaluation/api-cost-latency.png)

These are **API-only planning estimates**, not a controlled latency benchmark or a quality-adjusted cost ranking. Jev's cost uses recorded token usage; other costs use stated hypothetical token counts and public prices. The latency ranges are illustrative scenarios, not confidence intervals.

| Online filtering/reranking | Estimated USD / 1,000 queries | Illustrative seconds / query |
|---|---:|---:|
| HippoRAG 2 + Llama-3.3-70B | 0.26–2.29 | 1.5–15 |
| HippoRAG 2 + GPT-4o-mini (API configuration option) | 0.42 | 2–5 |
| Vector Graph RAG + GPT-4o-mini | 3.30 | 4–10 |
| Vector Graph RAG + GPT-5-mini | 6–9 | 10–30 |
| Vector Graph RAG + Jev | 3.15 | 1.2–4 |

Jev looks attractive against the longer generative reranking prompts, especially GPT-5-mini, but these assumptions **do not establish a cost advantage over HippoRAG 2's shorter filtering prompt**. Autoregressive decoding adds sequential output work; it does not guarantee every provider/model configuration is slower than Jev's repeated-context, multi-request workload.

The Jev estimate uses 56,716,079 input tokens over 756 newly scored queries and 1,881 successful requests: $2.3821 at $0.042/M input tokens, or $3.1509/1,000 queries. The other 244 queries reused prior scores and are excluded from that denominator. Retry/failure overhead is excluded from the chart; a conservative reservation accounting including five failed requests was approximately $2.3958 for the new calls. Per-request median latency was 1.222 seconds, not a per-query latency measurement.

See [cost/latency assumptions and primary sources](api-cost-latency.md) for provider prices, token formulas, cache treatment and speed references. Embeddings, graph/database work, indexing and final answers are excluded. The optional HippoRAG 2 GPT configuration must not be paired with the paper's Llama retrieval score as though it were measured.
