# Jev retrieval evaluation

Jev evaluation began in September 2026; the full two-stage results below were added in October 2026. Jev 1.14 was added on October 16, 2026. The comparison includes GPT-6 Luna and DeepSeek V4.1 Flash.

## Two-stage Jev reaches the quality–latency Pareto frontier

Across **MuSiQue and 2Wiki, 1,000 questions each**, two-stage Jev 1.14 achieves **86.61% average Recall@5**, compared with **82.70% for GPT-6 Luna** and **83.34% for DeepSeek V4.1 Flash**. Their additional model-call means are **≈4.9 s** (estimated direct), **5.35 s** and **10.97 s**, respectively. Different test conditions and external reference points limit direct speed comparisons.

![Quality and latency](../../docs/assets/evaluation/quality-latency.png)

The two-stage pipeline adds passage reranking after relation selection. Jev 1.14 improves average Recall@5 by **0.55 percentage points** over 1.13 on the same 2,000 questions; the additional 1,800 queries beyond the pilot confirm a **0.54-point** improvement. Most of the gain comes from MuSiQue; Recall@10 is essentially unchanged. Both versions remain on the observed Pareto frontier: 1.14 is the highest-quality point, while 1.13 retains its recorded 3.13-second timing. These are separate service runs, not a controlled speed comparison.

See the [full evaluation report](two-stage/README.md) for dataset scores, measured timing scope, Flash audit and chart reproduction. Enable the pipeline with `reranker_model="jev"`; the [reranking guide](../../docs/guides/reranking.md) covers configuration.

The [frozen Jev recipe and score reproduction](two-stage/README.md) support these results. This two-dataset comparison is separate from the project's historical three-dataset average.
