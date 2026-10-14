# Jev retrieval evaluation

Jev evaluation began in September 2026; the full two-stage results below were added in October 2026. The latest comparison includes GPT-6 Luna and DeepSeek V4.1 Flash.

## Current quality–latency comparison

Across **MuSiQue and 2Wiki, 1,000 questions each**, two-stage Jev achieves **86.07% average Recall@5**, compared with **82.70% for GPT-6 Luna** and **83.34% for DeepSeek V4.1 Flash**. Their measured additional model-call means are **3.13 s**, **5.35 s** and **10.97 s**, respectively. Different test conditions and external reference points limit direct speed comparisons.

![Quality and latency](../../docs/assets/evaluation/quality-latency.png)

Luna uses **medium** reasoning: xhigh had similar recall (82.53% versus 82.70%) but higher mean latency (8.86 s versus 5.35 s); Flash uses **low** because high produced very long reasoning and exhausted the output limit on some questions.

The two-stage pipeline adds passage reranking after relation selection. Relative to relation-only Jev, average Recall@5 rises by **6.03 percentage points**, while recorded additional model-call time increases from **2.28 to 3.13 seconds** on the same timing basis. Jev is the highest-quality point on the observed frontier; this is additional model time, not end-to-end search latency.

See the [current evaluation report](current/README.md) for dataset scores, measured timing scope, Flash audit and chart reproduction. Enable the pipeline with `reranker_model="jev"`; the [reranking guide](../../docs/guides/reranking.md) covers configuration.

The [frozen Jev recipe and score reproduction](two-stage/README.md) support the current results. This two-dataset comparison is separate from the project's historical three-dataset average.
