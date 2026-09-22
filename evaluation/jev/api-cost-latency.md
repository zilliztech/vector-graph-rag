# API cost and latency comparison: assumptions and sources

This is an illustrative planning chart, not a controlled benchmark. It compares only the model API portion of online relation filtering/reranking. It excludes embeddings, graph/database operations, indexing, final answer generation, retries, taxes and gateway credit-purchase fees. It does not compare retrieval quality. Latency spans are neither confidence intervals nor guaranteed service bounds. Cost and latency endpoints need not describe the same provider/configuration.

## Cost scenarios

No cross-request prompt-cache discounts are assumed. Ordinary internal KV-cache use is not an additional line item in these per-token API prices.

- HippoRAG 2: assumed 2,000 input and 200 output tokens per query, including instructions/demonstrations. These counts have not been measured on its evaluation corpus. Llama provider range: DeepInfra FP8 $0.10/$0.32 to Together $1.04/$1.04 per million input/output tokens; calculated $0.264 to $2.288 per 1,000 queries. Provider quantization differs; equal quality is not established. An optional GPT-4o-mini configuration calculates to $0.42 per 1,000 queries and does not inherit the paper's Llama evaluation scores.
- Vector Graph RAG + GPT-4o-mini: assumed 20,000 input and 500 output tokens at $0.15/$0.60 per million; $3.30 per 1,000 queries. These counts are illustrative, not measured token totals from the old cache.
- Vector Graph RAG + GPT-5-mini: assumed 20,000 input and 500–2,000 total billed output tokens, including reasoning, at $0.25/$2.00 per million; $6–9 per 1,000 queries. The latency scenario below assumes medium reasoning; tokens and timing are not linked measurements.
- Jev: 56,716,079 reported input tokens, 756 newly scored queries, 1,881 successful requests. At $0.042 per million input tokens, estimated $2.382075318 total or $3.150893 per 1,000 queries. Output is free. Approximately 75,021 cumulative input tokens and 2.488 requests per query; repeated state across batches is included. Cached/reused queries are excluded from the denominator.

## Latency scenario construction

- HippoRAG 2 + Llama: 1.5–15 seconds is an illustrative fast-to-slower endpoint span for 200 output tokens. Public provider measurements show approximately 288 tokens/s at Groq and 18 tokens/s at DeepInfra, with additional first-token latency. This is not an observed HippoRAG latency distribution and does not encompass all providers or tails.
- HippoRAG 2 + GPT-4o-mini: 2–5 seconds scenario. Public first-party API measurements give approximately 0.9-second TTFT and 140 tokens/s; 200 output tokens imply roughly 2.3 seconds before workload/network variation.
- Vector Graph RAG + GPT-4o-mini: 4–10 seconds scenario. At the same public speed reference, 500 output tokens imply roughly 4.5 seconds; longer input and serving variation motivate the wider planning span.
- Vector Graph RAG + GPT-5-mini: 10–30 seconds scenario for medium reasoning. Public measurements report approximately 12-second time to first answer token and 100 output tokens/s; 500 visible output tokens imply roughly 17 seconds in that separate benchmark. The range is not a measurement of our reranker. High reasoning can be much slower; minimal reasoning can be faster.
- Jev: 1.2–4 seconds scenario derived from our 1.222-second median per request and typical multi-batch scheduling. Approximately 1.2 seconds for concurrent calls versus 3.7 seconds for three sequential calls. This is not a query-level percentile interval, and large candidates/retries can exceed it.

## Sources

Price snapshot checked 2026-09-21:

- HippoRAG 2 paper: https://arxiv.org/html/2502.14802v1
- DeepInfra FP8 price: https://deepinfra.com/meta-llama/Llama-3.3-70B-Instruct-Turbo
- Together price: https://www.together.ai/pricing
- Additional provider rates from gateway's own endpoint listing: https://openrouter.ai/api/v1/models/meta-llama/llama-3.3-70b-instruct/endpoints
- GPT-4o-mini price: https://developers.openai.com/api/docs/models/gpt-4o-mini
- GPT-5-mini price: https://developers.openai.com/api/docs/models/gpt-5-mini
- OpenAI caching: https://developers.openai.com/api/docs/guides/prompt-caching
- Jev price and vendor latency claims: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Primary measurement publisher, Llama endpoints: https://artificialanalysis.ai/models/llama-3-3-instruct-70b/providers/
- Primary measurement publisher, GPT-4o-mini: https://artificialanalysis.ai/models/gpt-4o-mini/
- Primary measurement publisher, GPT-5-mini medium: https://artificialanalysis.ai/models/gpt-5-mini-medium
- Groq official token speed: https://console.groq.com/docs/models
- Community Jev measurements, small fixtures and prompt/schema limitations: https://github.com/WallerChen/jev-measured

For publication alongside the empirical retrieval chart, retain the explicit scenario-estimate label and link this methodology. Do not present these intervals as experimentally measured speedups or a quality-adjusted cost ranking.
