# Changelog

## 0.4.0

### What's changed

- Default generation tasks now use GPT-6 Luna through Responses with medium reasoning: triplet extraction, entity extraction, relationship reranking and answer generation.
- Cache keys distinguish the Responses interface and reasoning effort. Incomplete, refused or empty Responses outputs are rejected instead of cached as successful answers.
- Retrieval evaluation now compares Jev with GPT-6 Luna and official DeepSeek V4.1 Flash. The report retains the September/October 2026 evaluation timeline, reproducible Jev evidence and a link to superseded experiments.
- Publishing now requires passing tests on Python 3.10 and 3.12, matching package/tag versions, and an isolated wheel installation check.

### Upgrading

The default model changed from GPT-4o-mini to GPT-6 Luna. Custom OpenAI-compatible endpoints must support Responses when using GPT-6 models. Explicitly configured non-GPT-6 models retain Chat Completions; set `VGRAG_LLM_MODEL` explicitly to retain your chosen model. The OpenAI SDK minimum is now 1.66.0.

Reasoning defaults to `medium` and can be configured with `VGRAG_LLM_REASONING_EFFORT`. Existing cache files are preserved, but GPT-6 Responses requests use separate cache keys. DeepSeek is an evaluation baseline, not the runtime default. Frozen benchmark results are unchanged by the runtime migration.
