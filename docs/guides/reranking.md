# Models and reranking

Choose models by task. The default is `gpt-6-luna` for extraction, generative reranking and answer generation; task-specific values override only that task. GPT-6 models use the Responses API with `medium` reasoning by default. Explicit non-GPT-6 model configurations retain the Chat Completions path.

```python
from vector_graph_rag import VectorGraphRAG

rag = VectorGraphRAG(
    llm_model="gpt-6-luna",
    reranker_model="jev",
)
```

Jev is included in the base installation; no extra package or model weights are needed. Set `TYPESAFE_API_KEY` for Jev and retain your existing embedding and generation credentials. Until a release containing this change is published, install from the merged source:

```bash
uv add 'vector-graph-rag @ git+https://github.com/zilliztech/vector-graph-rag.git'
export TYPESAFE_API_KEY=your-typesafe-key
export OPENAI_API_KEY=your-generation-key
```

## Separate task models

```python
rag = VectorGraphRAG(
    extractor_model="gpt-6-luna",
    reranker_model="jev-1.13.0",
    answer_model="gpt-6-luna",
)
```

| Parameter | Role | When omitted |
|---|---|---|
| `llm_model` | Shared default generative model | `gpt-6-luna` |
| `extractor_model` | Document triplets and query entities | Inherits `llm_model` |
| `reranker_model` | Relation reranking, plus passage reranking for Jev | Inherits `llm_model` |
| `answer_model` | Final answer generation | Inherits `llm_model` |

A non-Jev `reranker_model` selects generative relation reranking. `jev` and `jev-*` names select the Jev two-stage pipeline; there is no strategy switch. The `jev` alias currently resolves to `jev-1.13.0` (a project recommendation, not an official latest-model alias). Pin the full name for reproducible runs. The [Jev 1.14 evaluation](../../evaluation/jev/two-stage/README.md#jev-114-paired-comparison) adds a full benchmark result via OpenRouter; it does not change this runtime alias or the direct-service configuration. To follow the official stable release, use `reranker_model="jev-latest"`; `jev-preview` is also accepted. These are the [official aliases](https://docs.typesafe.ai/models), not the bare string `latest`. Moving aliases bypass the local response cache so a previous release cannot silently supply the result. Jev extraction and answer generation are not supported and are rejected during configuration.

`retrieve()` skips answer generation; `query()` generates an answer from the same selected passages. Both still perform query entity extraction. The facade constructs all task clients, so their credentials must be configured even when only retrieval is requested.

## Custom services and credentials

Use a model name for the shared connection, or `ModelConfig` for a task-specific OpenAI-compatible service:

```python
import os
from vector_graph_rag import ModelConfig, VectorGraphRAG

rag = VectorGraphRAG(
    llm_model=ModelConfig(
        model="your-default-model",
        base_url="https://your-service.example/v1",
        api_key=os.environ["MY_MODEL_API_KEY"],
    ),
    reranker_model="jev",
    answer_model=ModelConfig(
        model="gpt-6-luna",
        base_url="https://your-answer-service.example/v1",
        api_key=os.environ["ANSWER_API_KEY"],
    ),
)
```

A string task model inherits the default generation connection. A task object may override its connection; when it specifies a different URL, provide its API key explicitly rather than inheriting a secret for another service. Jev always resolves its own URL and credential, never the generative connection. A Jev `ModelConfig` can specify a custom TypeSafe-compatible base URL and API key; the client appends `/systemone`.

Embedding configuration remains independent (`embedding_provider`, `embedding_model`, `embedding_api_key`, `embedding_base_url`). Legacy `openai_api_key` / `openai_base_url` remain shared defaults for generation and OpenAI embeddings. Changing a task's connection does not redirect embedding requests. Custom generative services must support the chat-completion features used by this project.

## Environment configuration and compatibility

The same fields work through `Settings` and `VGRAG_` environment variables:

```bash
export VGRAG_LLM_MODEL=gpt-6-luna
export VGRAG_RERANKER_MODEL=jev
export VGRAG_EXTRACTOR_MODEL=gpt-6-luna
export VGRAG_ANSWER_MODEL=gpt-6-luna
```

Task environment values can also be JSON objects with `model`, `base_url` and `api_key`. Avoid putting real credentials in shell history. `ModelConfig.api_key` is redacted in representations and serialization. The legacy OpenAI key field retains its existing behavior.

Existing `reranker_provider="jev"` and `jev_model=...` settings are still accepted and now select two stages. Conflicting old/new provider or model settings raise an error. `jev_threshold` is retained for loading old configurations but is no longer used: relation selection takes the top 64 and document selection follows the final rank. The `[jev]` installation extra remains an empty compatibility alias. Default generative reranking behavior is unchanged.

## Two-stage retrieval

```text
Expanded relations → Jev scores → top 64 → round-robin source passages
                                                    ↓
Direct vector-search top 10 → alternate and deduplicate → at most 16
                                                    ↓
                                Jev full-passage scores → final top K
```

For example, a company-founder relation can lead to a passage identifying the founder, while another passage supplies the founder's birthplace. Relation scoring gathers possible bridges; passage scoring checks their actual supporting text. Dense candidates provide a second route when a useful passage is poorly represented by extracted relations. No RRF or score threshold is applied. Ties preserve input order.

Metadata filters apply to both candidate branches. Passage text is never silently truncated. Requests share context but scores are independent judgments, not a listwise ranking objective. The exact prompts and criteria are in [`llm/jev.py`](https://github.com/zilliztech/vector-graph-rag/blob/main/src/vector_graph_rag/llm/jev.py).

## Batching, caching and errors

Relation questions use the existing token-budgeted batching; each batch repeats the shared graph. The second stage sends at most 16 complete passages. Proxy token limits reject oversized inputs with an error; they do not claim to be the provider's official tokenizer or context limit. Use smaller source chunks when needed. The candidate budget is 16, so a larger requested `top_k` cannot produce more than 16 results.

`VGRAG_JEV_TIMEOUT` defaults to 60 seconds and `VGRAG_JEV_MAX_CONCURRENCY` to 3. Transient transport and retryable status failures receive up to three attempts. Invalid scores, authentication errors and exhausted credit propagate as failures rather than empty successful retrievals. Cache keys include complete prompts and model names; custom endpoints are separated. Cache reads do not represent model latency.

## Evaluation

The [full evaluation](https://github.com/zilliztech/vector-graph-rag/blob/main/evaluation/jev/two-stage/README.md) covers MuSiQue and 2Wiki, 1,000 questions each. It includes model comparisons, method details and the timing scope. Frozen evaluation candidates use historical Contriever graph retrieval plus BGE dense retrieval; a fresh index with a different configuration need not reproduce the exact scores.

## Runtime API settings

Set `VGRAG_LLM_REASONING_EFFORT` (or `Settings.llm_reasoning_effort`) to change the GPT-6 Responses effort; the default is `medium`. This setting applies to GPT-6 tasks, and is included with the API route in response-cache identities. GPT-6 requests omit `temperature` and `stop`, read only final message text, and reject incomplete, refused or empty responses before caching. Existing task error handling still applies. Non-GPT-6 providers retain their Chat Completions behavior and must accept that API's request format.

Custom GPT-6 endpoints must support `/responses`; no silent fallback to Chat Completions is performed. See the [official Luna model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna). These runtime settings do not retroactively change the frozen evaluation recipe or its published scores.
