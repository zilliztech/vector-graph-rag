# Relation reranking

The default reranker uses `Settings.llm_model` (GPT-4o-mini by default). Jev is an optional alternative for scoring candidate relations after graph expansion. Entity/triplet extraction and answer generation continue to use the configured OpenAI-compatible model.

## Enable Jev

```bash
uv add 'vector-graph-rag[jev]'
export TYPESAFE_API_KEY='your-typesafe-key'
export OPENAI_API_KEY='your-openai-key'
```

```python
from vector_graph_rag import VectorGraphRAG
from vector_graph_rag.config import Settings

rag = VectorGraphRAG(settings=Settings(
    reranker_provider="jev",
    jev_model="jev-1.13.0",
    jev_threshold=0.5,
))
# Add documents or connect to an existing populated index first.
result = rag.retrieve("Which city was the founder of this company born in?", top_k=5)
```

For an existing application, set `VGRAG_RERANKER_PROVIDER=jev`. Other environment settings are `VGRAG_JEV_MODEL`, `VGRAG_JEV_THRESHOLD`, `VGRAG_JEV_TIMEOUT` (60 seconds), and `VGRAG_JEV_MAX_CONCURRENCY` (3). `VGRAG_JEV_API_KEY` overrides the `TYPESAFE_API_KEY` fallback. The key is stored as a Pydantic secret and is excluded from the settings representation. The optional extra declares `httpx`; no additional model weights are downloaded.

## Selection and passage ordering

Each request shares the user question and the full candidate relation table in `state`. Its `questions` map contains one `noul` score request per relation, including the relation text, its ID, and explicit true/false criteria. The criterion accepts facts directly answering the question **and necessary intermediate links in a multi-hop answer**. Shared context does not imply a listwise ranking objective or competition between scores.

For example, for a question asking where a company's founder was born, `company → founded by → person` can be a useful intermediate link alongside `person → born in → city`. A relation merely sharing the company's topic is insufficient. The shipped recipe is zero-shot: these examples explain the behavior here, but are not added to the request. The exact evaluated prompt lives in [`llm/jev.py`](https://github.com/zilliztech/vector-graph-rag/blob/main/src/vector_graph_rag/llm/jev.py).

Scores are sorted descending, with original candidate order breaking ties. All relations with scores **greater than or equal to 0.5** are retained by default; there is no forced top-five relation count. Relations expand into passages in the selected relation order, retaining each passage on its first occurrence. `retrieve()` fills short results with deduplicated vector-search passages, after graph passages; `query()` does the same when Jev reranking is enabled. The final `top_k` limits passages, not relations. Metadata filters apply to both graph expansion into passages and fallback retrieval.

## Batching, caching, and failures

Large question maps are split into token-budgeted requests, each repeating the full shared state. There is no silent truncation of relations or text. The implementation uses `cl100k_base` as a proxy with conservative limits (23k for shared state plus a question, 38k batch budget, and a final 40k payload check). These are implementation safety margins, not claims about Jev's tokenizer or official context window. Oversized inputs raise an error; reduce candidate retrieval limits explicitly if needed.

At most three requests run concurrently by default. Transient transport failures and selected retryable HTTP statuses receive up to three attempts. Authentication, exhausted credit, missing scores, and invalid scores raise errors rather than masquerading as an empty successful selection. An actual empty selection can still use passage fallback.

Validated responses use the existing local response cache when `use_llm_cache=True`. Cache keys include the model, complete state, instructions and criteria; changing only the threshold reuses raw scores. Cache files contain query and relation-derived responses, so apply the same access controls as for the existing response cache.

## Evaluation and compatibility

See the [evaluation results](../evaluation.md#jev-reranker-evaluation) for the frozen 500-row comparisons and API cost/latency scenarios. Jev is opt-in and the default model prompt is unchanged.

A relation-to-passage ordering fix applies to **both** rerankers: database ID lookups do not guarantee input order, so fetched relations are restored to the reranker's order before passage expansion. This can change which passages appear in top-k results, even with identical cached model responses. Historical published tables are retained as historical results; new evaluations use the corrected order and are reported separately.
