# Guides

These guides cover production-oriented workflows that go beyond the first quick start.

| Guide | Use it when you need to... |
|---|---|
| [Document Import](document-import.md) | Load files, URLs, Markdown, PDFs, DOCX files, or use parser adapters such as Docling and MinerU. |
| [Incremental Updates](incremental-updates.md) | Create, update, or delete one external source without rebuilding unrelated graph data. |
| [Observability](observability.md) | Export OpenTelemetry traces and connect them to systems such as Azure Application Insights. |
| [Embedding Providers](embedding-providers.md) | Choose and configure OpenAI, HuggingFace, Google, Voyage, Jina, Mistral, Ollama, local, or ONNX embeddings. |
| [Multi-Tenancy](multi-tenancy.md) | Isolate datasets with collection prefixes, Milvus databases, API graph names, and metadata filters. |
| [Frontend Visualization](frontend.md) | Run and customize the React graph visualization UI. |

For exact constructor arguments, return types, and method signatures, see the [Python API Reference](../reference/python-api.md).

## Relation reranking

Use the default model or opt into [Jev relation scoring](reranking.md), including threshold selection and passage fallback.
