# Incremental Updates

Use source-level incremental updates when your upstream system can tell you
which external object changed. A source can be a file, web page, email message,
database row, or any other stable business object.

In Vector Graph RAG, a LangChain `Document` represents one passage or chunk.
The source object is tracked through metadata, not through `Document.id`.

## Core Contract

Each incremental update call must contain chunks from exactly one source.

```python
from langchain_core.documents import Document

chunks = [
    Document(
        page_content="Q2 revenue increased in the enterprise segment.",
        metadata={
            "source": "file:file-123",
            "page": 1,
            "chunk_index": 0,
        },
    ),
    Document(
        page_content="Renewal rates improved after support response times dropped.",
        metadata={
            "source": "file:file-123",
            "page": 2,
            "chunk_index": 1,
        },
    ),
]

rag.upsert_documents_by_source(chunks)
```

The default grouping field is `metadata["source"]`. You can also pass the
source explicitly:

```python
rag.upsert_documents_by_source(
    documents=chunks,
    source="file:file-123",
)
```

Use a custom field when your application already has a different metadata name:

```python
rag.upsert_documents_by_source(
    documents=chunks,
    source="file-123",
    source_field="file_id",
)

rag.delete_documents_by_source("file-123", source_field="file_id")
```

## Parser Or Loader Output

Document parsing is intentionally outside the core incremental API. Use your
existing loader or parser to produce text chunks, then attach the same stable
source value to every chunk from the same external object.

```python
from langchain_core.documents import Document
from vector_graph_rag import VectorGraphRAG


def parse_file_to_chunks(path: str) -> list[str]:
    """Return text chunks from your parser or document processing pipeline."""
    return [
        "The handbook describes the support escalation policy.",
        "Escalated issues must include the customer impact and owner.",
    ]


def index_file(rag: VectorGraphRAG, path: str, file_id: str) -> None:
    chunks = [
        Document(
            page_content=text,
            metadata={
                "source": f"file:{file_id}",
                "chunk_index": index,
                "parser": "internal",
            },
        )
        for index, text in enumerate(parse_file_to_chunks(path))
    ]

    rag.upsert_documents_by_source(chunks, extract_triplets=True)
```

If your parser also extracts knowledge graph triplets, store them in each
chunk's `metadata["triplets"]` and disable LLM triplet extraction:

```python
chunks = [
    Document(
        page_content="The handbook describes the support escalation policy.",
        metadata={
            "source": "file:file-123",
            "triplets": [
                ["handbook", "describes", "support escalation policy"],
                ["escalated issues", "include", "customer impact"],
            ],
        },
    )
]

rag.upsert_documents_by_source(chunks, extract_triplets=False)
```

## Create, Update, Delete

Use the same upsert API for both create and update.

```python
# Create or replace one source.
rag.upsert_documents_by_source(
    documents=chunks,
    source="file:file-123",
)

# Delete one source.
rag.delete_documents_by_source("file:file-123")
```

When a source already exists, `upsert_documents_by_source()` removes the old
chunks and graph references for that source, then inserts the new chunks. Other
sources under the same collection prefix are preserved.

## Large Graph Metadata

Milvus limits each dynamic metadata field to 65,536 bytes. Vector Graph RAG
keeps serialized graph metadata below a 60,000-byte budget by default. When a
record exceeds the budget, it trims the oldest adjacency IDs and keeps the
newest IDs in insertion order. This can make older graph links unavailable for
retrieval. Configure a lower budget with `VGRAG_MAX_DYNAMIC_FIELD_BYTES` or
`Settings(max_dynamic_field_bytes=...)`; the maximum supported value is 60,000.
If the non-adjacency metadata alone exceeds the configured budget, the write
raises `ValueError` instead of silently dropping user metadata.

For multi-source updates, call the API once per source:

```python
for source, chunks in changed_sources:
    rag.upsert_documents_by_source(chunks, source=source)
```

## Retry Behavior

Source-level updates perform a multi-step cascade across passages, relations,
and entities. Milvus does not provide a transaction that spans those logical
records, so these writes are not transactionally atomic. If the process is
interrupted, queries may observe intermediate state until the update is retried.

The source-level APIs are designed to be retryable. If an upsert or delete
raises an exception, rerun the same operation for the same source.

```python
try:
    rag.upsert_documents_by_source(
        documents=chunks,
        source="file:file-123",
    )
except Exception:
    # Log the failed source in your application job state, then retry it.
    rag.upsert_documents_by_source(
        documents=chunks,
        source="file:file-123",
    )
```

For delete:

```python
try:
    rag.delete_documents_by_source("file:file-123")
except Exception:
    rag.delete_documents_by_source("file:file-123")
```

A production ingestion job should track which source keys succeeded or failed.
If a retry sees no matching passages during delete, the source may already have
been removed by the previous attempt.

## ID Guidance

Use `source` for the external object identity. Use `Document.id` only when you
need to control the passage ID.

If `Document.id` is omitted, `upsert_documents_by_source()` generates stable
passage IDs from the source value and chunk index. If you provide `Document.id`,
make sure it is unique to that passage and does not belong to another source.

Good source values are stable and globally meaningful in your application:

```text
file:file-123
url:https://example.com/docs/pricing
message:message-456
record:account-789
```

Avoid using a chunk ID, page number, or transient parser output path as the
source. Those values identify a passage or a local processing artifact, not the
external object that should be replaced as a unit.

## Metadata And Filtering

Additional metadata is stored on passages and can be used for filtering.

```python
rag.upsert_documents_by_source(
    documents=chunks,
    source="file:file-123",
    metadata={
        "tenant_id": "tenant-a",
        "workspace_id": "finance",
    },
)

result = rag.query(
    "What should escalated issues include?",
    filter='tenant_id == "tenant-a" and workspace_id == "finance"',
)
```

The source metadata is also stored on each passage, so you can filter by source
when needed:

```python
result = rag.query(
    "What should escalated issues include?",
    filter='source == "file:file-123"',
)
```

## Initial Bulk Load

For a first-time corpus load, you can either use `rebuild_documents()` or call
`upsert_documents_by_source()` once per source. Use `rebuild_documents()` when
you intentionally want a full refresh of the collection prefix. Use
`upsert_documents_by_source()` when you want the same ingestion path for initial
load and later CUD events.

```python
for source, chunks in all_sources:
    rag.upsert_documents_by_source(chunks, source=source)
```

Avoid the legacy `add_*` helpers for new ingestion code. They keep their old
full-rebuild behavior and are planned for removal in v1.0.0.
