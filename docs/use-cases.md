# Use Cases

Vector Graph RAG is designed for **knowledge-intensive domains** where documents contain dense factual relationships and answers often require connecting information across multiple sources.

## When to Use Graph RAG vs Naive RAG

**Use Naive RAG when:**

- Questions can be answered from a single passage
- Content is self-contained (e.g., FAQ, product docs)
- Low latency is critical and accuracy trade-off is acceptable

**Use Vector Graph RAG when:**

- Answers require connecting facts across multiple documents
- Content has rich entity relationships (people, organizations, concepts)
- Multi-hop reasoning is needed ("Who worked with X at Y?")
- Domain has dense factual knowledge

## Domain Examples

### Legal

Legal documents are full of cross-references: statutes cite other statutes, court opinions reference precedents, contracts refer to defined terms across sections.

```python
rag = VectorGraphRAG(collection_prefix="legal_contracts")

rag.add_texts([
    "Section 3.1 defines the indemnification obligations of the Seller.",
    "Under Section 5.2, breach of Section 3.1 triggers termination rights.",
    "The Buyer may exercise termination rights within 30 days of notice.",
])

result = rag.query("What happens if the Seller breaches indemnification obligations?")
# Graph connects: Seller → indemnification (3.1) → breach triggers termination (5.2) → 30 days
```

### Finance

Financial data forms natural graphs: companies own subsidiaries, executives serve on boards, transactions flow between entities.

```python
rag = VectorGraphRAG(collection_prefix="financial_reports")

rag.add_texts([
    "Berkshire Hathaway acquired See's Candies in 1972 for $25 million.",
    "See's Candies generated $383 million in pre-tax earnings by 2007.",
    "Warren Buffett has called See's the ideal business.",
])

result = rag.query("How has Berkshire's candy acquisition performed?")
# Graph connects: Berkshire → acquired See's → earnings growth → Buffett's assessment
```

### Medical & Biomedical

Drug interactions, symptom-disease-treatment pathways, and clinical trial relationships are inherently relational.

```python
rag = VectorGraphRAG(collection_prefix="medical_literature")

rag.add_texts([
    "Metformin is the first-line treatment for type 2 diabetes.",
    "Patients on metformin should have renal function monitored.",
    "Impaired renal function may require dose adjustment or alternative therapy.",
])

result = rag.query("What monitoring is needed for first-line diabetes treatment?")
# Graph connects: diabetes → metformin (first-line) → renal monitoring → dose adjustment
```

### Literature & Novels

Character relationships, plot events, and thematic connections across chapters benefit from graph representation.

```python
from vector_graph_rag.loaders import DocumentImporter

importer = DocumentImporter(chunk_size=1500, chunk_overlap=200)
result = importer.import_sources(["/path/to/novel.pdf"])

rag = VectorGraphRAG(collection_prefix="novel_analysis")
rag.add_documents(result.documents, extract_triplets=True)

result = rag.query("How does the protagonist's relationship with the antagonist evolve?")
# Graph captures character interactions across the entire novel
```

### Academic Research

Citation networks, concept dependencies, and cross-paper methodology comparisons.

```python
from vector_graph_rag.loaders import DocumentImporter

importer = DocumentImporter(chunk_size=1000, chunk_overlap=200)
result = importer.import_sources([
    "/path/to/paper1.pdf",
    "/path/to/paper2.pdf",
    "/path/to/paper3.pdf",
])

rag = VectorGraphRAG(collection_prefix="research_survey")
rag.add_documents(result.documents, extract_triplets=True)

result = rag.query("What methods achieve the best performance on this task?")
# Graph connects methods, results, and comparisons across papers
```

## Organizing Multiple Knowledge Bases

Use `collection_prefix` to separate different document sets in the same Milvus instance:

```python
# Each domain gets its own isolated graph
legal_rag = VectorGraphRAG(milvus_uri="http://localhost:19530", collection_prefix="legal")
finance_rag = VectorGraphRAG(milvus_uri="http://localhost:19530", collection_prefix="finance")
medical_rag = VectorGraphRAG(milvus_uri="http://localhost:19530", collection_prefix="medical")
```

Or use `milvus_db` for database-level isolation:

```python
rag = VectorGraphRAG(
    milvus_uri="http://localhost:19530",
    milvus_db="production",
    collection_prefix="legal_v2",
)
```
