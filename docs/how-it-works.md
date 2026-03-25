# How It Works

## Overview

Vector Graph RAG processes documents and answers questions through two main pipelines: **indexing** (offline) and **querying** (online).

```mermaid
flowchart TB
    subgraph Indexing["Indexing Pipeline (offline)"]
        direction LR
        D[Documents] --> TE["Triplet\nExtraction"]
        TE --> ER["Entities +\nRelations"]
        ER --> EMB[Embedding]
        EMB --> MV[(Milvus)]
    end

    subgraph Querying["Query Pipeline (online)"]
        direction LR
        Q[Question] --> EE["Entity\nExtraction"]
        EE --> VS["Vector\nSearch"]
        VS --> SE["Subgraph\nExpansion"]
        SE --> LR["LLM\nRerank"]
        LR --> AG[Answer]
    end

    MV -.-> VS
```

---

## Indexing Pipeline

### Step 1: Triplet Extraction

An LLM extracts `(subject, predicate, object)` triplets from each document passage.

```mermaid
flowchart LR
    P["'Einstein developed relativity\nat Princeton.'"] --> LLM["LLM\nExtraction"]
    LLM --> T1["(Einstein, developed, relativity)"]
    LLM --> T2["(Einstein, worked at, Princeton)"]
```

!!! example "Input → Output"
    **Input passage:** *"Einstein developed the theory of relativity at Princeton."*

    **Extracted triplets:**

    | Subject | Predicate | Object |
    |---------|-----------|--------|
    | Einstein | developed | theory of relativity |
    | Einstein | worked at | Princeton |

### Step 2: Knowledge Graph Construction

Triplets are decomposed into three types of objects, each stored as a separate Milvus collection:

| Collection | Contents | What Gets Embedded |
|------------|----------|--------------------|
| **Entities** | Unique entity names | Entity name text |
| **Relations** | Triplet text + linked entity IDs | Full relation text (e.g., "Einstein developed relativity") |
| **Passages** | Original document text + linked entity/relation IDs | Passage text |

```mermaid
graph TD
    subgraph Milvus
        E1["Entity: Einstein"] --- R1["Relation: Einstein developed relativity"]
        R1 --- E2["Entity: relativity"]
        E1 --- R2["Relation: Einstein worked at Princeton"]
        R2 --- E3["Entity: Princeton"]
        R1 -.- P1["Passage: 'Einstein developed...'"]
        R2 -.- P1
    end
```

!!! info "Cross-referencing"
    Each relation stores the IDs of its subject and object entities. Each passage stores the IDs of entities and relations extracted from it. This cross-referencing enables the subgraph expansion step during querying.

### Step 3: Embedding and Indexing

All text is embedded using the configured embedding model (default: `text-embedding-3-large`) and indexed in Milvus for vector similarity search.

---

## Query Pipeline

### Step 1: Entity Extraction

Key entities are extracted from the user's question using the LLM.

!!! example
    **Question:** *"What did Einstein develop?"*

    **Extracted entities:** `["Einstein"]`

### Step 2: Vector Search (Seed Retrieval)

The extracted entities are used to search Milvus for:

- **Similar entities** — vector search on the entity collection
- **Similar relations** — vector search on the relation collection

This produces the **seed set**: initial entities and relations that match the query.

```mermaid
flowchart LR
    QE["Query: 'Einstein'"] --> VS["Vector Search"]
    VS --> SE["Seed Entities:\nEinstein (0.95)\nAlbert Einstein (0.91)"]
    VS --> SR["Seed Relations:\n'Einstein developed relativity' (0.88)\n'Einstein worked at Princeton' (0.72)"]
```

### Step 3: Subgraph Expansion

Starting from the seed entities and relations, the algorithm expands outward through the graph:

```mermaid
flowchart TD
    subgraph "Hop 0 (Seeds)"
        E1["Einstein 🟠"]
    end

    subgraph "Hop 1 (Expansion)"
        E1 --> R1["developed 🔵"]
        E1 --> R2["worked at 🔵"]
        R1 --> E2["relativity 🔵"]
        R2 --> E3["Princeton 🔵"]
    end

    subgraph "Hop 2 (if degree=2)"
        E2 --> R3["revolutionized 🔵"]
        R3 --> E4["physics 🔵"]
    end
```

The expansion follows these links:

1. **Seed relations** → find their **entity IDs** → add those entities
2. **New entities** → find **relations** that reference them → add those relations
3. Repeat for `expansion_degree` hops (default: 1)

!!! tip "Tuning expansion"
    - `expansion_degree=1` (default): Good for most 2-hop questions
    - `expansion_degree=2`: Better for 3-4 hop questions but retrieves more noise
    - Higher values increase recall but may decrease precision

### Step 4: LLM Reranking

The expanded subgraph typically contains many candidate relations. A single LLM call selects the most relevant ones:

```mermaid
flowchart LR
    subgraph Candidates["Candidate Relations (e.g., 83)"]
        R1["Einstein developed relativity"]
        R2["Einstein worked at Princeton"]
        R3["Einstein born in Ulm"]
        R4["..."]
    end

    Candidates --> LLM["LLM Reranking\n(single pass)"]

    LLM --> Selected["Selected (e.g., 5):\n✅ Einstein developed relativity\n✅ relativity revolutionized physics"]
```

!!! note "Why single-pass works"
    The combination of vector search + subgraph expansion already produces high-quality candidates. A single reranking pass is sufficient to filter the best results — no need for expensive iterative retrieval.

### Step 5: Answer Generation

The selected relations and their associated passages are used as context for the final LLM answer generation call.

---

## Worked Example

### Full Query Flow

For the question: *"What did Einstein develop?"*

```mermaid
flowchart TD
    Q["What did Einstein develop?"] --> E1["Extract entity:\nEinstein"]
    E1 --> VS["Vector search\n→ seed entities & relations"]
    VS --> SE["Subgraph expansion\n→ 15 entities, 20 relations"]
    SE --> LR["LLM reranking\n→ select top 5 relations"]
    LR --> R1["✅ (Einstein, developed, theory of relativity)"]
    LR --> R2["✅ (theory of relativity, revolutionized, physics)"]
    LR --> R3["❌ (Einstein, worked at, Princeton)"]
    R1 --> AG["Generate answer"]
    R2 --> AG
    AG --> A["'Einstein developed the theory of relativity,\nwhich revolutionized physics.'"]

    style R1 fill:#e8f5e9,stroke:#2e7d32
    style R2 fill:#e8f5e9,stroke:#2e7d32
    style R3 fill:#fce4ec,stroke:#c62828
```

<span class="step-badge">1</span> Extract entity: `Einstein`
<span class="step-badge">2</span> Vector search finds seed entities and relations in Milvus
<span class="step-badge">3</span> Subgraph expansion discovers connected entities and relations
<span class="step-badge">4</span> **LLM reranking** selects the most relevant relations — one call, no iteration
<span class="step-badge">5</span> Generate answer from selected context

---

## Comparison with Other Approaches

| Approach | Graph DB | LLM Calls / Query | Iterative | Multi-hop | Complexity |
|----------|----------|---------------------|-----------|-----------|------------|
| **Naive RAG** | No | 1 (generation) | No | Limited | Low |
| **IRCoT** | No | 3-5+ (retrieve + reason) | Yes | Yes | High |
| **HippoRAG** | No | 1-2 | No | Yes | Medium |
| **Microsoft GraphRAG** | Yes (Neo4j) | Multiple | Yes | Yes | High |
| **Vector Graph RAG** | **No** | **2** (rerank + gen) | **No** | **Yes** | **Low** |

!!! info "Learn more"
    See the [Design Philosophy](design-philosophy.md) page for an in-depth discussion of why these architectural decisions were made and the trade-offs involved.

---

## Known Limitations

!!! warning "LLM Dependency"
    Triplet extraction and reranking quality depends heavily on the underlying LLM's capability. Weaker models may produce incomplete or inaccurate triplets, which directly affects retrieval quality.

!!! warning "Graph Consistency"
    The knowledge graph topology is maintained as a logical layer on top of Milvus via cross-referenced ID fields. Since Milvus does not support transactions, multi-step mutations (e.g., cascade deletes) are not atomic and may leave inconsistent state if interrupted. For best results, prefer batch construction via `add_documents()` over frequent incremental updates.
