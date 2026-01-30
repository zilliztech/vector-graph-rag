# LLM Reranker Prompt Optimization Plan

## Goal

Optimize the LLM reranker prompt to improve **Recall@5** on the MuSiQue dataset (first 500 samples).

- **Current Baseline**: Contriever embedding model + HippoRAG-style reranking prompt
- **LLM Model**: gpt-4o-mini
- **Metric**: Recall@5 (percentage of supporting passages found in top 5 retrieved results)

---

## Current Prompt Analysis

### Location
`src/vector_graph_rag/llm/reranker.py`

### Current Prompt Structure (HippoRAG)
```
System: No explicit system prompt
User:
  - Task description: "Select 5 relationships useful to answer the question"
  - One-shot example: Question about "Lothair II's mother" + expected JSON output
  - Current query: Question + Relationship descriptions
Output Format: JSON with "thought_process" and "useful_relations"
```

### Potential Optimization Areas
1. **Task Description** - Make instructions clearer/more specific
2. **Few-shot Examples** - Add more examples or use better examples
3. **Chain-of-Thought** - Encourage step-by-step reasoning
4. **Output Format** - Optimize JSON structure or parsing
5. **Scoring/Ranking** - Ask for relevance scores instead of binary selection
6. **Persona** - Add system prompt with expert persona

---

## Experiment Groups

### Group 0: Baseline (Current Implementation)
- **Description**: Run current HippoRAG prompt as baseline
- **Expected**: Get baseline Recall@5 score for comparison

### Group 1: Enhanced Task Description
- **Description**: Improve the instruction clarity
- **Changes**:
  - Add explicit criteria for what makes a relation "useful"
  - Emphasize multi-hop reasoning requirements
  - Clarify that selected relations should form reasoning chains

### Group 2: Better Few-shot Examples
- **Description**: Use more relevant/diverse examples
- **Changes**:
  - Add 2-3 carefully selected examples from MuSiQue-style questions
  - Include examples with different hop counts (2-hop, 3-hop, 4-hop)
  - Show examples of both good and bad relation selections

### Group 3: Chain-of-Thought (CoT) Prompting
- **Description**: Encourage explicit step-by-step reasoning
- **Changes**:
  - Ask model to first decompose the question into sub-questions
  - Then identify which relations answer each sub-question
  - Finally select the most relevant relations

### Group 4: Scoring-based Selection
- **Description**: Ask for relevance scores instead of binary selection
- **Changes**:
  - Request a relevance score (0-10) for each relation
  - Select top-K based on scores
  - May help with edge cases where binary selection is ambiguous

### Group 5: Expert Persona + Detailed Guidelines
- **Description**: Add system prompt with expert role
- **Changes**:
  - System prompt: "You are an expert in knowledge graph reasoning..."
  - Add detailed guidelines about multi-hop QA
  - Include common pitfalls to avoid

### Group 6: Structured Reasoning Template
- **Description**: Provide a structured template for reasoning
- **Changes**:
  - Ask model to identify: (1) Question entities, (2) Target information, (3) Reasoning path
  - Use template to guide relation selection
  - May improve consistency

### Group 7: Combined Best Approaches
- **Description**: Combine the best-performing techniques from Groups 1-6
- **Changes**: TBD based on individual group results

---

## Experiment Protocol

### Test Configuration
```python
dataset: MuSiQue
samples: 500 (first 500)
metric: Recall@5
reranking: enabled
llm_model: gpt-4o-mini
top_k: 10  # retrieve 10, evaluate recall@5
```

### Evaluation Script
Create `experiments/prompt_optimization.py`:
1. Load MuSiQue dataset (500 samples)
2. For each prompt variant:
   - Run retrieval + reranking
   - Calculate Recall@5
   - Log results
3. Compare all variants

### Output Format
```json
{
  "experiment_name": "group_1_enhanced_description",
  "dataset": "musique",
  "num_samples": 500,
  "recall_at_5": 0.xxx,
  "prompt_config": { ... },
  "timestamp": "..."
}
```

---

## Implementation Tasks

- [x] **Task 1**: Run baseline (Group 0) to get current Recall@5
- [x] **Task 2**: Create experiment framework (`experiments/prompt_optimization.py`)
- [x] **Task 3**: Implement Group 1 - Enhanced Task Description
- [x] **Task 4**: Implement Group 2 - Better Few-shot Examples
- [x] **Task 5**: Implement Group 3 - Chain-of-Thought
- [x] **Task 6**: Implement Group 4 - Scoring-based Selection
- [x] **Task 7**: Implement Group 5 - Expert Persona
- [x] **Task 8**: Implement Group 6 - Structured Reasoning
- [x] **Task 9**: Analyze results (Group 2 was best, no need for combined approach)
- [x] **Task 10**: Update production code with best prompt

---

## Experiment Results (Completed 2026-01-29)

### Final Results - MuSiQue 500 samples

| Rank | Experiment Group | Recall@5 | vs Baseline | Recall@1 | Recall@10 |
|------|------------------|----------|-------------|----------|-----------|
| **1** | **Group 2 (Better Fewshot)** | **0.6287** | **+2.85%** ✓ | 0.2002 | 0.7012 |
| 2 | Group 1 (Enhanced Description) | 0.6172 | +1.70% | 0.1953 | 0.6907 |
| 3 | Group 6 (Structured Reasoning) | 0.6162 | +1.60% | 0.1887 | 0.6910 |
| 4 | Group 5 (Expert Persona) | 0.6107 | +1.05% | 0.2000 | 0.6913 |
| 5 | Group 3 (Chain of Thought) | 0.6095 | +0.93% | 0.1727 | 0.6937 |
| 6 | Group 0 (Baseline - HippoRAG) | 0.6002 | - | 0.2035 | 0.6830 |
| 7 | Group 4 (Scoring) | 0.5950 | -0.52% ✗ | 0.1930 | 0.6712 |

### Key Findings

1. **Winner: Group 2 (Better Fewshot)** - Using diverse few-shot examples (2-hop, 3-hop questions) achieved the best performance with **+2.85% improvement** on Recall@5.

2. **Scoring-based approach underperformed** - Asking the model to score relations (0-10) actually hurt performance, likely due to added complexity and potential inconsistency in scoring.

3. **Chain-of-Thought had minimal impact** - Explicit reasoning decomposition didn't significantly improve results, possibly because the model already reasons implicitly.

4. **Enhanced descriptions and structured reasoning helped** - Clear task descriptions and structured templates both showed consistent improvements.

5. **System prompts (Expert Persona) showed moderate gains** - Adding expert context helped but wasn't as effective as better examples.

### Cross-Dataset Validation (2026-01-30)

| Dataset | Baseline R@5 | Best Prompt R@5 | Improvement |
|---------|--------------|-----------------|-------------|
| MuSiQue | 0.6002 | 0.6287 | **+2.85%** |
| HotpotQA | 0.8860 | 0.8930 | **+0.79%** |
| 2WikiMultiHopQA | 0.9070 | 0.9065 | -0.06% (flat) |

The best prompt generalizes well to HotpotQA and doesn't regress on 2WikiMultiHopQA (where baseline is already very high).

### Best Prompt Configuration

The winning configuration (Group 2 - Better Fewshot) uses:
- **3 diverse few-shot examples** covering different question types (2-hop and 3-hop)
- **Concise task description** with clear JSON output format
- **No system prompt** (kept simple)

See `experiments/prompt_configs.py` for the full prompt configuration.

---

## Success Criteria

- **Minimum**: Find a prompt that improves Recall@5 by at least 2% over baseline ✓ **ACHIEVED (2.85%)**
- **Target**: Find a prompt that improves Recall@5 by 5%+ over baseline ✗ NOT ACHIEVED
- **Stretch**: Find a prompt that improves Recall@5 by 10%+ over baseline ✗ NOT ACHIEVED

---

## Notes

- Best prompt (Group 2 - Better Fewshot) has been applied to `src/vector_graph_rag/llm/reranker.py`
- Experiment files and temporary databases have been cleaned up
- LLM cache enabled to avoid redundant API calls
