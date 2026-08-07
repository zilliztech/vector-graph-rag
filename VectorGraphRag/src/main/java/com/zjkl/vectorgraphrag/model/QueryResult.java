package com.zjkl.vectorgraphrag.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.ArrayList;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class QueryResult {
    private String query;
    private String answer;
    @Builder.Default
    private List<String> queryEntities = new ArrayList<>();
    @Builder.Default
    private List<String> retrievedPassages = new ArrayList<>();
    @Builder.Default
    private List<String> retrievedRelations = new ArrayList<>();
    @Builder.Default
    private List<String> expandedRelations = new ArrayList<>();
    @Builder.Default
    private List<String> rerankedRelations = new ArrayList<>();
    private Object subgraph;
    @Builder.Default
    private List<String> passages = new ArrayList<>();
    private RetrievalDetail retrievalDetail;
    private RerankResult rerankResult;
    private EvictionResult evictionResult;
}
