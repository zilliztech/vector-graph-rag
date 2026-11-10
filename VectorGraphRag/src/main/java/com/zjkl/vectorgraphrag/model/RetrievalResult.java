package com.zjkl.vectorgraphrag.model;

import com.zjkl.vectorgraphrag.graph.SubGraph;
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
public class RetrievalResult {
    @Builder.Default
    private List<String> entityIds = new ArrayList<>();
    @Builder.Default
    private List<String> entityTexts = new ArrayList<>();
    @Builder.Default
    private List<Float> entityScores = new ArrayList<>();
    @Builder.Default
    private List<String> relationIds = new ArrayList<>();
    @Builder.Default
    private List<String> relationTexts = new ArrayList<>();
    @Builder.Default
    private List<Float> relationScores = new ArrayList<>();
    private SubGraph subgraph;
    @Builder.Default
    private List<String> expandedRelationIds = new ArrayList<>();
    @Builder.Default
    private List<String> expandedRelationTexts = new ArrayList<>();
    private String query;
    @Builder.Default
    private List<String> queryEntities = new ArrayList<>();
    @Builder.Default
    private boolean evictionOccurred = false;
    @Builder.Default
    private int evictionBeforeCount = 0;
    @Builder.Default
    private int evictionAfterCount = 0;
}
