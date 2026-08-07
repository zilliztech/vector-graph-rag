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
public class RetrievalDetail {
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
}
