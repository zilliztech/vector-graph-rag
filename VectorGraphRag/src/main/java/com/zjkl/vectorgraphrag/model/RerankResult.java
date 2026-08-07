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
public class RerankResult {
    @Builder.Default
    private List<String> selectedRelationIds = new ArrayList<>();
    @Builder.Default
    private List<String> selectedRelationTexts = new ArrayList<>();
}
