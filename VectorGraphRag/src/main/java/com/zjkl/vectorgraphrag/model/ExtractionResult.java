package com.zjkl.vectorgraphrag.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExtractionResult {
    @Builder.Default
    private List<String> documents = new ArrayList<>();
    @Builder.Default
    private List<Entity> entities = new ArrayList<>();
    @Builder.Default
    private List<Relation> relations = new ArrayList<>();
    @Builder.Default
    private Map<String, List<String>> entityToRelationIds = new HashMap<>();
    @Builder.Default
    private Map<String, List<String>> relationToPassageIds = new HashMap<>();
}
