package com.zjkl.vectorgraphrag.graph;

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
public class SubGraphPassage {
    private String id;
    private String text;
    @Builder.Default
    private List<String> entityIds = new ArrayList<>();
    @Builder.Default
    private List<String> relationIds = new ArrayList<>();
}
