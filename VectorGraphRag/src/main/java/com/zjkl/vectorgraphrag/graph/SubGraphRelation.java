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
public class SubGraphRelation {
    private String id;
    private String text;
    private String subject;
    private String predicate;
    private String object;
    @Builder.Default
    private List<String> entityIds = new ArrayList<>();
    @Builder.Default
    private List<String> passageIds = new ArrayList<>();
}
