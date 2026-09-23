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
public class Relation {
    private String id;
    private String text;
    private Triplet triplet;
    @Builder.Default
    private List<String> sourcePassageIds = new ArrayList<>();
    private List<Float> embedding;
}
