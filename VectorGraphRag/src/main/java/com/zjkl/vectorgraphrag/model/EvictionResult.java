package com.zjkl.vectorgraphrag.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class EvictionResult {
    @Builder.Default
    private boolean occurred = false;
    @Builder.Default
    private int beforeCount = 0;
    @Builder.Default
    private int afterCount = 0;
}
