package com.zjkl.vectorgraphrag.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Component
public class VectorGraphRagSettings {
    // OpenAI Settings
    @Builder.Default
    private String openaiApiKey = System.getenv().getOrDefault("OPENAI_API_KEY", "");

    @Builder.Default
    private String openaiBaseUrl = "";

    // Model Settings
    @Builder.Default
    private String llmModel = "gpt-4o-mini";

    @Builder.Default
    private String embeddingModel = "text-embedding-3-large";

    @Builder.Default
    private int embeddingDimension = 3072;

    // Milvus Settings
    @Builder.Default
    private String milvusUri = "./vector_graph_rag.db";

    @Builder.Default
    private String milvusToken = "";

    @Builder.Default
    private String milvusDb = "";

    // Milvus Index Settings
    @Builder.Default
    private String milvusIndexType = "AUTOINDEX";

    @Builder.Default
    private String milvusMetricType = "IP";

    private Map<String, Object> milvusIndexParams;

    @Builder.Default
    private String milvusConsistencyLevel = "Bounded";

    // Collection prefix
    @Builder.Default
    private String collectionPrefix = "";

    // Collection Names
    @Builder.Default
    private String entityCollection = "vgrag_entities";

    @Builder.Default
    private String relationCollection = "vgrag_relations";

    @Builder.Default
    private String passageCollection = "vgrag_passages";

    // Retrieval Settings
    @Builder.Default
    private int entityTopK = 20;

    @Builder.Default
    private int relationTopK = 20;

    @Builder.Default
    private float entitySimilarityThreshold = 0.9f;

    @Builder.Default
    private float relationSimilarityThreshold = -1.0f;

    @Builder.Default
    private int expansionDegree = 1;

    @Builder.Default
    private int relationNumberThreshold = 1000;

    @Builder.Default
    private int finalTopK = 3;

    // LLM Settings
    @Builder.Default
    private double llmTemperature = 0.0;

    @Builder.Default
    private int llmMaxRetries = 3;

    @Builder.Default
    private boolean useLlmCache = true;

    // Processing Settings
    @Builder.Default
    private int batchSize = 32;

    public boolean hasCollectionPrefix() {
        return collectionPrefix != null && !collectionPrefix.isEmpty();
    }
}
