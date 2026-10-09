package com.zjkl.vectorgraphrag.storage;

import com.zjkl.vectorgraphrag.config.VectorGraphRagSettings;
import com.zjkl.vectorgraphrag.llm.OpenAiClient;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Unified embedding model wrapper.
 * Supports OpenAI embedding models.
 */
@Slf4j
public class EmbeddingClient {

    private final VectorGraphRagSettings settings;
    private final OpenAiClient openAiClient;
    private Integer dimension;

    public EmbeddingClient(VectorGraphRagSettings settings, OpenAiClient openAiClient) {
        this.settings = settings;
        this.openAiClient = openAiClient;
    }

    public int getDimension() {
        if (dimension == null) {
            List<Float> test = embed("test");
            dimension = test.size();
        }
        return dimension;
    }

    public List<Float> embed(String text) {
        List<Float> raw = openAiClient.embed(text);
        return l2Normalize(raw);
    }

    public List<List<Float>> embedBatch(List<String> texts) {
        return embedBatch(texts, false);
    }

    public List<List<Float>> embedBatch(List<String> texts, boolean showProgress) {
        List<List<Float>> results = new ArrayList<>();
        for (int i = 0; i < texts.size(); i += settings.getBatchSize()) {
            int end = Math.min(i + settings.getBatchSize(), texts.size());
            List<String> batch = texts.subList(i, end);
            try {
                // Use batch API: all texts in one HTTP call
                List<List<Float>> batchResults = openAiClient.embedBatch(batch);
                for (List<Float> vec : batchResults) {
                    results.add(vec != null ? l2Normalize(vec) : fallbackVector());
                }
            } catch (Exception e) {
                log.warn("Batch embedding failed, falling back to single: {}", e.getMessage());
                for (String text : batch) {
                    try {
                        results.add(embed(text));
                    } catch (Exception e2) {
                        log.warn("Embedding failed for text, using zero vector: {}", e2.getMessage());
                        results.add(fallbackVector());
                    }
                }
            }
            if (showProgress) {
                log.info("Embedded {}/{} texts", end, texts.size());
            }
        }
        return results;
    }

    private List<Float> fallbackVector() {
        List<Float> zero = new ArrayList<>();
        for (int j = 0; j < getDimension(); j++) zero.add(0.0f);
        return zero;
    }

    private List<Float> l2Normalize(List<Float> vector) {
        double sum = 0.0;
        for (float v : vector) {
            sum += v * v;
        }
        double norm = Math.sqrt(sum);
        if (norm == 0) return vector;

        List<Float> normalized = new ArrayList<>(vector.size());
        for (float v : vector) {
            normalized.add((float) (v / norm));
        }
        return normalized;
    }
}
