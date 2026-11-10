package com.zjkl.vectorgraphrag.graph;

import com.zjkl.vectorgraphrag.config.VectorGraphRagSettings;
import com.zjkl.vectorgraphrag.llm.EntityExtractor;
import com.zjkl.vectorgraphrag.model.RetrievalResult;
import com.zjkl.vectorgraphrag.storage.EmbeddingClient;
import com.zjkl.vectorgraphrag.storage.MilvusStore;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Graph-based retriever using multi-way vector search.
 * Performs entity and relation retrieval, then expands the subgraph
 * with lazy loading from Milvus.
 */
@Slf4j
public class GraphRetriever {

    private final VectorGraphRagSettings settings;
    private final MilvusStore store;
    private final EmbeddingClient embeddingClient;
    private final EntityExtractor entityExtractor;

    public GraphRetriever(VectorGraphRagSettings settings, MilvusStore store,
                          EmbeddingClient embeddingClient, EntityExtractor entityExtractor) {
        this.settings = settings;
        this.store = store;
        this.embeddingClient = embeddingClient;
        this.entityExtractor = entityExtractor;
    }

    public RetrievalResult retrieve(String query) {
        return retrieve(query, null, null, null, null, null, null);
    }

    public RetrievalResult retrieve(String query,
                                     Integer entityTopK,
                                     Integer relationTopK,
                                     Float entitySimilarityThreshold,
                                     Float relationSimilarityThreshold,
                                     Integer expansionDegree,
                                     String filter) {
        Set<String> allowedPassageIds = getAllowedPassageIds(filter);

        // Extract query entities
        List<String> queryEntities = entityExtractor.extract(query);

        // Retrieve entities
        List<String> entityIds = new ArrayList<>();
        List<String> entityTexts = new ArrayList<>();
        List<Float> entityScores = new ArrayList<>();

        if (!queryEntities.isEmpty()) {
            List<List<Float>> queryEmbeddings = embeddingClient.embedBatch(queryEntities);
            int eTopK = entityTopK != null ? entityTopK : settings.getEntityTopK();
            float eThreshold = entitySimilarityThreshold != null ? entitySimilarityThreshold
                    : settings.getEntitySimilarityThreshold();

            List<Map<String, Object>> searchResults = store.searchEntities(queryEmbeddings, eTopK);
            Set<String> seenIds = new LinkedHashSet<>();

            for (Map<String, Object> result : searchResults) {
                double distance = (double) result.get("distance");
                if (distance <= eThreshold) continue;

                @SuppressWarnings("unchecked")
                Map<String, Object> entity = (Map<String, Object>) result.get("entity");
                String eid = entity.get("id").toString();
                if (seenIds.add(eid)) {
                    entityIds.add(eid);
                    entityTexts.add(entity.get("text").toString());
                    entityScores.add((float) distance);
                }
            }
        }

        // Retrieve relations
        int rTopK = relationTopK != null ? relationTopK : settings.getRelationTopK();
        float rThreshold = relationSimilarityThreshold != null ? relationSimilarityThreshold
                : settings.getRelationSimilarityThreshold();

        List<Float> queryEmbedding = embeddingClient.embed(query);
        List<Map<String, Object>> relationResults = store.searchRelations(queryEmbedding, rTopK);

        List<String> relationIds = new ArrayList<>();
        List<String> relationTexts = new ArrayList<>();
        List<Float> relationScores = new ArrayList<>();

        for (Map<String, Object> result : relationResults) {
            double distance = (double) result.get("distance");
            if (distance <= rThreshold) continue;

            @SuppressWarnings("unchecked")
            Map<String, Object> entity = (Map<String, Object>) result.get("entity");
            String rid = entity.get("id").toString();
            relationIds.add(rid);
            relationTexts.add(entity.get("text").toString());
            relationScores.add((float) distance);
        }

        // Filter relations by allowed passage IDs
        if (allowedPassageIds != null) {
            List<String> filteredIds = filterRelationsByPassageIds(relationIds, allowedPassageIds);
            // Rebuild parallel lists
            Set<String> allowed = new HashSet<>(filteredIds);
            List<String> keptIds = new ArrayList<>();
            List<String> keptTexts = new ArrayList<>();
            List<Float> keptScores = new ArrayList<>();
            for (int i = 0; i < relationIds.size(); i++) {
                if (allowed.contains(relationIds.get(i))) {
                    keptIds.add(relationIds.get(i));
                    keptTexts.add(relationTexts.get(i));
                    keptScores.add(relationScores.get(i));
                }
            }
            relationIds = keptIds;
            relationTexts = keptTexts;
            relationScores = keptScores;
        }

        // Expand subgraph
        SubGraph subgraph = expandSubGraph(entityIds, relationIds, expansionDegree);

        // Apply eviction
        int threshold = settings.getRelationNumberThreshold();
        List<String> filteredExpandedIds = filterRelationsByPassageIds(
                new ArrayList<>(subgraph.getRelationIds()), allowedPassageIds);
        List<String> expandedIds = new ArrayList<>();
        List<String> expandedTexts = new ArrayList<>();
        boolean evictionOccurred = false;
        int evictionBefore = filteredExpandedIds.size();

        if (!filteredExpandedIds.isEmpty()) {
            if (filteredExpandedIds.size() <= threshold) {
                // No eviction needed
                expandedIds = filteredExpandedIds;
                expandedTexts = filteredExpandedIds.stream()
                        .map(rid -> {
                            SubGraphRelation rel = null;
                            for (SubGraphRelation r : subgraph.getRelations()) {
                                if (r.getId().equals(rid)) { rel = r; break; }
                            }
                            return rel != null ? rel.getText() : "";
                        })
                        .collect(Collectors.toList());
            } else {
                // Eviction: use vector search to filter
                log.info("Use Eviction Strategy. ({} -> {})", evictionBefore, threshold);
                List<Float> qEmbed = embeddingClient.embed(query);
                String idsStr = filteredExpandedIds.stream()
                        .map(id -> "\"" + id + "\"")
                        .collect(Collectors.joining(", "));
                String filterExpr = "id in [" + idsStr + "]";

                List<Map<String, Object>> searchRes = store.searchRelationsWithFilter(qEmbed, threshold, filterExpr);
                evictionOccurred = true;
                for (Map<String, Object> r : searchRes) {
                    expandedIds.add(r.get("id").toString());
                    expandedTexts.add(r.get("text") != null ? r.get("text").toString() : "");
                }
            }
        }

        return RetrievalResult.builder()
                .entityIds(entityIds)
                .entityTexts(entityTexts)
                .entityScores(entityScores)
                .relationIds(relationIds)
                .relationTexts(relationTexts)
                .relationScores(relationScores)
                .subgraph(subgraph)
                .expandedRelationIds(expandedIds)
                .expandedRelationTexts(expandedTexts)
                .query(query)
                .queryEntities(queryEntities)
                .evictionOccurred(evictionOccurred)
                .evictionBeforeCount(evictionBefore)
                .evictionAfterCount(expandedIds.size())
                .build();
    }

    public List<String> retrievePassagesNaive(String query, int topK, String filter) {
        List<Float> queryEmbedding = embeddingClient.embed(query);
        List<Map<String, Object>> results = store.searchPassages(queryEmbedding, topK, filter);
        return results.stream()
                .map(r -> {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> entity = (Map<String, Object>) r.get("entity");
                    return entity.get("text").toString();
                })
                .collect(Collectors.toList());
    }

    private SubGraph expandSubGraph(List<String> entityIds, List<String> relationIds, Integer degree) {
        int deg = degree != null ? degree : settings.getExpansionDegree();
        SubGraph subgraph = new SubGraph(store);
        subgraph.addEntities(entityIds);
        subgraph.addRelations(relationIds);
        subgraph.expand(deg);
        return subgraph;
    }

    private Set<String> getAllowedPassageIds(String filter) {
        if (filter == null || filter.trim().isEmpty()) return null;
        return new HashSet<>(store.queryPassageIds(filter));
    }

    private List<String> filterRelationsByPassageIds(List<String> relationIds, Set<String> allowedPassageIds) {
        if (allowedPassageIds == null) return relationIds;
        if (relationIds == null || relationIds.isEmpty()) return List.of();

        List<Map<String, Object>> relationData = store.getRelationsByIds(relationIds);
        Set<String> allowed = new HashSet<>();
        for (Map<String, Object> r : relationData) {
            List<String> pids = safeGetList(r, "passage_ids");
            for (String pid : pids) {
                if (allowedPassageIds.contains(pid)) {
                    allowed.add(r.get("id").toString());
                    break;
                }
            }
        }

        return relationIds.stream().filter(allowed::contains).collect(Collectors.toList());
    }

    @SuppressWarnings("unchecked")
    private List<String> safeGetList(Map<String, Object> map, String key) {
        Object val = map.get(key);
        if (val instanceof List) {
            return ((List<Object>) val).stream().map(Object::toString).collect(Collectors.toList());
        }
        return List.of();
    }
}
