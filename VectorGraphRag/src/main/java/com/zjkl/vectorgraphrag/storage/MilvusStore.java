package com.zjkl.vectorgraphrag.storage;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.zjkl.vectorgraphrag.config.VectorGraphRagSettings;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.common.ConsistencyLevel;
import io.milvus.v2.common.DataType;
import io.milvus.v2.common.IndexParam;
import io.milvus.v2.service.vector.request.data.BaseVector;
import io.milvus.v2.service.vector.request.data.FloatVec;
import io.milvus.v2.service.collection.request.CreateCollectionReq;
import io.milvus.v2.service.collection.request.DropCollectionReq;
import io.milvus.v2.service.collection.request.HasCollectionReq;
import io.milvus.v2.service.index.request.CreateIndexReq;
import io.milvus.v2.service.vector.request.InsertReq;
import io.milvus.v2.service.vector.request.QueryReq;
import io.milvus.v2.service.vector.request.SearchReq;
import io.milvus.v2.service.vector.request.UpsertReq;
import io.milvus.v2.service.vector.response.InsertResp;
import io.milvus.v2.service.vector.response.QueryResp;
import io.milvus.v2.service.vector.response.SearchResp;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Milvus vector store for entities, relations, and passages.
 *
 * Manages three collections:
 * - Entity collection: entities with embeddings and adjacency metadata
 * - Relation collection: relations with embeddings and triplet fields
 * - Passage collection: passages with embeddings
 */
@Slf4j
public class MilvusStore {

    private final VectorGraphRagSettings settings;
    private final EmbeddingClient embeddingClient;
    private final MilvusClientV2 client;
    private final Gson gson;

    private final String entityCollection;
    private final String relationCollection;
    private final String passageCollection;

    public MilvusStore(VectorGraphRagSettings settings, EmbeddingClient embeddingClient) {
        this.settings = settings;
        this.embeddingClient = embeddingClient;
        this.gson = new Gson();

        ConnectConfig.ConnectConfigBuilder builder = ConnectConfig.builder()
                .uri(settings.getMilvusUri());
        if (settings.getMilvusToken() != null && !settings.getMilvusToken().isEmpty()) {
            builder.token(settings.getMilvusToken());
        }
        if (settings.getMilvusDb() != null && !settings.getMilvusDb().isEmpty()) {
            builder.dbName(settings.getMilvusDb());
        }
        this.client = new MilvusClientV2(builder.build());

        String prefix = settings.hasCollectionPrefix() ? settings.getCollectionPrefix() + "_" : "";
        this.entityCollection = prefix + settings.getEntityCollection();
        this.relationCollection = prefix + settings.getRelationCollection();
        this.passageCollection = prefix + settings.getPassageCollection();
    }

    // ==================== Collection Management ====================

    public void createCollections(boolean dropExisting) {
        for (String name : List.of(entityCollection, relationCollection, passageCollection)) {
            createCollection(name, dropExisting);
        }
    }

    public void dropCollections() {
        for (String name : List.of(entityCollection, relationCollection, passageCollection)) {
            try {
                if (client.hasCollection(HasCollectionReq.builder().collectionName(name).build())) {
                    client.dropCollection(DropCollectionReq.builder().collectionName(name).build());
                    log.info("Dropped collection: {}", name);
                }
            } catch (Exception e) {
                log.warn("Error dropping collection {}: {}", name, e.getMessage());
            }
        }
    }

    private void createCollection(String collectionName, boolean dropExisting) {
        try {
            if (client.hasCollection(HasCollectionReq.builder().collectionName(collectionName).build())) {
                if (dropExisting) {
                    client.dropCollection(DropCollectionReq.builder().collectionName(collectionName).build());
                } else {
                    return;
                }
            }
        } catch (Exception e) {
            log.warn("Error checking collection {}: {}", collectionName, e.getMessage());
        }

        int dim = settings.getEmbeddingDimension() > 0
                ? settings.getEmbeddingDimension()
                : embeddingClient.getDimension();

        CreateCollectionReq.CollectionSchema schema = CreateCollectionReq.CollectionSchema.builder().build();
        schema.addField(io.milvus.v2.service.collection.request.AddFieldReq.builder()
                .fieldName("id")
                .dataType(DataType.VarChar)
                .maxLength(64)
                .isPrimaryKey(true)
                .autoID(false)
                .build());
        schema.addField(io.milvus.v2.service.collection.request.AddFieldReq.builder()
                .fieldName("vector")
                .dataType(DataType.FloatVector)
                .dimension(dim)
                .build());
        schema.addField(io.milvus.v2.service.collection.request.AddFieldReq.builder()
                .fieldName("text")
                .dataType(DataType.VarChar)
                .maxLength(65535)
                .build());
        // enableDynamicField is true by default

        client.createCollection(CreateCollectionReq.builder()
                .collectionName(collectionName)
                .collectionSchema(schema)
                .consistencyLevel(ConsistencyLevel.fromName(settings.getMilvusConsistencyLevel()))
                .build());

        // Create index
        IndexParam.IndexParamBuilder paramBuilder = IndexParam.builder()
                .fieldName("vector")
                .indexType(IndexParam.IndexType.valueOf(settings.getMilvusIndexType()))
                .metricType(IndexParam.MetricType.valueOf(settings.getMilvusMetricType()));
        if (settings.getMilvusIndexParams() != null && !settings.getMilvusIndexParams().isEmpty()) {
            paramBuilder.indexName("vector_index");
            paramBuilder.extraParams(settings.getMilvusIndexParams());
        }
        client.createIndex(CreateIndexReq.builder()
                .collectionName(collectionName)
                .indexParams(List.of(paramBuilder.build()))
                .build());

        log.info("Created collection: {} (dim={})", collectionName, dim);
    }

    // ==================== Generic Insert ====================

    private List<String> insertData(String collectionName, List<String> ids, List<String> texts,
                                    List<List<Float>> embeddings, List<Map<String, Object>> metadatas,
                                    boolean showProgress) {
        if (ids == null || ids.isEmpty()) return List.of();

        int batchSize = settings.getBatchSize();
        List<String> resultIds = new ArrayList<>();

        for (int i = 0; i < ids.size(); i += batchSize) {
            int end = Math.min(i + batchSize, ids.size());
            List<JsonObject> batch = new ArrayList<>();

            for (int j = i; j < end; j++) {
                JsonObject row = new JsonObject();
                row.addProperty("id", ids.get(j));
                row.addProperty("text", texts.get(j));

                JsonArray vectorArr = new JsonArray();
                List<Float> vec = embeddings.get(j);
                for (float v : vec) vectorArr.add(v);
                row.add("vector", vectorArr);

                if (metadatas != null && j < metadatas.size() && metadatas.get(j) != null) {
                    Map<String, Object> meta = metadatas.get(j);
                    for (Map.Entry<String, Object> entry : meta.entrySet()) {
                        if (entry.getValue() instanceof List) {
                            JsonArray arr = new JsonArray();
                            for (Object item : (List<?>) entry.getValue()) {
                                arr.add(item.toString());
                            }
                            row.add(entry.getKey(), arr);
                        } else if (entry.getValue() instanceof String) {
                            row.addProperty(entry.getKey(), (String) entry.getValue());
                        } else if (entry.getValue() instanceof Number) {
                            row.addProperty(entry.getKey(), (Number) entry.getValue());
                        } else if (entry.getValue() instanceof Boolean) {
                            row.addProperty(entry.getKey(), (Boolean) entry.getValue());
                        }
                    }
                }

                batch.add(row);
                resultIds.add(ids.get(j));
            }

            InsertReq insertReq = InsertReq.builder()
                    .collectionName(collectionName)
                    .data(batch)
                    .build();
            client.insert(insertReq);

            if (showProgress) {
                log.info("Inserted {}/{} into {}", end, ids.size(), collectionName);
            }
        }

        return resultIds;
    }

    // ==================== Entity Operations ====================

    public List<String> insertEntities(List<String> entityTexts, List<String> ids,
                                       List<List<Float>> embeddings, List<Map<String, Object>> metadatas,
                                       boolean showProgress) {
        if (entityTexts == null || entityTexts.isEmpty()) return List.of();

        List<List<Float>> embeds = embeddings != null ? embeddings
                : embeddingClient.embedBatch(entityTexts, showProgress);
        List<String> finalIds = ids != null ? ids : entityTexts.stream()
                .map(t -> UUID.randomUUID().toString()).collect(Collectors.toList());

        return insertData(entityCollection, finalIds, entityTexts, embeds, metadatas, showProgress);
    }

    public List<Map<String, Object>> searchEntities(List<List<Float>> queryEmbeddings, int topK) {
        List<BaseVector> floatVecs = queryEmbeddings.stream()
                .<BaseVector>map(FloatVec::new)
                .collect(Collectors.toList());
        SearchReq searchReq = SearchReq.builder()
                .collectionName(entityCollection)
                .data(floatVecs)
                .topK(topK)
                .outputFields(List.of("id", "text", "relation_ids", "passage_ids"))
                .build();

        SearchResp resp = client.search(searchReq);
        return convertSearchResults(resp);
    }

    public List<Map<String, Object>> searchRelations(List<Float> queryEmbedding, int topK) {
        SearchReq searchReq = SearchReq.builder()
                .collectionName(relationCollection)
                .data(Collections.singletonList(new FloatVec(queryEmbedding)))
                .topK(topK)
                .outputFields(List.of("id", "text", "entity_ids", "passage_ids", "subject", "predicate", "object"))
                .build();

        SearchResp resp = client.search(searchReq);
        return convertSearchResults(resp);
    }

    public List<Map<String, Object>> searchPassages(List<Float> queryEmbedding, int topK, String filter) {
        SearchReq.SearchReqBuilder builder = SearchReq.builder()
                .collectionName(passageCollection)
                .data(Collections.singletonList(new FloatVec(queryEmbedding)))
                .topK(topK)
                .outputFields(List.of("id", "text", "entity_ids", "relation_ids"));
        if (filter != null && !filter.isEmpty()) {
            builder.filter(filter);
        }

        SearchResp resp = client.search(builder.build());
        return convertSearchResults(resp);
    }

    // ==================== Relation Operations ====================

    public List<String> insertRelations(List<String> relationTexts, List<String> ids,
                                        List<List<Float>> embeddings, List<Map<String, Object>> metadatas,
                                        boolean showProgress) {
        if (relationTexts == null || relationTexts.isEmpty()) return List.of();

        List<List<Float>> embeds = embeddings != null ? embeddings
                : embeddingClient.embedBatch(relationTexts, showProgress);
        List<String> finalIds = ids != null ? ids : relationTexts.stream()
                .map(t -> UUID.randomUUID().toString()).collect(Collectors.toList());

        return insertData(relationCollection, finalIds, relationTexts, embeds, metadatas, showProgress);
    }

    // ==================== Passage Operations ====================

    public List<String> insertPassages(List<String> passageTexts, List<String> ids,
                                       List<List<Float>> embeddings, List<Map<String, Object>> metadatas,
                                       boolean showProgress) {
        if (passageTexts == null || passageTexts.isEmpty()) return List.of();

        List<List<Float>> embeds = embeddings != null ? embeddings
                : embeddingClient.embedBatch(passageTexts, showProgress);
        List<String> finalIds = ids != null ? ids : passageTexts.stream()
                .map(t -> UUID.randomUUID().toString()).collect(Collectors.toList());

        return insertData(passageCollection, finalIds, passageTexts, embeds, metadatas, showProgress);
    }

    // ==================== Query by ID ====================

    public List<Map<String, Object>> getEntitiesByIds(List<String> entityIds) {
        if (entityIds == null || entityIds.isEmpty()) return List.of();

        String idsStr = entityIds.stream()
                .map(id -> "\"" + id + "\"")
                .collect(Collectors.joining(", "));
        QueryResp resp = client.query(QueryReq.builder()
                .collectionName(entityCollection)
                .filter("id in [" + idsStr + "]")
                .outputFields(List.of("id", "text", "relation_ids", "passage_ids"))
                .build());
        return convertQueryResults(resp);
    }

    public List<Map<String, Object>> getRelationsByIds(List<String> relationIds) {
        if (relationIds == null || relationIds.isEmpty()) return List.of();

        String idsStr = relationIds.stream()
                .map(id -> "\"" + id + "\"")
                .collect(Collectors.joining(", "));
        QueryResp resp = client.query(QueryReq.builder()
                .collectionName(relationCollection)
                .filter("id in [" + idsStr + "]")
                .outputFields(List.of("id", "text", "entity_ids", "passage_ids", "subject", "predicate", "object"))
                .build());
        return convertQueryResults(resp);
    }

    public List<Map<String, Object>> getPassagesByIds(List<String> passageIds) {
        return getPassagesByIds(passageIds, null);
    }

    public List<Map<String, Object>> getPassagesByIds(List<String> passageIds, String filter) {
        if (passageIds == null || passageIds.isEmpty()) return List.of();

        String idsStr = passageIds.stream()
                .map(id -> "\"" + id + "\"")
                .collect(Collectors.joining(", "));
        String filterExpr = "id in [" + idsStr + "]";
        if (filter != null && !filter.isEmpty()) {
            filterExpr = "(" + filterExpr + ") and (" + filter + ")";
        }

        QueryResp resp = client.query(QueryReq.builder()
                .collectionName(passageCollection)
                .filter(filterExpr)
                .outputFields(List.of("id", "text", "entity_ids", "relation_ids"))
                .build());
        return convertQueryResults(resp);
    }

    public List<String> queryPassageIds(String filter) {
        if (filter == null || filter.trim().isEmpty()) return List.of();

        QueryResp resp = client.query(QueryReq.builder()
                .collectionName(passageCollection)
                .filter(filter)
                .outputFields(List.of("id"))
                .build());
        return resp.getQueryResults().stream()
                .map(r -> r.getEntity().get("id").toString())
                .collect(Collectors.toList());
    }

    // ==================== Update / Upsert ====================

    public boolean upsertEntity(String entityId, String text, List<Float> embedding,
                                List<String> relationIds, List<String> passageIds) {
        List<Map<String, Object>> existing = getEntitiesByIds(List.of(entityId));
        if (existing.isEmpty()) return false;

        Map<String, Object> data = existing.get(0);
        JsonObject update = new JsonObject();
        update.addProperty("id", entityId);
        update.addProperty("text", text != null ? text : (String) data.get("text"));

        if (embedding != null) {
            JsonArray arr = new JsonArray();
            for (float v : embedding) arr.add(v);
            update.add("vector", arr);
        } else {
            List<Float> newEmbed = embeddingClient.embed(update.get("text").getAsString());
            JsonArray arr = new JsonArray();
            for (float v : newEmbed) arr.add(v);
            update.add("vector", arr);
        }

        if (relationIds != null) {
            JsonArray arr = new JsonArray();
            for (String id : relationIds) arr.add(id);
            update.add("relation_ids", arr);
        }
        if (passageIds != null) {
            JsonArray arr = new JsonArray();
            for (String id : passageIds) arr.add(id);
            update.add("passage_ids", arr);
        }

        client.upsert(UpsertReq.builder().collectionName(entityCollection).data(List.of(update)).build());
        return true;
    }

    public boolean upsertRelation(String relationId, String text, List<Float> embedding,
                                  List<String> entityIds, List<String> passageIds,
                                  String subject, String predicate, String object) {
        List<Map<String, Object>> existing = getRelationsByIds(List.of(relationId));
        if (existing.isEmpty()) return false;

        Map<String, Object> data = existing.get(0);
        JsonObject update = new JsonObject();
        update.addProperty("id", relationId);
        update.addProperty("text", text != null ? text : (String) data.get("text"));

        String finalText = update.get("text").getAsString();
        if (embedding != null) {
            JsonArray arr = new JsonArray();
            for (float v : embedding) arr.add(v);
            update.add("vector", arr);
        } else {
            List<Float> newEmbed = embeddingClient.embed(finalText);
            JsonArray arr = new JsonArray();
            for (float v : newEmbed) arr.add(v);
            update.add("vector", arr);
        }

        if (entityIds != null) {
            JsonArray arr = new JsonArray();
            for (String id : entityIds) arr.add(id);
            update.add("entity_ids", arr);
        }
        if (passageIds != null) {
            JsonArray arr = new JsonArray();
            for (String id : passageIds) arr.add(id);
            update.add("passage_ids", arr);
        }
        if (subject != null) update.addProperty("subject", subject);
        if (predicate != null) update.addProperty("predicate", predicate);
        if (object != null) update.addProperty("object", object);

        client.upsert(UpsertReq.builder().collectionName(relationCollection).data(List.of(update)).build());
        return true;
    }

    public boolean upsertPassage(String passageId, String text, List<Float> embedding,
                                 List<String> entityIds, List<String> relationIds) {
        List<Map<String, Object>> existing = getPassagesByIds(List.of(passageId));
        if (existing.isEmpty()) return false;

        Map<String, Object> data = existing.get(0);
        JsonObject update = new JsonObject();
        update.addProperty("id", passageId);
        update.addProperty("text", text != null ? text : (String) data.get("text"));

        String finalText = update.get("text").getAsString();
        if (embedding != null) {
            JsonArray arr = new JsonArray();
            for (float v : embedding) arr.add(v);
            update.add("vector", arr);
        } else {
            List<Float> newEmbed = embeddingClient.embed(finalText);
            JsonArray arr = new JsonArray();
            for (float v : newEmbed) arr.add(v);
            update.add("vector", arr);
        }

        if (entityIds != null) {
            JsonArray arr = new JsonArray();
            for (String id : entityIds) arr.add(id);
            update.add("entity_ids", arr);
        }
        if (relationIds != null) {
            JsonArray arr = new JsonArray();
            for (String id : relationIds) arr.add(id);
            update.add("relation_ids", arr);
        }

        client.upsert(UpsertReq.builder().collectionName(passageCollection).data(List.of(update)).build());
        return true;
    }

    // ==================== Delete Operations ====================

    public boolean deleteEntity(String entityId) {
        return deleteById(entityCollection, entityId);
    }

    public boolean deleteRelation(String relationId) {
        return deleteById(relationCollection, relationId);
    }

    public boolean deletePassage(String passageId) {
        return deleteById(passageCollection, passageId);
    }

    public int deleteEntities(List<String> entityIds) {
        return deleteByIds(entityCollection, entityIds);
    }

    public int deleteRelations(List<String> relationIds) {
        return deleteByIds(relationCollection, relationIds);
    }

    public int deletePassages(List<String> passageIds) {
        return deleteByIds(passageCollection, passageIds);
    }

    private boolean deleteById(String collection, String id) {
        try {
            client.delete(io.milvus.v2.service.vector.request.DeleteReq.builder()
                    .collectionName(collection)
                    .filter("id == \"" + id + "\"")
                    .build());
            return true;
        } catch (Exception e) {
            log.warn("Delete failed for {} in {}: {}", id, collection, e.getMessage());
            return false;
        }
    }

    private int deleteByIds(String collection, List<String> ids) {
        if (ids == null || ids.isEmpty()) return 0;
        String idsStr = ids.stream().map(id -> "\"" + id + "\"").collect(Collectors.joining(", "));
        client.delete(io.milvus.v2.service.vector.request.DeleteReq.builder()
                .collectionName(collection)
                .filter("id in [" + idsStr + "]")
                .build());
        return ids.size();
    }

    // ==================== Utility ====================

    public Map<String, Integer> getCollectionStats() {
        Map<String, Integer> stats = new HashMap<>();
        stats.put("entityCount", 0);
        stats.put("relationCount", 0);
        stats.put("passageCount", 0);

        try {
            // MilvusClientV2 doesn't have getCollectionStats directly;
            // We'll do a rough count by querying all IDs
        } catch (Exception e) {
            log.warn("Error getting collection stats: {}", e.getMessage());
        }
        return stats;
    }

    /**
     * Search relations with a custom filter expression (for eviction strategy).
     */
    public List<Map<String, Object>> searchRelationsWithFilter(List<Float> queryEmbedding,
                                                                int topK, String filter) {
        SearchReq.SearchReqBuilder builder = SearchReq.builder()
                .collectionName(relationCollection)
                .data(Collections.singletonList(new FloatVec(queryEmbedding)))
                .topK(topK)
                .outputFields(List.of("id", "text"));
        if (filter != null && !filter.isEmpty()) {
            builder.filter(filter);
        }

        SearchResp resp = client.search(builder.build());
        return convertSearchResults(resp);
    }

    public MilvusClientV2 getClient() {
        return client;
    }

    public String getRelationCollection() {
        return relationCollection;
    }

    // ==================== Result Conversion ====================

    private List<Map<String, Object>> convertSearchResults(SearchResp resp) {
        List<Map<String, Object>> results = new ArrayList<>();
        if (resp.getSearchResults() == null) return results;

        for (List<SearchResp.SearchResult> resultList : resp.getSearchResults()) {
            for (SearchResp.SearchResult r : resultList) {
                Map<String, Object> map = new HashMap<>();
                map.put("id", r.getId());
                map.put("distance", r.getScore());

                Map<String, Object> entity = r.getEntity();
                Map<String, Object> flat = new HashMap<>();
                if (entity != null) {
                    for (Map.Entry<String, Object> e : entity.entrySet()) {
                        flat.put(e.getKey(), e.getValue());
                    }
                }
                map.put("entity", flat);
                results.add(map);
            }
        }
        return results;
    }

    private List<Map<String, Object>> convertQueryResults(QueryResp resp) {
        List<Map<String, Object>> results = new ArrayList<>();
        if (resp.getQueryResults() == null) return results;

        for (QueryResp.QueryResult r : resp.getQueryResults()) {
            Map<String, Object> map = new HashMap<>();
            Map<String, Object> entity = r.getEntity();
            if (entity != null) {
                map.putAll(entity);
            }
            results.add(map);
        }
        return results;
    }

    public static String combineFilters(String... filters) {
        return Arrays.stream(filters)
                .filter(f -> f != null && !f.trim().isEmpty())
                .map(String::trim)
                .map(f -> "(" + f + ")")
                .collect(Collectors.joining(" and "));
    }
}
