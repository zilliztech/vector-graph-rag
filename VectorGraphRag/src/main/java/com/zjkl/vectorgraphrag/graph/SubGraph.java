package com.zjkl.vectorgraphrag.graph;

import com.zjkl.vectorgraphrag.storage.MilvusStore;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Lazy-loading subgraph for graph expansion.
 * Starts from seed entities/relations and expands by fetching
 * neighbor data on-demand from Milvus storage.
 */
@Slf4j
public class SubGraph {

    private final MilvusStore store;

    private final Set<String> entityIds = new LinkedHashSet<>();
    private final Set<String> relationIds = new LinkedHashSet<>();
    private final Set<String> passageIds = new LinkedHashSet<>();

    private final Map<String, SubGraphEntity> entityMap = new HashMap<>();
    private final Map<String, SubGraphRelation> relationMap = new HashMap<>();
    private final Map<String, SubGraphPassage> passageMap = new HashMap<>();

    @Getter
    private final List<Map<String, Object>> expansionHistory = new ArrayList<>();

    public SubGraph(MilvusStore store) {
        this.store = store;
    }

    // ==================== Add Initial Nodes ====================

    public SubGraph addEntities(List<String> ids) {
        Set<String> newIds = new LinkedHashSet<>(ids);
        newIds.removeAll(entityIds);
        if (!newIds.isEmpty()) {
            entityIds.addAll(newIds);
            fetchEntities(new ArrayList<>(newIds));

            if (expansionHistory.isEmpty()) {
                Map<String, Object> step = new LinkedHashMap<>();
                step.put("step", 0);
                step.put("operation", "init");
                step.put("addedEntityIds", new ArrayList<>(newIds));
                step.put("addedRelationIds", List.of());
                expansionHistory.add(step);
            } else {
                @SuppressWarnings("unchecked")
                List<String> added = (List<String>) expansionHistory.get(0).get("addedEntityIds");
                added.addAll(newIds);
            }
        }
        return this;
    }

    public SubGraph addRelations(List<String> ids) {
        Set<String> newIds = new LinkedHashSet<>(ids);
        newIds.removeAll(relationIds);
        if (!newIds.isEmpty()) {
            relationIds.addAll(newIds);
            fetchRelations(new ArrayList<>(newIds));

            if (expansionHistory.isEmpty()) {
                Map<String, Object> step = new LinkedHashMap<>();
                step.put("step", 0);
                step.put("operation", "init");
                step.put("addedEntityIds", List.of());
                step.put("addedRelationIds", new ArrayList<>(newIds));
                expansionHistory.add(step);
            } else {
                @SuppressWarnings("unchecked")
                List<String> added = (List<String>) expansionHistory.get(0).get("addedRelationIds");
                added.addAll(newIds);
            }
        }
        return this;
    }

    // ==================== Expansion ====================

    public SubGraph expand(int degree) {
        // Step 0: From initial entities -> relations, merge with initial relations
        Set<String> initNewRelations = new LinkedHashSet<>();
        for (String eid : entityIds) {
            SubGraphEntity entity = entityMap.get(eid);
            if (entity != null) {
                for (String rid : entity.getRelationIds()) {
                    if (!relationIds.contains(rid)) {
                        initNewRelations.add(rid);
                    }
                }
            }
        }

        if (!initNewRelations.isEmpty()) {
            fetchRelations(new ArrayList<>(initNewRelations));
            relationIds.addAll(initNewRelations);
        }

        Map<String, Object> initStep = new LinkedHashMap<>();
        initStep.put("step", expansionHistory.size());
        initStep.put("operation", "init_merge");
        initStep.put("newRelationIds", new ArrayList<>(initNewRelations));
        initStep.put("totalEntities", entityIds.size());
        initStep.put("totalRelations", relationIds.size());
        expansionHistory.add(initStep);

        // For each degree: relations -> entities -> relations
        for (int step = 0; step < degree; step++) {
            Set<String> stepNewEntities = new LinkedHashSet<>();
            Set<String> stepNewRelations = new LinkedHashSet<>();

            // From current relations -> entities
            for (String rid : relationIds) {
                SubGraphRelation relation = relationMap.get(rid);
                if (relation != null) {
                    for (String eid : relation.getEntityIds()) {
                        if (!entityIds.contains(eid)) {
                            stepNewEntities.add(eid);
                        }
                    }
                }
            }

            // Fetch new entities
            if (!stepNewEntities.isEmpty()) {
                fetchEntities(new ArrayList<>(stepNewEntities));
                entityIds.addAll(stepNewEntities);
            }

            // From new entities -> relations
            for (String eid : stepNewEntities) {
                SubGraphEntity entity = entityMap.get(eid);
                if (entity != null) {
                    for (String rid : entity.getRelationIds()) {
                        if (!relationIds.contains(rid)) {
                            stepNewRelations.add(rid);
                        }
                    }
                }
            }

            // Fetch new relations
            if (!stepNewRelations.isEmpty()) {
                fetchRelations(new ArrayList<>(stepNewRelations));
                relationIds.addAll(stepNewRelations);
            }

            Map<String, Object> stepRecord = new LinkedHashMap<>();
            stepRecord.put("step", expansionHistory.size());
            stepRecord.put("operation", "expand_degree_" + (step + 1));
            stepRecord.put("newEntityIds", new ArrayList<>(stepNewEntities));
            stepRecord.put("newRelationIds", new ArrayList<>(stepNewRelations));
            stepRecord.put("totalEntities", entityIds.size());
            stepRecord.put("totalRelations", relationIds.size());
            expansionHistory.add(stepRecord);
        }

        // Collect passages from all relations
        for (String rid : relationIds) {
            SubGraphRelation relation = relationMap.get(rid);
            if (relation != null) {
                passageIds.addAll(relation.getPassageIds());
            }
        }

        // Fetch passages
        if (!passageIds.isEmpty()) {
            fetchPassages(new ArrayList<>(passageIds));
        }

        return this;
    }

    // ==================== Data Fetching ====================

    private void fetchEntities(List<String> ids) {
        List<String> toFetch = ids.stream()
                .filter(id -> !entityMap.containsKey(id))
                .collect(Collectors.toList());
        if (toFetch.isEmpty()) return;

        List<Map<String, Object>> results = store.getEntitiesByIds(toFetch);
        for (Map<String, Object> r : results) {
            SubGraphEntity entity = SubGraphEntity.builder()
                    .id(safeGet(r, "id"))
                    .name(safeGet(r, "text"))
                    .relationIds(safeGetList(r, "relation_ids"))
                    .passageIds(safeGetList(r, "passage_ids"))
                    .build();
            entityMap.put(entity.getId(), entity);
        }
    }

    private void fetchRelations(List<String> ids) {
        List<String> toFetch = ids.stream()
                .filter(id -> !relationMap.containsKey(id))
                .collect(Collectors.toList());
        if (toFetch.isEmpty()) return;

        List<Map<String, Object>> results = store.getRelationsByIds(toFetch);
        for (Map<String, Object> r : results) {
            String text = safeGet(r, "text");
            String subject = r.getOrDefault("subject", "").toString();
            String predicate = r.getOrDefault("predicate", "").toString();
            String obj = r.getOrDefault("object", "").toString();

            // Fallback parsing
            if (subject.isEmpty() && predicate.isEmpty() && obj.isEmpty()) {
                String[] parts = text.split(" ", 3);
                subject = parts.length > 0 ? parts[0] : "";
                predicate = parts.length > 1 ? parts[1] : "";
                obj = parts.length > 2 ? parts[2] : "";
            }

            SubGraphRelation relation = SubGraphRelation.builder()
                    .id(safeGet(r, "id"))
                    .text(text)
                    .subject(subject)
                    .predicate(predicate)
                    .object(obj)
                    .entityIds(safeGetList(r, "entity_ids"))
                    .passageIds(safeGetList(r, "passage_ids"))
                    .build();
            relationMap.put(relation.getId(), relation);
        }
    }

    private void fetchPassages(List<String> ids) {
        List<String> toFetch = ids.stream()
                .filter(id -> !passageMap.containsKey(id))
                .collect(Collectors.toList());
        if (toFetch.isEmpty()) return;

        List<Map<String, Object>> results = store.getPassagesByIds(toFetch);
        for (Map<String, Object> r : results) {
            SubGraphPassage passage = SubGraphPassage.builder()
                    .id(safeGet(r, "id"))
                    .text(safeGet(r, "text"))
                    .entityIds(safeGetList(r, "entity_ids"))
                    .relationIds(safeGetList(r, "relation_ids"))
                    .build();
            passageMap.put(passage.getId(), passage);
        }
    }

    // ==================== Accessors ====================

    public Set<String> getEntityIds() { return Collections.unmodifiableSet(entityIds); }
    public Set<String> getRelationIds() { return Collections.unmodifiableSet(relationIds); }
    public Set<String> getPassageIds() { return Collections.unmodifiableSet(passageIds); }

    public List<SubGraphEntity> getEntities() {
        return entityIds.stream().map(entityMap::get).filter(Objects::nonNull).collect(Collectors.toList());
    }

    public List<SubGraphRelation> getRelations() {
        return relationIds.stream().map(relationMap::get).filter(Objects::nonNull).collect(Collectors.toList());
    }

    public List<SubGraphPassage> getPassages() {
        return passageIds.stream().map(passageMap::get).filter(Objects::nonNull).collect(Collectors.toList());
    }

    public List<String> getEntityNames() {
        return getEntities().stream().map(SubGraphEntity::getName).collect(Collectors.toList());
    }

    public List<String> getRelationTexts() {
        return getRelations().stream().map(SubGraphRelation::getText).collect(Collectors.toList());
    }

    public List<String> getPassageTexts() {
        return getPassages().stream().map(SubGraphPassage::getText).collect(Collectors.toList());
    }

    // ==================== Helpers ====================

    @SuppressWarnings("unchecked")
    private List<String> safeGetList(Map<String, Object> map, String key) {
        Object val = map.get(key);
        if (val instanceof List) {
            return ((List<Object>) val).stream()
                    .map(Object::toString)
                    .collect(Collectors.toList());
        }
        return List.of();
    }

    private String safeGet(Map<String, Object> map, String key) {
        Object val = map.get(key);
        return val != null ? val.toString() : "";
    }
}
