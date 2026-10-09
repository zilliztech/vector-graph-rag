package com.zjkl.vectorgraphrag.graph;

import com.zjkl.vectorgraphrag.config.VectorGraphRagSettings;
import com.zjkl.vectorgraphrag.model.*;
import lombok.Getter;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Builds in-memory graph structures from documents with extracted triplets.
 * Manages entity-relation-passage adjacency mappings.
 */
public class GraphBuilder {

    private final VectorGraphRagSettings settings;

    // id -> text
    @Getter
    private final Map<String, String> entities = new LinkedHashMap<>();
    @Getter
    private final Map<String, String> relations = new LinkedHashMap<>();
    @Getter
    private final Map<String, String> passages = new LinkedHashMap<>();

    // ordered IDs
    @Getter
    private final List<String> entityIds = new ArrayList<>();
    @Getter
    private final List<String> relationIds = new ArrayList<>();
    @Getter
    private final List<String> passageIds = new ArrayList<>();

    // deduplication
    private final Map<String, String> entityNameToId = new HashMap<>();
    private final Map<String, String> relationTextToId = new HashMap<>();

    // triplet storage
    @Getter
    private final Map<String, Triplet> relationIdToTriplet = new HashMap<>();

    // adjacency
    @Getter
    private final Map<String, List<String>> entityToRelationIds = new HashMap<>();
    @Getter
    private final Map<String, List<String>> entityToPassageIds = new HashMap<>();
    @Getter
    private final Map<String, List<String>> relationToPassageIds = new HashMap<>();
    @Getter
    private final Map<String, List<String>> relationToEntityIds = new HashMap<>();
    @Getter
    private final Map<String, List<String>> passageToEntityIds = new HashMap<>();
    @Getter
    private final Map<String, List<String>> passageToRelationIds = new HashMap<>();

    public GraphBuilder(VectorGraphRagSettings settings) {
        this.settings = settings;
    }

    public void clear() {
        entities.clear();
        relations.clear();
        passages.clear();
        entityIds.clear();
        relationIds.clear();
        passageIds.clear();
        entityNameToId.clear();
        relationTextToId.clear();
        relationIdToTriplet.clear();
        entityToRelationIds.clear();
        entityToPassageIds.clear();
        relationToPassageIds.clear();
        relationToEntityIds.clear();
        passageToEntityIds.clear();
        passageToRelationIds.clear();
    }

    public ExtractionResult buildFromDocuments(List<Document> documents) {
        clear();

        for (Document doc : documents) {
            String passageId = doc.getId() != null ? doc.getId() : UUID.randomUUID().toString();
            passages.put(passageId, doc.getText());
            passageIds.add(passageId);
            if (doc.getId() == null) doc.setId(passageId);

            // Process triplets
            List<Triplet> triplets = doc.getTriplets();
            if (triplets == null) triplets = List.of();

            // Also check metadata for raw triplets
            Object rawTriplets = doc.getMetadata().get("triplets");
            if (rawTriplets instanceof List) {
                for (Object raw : (List<?>) rawTriplets) {
                    if (raw instanceof List && ((List<?>) raw).size() >= 3) {
                        List<?> parts = (List<?>) raw;
                        Triplet t = new Triplet(
                                String.valueOf(parts.get(0)),
                                String.valueOf(parts.get(1)),
                                String.valueOf(parts.get(2))
                        );
                        addRelation(t, passageId);
                    } else if (raw instanceof Triplet) {
                        addRelation((Triplet) raw, passageId);
                    }
                }
            } else {
                for (Triplet t : triplets) {
                    addRelation(t, passageId);
                }
            }
        }

        // Build result
        List<Entity> entityList = entityIds.stream()
                .map(eid -> Entity.builder().id(eid).name(entities.get(eid)).build())
                .collect(Collectors.toList());

        List<Relation> relationList = relationIds.stream()
                .map(rid -> Relation.builder()
                        .id(rid)
                        .text(relations.get(rid))
                        .triplet(relationIdToTriplet.get(rid))
                        .sourcePassageIds(new ArrayList<>(relationToPassageIds.getOrDefault(rid, List.of())))
                        .build())
                .collect(Collectors.toList());

        return ExtractionResult.builder()
                .documents(documents.stream().map(Document::getText).collect(Collectors.toList()))
                .entities(entityList)
                .relations(relationList)
                .entityToRelationIds(new HashMap<>(entityToRelationIds))
                .relationToPassageIds(new HashMap<>(relationToPassageIds))
                .build();
    }

    private String addRelation(Triplet triplet, String passageId) {
        String subject = normalizePhrase(triplet.getSubject());
        String predicate = normalizePhrase(triplet.getPredicate());
        String obj = normalizePhrase(triplet.getObject());
        String relationText = subject + " " + predicate + " " + obj;

        String relationId = relationTextToId.get(relationText);
        if (relationId == null) {
            relationId = UUID.randomUUID().toString();
            relations.put(relationId, relationText);
            relationIds.add(relationId);
            relationTextToId.put(relationText, relationId);
            relationIdToTriplet.put(relationId, triplet);

            String subjectId = addEntity(triplet.getSubject(), passageId);
            String objectId = addEntity(triplet.getObject(), passageId);

            entityToRelationIds.computeIfAbsent(subjectId, k -> new ArrayList<>()).add(relationId);
            entityToRelationIds.computeIfAbsent(objectId, k -> new ArrayList<>()).add(relationId);
            relationToEntityIds.put(relationId, List.of(subjectId, objectId));
        }

        // Link relation to passage
        relationToPassageIds.computeIfAbsent(relationId, k -> new ArrayList<>()).add(passageId);
        passageToRelationIds.computeIfAbsent(passageId, k -> new ArrayList<>()).add(relationId);

        return relationId;
    }

    private String addEntity(String entityName, String passageId) {
        String normalized = normalizePhrase(entityName);
        String entityId = entityNameToId.get(normalized);
        if (entityId == null) {
            entityId = UUID.randomUUID().toString();
            entities.put(entityId, normalized);
            entityIds.add(entityId);
            entityNameToId.put(normalized, entityId);
        }

        // Link entity to passage
        entityToPassageIds.computeIfAbsent(entityId, k -> new ArrayList<>()).add(passageId);
        passageToEntityIds.computeIfAbsent(passageId, k -> new ArrayList<>()).add(entityId);

        return entityId;
    }

    public static String normalizePhrase(String phrase) {
        if (phrase == null) return "";
        // Replace all non-alphanumeric chars with space, lowercase, trim
        return phrase.replaceAll("[^A-Za-z0-9 ]", " ").toLowerCase().trim();
    }

    public List<String> getEntityTexts() {
        return entityIds.stream().map(entities::get).collect(Collectors.toList());
    }

    public List<String> getRelationTexts() {
        return relationIds.stream().map(relations::get).collect(Collectors.toList());
    }

    public List<String> getPassageTexts() {
        return passageIds.stream().map(passages::get).collect(Collectors.toList());
    }
}
