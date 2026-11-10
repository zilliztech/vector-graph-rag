package com.zjkl.vectorgraphrag;

import com.zjkl.vectorgraphrag.config.VectorGraphRagSettings;
import com.zjkl.vectorgraphrag.graph.Graph;
import com.zjkl.vectorgraphrag.graph.GraphBuilder;
import com.zjkl.vectorgraphrag.graph.GraphRetriever;
import com.zjkl.vectorgraphrag.graph.SubGraph;
import com.zjkl.vectorgraphrag.llm.AnswerGenerator;
import com.zjkl.vectorgraphrag.llm.EntityExtractor;
import com.zjkl.vectorgraphrag.llm.LLMCache;
import com.zjkl.vectorgraphrag.llm.LLMReranker;
import com.zjkl.vectorgraphrag.llm.OpenAiClient;
import com.zjkl.vectorgraphrag.llm.TripletExtractor;
import com.zjkl.vectorgraphrag.model.*;
import com.zjkl.vectorgraphrag.storage.EmbeddingClient;
import com.zjkl.vectorgraphrag.storage.MilvusStore;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Vector Graph RAG - Graph RAG using pure vector search with Milvus.
 *
 * Main entry point for building and querying a Graph RAG system.
 * Implements the full pipeline: triplet extraction → graph building →
 * embedding → Milvus indexing → multi-way retrieval → subgraph expansion →
 * LLM reranking → answer generation.
 *
 * Example:
 * <pre>{@code
 * VectorGraphRagSettings settings = VectorGraphRagSettings.builder()
 *     .openaiApiKey(System.getenv("OPENAI_API_KEY"))
 *     .milvusUri("./my_graph.db")
 *     .build();
 *
 * VectorGraphRAG rag = new VectorGraphRAG(settings);
 *
 * rag.addTexts(List.of(
 *     "Albert Einstein developed the theory of relativity.",
 *     "The theory of relativity revolutionized physics."
 * ));
 *
 * QueryResult result = rag.query("What did Einstein develop?");
 * System.out.println(result.getAnswer());
 * }</pre>
 */
@Slf4j
public class VectorGraphRAG {

    private final VectorGraphRagSettings settings;

    private final EmbeddingClient embeddingClient;
    private final MilvusStore store;
    private final GraphBuilder graphBuilder;
    private final TripletExtractor tripletExtractor;
    private final EntityExtractor entityExtractor;
    private final LLMReranker reranker;
    private final AnswerGenerator answerGenerator;
    private final LLMCache llmCache;
    private final OpenAiClient openAiClient;
    private final Graph graph;

    @Getter
    private GraphRetriever retriever;

    @Getter
    private ExtractionResult extractionResult;

    /**
     * Create VectorGraphRAG with custom settings.
     */
    public VectorGraphRAG(VectorGraphRagSettings settings) {
        this.settings = settings;
        this.llmCache = settings.isUseLlmCache() ? new LLMCache() : null;
        this.openAiClient = new OpenAiClient(settings, llmCache);
        this.embeddingClient = new EmbeddingClient(settings, openAiClient);
        this.store = new MilvusStore(settings, embeddingClient);
        this.graphBuilder = new GraphBuilder(settings);
        this.tripletExtractor = new TripletExtractor(settings, openAiClient);
        this.entityExtractor = new EntityExtractor(settings, openAiClient);
        this.reranker = new LLMReranker(settings, openAiClient);
        this.answerGenerator = new AnswerGenerator(settings, openAiClient);
        this.graph = new Graph(settings, store, embeddingClient);

        // Create collections if they don't exist
        store.createCollections(false);
    }

    /**
     * Create VectorGraphRAG with default settings (reads OPENAI_API_KEY from environment).
     */
    public static VectorGraphRAG createDefault() {
        VectorGraphRagSettings settings = VectorGraphRagSettings.builder()
                .openaiApiKey(System.getenv().getOrDefault("OPENAI_API_KEY", ""))
                .build();
        return new VectorGraphRAG(settings);
    }

    // ==================== Indexing ====================

    /**
     * Add text strings to the knowledge base.
     */
    public ExtractionResult addTexts(List<String> texts) {
        return addTexts(texts, null, true);
    }

    /**
     * Add text strings with extraction control.
     */
    public ExtractionResult addTexts(List<String> texts, List<String> ids, boolean extractTriplets) {
        List<Document> documents = new ArrayList<>();
        for (int i = 0; i < texts.size(); i++) {
            String docId = (ids != null && i < ids.size()) ? ids.get(i) : UUID.randomUUID().toString();
            documents.add(Document.builder()
                    .id(docId)
                    .text(texts.get(i))
                    .build());
        }
        return addDocuments(documents, extractTriplets, true);
    }

    /**
     * Add Document objects to the knowledge base.
     * Full indexing pipeline: extract triplets → build graph → embed → index in Milvus.
     */
    public ExtractionResult addDocuments(List<Document> documents, boolean extractTriplets, boolean showProgress) {
        // Ensure all documents have IDs
        for (Document doc : documents) {
            if (doc.getId() == null) {
                doc.setId(UUID.randomUUID().toString());
            }
        }

        // Extract triplets
        if (extractTriplets) {
            documents = tripletExtractor.extractFromDocuments(documents, showProgress);
        }

        // Build graph
        this.extractionResult = graphBuilder.buildFromDocuments(documents);

        // Generate embeddings
        if (showProgress) log.info("Generating embeddings...");

        List<String> entityTexts = graphBuilder.getEntityTexts();
        List<String> relationTexts = graphBuilder.getRelationTexts();
        List<String> passageTexts = graphBuilder.getPassageTexts();

        List<List<Float>> entityEmbeddings = entityTexts.isEmpty() ? List.of()
                : embeddingClient.embedBatch(entityTexts, showProgress);
        List<List<Float>> relationEmbeddings = relationTexts.isEmpty() ? List.of()
                : embeddingClient.embedBatch(relationTexts, showProgress);
        List<List<Float>> passageEmbeddings = passageTexts.isEmpty() ? List.of()
                : embeddingClient.embedBatch(passageTexts, showProgress);

        // Build metadata for adjacency
        List<Map<String, Object>> entityMetadatas = new ArrayList<>();
        for (String eid : graphBuilder.getEntityIds()) {
            Map<String, Object> meta = new HashMap<>();
            meta.put("relation_ids", graphBuilder.getEntityToRelationIds()
                    .getOrDefault(eid, List.of()));
            meta.put("passage_ids", graphBuilder.getEntityToPassageIds()
                    .getOrDefault(eid, List.of()));
            entityMetadatas.add(meta);
        }

        List<Map<String, Object>> relationMetadatas = new ArrayList<>();
        for (String rid : graphBuilder.getRelationIds()) {
            Map<String, Object> meta = new HashMap<>();
            meta.put("entity_ids", graphBuilder.getRelationToEntityIds()
                    .getOrDefault(rid, List.of()));
            meta.put("passage_ids", graphBuilder.getRelationToPassageIds()
                    .getOrDefault(rid, List.of()));

            Triplet triplet = graphBuilder.getRelationIdToTriplet().get(rid);
            if (triplet != null) {
                meta.put("subject", triplet.getSubject());
                meta.put("predicate", triplet.getPredicate());
                meta.put("object", triplet.getObject());
            }
            relationMetadatas.add(meta);
        }

        List<Map<String, Object>> passageMetadatas = new ArrayList<>();
        for (String pid : graphBuilder.getPassageIds()) {
            Map<String, Object> meta = new HashMap<>();
            meta.put("entity_ids", graphBuilder.getPassageToEntityIds()
                    .getOrDefault(pid, List.of()));
            meta.put("relation_ids", graphBuilder.getPassageToRelationIds()
                    .getOrDefault(pid, List.of()));
            passageMetadatas.add(meta);
        }

        // Drop and recreate for idempotent indexing (matching Python reference behavior)
        store.dropCollections();
        store.createCollections(true);
        if (showProgress) log.info("Inserting into Milvus...");

        store.insertEntities(entityTexts, graphBuilder.getEntityIds(),
                entityEmbeddings, entityMetadatas, showProgress);
        store.insertRelations(relationTexts, graphBuilder.getRelationIds(),
                relationEmbeddings, relationMetadatas, showProgress);
        store.insertPassages(passageTexts, graphBuilder.getPassageIds(),
                passageEmbeddings, passageMetadatas, showProgress);

        // Reset retriever
        this.retriever = null;

        return extractionResult;
    }

    /**
     * Add documents with pre-extracted triplets (skip LLM extraction).
     */
    public ExtractionResult addDocumentsWithTriplets(List<Map<String, Object>> documents) {
        List<Document> docs = new ArrayList<>();
        for (Map<String, Object> docData : documents) {
            String passage = (String) docData.getOrDefault("passage", docData.get("text"));
            if (passage == null) {
                throw new IllegalArgumentException("Each document must include 'passage' or 'text'");
            }

            String docId = (String) docData.getOrDefault("id", UUID.randomUUID().toString());

            @SuppressWarnings("unchecked")
            List<List<String>> rawTriplets = (List<List<String>>) docData.get("triplets");
            List<Triplet> triplets = new ArrayList<>();
            if (rawTriplets != null) {
                for (List<String> raw : rawTriplets) {
                    if (raw.size() >= 3) {
                        triplets.add(new Triplet(raw.get(0), raw.get(1), raw.get(2)));
                    }
                }
            }

            Document doc = Document.builder()
                    .id(docId)
                    .text(passage)
                    .triplets(triplets)
                    .build();
            // Store triplets in metadata for graph builder
            List<List<String>> metadataTriplets = triplets.stream()
                    .map(t -> List.of(t.getSubject(), t.getPredicate(), t.getObject()))
                    .collect(Collectors.toList());
            doc.getMetadata().put("triplets", metadataTriplets);

            docs.add(doc);
        }

        return addDocuments(docs, false, true);
    }

    // ==================== Query ====================

    /**
     * Full Graph RAG query pipeline.
     */
    public QueryResult query(String question) {
        return query(question, true, null, null, null, null, null);
    }

    /**
     * Query with custom parameters.
     */
    public QueryResult query(String question, boolean useReranking,
                              Integer entityTopK, Integer relationTopK,
                              Float entitySimilarityThreshold, Float relationSimilarityThreshold,
                              String filter) {
        GraphRetriever ret = getOrCreateRetriever();

        RetrievalResult retrievalResult = ret.retrieve(question, entityTopK, relationTopK,
                entitySimilarityThreshold, relationSimilarityThreshold,
                null, filter);

        // Build retrieval detail
        RetrievalDetail retrievalDetail = RetrievalDetail.builder()
                .entityIds(retrievalResult.getEntityIds())
                .entityTexts(retrievalResult.getEntityTexts())
                .entityScores(retrievalResult.getEntityScores())
                .relationIds(retrievalResult.getRelationIds())
                .relationTexts(retrievalResult.getRelationTexts())
                .relationScores(retrievalResult.getRelationScores())
                .build();

        // Get candidate relations from subgraph
        List<String> candidateIds = retrievalResult.getExpandedRelationIds();
        List<String> candidateTexts = retrievalResult.getExpandedRelationTexts();

        // Rerank
        RerankResult rerankResult = null;
        List<String> rerankedIds;
        List<String> rerankedTexts;

        if (useReranking && !candidateIds.isEmpty()) {
            Map.Entry<List<String>, List<String>> result = reranker.rerank(
                    question, candidateIds, candidateTexts);
            rerankedIds = result.getKey();
            rerankedTexts = result.getValue();
            rerankResult = RerankResult.builder()
                    .selectedRelationIds(rerankedIds)
                    .selectedRelationTexts(rerankedTexts)
                    .build();
        } else {
            rerankedIds = candidateIds.size() > settings.getFinalTopK()
                    ? candidateIds.subList(0, settings.getFinalTopK())
                    : candidateIds;
            rerankedTexts = candidateTexts.size() > settings.getFinalTopK()
                    ? candidateTexts.subList(0, settings.getFinalTopK())
                    : candidateTexts;
        }

        // Get passages from reranked relations
        List<String> finalPassages = getPassagesFromRelations(rerankedIds, filter);

        // Hybrid fallback: if graph retrieval yields fewer passages than needed,
        // supplement with naive vector search on passages directly
        if (finalPassages.size() < settings.getFinalTopK()) {
            int needed = settings.getFinalTopK() - finalPassages.size();
            Set<String> existing = new HashSet<>(finalPassages);
            List<String> naivePassages = ret.retrievePassagesNaive(question, needed, filter);
            for (String p : naivePassages) {
                if (existing.add(p)) {
                    finalPassages.add(p);
                    if (finalPassages.size() >= settings.getFinalTopK()) break;
                }
            }
        }

        if (finalPassages.size() > settings.getFinalTopK()) {
            finalPassages = finalPassages.subList(0, settings.getFinalTopK());
        }

        // Generate answer
        String answer = answerGenerator.generate(question, finalPassages);

        return QueryResult.builder()
                .query(question)
                .answer(answer)
                .queryEntities(retrievalResult.getQueryEntities())
                .retrievedPassages(finalPassages)
                .retrievedRelations(retrievalResult.getRelationTexts())
                .expandedRelations(candidateTexts)
                .rerankedRelations(rerankedTexts)
                .subgraph(retrievalResult.getSubgraph())
                .passages(finalPassages)
                .retrievalDetail(retrievalDetail)
                .rerankResult(rerankResult)
                .evictionResult(EvictionResult.builder()
                        .occurred(retrievalResult.isEvictionOccurred())
                        .beforeCount(retrievalResult.getEvictionBeforeCount())
                        .afterCount(retrievalResult.getEvictionAfterCount())
                        .build())
                .build();
    }

    /**
     * Simple query that returns just the answer string.
     */
    public String querySimple(String question) {
        return query(question).getAnswer();
    }

    /**
     * Naive RAG query (direct passage retrieval, no graph).
     */
    public QueryResult queryNaive(String question, String filter) {
        GraphRetriever ret = getOrCreateRetriever();
        List<String> passages = ret.retrievePassagesNaive(question, settings.getFinalTopK(), filter);
        String answer = answerGenerator.generate(question, passages);

        return QueryResult.builder()
                .query(question)
                .answer(answer)
                .retrievedPassages(passages)
                .build();
    }

    // ==================== Utility ====================

    /**
     * Get passages associated with given relation IDs.
     */
    private List<String> getPassagesFromRelations(List<String> relationIds, String filter) {
        if (relationIds == null || relationIds.isEmpty()) return List.of();

        List<Map<String, Object>> relationData = store.getRelationsByIds(relationIds);
        Set<String> passageIds = new LinkedHashSet<>();
        for (Map<String, Object> rel : relationData) {
            @SuppressWarnings("unchecked")
            List<String> pids = (List<String>) rel.get("passage_ids");
            if (pids != null) passageIds.addAll(pids);
        }

        if (passageIds.isEmpty()) return List.of();

        List<Map<String, Object>> passageData = store.getPassagesByIds(new ArrayList<>(passageIds), filter);
        Map<String, String> idToText = new LinkedHashMap<>();
        for (Map<String, Object> p : passageData) {
            idToText.put(p.get("id").toString(), p.get("text").toString());
        }

        return passageIds.stream()
                .map(idToText::get)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
    }

    /**
     * Get knowledge base statistics.
     */
    public Map<String, Integer> getStats() {
        if (extractionResult == null) {
            Map<String, Integer> stats = new HashMap<>();
            stats.put("entities", 0);
            stats.put("relations", 0);
            stats.put("passages", 0);
            return stats;
        }
        Map<String, Integer> stats = new HashMap<>();
        stats.put("entities", extractionResult.getEntities().size());
        stats.put("relations", extractionResult.getRelations().size());
        stats.put("passages", extractionResult.getDocuments().size());
        return stats;
    }

    /**
     * Reset the knowledge base, removing all data.
     */
    public void reset() {
        store.dropCollections();
        store.createCollections(true);
        this.extractionResult = null;
        this.retriever = null;
    }

    private GraphRetriever getOrCreateRetriever() {
        if (retriever == null) {
            retriever = new GraphRetriever(settings, store, embeddingClient, entityExtractor);
        }
        return retriever;
    }

    // ==================== Accessors ====================

    public Graph getGraph() { return graph; }
    public MilvusStore getStore() { return store; }
    public EmbeddingClient getEmbeddingClient() { return embeddingClient; }
}
