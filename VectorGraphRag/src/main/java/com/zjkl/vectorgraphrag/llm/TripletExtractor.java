package com.zjkl.vectorgraphrag.llm;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.zjkl.vectorgraphrag.config.VectorGraphRagSettings;
import com.zjkl.vectorgraphrag.model.Document;
import com.zjkl.vectorgraphrag.model.Triplet;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Extracts knowledge triplets (subject-predicate-object) from text using LLM.
 */
@Slf4j
public class TripletExtractor {

    private static final String SYSTEM_PROMPT =
            "You are an expert knowledge graph builder. Your task is to extract knowledge triplets from the given text.\n" +
            "\n" +
            "A triplet consists of:\n" +
            "- Subject: An entity (person, place, thing, concept, etc.)\n" +
            "- Predicate: The relationship between subject and object\n" +
            "- Object: Another entity\n" +
            "\n" +
            "Guidelines:\n" +
            "1. Extract all meaningful relationships from the text\n" +
            "2. Keep entities concise but complete\n" +
            "3. Use clear, specific predicates\n" +
            "4. Extract both explicit and implicit relationships\n" +
            "5. Ensure triplets are factually accurate based on the text\n" +
            "\n" +
            "Return your response as a JSON object with a \"triplets\" array, where each triplet is an array of [subject, predicate, object].";

    private static final String EXAMPLE_INPUT =
            "Text: Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of relativity, which revolutionized physics. Einstein worked at the Institute for Advanced Study in Princeton.";

    private static final String EXAMPLE_OUTPUT =
            "{\"triplets\": [[\"Albert Einstein\", \"was born in\", \"Ulm, Germany\"], [\"Albert Einstein\", \"was born in\", \"1879\"], [\"Albert Einstein\", \"developed\", \"the theory of relativity\"], [\"the theory of relativity\", \"revolutionized\", \"physics\"], [\"Albert Einstein\", \"worked at\", \"the Institute for Advanced Study\"], [\"the Institute for Advanced Study\", \"is located in\", \"Princeton\"]]}";

    private final OpenAiClient openAiClient;
    private final Gson gson;

    public TripletExtractor(VectorGraphRagSettings settings, OpenAiClient openAiClient) {
        this.openAiClient = openAiClient;
        this.gson = new Gson();
    }

    public List<Triplet> extract(String text) {
        if (text == null || text.trim().isEmpty()) return List.of();

        try {
            List<Map<String, String>> examples = List.of(
                    Map.of("user", EXAMPLE_INPUT, "assistant", EXAMPLE_OUTPUT)
            );
            String response = openAiClient.chat(SYSTEM_PROMPT, examples, "Text: " + text);
            return parseResponse(response);
        } catch (Exception e) {
            log.warn("Triplet extraction failed: {}", e.getMessage());
            return List.of();
        }
    }

    public List<Document> extractFromDocuments(List<Document> documents, boolean showProgress) {
        for (int i = 0; i < documents.size(); i++) {
            Document doc = documents.get(i);
            List<Triplet> triplets = extract(doc.getText());
            doc.setTriplets(triplets);
            // Store as raw arrays in metadata for backward compat
            List<List<String>> rawTriplets = triplets.stream()
                    .map(t -> List.of(t.getSubject(), t.getPredicate(), t.getObject()))
                    .collect(Collectors.toList());
            doc.getMetadata().put("triplets", rawTriplets);
            if (showProgress) {
                log.info("Extracted triplets for doc {}/{}: {} triplets", i + 1, documents.size(), triplets.size());
            }
        }
        return documents;
    }

    private List<Triplet> parseResponse(String response) {
        try {
            JsonObject json = gson.fromJson(response, JsonObject.class);
            JsonArray tripletsArray = json.getAsJsonArray("triplets");
            if (tripletsArray == null) return List.of();

            List<Triplet> results = new ArrayList<>();
            for (int i = 0; i < tripletsArray.size(); i++) {
                JsonArray arr = tripletsArray.get(i).getAsJsonArray();
                if (arr.size() >= 3) {
                    String subject = arr.get(0).getAsString().trim();
                    String predicate = arr.get(1).getAsString().trim();
                    String object = arr.get(2).getAsString().trim();
                    if (!subject.isEmpty() && !predicate.isEmpty() && !object.isEmpty()) {
                        results.add(new Triplet(subject, predicate, object));
                    }
                }
            }
            return results;
        } catch (Exception e) {
            log.warn("Failed to parse triplet response: {}", e.getMessage());
            return List.of();
        }
    }
}
