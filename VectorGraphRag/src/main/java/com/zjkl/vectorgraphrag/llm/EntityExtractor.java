package com.zjkl.vectorgraphrag.llm;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.zjkl.vectorgraphrag.config.VectorGraphRagSettings;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.zjkl.vectorgraphrag.graph.GraphBuilder.normalizePhrase;

/**
 * Extracts named entities from text for query processing (NER).
 * Used during query time to identify entities in the user's question.
 */
@Slf4j
public class EntityExtractor {

    private static final String SYSTEM_PROMPT =
            "You're a very effective entity extraction system.";

    private static final String ONE_SHOT_INPUT =
            "Please extract all named entities that are important for solving the questions below.\n" +
            "Place the named entities in json format.\n\n" +
            "Question: Which magazine was started first Arthur's Magazine or First for Women?";

    private static final String ONE_SHOT_OUTPUT =
            "{\"named_entities\": [\"First for Women\", \"Arthur's Magazine\"]}";

    private static final String TEMPLATE = "\nQuestion: {}\n";

    private final OpenAiClient openAiClient;
    private final Gson gson;

    public EntityExtractor(VectorGraphRagSettings settings, OpenAiClient openAiClient) {
        this.openAiClient = openAiClient;
        this.gson = new Gson();
    }

    public List<String> extract(String question) {
        if (question == null || question.trim().isEmpty()) return List.of();

        try {
            List<Map<String, String>> examples = List.of(
                    Map.of("user", ONE_SHOT_INPUT, "assistant", ONE_SHOT_OUTPUT)
            );
            String response = openAiClient.chat(SYSTEM_PROMPT, examples,
                    TEMPLATE.replace("{}", question));

            return parseResponse(response);
        } catch (Exception e) {
            log.warn("Entity extraction failed, returning empty: {}", e.getMessage());
            return List.of();
        }
    }

    private List<String> parseResponse(String response) {
        try {
            JsonObject json = gson.fromJson(response, JsonObject.class);
            JsonArray entities = json.getAsJsonArray("named_entities");
            if (entities == null) {
                entities = json.getAsJsonArray("entities");
            }
            if (entities == null) return List.of();

            List<String> results = new ArrayList<>();
            for (int i = 0; i < entities.size(); i++) {
                String entity = entities.get(i).getAsString();
                if (entity != null && !entity.trim().isEmpty()) {
                    results.add(normalizePhrase(entity));
                }
            }
            return results;
        } catch (Exception e) {
            log.warn("Failed to parse NER response: {}", e.getMessage());
            return List.of();
        }
    }
}
