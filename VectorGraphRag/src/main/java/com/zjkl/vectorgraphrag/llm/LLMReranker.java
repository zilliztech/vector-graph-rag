package com.zjkl.vectorgraphrag.llm;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.zjkl.vectorgraphrag.config.VectorGraphRagSettings;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.stream.Collectors;

/**
 * LLM-based reranker for candidate relations.
 * Uses few-shot prompting with chain-of-thought to select the most relevant relations.
 */
@Slf4j
public class LLMReranker {

    // 3 few-shot examples for 1-hop, 2-hop, and 3-hop reasoning
    private static final String EXAMPLE_1_INPUT =
            "I will provide you with a set of relationship descriptions from a knowledge graph. Select exactly 5 relationships most useful for answering this multi-hop question.\n\n" +
            "Return JSON with \"thought_process\" and \"useful_relations\" (list of 5 relation lines, most useful first).\n\n" +
            "Question:\nWhen did Lothair Ii's mother die?\n\n" +
            "Relationship descriptions:\n" +
            "[53] bertha married to theobald of arles\n" +
            "[54] bertha married to adalbert ii of tuscany\n" +
            "[42] lothair ii son of ermengarde of tours\n" +
            "[43] lothair ii married to teutberga\n" +
            "[41] lothair ii son of emperor lothair i\n" +
            "[60] lothair ii husband of waldrada\n" +
            "[67] waldrada was mistress of lothair ii\n";

    private static final String EXAMPLE_1_OUTPUT =
            "{\"thought_process\": \"2-hop question: First find Lothair II's mother (relation [42]: Ermengarde of Tours), then find death date. [41] gives father for family context.\", " +
            "\"useful_relations\": [\"[42] lothair ii son of ermengarde of tours\", \"[41] lothair ii son of emperor lothair i\", \"[43] lothair ii married to teutberga\", \"[60] lothair ii husband of waldrada\", \"[67] waldrada was mistress of lothair ii\"]}";

    private static final String EXAMPLE_2_INPUT =
            "I will provide you with a set of relationship descriptions from a knowledge graph. Select exactly 5 relationships most useful for answering this multi-hop question.\n\n" +
            "Return JSON with \"thought_process\" and \"useful_relations\" (list of 5 relation lines, most useful first).\n\n" +
            "Question:\nWhat country is the composer of \"Erta Eterna\" from?\n\n" +
            "Relationship descriptions:\n" +
            "[12] terra eterna composed by paulo flores\n" +
            "[15] paulo flores born in angola\n" +
            "[18] paulo flores genre is semba\n" +
            "[22] angola located in africa\n" +
            "[25] semba originated in angola\n" +
            "[30] paulo flores nationality angolan\n";

    private static final String EXAMPLE_2_OUTPUT =
            "{\"thought_process\": \"2-hop question: First find composer of Terra Eterna ([12]: Paulo Flores), then find his country ([15] born in Angola or [30] nationality Angolan).\", " +
            "\"useful_relations\": [\"[12] terra eterna composed by paulo flores\", \"[15] paulo flores born in angola\", \"[30] paulo flores nationality angolan\", \"[22] angola located in africa\", \"[25] semba originated in angola\"]}";

    private static final String EXAMPLE_3_INPUT =
            "I will provide you with a set of relationship descriptions from a knowledge graph. Select exactly 5 relationships most useful for answering this multi-hop question.\n\n" +
            "Return JSON with \"thought_process\" and \"useful_relations\" (list of 5 relation lines, most useful first).\n\n" +
            "Question:\nWho is the director of the film that won the award also won by \"The Hurt Locker\"?\n\n" +
            "Relationship descriptions:\n" +
            "[5] the hurt locker won academy award best picture\n" +
            "[8] the hurt locker directed by kathryn bigelow\n" +
            "[12] moonlight won academy award best picture\n" +
            "[15] moonlight directed by barry jenkins\n" +
            "[20] la la land won golden globe best musical\n" +
            "[25] barry jenkins born in miami\n";

    private static final String EXAMPLE_3_OUTPUT =
            "{\"thought_process\": \"3-hop question: (1) Find award won by The Hurt Locker ([5]: Academy Award Best Picture), (2) Find another film with same award ([12]: Moonlight), (3) Find director ([15]: Barry Jenkins).\", " +
            "\"useful_relations\": [\"[5] the hurt locker won academy award best picture\", \"[12] moonlight won academy award best picture\", \"[15] moonlight directed by barry jenkins\", \"[8] the hurt locker directed by kathryn bigelow\", \"[25] barry jenkins born in miami\"]}";

    private final OpenAiClient openAiClient;
    private final Gson gson;
    private final VectorGraphRagSettings settings;

    public LLMReranker(VectorGraphRagSettings settings, OpenAiClient openAiClient) {
        this.settings = settings;
        this.openAiClient = openAiClient;
        this.gson = new Gson();
    }

    public Map.Entry<List<String>, List<String>> rerank(String query,
                                                         List<String> relationIds,
                                                         List<String> relationTexts) {
        if (relationIds == null || relationIds.isEmpty()) {
            return Map.entry(List.of(), List.of());
        }

        // Format relations
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < relationIds.size(); i++) {
            sb.append("[").append(relationIds.get(i)).append("] ")
              .append(relationTexts.get(i)).append("\n");
        }
        String relationDescriptions = sb.toString();

        // Build prompt
        String userPrompt = "Question:\n" + query + "\n\nRelationship descriptions:\n" + relationDescriptions;

        try {
            List<Map<String, String>> messages = new ArrayList<>();
            messages.add(Map.of("role", "user", "content", EXAMPLE_1_INPUT));
            messages.add(Map.of("role", "assistant", "content", EXAMPLE_1_OUTPUT));
            messages.add(Map.of("role", "user", "content", EXAMPLE_2_INPUT));
            messages.add(Map.of("role", "assistant", "content", EXAMPLE_2_OUTPUT));
            messages.add(Map.of("role", "user", "content", EXAMPLE_3_INPUT));
            messages.add(Map.of("role", "assistant", "content", EXAMPLE_3_OUTPUT));
            messages.add(Map.of("role", "user", "content", userPrompt));

            String response = openAiClient.chatWithMessages(messages);
            return parseResponse(response, new HashSet<>(relationIds), relationIds, relationTexts);

        } catch (Exception e) {
            log.warn("Reranking failed, using top relations: {}", e.getMessage());
            int limit = Math.min(settings.getFinalTopK(), relationIds.size());
            return Map.entry(
                    relationIds.subList(0, limit),
                    relationTexts.subList(0, limit)
            );
        }
    }

    private Map.Entry<List<String>, List<String>> parseResponse(String response,
                                                                  Set<String> validIds,
                                                                  List<String> allIds,
                                                                  List<String> allTexts) {
        try {
            JsonObject json = gson.fromJson(response, JsonObject.class);
            JsonArray usefulRelations = json.getAsJsonArray("useful_relations");
            if (usefulRelations == null) {
                return Map.entry(List.of(), List.of());
            }

            // Build id->text lookup
            Map<String, String> idToText = new LinkedHashMap<>();
            for (int i = 0; i < allIds.size(); i++) {
                idToText.put(allIds.get(i), allTexts.get(i));
            }

            List<String> selectedIds = new ArrayList<>();
            Set<String> seen = new HashSet<>();

            for (int i = 0; i < usefulRelations.size(); i++) {
                String line = usefulRelations.get(i).getAsString();
                String relId = extractIdFromLine(line);
                if (relId == null) continue;

                if (validIds.contains(relId) && seen.add(relId)) {
                    selectedIds.add(relId);
                } else if (!validIds.contains(relId)) {
                    // Try to correct by text matching
                    String lineText = line.substring(line.indexOf("]") + 1).trim();
                    for (int j = 0; j < allTexts.size(); j++) {
                        if (allTexts.get(j).trim().equalsIgnoreCase(lineText)
                                && seen.add(allIds.get(j))) {
                            selectedIds.add(allIds.get(j));
                            break;
                        }
                    }
                }
            }

            List<String> selectedTexts = selectedIds.stream()
                    .map(id -> idToText.getOrDefault(id, ""))
                    .collect(Collectors.toList());

            return Map.entry(selectedIds, selectedTexts);

        } catch (Exception e) {
            log.warn("Failed to parse reranker response: {}", e.getMessage());
            return Map.entry(List.of(), List.of());
        }
    }

    private String extractIdFromLine(String line) {
        int start = line.indexOf("[");
        int end = line.indexOf("]");
        if (start >= 0 && end > start) {
            return line.substring(start + 1, end).trim();
        }
        return null;
    }
}
