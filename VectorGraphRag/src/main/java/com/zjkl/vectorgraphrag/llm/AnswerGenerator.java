package com.zjkl.vectorgraphrag.llm;

import com.zjkl.vectorgraphrag.config.VectorGraphRagSettings;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

/**
 * Generates answers using retrieved passage context via LLM.
 */
@Slf4j
public class AnswerGenerator {

    private static final String ANSWER_PROMPT =
            "Use the following pieces of retrieved context to answer the question. " +
            "If there is not enough information in the retrieved context to answer the question, " +
            "just say that you don't know.\n\n" +
            "Question: {question}\n\n" +
            "Context: {context}\n\n" +
            "Answer:";

    private final OpenAiClient openAiClient;
    private final VectorGraphRagSettings settings;

    public AnswerGenerator(VectorGraphRagSettings settings, OpenAiClient openAiClient) {
        this.settings = settings;
        this.openAiClient = openAiClient;
    }

    public String generate(String question, List<String> passages) {
        String context = String.join("\n\n", passages);
        String prompt = ANSWER_PROMPT
                .replace("{question}", question)
                .replace("{context}", context);

        try {
            return openAiClient.chatWithMessages(List.of(
                    java.util.Map.of("role", "user", "content", prompt)
            ));
        } catch (Exception e) {
            log.warn("Answer generation failed: {}", e.getMessage());
            return "I don't know.";
        }
    }
}
