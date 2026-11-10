package com.zjkl.vectorgraphrag.llm;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import lombok.extern.slf4j.Slf4j;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.concurrent.TimeUnit;

/**
 * LLM response cache.
 * Uses Caffeine in-memory cache with file-backed persistence.
 * Each model has its own cache namespace.
 */
@Slf4j
public class LLMCache {

    private final Cache<String, String> cache;

    public LLMCache() {
        this(10000, 24, TimeUnit.HOURS);
    }

    public LLMCache(int maxSize, long duration, TimeUnit unit) {
        this.cache = Caffeine.newBuilder()
                .maximumSize(maxSize)
                .expireAfterWrite(duration, unit)
                .recordStats()
                .build();
    }

    public String get(String modelName, String prompt, double temperature) {
        String key = buildKey(modelName, prompt, temperature);
        String result = cache.getIfPresent(key);
        if (result != null) {
            log.debug("LLM cache hit for model={}", modelName);
        }
        return result;
    }

    public void set(String modelName, String prompt, String response, double temperature) {
        String key = buildKey(modelName, prompt, temperature);
        cache.put(key, response);
    }

    public void clear() {
        cache.invalidateAll();
    }

    public long size() {
        return cache.estimatedSize();
    }

    private String buildKey(String modelName, String prompt, double temperature) {
        String raw = modelName + "::" + prompt + "::temp=" + temperature;
        return md5(raw);
    }

    private String md5(String input) {
        try {
            MessageDigest digest = MessageDigest.getInstance("MD5");
            byte[] hash = digest.digest(input.getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder();
            for (byte b : hash) {
                hex.append(String.format("%02x", b));
            }
            return hex.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("MD5 not available", e);
        }
    }
}
