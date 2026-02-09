"""
LLM response caching mechanism.

Caches LLM responses to avoid repeated API calls for the same inputs.
Uses MD5 hashing of the prompt to create unique cache keys.
"""

import os
import json
import hashlib
from typing import Optional, Dict, Any
from pathlib import Path


class LLMCache:
    """
    File-based LLM response cache.
    
    Caches LLM responses using MD5 hash of the prompt as the key.
    Each model has its own cache file.
    
    Example:
        >>> cache = LLMCache()
        >>> cache.get("gpt-4o-mini", "What is 2+2?", temperature=0)
        None  # Cache miss
        >>> cache.set("gpt-4o-mini", "What is 2+2?", "4", temperature=0)
        >>> cache.get("gpt-4o-mini", "What is 2+2?", temperature=0)
        "4"  # Cache hit
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the LLM cache.
        
        Args:
            cache_dir: Directory to store cache files. Defaults to ./llm_cache
        """
        self.cache_dir = Path(cache_dir or "./llm_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, str]] = {}
    
    def _get_cache_file(self, model_name: str) -> Path:
        """Get the cache file path for a model."""
        safe_model_name = model_name.replace("/", "-").replace(".", "_")
        return self.cache_dir / f"{safe_model_name}.json"
    
    def _load_cache(self, model_name: str) -> Dict[str, str]:
        """Load cache from file for a model."""
        if model_name in self._cache:
            return self._cache[model_name]
        
        cache_file = self._get_cache_file(model_name)
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    self._cache[model_name] = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache[model_name] = {}
        else:
            self._cache[model_name] = {}
        
        return self._cache[model_name]
    
    def _save_cache(self, model_name: str) -> None:
        """Save cache to file for a model."""
        cache_file = self._get_cache_file(model_name)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(self._cache.get(model_name, {}), f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def _compute_hash(prompt: str, temperature: float = 0.0, **kwargs) -> str:
        """Compute MD5 hash for a prompt with parameters."""
        cache_key = f"{prompt}\n\ntemperature={temperature}"
        for k, v in sorted(kwargs.items()):
            cache_key += f"\n{k}={v}"
        
        return hashlib.md5(cache_key.encode("utf-8")).hexdigest()
    
    def get(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.0,
        **kwargs
    ) -> Optional[str]:
        """
        Get cached response for a prompt.
        
        Args:
            model_name: The model name.
            prompt: The prompt text.
            temperature: Temperature setting.
            **kwargs: Additional parameters to include in cache key.
        
        Returns:
            Cached response if exists, None otherwise.
        """
        cache = self._load_cache(model_name)
        hash_key = self._compute_hash(prompt, temperature, **kwargs)
        return cache.get(hash_key)
    
    def set(
        self,
        model_name: str,
        prompt: str,
        response: str,
        temperature: float = 0.0,
        **kwargs
    ) -> None:
        """
        Cache a response for a prompt.
        
        Args:
            model_name: The model name.
            prompt: The prompt text.
            response: The response to cache.
            temperature: Temperature setting.
            **kwargs: Additional parameters to include in cache key.
        """
        cache = self._load_cache(model_name)
        hash_key = self._compute_hash(prompt, temperature, **kwargs)
        cache[hash_key] = response
        self._cache[model_name] = cache
        self._save_cache(model_name)
    
    def clear(self, model_name: Optional[str] = None) -> None:
        """
        Clear cache.
        
        Args:
            model_name: If provided, clear only that model's cache.
                       If None, clear all caches.
        """
        if model_name:
            cache_file = self._get_cache_file(model_name)
            if cache_file.exists():
                cache_file.unlink()
            if model_name in self._cache:
                del self._cache[model_name]
        else:
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        stats = {}
        for cache_file in self.cache_dir.glob("*.json"):
            model_name = cache_file.stem
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    stats[model_name] = len(data)
            except (json.JSONDecodeError, IOError):
                stats[model_name] = 0
        return stats


# Global cache instance
_global_cache: Optional[LLMCache] = None


def get_llm_cache(cache_dir: Optional[str] = None) -> LLMCache:
    """Get or create the global LLM cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = LLMCache(cache_dir)
    return _global_cache


def set_llm_cache(cache: LLMCache) -> None:
    """Set the global LLM cache instance."""
    global _global_cache
    _global_cache = cache
