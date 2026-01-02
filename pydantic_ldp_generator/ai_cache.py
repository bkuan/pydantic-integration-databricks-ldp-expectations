"""
AI Query Cache Module

Provides caching functionality for AI queries to avoid duplicate prompts and improve performance.
Shared between ValidatorAnalyzer and Generator classes.
"""

import hashlib
import json
import logging
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class CacheEntry:
    """Represents a cached AI response with metadata."""
    response: str
    timestamp: datetime
    model: str
    prompt_hash: str
    hit_count: int = 0


class AIQueryCache:
    """
    Thread-safe cache for AI query responses.
    
    Cache keys are based on:
    - Prompt content (deterministic for same input)
    - AI model name (different models may give different responses)
    - Validator source code hash (cache invalidation when validator changes)
    """
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        """
        Initialize AI query cache.
        
        Args:
            max_size: Maximum number of entries to cache
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self._cache: Dict[str, CacheEntry] = {}
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
    def _create_cache_key(self, prompt: str, model: str, validator_source_hash: Optional[str] = None) -> str:
        """
        Create a deterministic cache key from prompt and configuration.
        
        Args:
            prompt: The AI prompt text
            model: AI model name
            validator_source_hash: Optional hash of validator source code
            
        Returns:
            Hexadecimal cache key
        """
        # Normalize prompt by removing extra whitespace
        normalized_prompt = ' '.join(prompt.split())
        
        # Create content for hashing
        cache_content = {
            'prompt': normalized_prompt,
            'model': model,
            'validator_hash': validator_source_hash or 'none'
        }
        
        # Create deterministic JSON string
        cache_json = json.dumps(cache_content, sort_keys=True)
        
        # Generate hash
        cache_key = hashlib.sha256(cache_json.encode()).hexdigest()[:16]
        
        return cache_key
    
    def _is_entry_valid(self, entry: CacheEntry) -> bool:
        """Check if a cache entry is still valid (not expired)."""
        if self.ttl_hours <= 0:
            return True  # No expiration
            
        expiry_time = entry.timestamp + timedelta(hours=self.ttl_hours)
        return datetime.now() < expiry_time
    
    def _cleanup_expired_entries(self):
        """Remove expired entries from cache."""
        if self.ttl_hours <= 0:
            return  # No expiration enabled
            
        current_time = datetime.now()
        expired_keys = []
        
        for key, entry in self._cache.items():
            expiry_time = entry.timestamp + timedelta(hours=self.ttl_hours)
            if current_time >= expiry_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _evict_lru_entries(self):
        """Evict least recently used entries if cache is full."""
        if len(self._cache) <= self.max_size:
            return
        
        # Sort by hit_count (LRU approximation) and timestamp
        entries_by_usage = sorted(
            self._cache.items(),
            key=lambda x: (x[1].hit_count, x[1].timestamp)
        )
        
        # Remove oldest, least used entries
        entries_to_remove = len(self._cache) - self.max_size + 1
        for i in range(entries_to_remove):
            key_to_remove = entries_by_usage[i][0]
            del self._cache[key_to_remove]
            
        self.logger.debug(f"Evicted {entries_to_remove} LRU cache entries")
    
    def get(self, prompt: str, model: str, validator_source_hash: Optional[str] = None) -> Optional[str]:
        """
        Retrieve cached AI response if available.
        
        Args:
            prompt: The AI prompt text
            model: AI model name  
            validator_source_hash: Optional hash of validator source code
            
        Returns:
            Cached AI response or None if not found/expired
        """
        cache_key = self._create_cache_key(prompt, model, validator_source_hash)
        
        # Clean up expired entries periodically
        self._cleanup_expired_entries()
        
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            
            # Check if entry is still valid
            if self._is_entry_valid(entry):
                # Update hit count and statistics
                entry.hit_count += 1
                self.hits += 1
                
                self.logger.debug(f"AI cache HIT for key {cache_key[:8]}... (hits: {entry.hit_count})")
                return entry.response
            else:
                # Remove expired entry
                del self._cache[cache_key]
                self.logger.debug(f"Removed expired cache entry {cache_key[:8]}...")
        
        self.misses += 1
        self.logger.debug(f"AI cache MISS for key {cache_key[:8]}...")
        return None
    
    def put(self, prompt: str, model: str, response: str, validator_source_hash: Optional[str] = None):
        """
        Store AI response in cache.
        
        Args:
            prompt: The AI prompt text
            model: AI model name
            response: AI response to cache
            validator_source_hash: Optional hash of validator source code
        """
        cache_key = self._create_cache_key(prompt, model, validator_source_hash)
        
        # Evict entries if cache is full
        self._evict_lru_entries()
        
        # Create cache entry
        entry = CacheEntry(
            response=response,
            timestamp=datetime.now(),
            model=model,
            prompt_hash=cache_key,
            hit_count=0
        )
        
        self._cache[cache_key] = entry
        self.logger.debug(f"Cached AI response for key {cache_key[:8]}... (cache size: {len(self._cache)})")
    
    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        self.logger.info("AI cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self._cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests,
            'ttl_hours': self.ttl_hours
        }
    
    def get_validator_source_hash(self, validator_info) -> Optional[str]:
        """
        Generate hash of validator source code for cache invalidation.
        
        Args:
            validator_info: ValidatorInfo object with file_path and name
            
        Returns:
            Hash of validator source code or None if unable to read
        """
        try:
            from pathlib import Path
            import ast
            
            validator_file = Path(validator_info.file_path)
            if not validator_file.exists():
                return None
                
            with open(validator_file, 'r') as f:
                source_code = f.read()
            
            # Parse and find the specific validator function
            tree = ast.parse(source_code)
            validator_source = ""
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == validator_info.name:
                    validator_source = ast.get_source_segment(source_code, node) or ""
                    break
            
            if validator_source:
                return hashlib.md5(validator_source.encode()).hexdigest()[:8]
                
        except Exception as e:
            self.logger.debug(f"Could not generate validator source hash: {e}")
            
        return None


# Global cache instance - shared across all analyzer and generator instances
_global_ai_cache: Optional[AIQueryCache] = None


def get_ai_cache(config=None) -> AIQueryCache:
    """
    Get the global AI cache instance.
    
    Args:
        config: Optional configuration object with AI cache settings
        
    Returns:
        Global AIQueryCache instance
    """
    global _global_ai_cache
    
    if _global_ai_cache is None:
        # Initialize cache with config settings if available
        max_size = 1000
        ttl_hours = 24
        
        if config and hasattr(config, 'validation') and hasattr(config.validation, 'ai'):
            ai_config = config.validation.ai
            if hasattr(ai_config, 'cache_size_limit'):
                max_size = ai_config.cache_size_limit
            # TTL could be added to config later if needed
            
        _global_ai_cache = AIQueryCache(max_size=max_size, ttl_hours=ttl_hours)
    
    return _global_ai_cache


def clear_ai_cache():
    """Clear the global AI cache."""
    global _global_ai_cache
    if _global_ai_cache:
        _global_ai_cache.clear()
