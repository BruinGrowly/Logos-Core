"""
Sensory Cache - Vectorization Caching Layer
============================================

Caches vectorized content to avoid expensive re-computation.
Uses MD5 hash of file header as cache key.

Storage: workspace/sensory_index.json

Usage:
    from memory.sensory_cache import SensoryCache
    
    cache = SensoryCache()
    
    # Check cache first
    vector = cache.get_vector(filepath)
    if vector is None:
        # Compute vector
        vector = memory.encode(content)
        cache.store_vector(filepath, vector)
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, List


class SensoryCache:
    """
    JSON-based cache for vectorized file content.
    
    Key: MD5 hash of first 1KB of file
    Value: Vector embedding (stored as list)
    """
    
    def __init__(self, cache_path: str = None):
        """
        Initialize the sensory cache.
        
        Args:
            cache_path: Path to cache file. Defaults to workspace/sensory_index.json
        """
        if cache_path is None:
            # Default to workspace folder relative to project root
            project_root = Path(__file__).parent.parent
            cache_dir = project_root / "workspace"
            cache_dir.mkdir(exist_ok=True)
            self.cache_path = cache_dir / "sensory_index.json"
        else:
            self.cache_path = Path(cache_path)
        
        self._cache = self._load_cache()
        
        # Track statistics
        self.hits = 0
        self.misses = 0
    
    def _load_cache(self) -> dict:
        """Load cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(self._cache, f)
        except IOError as e:
            print(f"[SensoryCache] Warning: Could not save cache: {e}")
    
    def _compute_hash(self, filepath: str) -> Optional[str]:
        """
        Compute MD5 hash of the first 1KB of a file.
        
        Args:
            filepath: Path to file
            
        Returns:
            MD5 hex digest, or None if unreadable
        """
        try:
            with open(filepath, 'rb') as f:
                first_kb = f.read(1024)
            return hashlib.md5(first_kb).hexdigest()
        except (IOError, OSError):
            return None
    
    def get_vector(self, filepath: str) -> Optional[List[float]]:
        """
        Get cached vector for a file.
        
        Args:
            filepath: Path to file
            
        Returns:
            Cached vector as list, or None if not cached
        """
        file_hash = self._compute_hash(filepath)
        if file_hash is None:
            return None
        
        if file_hash in self._cache:
            self.hits += 1
            return self._cache[file_hash]
        
        self.misses += 1
        return None
    
    def store_vector(self, filepath: str, vector) -> bool:
        """
        Store a vector in the cache.
        
        Args:
            filepath: Path to the source file
            vector: Vector embedding (numpy array or list)
            
        Returns:
            True if stored successfully, False otherwise
        """
        file_hash = self._compute_hash(filepath)
        if file_hash is None:
            return False
        
        # Convert numpy array to list if needed
        if hasattr(vector, 'tolist'):
            vector = vector.tolist()
        
        self._cache[file_hash] = vector
        self._save_cache()
        return True
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache = {}
        self._save_cache()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total,
            'hit_rate_percent': round(hit_rate, 1),
            'cached_entries': len(self._cache)
        }
    
    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)


if __name__ == "__main__":
    # Quick test
    cache = SensoryCache()
    print(f"[SensoryCache] Loaded {len(cache)} cached entries")
    print(f"[SensoryCache] Cache file: {cache.cache_path}")
