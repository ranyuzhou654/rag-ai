# src/caching/multilayer_cache.py
"""
Multi-layer caching architecture for RAG-AI system
Implements 4-tier caching strategy as outlined in research document
"""

import asyncio
import json
import hashlib
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod

import redis
import numpy as np
from cachetools import LRUCache, TTLCache
from loguru import logger


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    def update_access(self):
        """Update access statistics"""
        self.last_accessed = datetime.now()
        self.access_count += 1


class CacheLayer(ABC):
    """Abstract base class for cache layers"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class MemoryCache(CacheLayer):
    """In-memory cache layer with LRU and TTL support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = LRUCache(maxsize=max_size)
        self.metadata: Dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        if key in self.cache:
            entry = self.metadata.get(key)
            if entry and entry.is_expired():
                await self.delete(key)
                self.misses += 1
                return None
            
            if entry:
                entry.update_access()
            
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache"""
        try:
            self.cache[key] = value
            
            # Calculate size estimate
            size_bytes = len(pickle.dumps(value)) if value is not None else 0
            
            self.metadata[key] = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl or self.default_ttl
            )
            
            return True
        except Exception as e:
            logger.error(f"❌ Error setting memory cache key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from memory cache"""
        try:
            self.cache.pop(key, None)
            self.metadata.pop(key, None)
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting memory cache key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear memory cache"""
        try:
            self.cache.clear()
            self.metadata.clear()
            self.hits = 0
            self.misses = 0
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing memory cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics"""
        total_size = sum(entry.size_bytes for entry in self.metadata.values())
        hit_rate = self.hits / max(self.hits + self.misses, 1)
        
        return {
            'type': 'memory',
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total_size_bytes': total_size,
            'average_size_bytes': total_size / max(len(self.cache), 1)
        }


class RedisCache(CacheLayer):
    """Redis-based distributed cache layer"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, 
                 password: Optional[str] = None, default_ttl: int = 3600):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self.client: Optional[redis.Redis] = None
        self.hits = 0
        self.misses = 0
        
        self._connect()
    
    def _connect(self):
        """Connect to Redis"""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False,  # Keep binary for pickle
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            self.client.ping()
            logger.info(f"✅ Connected to Redis: {self.host}:{self.port}/{self.db}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to connect to Redis: {e}")
            self.client = None
    
    def _is_connected(self) -> bool:
        """Check if Redis is connected"""
        try:
            return self.client is not None and self.client.ping()
        except:
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self._is_connected():
            self.misses += 1
            return None
        
        try:
            data = self.client.get(key)
            if data:
                self.hits += 1
                return pickle.loads(data)
            else:
                self.misses += 1
                return None
        except Exception as e:
            logger.error(f"❌ Error getting Redis cache key {key}: {e}")
            self.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        if not self._is_connected():
            return False
        
        try:
            data = pickle.dumps(value)
            return self.client.setex(key, ttl or self.default_ttl, data)
        except Exception as e:
            logger.error(f"❌ Error setting Redis cache key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        if not self._is_connected():
            return False
        
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"❌ Error deleting Redis cache key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear Redis cache"""
        if not self._is_connected():
            return False
        
        try:
            self.client.flushdb()
            self.hits = 0
            self.misses = 0
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing Redis cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        if not self._is_connected():
            return {'type': 'redis', 'connected': False}
        
        try:
            info = self.client.info()
            hit_rate = self.hits / max(self.hits + self.misses, 1)
            
            return {
                'type': 'redis',
                'connected': True,
                'size': info.get('db0', {}).get('keys', 0),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'memory_usage': info.get('used_memory', 0),
                'memory_usage_human': info.get('used_memory_human', '0B')
            }
        except Exception as e:
            logger.error(f"❌ Error getting Redis stats: {e}")
            return {'type': 'redis', 'connected': False, 'error': str(e)}


class FileCache(CacheLayer):
    """File-based persistent cache layer"""
    
    def __init__(self, cache_dir: Path, max_size_mb: int = 1000, default_ttl: int = 86400):
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cache_dir.mkdir(exist_ok=True)
        self.hits = 0
        self.misses = 0
        
        # Index file to track cache entries
        self.index_file = self.cache_dir / 'cache_index.json'
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict]:
        """Load cache index from disk"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"❌ Error loading cache index: {e}")
        return {}
    
    def _save_index(self):
        """Save cache index to disk"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, default=str, indent=2)
        except Exception as e:
            logger.error(f"❌ Error saving cache index: {e}")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key"""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.index:
            return True
        
        entry = self.index[key]
        if 'ttl_seconds' not in entry or entry['ttl_seconds'] is None:
            return False
        
        created_at = datetime.fromisoformat(entry['created_at'])
        return (datetime.now() - created_at).total_seconds() > entry['ttl_seconds']
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from file cache"""
        if key not in self.index or self._is_expired(key):
            self.misses += 1
            if key in self.index:
                await self.delete(key)
            return None
        
        try:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)
                
                # Update access statistics
                self.index[key]['last_accessed'] = datetime.now().isoformat()
                self.index[key]['access_count'] = self.index[key].get('access_count', 0) + 1
                
                self.hits += 1
                return value
            else:
                self.misses += 1
                # Remove from index if file doesn't exist
                del self.index[key]
                self._save_index()
                return None
        except Exception as e:
            logger.error(f"❌ Error getting file cache key {key}: {e}")
            self.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in file cache"""
        try:
            cache_path = self._get_cache_path(key)
            
            # Serialize and save
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Update index
            file_size = cache_path.stat().st_size
            now = datetime.now()
            
            self.index[key] = {
                'created_at': now.isoformat(),
                'last_accessed': now.isoformat(),
                'access_count': 1,
                'size_bytes': file_size,
                'ttl_seconds': ttl or self.default_ttl,
                'file_path': str(cache_path)
            }
            
            self._save_index()
            
            # Check if we need to cleanup old entries
            await self._cleanup_if_needed()
            
            return True
        except Exception as e:
            logger.error(f"❌ Error setting file cache key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from file cache"""
        try:
            if key in self.index:
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    cache_path.unlink()
                del self.index[key]
                self._save_index()
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting file cache key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear file cache"""
        try:
            for cache_file in self.cache_dir.glob('*.cache'):
                cache_file.unlink()
            
            self.index.clear()
            self._save_index()
            self.hits = 0
            self.misses = 0
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing file cache: {e}")
            return False
    
    async def _cleanup_if_needed(self):
        """Cleanup old entries if cache size exceeds limit"""
        total_size = sum(entry.get('size_bytes', 0) for entry in self.index.values())
        
        if total_size > self.max_size_bytes:
            # Sort by last accessed time (oldest first)
            sorted_entries = sorted(
                self.index.items(),
                key=lambda x: x[1].get('last_accessed', '1970-01-01')
            )
            
            # Remove oldest entries until under limit
            for key, entry in sorted_entries:
                if total_size <= self.max_size_bytes * 0.8:  # Keep 20% buffer
                    break
                
                await self.delete(key)
                total_size -= entry.get('size_bytes', 0)
            
            logger.info(f"🧹 File cache cleanup completed, size: {total_size / 1024 / 1024:.1f}MB")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get file cache statistics"""
        total_size = sum(entry.get('size_bytes', 0) for entry in self.index.values())
        hit_rate = self.hits / max(self.hits + self.misses, 1)
        
        return {
            'type': 'file',
            'size': len(self.index),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / 1024 / 1024,
            'max_size_mb': self.max_size_bytes / 1024 / 1024,
            'cache_dir': str(self.cache_dir)
        }


class MultiLayerCache:
    """
    Multi-layer cache system implementing 4-tier caching strategy:
    1. Memory Cache (L1) - Fastest, for frequently accessed data
    2. Redis Cache (L2) - Distributed, for shared data across instances
    3. File Cache (L3) - Persistent, for larger objects
    4. Vector Index Cache (L4) - Specialized for embeddings and search results
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize cache layers
        self.memory_cache = MemoryCache(
            max_size=config.get('memory_max_size', 1000),
            default_ttl=config.get('memory_ttl', 3600)
        )
        
        # Redis cache (optional)
        self.redis_cache = None
        if config.get('redis_enabled', True):
            try:
                self.redis_cache = RedisCache(
                    host=config.get('redis_host', 'localhost'),
                    port=config.get('redis_port', 6379),
                    db=config.get('redis_db', 0),
                    password=config.get('redis_password'),
                    default_ttl=config.get('redis_ttl', 3600)
                )
            except Exception as e:
                logger.warning(f"⚠️ Redis cache disabled: {e}")
        
        # File cache
        cache_dir = Path(config.get('file_cache_dir', './cache'))
        self.file_cache = FileCache(
            cache_dir=cache_dir,
            max_size_mb=config.get('file_max_size_mb', 1000),
            default_ttl=config.get('file_ttl', 86400)
        )
        
        # Vector cache for embeddings and search results
        self.vector_cache = MemoryCache(
            max_size=config.get('vector_max_size', 500),
            default_ttl=config.get('vector_ttl', 1800)
        )
        
        # Cache statistics
        self.total_requests = 0
        self.layer_hits = {'memory': 0, 'redis': 0, 'file': 0, 'vector': 0}
        
        logger.info("✅ Multi-layer cache system initialized")
    
    async def get(self, key: str, cache_type: str = 'general') -> Optional[Any]:
        """Get value from appropriate cache layer"""
        self.total_requests += 1
        
        # For vector/embedding data, use specialized cache
        if cache_type == 'vector':
            value = await self.vector_cache.get(key)
            if value is not None:
                self.layer_hits['vector'] += 1
                return value
        
        # Try memory cache first (L1)
        value = await self.memory_cache.get(key)
        if value is not None:
            self.layer_hits['memory'] += 1
            logger.debug(f"🎯 Memory cache hit: {key}")
            return value
        
        # Try Redis cache (L2)
        if self.redis_cache:
            value = await self.redis_cache.get(key)
            if value is not None:
                self.layer_hits['redis'] += 1
                logger.debug(f"🎯 Redis cache hit: {key}")
                # Promote to memory cache
                await self.memory_cache.set(key, value)
                return value
        
        # Try file cache (L3)
        value = await self.file_cache.get(key)
        if value is not None:
            self.layer_hits['file'] += 1
            logger.debug(f"🎯 File cache hit: {key}")
            # Promote to higher layers
            await self.memory_cache.set(key, value)
            if self.redis_cache:
                await self.redis_cache.set(key, value)
            return value
        
        logger.debug(f"❌ Cache miss: {key}")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                  cache_type: str = 'general', layers: Optional[List[str]] = None) -> bool:
        """Set value in appropriate cache layers"""
        
        # Default to all layers if not specified
        if layers is None:
            if cache_type == 'vector':
                layers = ['vector']
            else:
                layers = ['memory', 'redis', 'file']
        
        success = True
        
        # Set in specified layers
        if 'memory' in layers:
            success &= await self.memory_cache.set(key, value, ttl)
        
        if 'redis' in layers and self.redis_cache:
            success &= await self.redis_cache.set(key, value, ttl)
        
        if 'file' in layers:
            success &= await self.file_cache.set(key, value, ttl)
        
        if 'vector' in layers:
            success &= await self.vector_cache.set(key, value, ttl)
        
        if success:
            logger.debug(f"💾 Cached {key} in layers: {layers}")
        
        return success
    
    async def delete(self, key: str, layers: Optional[List[str]] = None) -> bool:
        """Delete key from cache layers"""
        if layers is None:
            layers = ['memory', 'redis', 'file', 'vector']
        
        success = True
        
        if 'memory' in layers:
            success &= await self.memory_cache.delete(key)
        
        if 'redis' in layers and self.redis_cache:
            success &= await self.redis_cache.delete(key)
        
        if 'file' in layers:
            success &= await self.file_cache.delete(key)
        
        if 'vector' in layers:
            success &= await self.vector_cache.delete(key)
        
        return success
    
    async def clear(self, layers: Optional[List[str]] = None) -> bool:
        """Clear cache layers"""
        if layers is None:
            layers = ['memory', 'redis', 'file', 'vector']
        
        success = True
        
        if 'memory' in layers:
            success &= await self.memory_cache.clear()
        
        if 'redis' in layers and self.redis_cache:
            success &= await self.redis_cache.clear()
        
        if 'file' in layers:
            success &= await self.file_cache.clear()
        
        if 'vector' in layers:
            success &= await self.vector_cache.clear()
        
        # Reset statistics
        self.total_requests = 0
        self.layer_hits = {'memory': 0, 'redis': 0, 'file': 0, 'vector': 0}
        
        logger.info(f"🧹 Cleared cache layers: {layers}")
        return success
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        redis_stats = self.redis_cache.get_stats() if self.redis_cache else {'type': 'redis', 'enabled': False}
        file_stats = self.file_cache.get_stats()
        vector_stats = self.vector_cache.get_stats()
        
        total_hits = sum(self.layer_hits.values())
        overall_hit_rate = total_hits / max(self.total_requests, 1)
        
        return {
            'overall': {
                'total_requests': self.total_requests,
                'total_hits': total_hits,
                'hit_rate': overall_hit_rate,
                'layer_hits': self.layer_hits
            },
            'layers': {
                'memory': memory_stats,
                'redis': redis_stats,
                'file': file_stats,
                'vector': vector_stats
            }
        }
    
    # Specialized methods for different cache types
    
    async def cache_query_embedding(self, query: str, embedding: np.ndarray, ttl: int = 1800) -> bool:
        """Cache query embedding vector"""
        key = f"query_emb:{hashlib.md5(query.encode()).hexdigest()}"
        return await self.set(key, embedding, ttl, cache_type='vector')
    
    async def get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get cached query embedding"""
        key = f"query_emb:{hashlib.md5(query.encode()).hexdigest()}"
        return await self.get(key, cache_type='vector')
    
    async def cache_search_results(self, query_hash: str, results: List[Dict], ttl: int = 3600) -> bool:
        """Cache search results"""
        key = f"search_results:{query_hash}"
        return await self.set(key, results, ttl, layers=['memory', 'redis'])
    
    async def get_search_results(self, query_hash: str) -> Optional[List[Dict]]:
        """Get cached search results"""
        key = f"search_results:{query_hash}"
        return await self.get(key)
    
    async def cache_generated_answer(self, context_hash: str, answer: str, citations: List[str], ttl: int = 7200) -> bool:
        """Cache generated answer with citations"""
        key = f"answer:{context_hash}"
        data = {'answer': answer, 'citations': citations, 'timestamp': datetime.now().isoformat()}
        return await self.set(key, data, ttl, layers=['memory', 'redis', 'file'])
    
    async def get_generated_answer(self, context_hash: str) -> Optional[Dict]:
        """Get cached generated answer"""
        key = f"answer:{context_hash}"
        return await self.get(key)
    
    async def cache_model_output(self, model_name: str, input_hash: str, output: Any, ttl: int = 86400) -> bool:
        """Cache model inference output"""
        key = f"model:{model_name}:{input_hash}"
        return await self.set(key, output, ttl, layers=['file'])
    
    async def get_model_output(self, model_name: str, input_hash: str) -> Optional[Any]:
        """Get cached model output"""
        key = f"model:{model_name}:{input_hash}"
        return await self.get(key)
    
    # Utility methods
    
    def generate_cache_key(self, *args, prefix: str = "") -> str:
        """Generate a consistent cache key from arguments"""
        combined = "|".join(str(arg) for arg in args)
        hash_key = hashlib.md5(combined.encode()).hexdigest()
        return f"{prefix}:{hash_key}" if prefix else hash_key
    
    async def warmup_cache(self, warmup_data: Dict[str, Any]):
        """Warmup cache with frequently accessed data"""
        logger.info("🔥 Starting cache warmup...")
        
        for key, data in warmup_data.items():
            await self.set(key, data['value'], data.get('ttl'), 
                          layers=data.get('layers', ['memory']))
        
        logger.info(f"✅ Cache warmup completed: {len(warmup_data)} entries")
    
    async def cleanup_expired_entries(self):
        """Cleanup expired entries across all layers"""
        logger.info("🧹 Starting cache cleanup...")
        
        # Memory and vector caches handle TTL automatically
        # File cache cleanup is handled in _cleanup_if_needed
        
        # For Redis, we rely on Redis's TTL mechanism
        
        logger.info("✅ Cache cleanup completed")


# Convenience function to create cache instance
def create_multilayer_cache(config: Dict[str, Any]) -> MultiLayerCache:
    """Create and configure multi-layer cache instance"""
    return MultiLayerCache(config)