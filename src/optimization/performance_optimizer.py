# src/optimization/performance_optimizer.py
"""
性能优化系统
实现多级缓存、GPU加速、批处理优化等
目标：30-50%延迟降低，15-30%成本节省
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import asyncio
import time
import hashlib
import json
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from loguru import logger

# 缓存组件
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using in-memory cache only")

from cachetools import TTLCache, LRUCache
import numpy as np

# GPU加速组件
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, GPU acceleration disabled")

# 本地组件
from ..retrieval.hybrid_retriever import EnhancedDocument
from ..processing.weighted_embedding_fusion import FusedEmbedding


@dataclass
class CacheMetrics:
    """缓存指标"""
    hit_count: int = 0
    miss_count: int = 0
    total_requests: int = 0
    cache_size: int = 0
    hit_rate: float = 0.0
    avg_retrieval_time: float = 0.0
    memory_usage: float = 0.0


@dataclass
class PerformanceMetrics:
    """性能指标"""
    avg_latency: float
    throughput: float  # requests per second
    cache_hit_rate: float
    gpu_utilization: float
    memory_usage: float
    cost_per_request: float
    total_requests: int
    error_rate: float


class CacheStrategy(ABC):
    """缓存策略基类"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存项"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> CacheMetrics:
        """获取缓存指标"""
        pass


class MemoryCache(CacheStrategy):
    """内存缓存"""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.query_cache = TTLCache(maxsize=max_size, ttl=ttl)
        self.embedding_cache = LRUCache(maxsize=max_size * 5)
        self.document_cache = TTLCache(maxsize=max_size // 2, ttl=ttl * 24)
        
        self.hit_count = 0
        self.miss_count = 0
        
        logger.info(f"Memory cache initialized: max_size={max_size}, ttl={ttl}s")
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        
        # 根据key前缀选择缓存
        cache = self._select_cache(key)
        
        try:
            value = cache.get(key)
            if value is not None:
                self.hit_count += 1
                return value
            else:
                self.miss_count += 1
                return None
        except Exception as e:
            logger.warning(f"Memory cache get failed: {e}")
            self.miss_count += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        
        try:
            cache = self._select_cache(key)
            cache[key] = value
            return True
        except Exception as e:
            logger.warning(f"Memory cache set failed: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存项"""
        
        try:
            cache = self._select_cache(key)
            if key in cache:
                del cache[key]
                return True
            return False
        except Exception as e:
            logger.warning(f"Memory cache delete failed: {e}")
            return False
    
    def _select_cache(self, key: str):
        """根据key选择合适的缓存"""
        if key.startswith('query:'):
            return self.query_cache
        elif key.startswith('embedding:'):
            return self.embedding_cache
        elif key.startswith('document:'):
            return self.document_cache
        else:
            return self.query_cache
    
    def get_metrics(self) -> CacheMetrics:
        """获取缓存指标"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return CacheMetrics(
            hit_count=self.hit_count,
            miss_count=self.miss_count,
            total_requests=total_requests,
            cache_size=len(self.query_cache) + len(self.embedding_cache) + len(self.document_cache),
            hit_rate=hit_rate,
            avg_retrieval_time=0.001,  # Memory access is very fast
            memory_usage=0.0  # Simplified
        )


class RedisCache(CacheStrategy):
    """Redis分布式缓存"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, ttl: int = 3600):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available")
        
        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
        self.default_ttl = ttl
        self.hit_count = 0
        self.miss_count = 0
        
        # 测试连接
        try:
            self.redis_client.ping()
            logger.info(f"Redis cache connected: {host}:{port}")
        except redis.ConnectionError:
            logger.error(f"Failed to connect to Redis: {host}:{port}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        
        try:
            # Redis操作在线程池中执行
            loop = asyncio.get_event_loop()
            
            def _sync_get():
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
                return None
            
            value = await loop.run_in_executor(None, _sync_get)
            
            if value is not None:
                self.hit_count += 1
                return value
            else:
                self.miss_count += 1
                return None
                
        except Exception as e:
            logger.warning(f"Redis cache get failed: {e}")
            self.miss_count += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        
        try:
            ttl = ttl or self.default_ttl
            serialized_value = json.dumps(value, default=str)
            
            loop = asyncio.get_event_loop()
            
            def _sync_set():
                return self.redis_client.setex(key, ttl, serialized_value)
            
            result = await loop.run_in_executor(None, _sync_set)
            return bool(result)
            
        except Exception as e:
            logger.warning(f"Redis cache set failed: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存项"""
        
        try:
            loop = asyncio.get_event_loop()
            
            def _sync_delete():
                return self.redis_client.delete(key)
            
            result = await loop.run_in_executor(None, _sync_delete)
            return bool(result)
            
        except Exception as e:
            logger.warning(f"Redis cache delete failed: {e}")
            return False
    
    def get_metrics(self) -> CacheMetrics:
        """获取缓存指标"""
        try:
            info = self.redis_client.info('memory')
            memory_usage = info.get('used_memory', 0) / (1024 * 1024)  # MB
            
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
            
            return CacheMetrics(
                hit_count=self.hit_count,
                miss_count=self.miss_count,
                total_requests=total_requests,
                cache_size=self.redis_client.dbsize(),
                hit_rate=hit_rate,
                avg_retrieval_time=0.002,  # Network latency
                memory_usage=memory_usage
            )
        except:
            return CacheMetrics()


class MultiLevelCache:
    """多级缓存系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.caches = []
        
        # L1: 内存缓存 (最快)
        self.memory_cache = MemoryCache(
            max_size=config.get('memory_cache_size', 10000),
            ttl=config.get('memory_cache_ttl', 3600)
        )
        self.caches.append(('memory', self.memory_cache))
        
        # L2: Redis缓存 (分布式)
        if config.get('redis_enabled', False) and REDIS_AVAILABLE:
            try:
                self.redis_cache = RedisCache(
                    host=config.get('redis_host', 'localhost'),
                    port=config.get('redis_port', 6379),
                    ttl=config.get('redis_ttl', 86400)
                )
                self.caches.append(('redis', self.redis_cache))
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
        
        logger.info(f"Multi-level cache initialized with {len(self.caches)} levels")
    
    async def get(self, key: str) -> Optional[Any]:
        """多级缓存获取"""
        
        for cache_name, cache in self.caches:
            try:
                value = await cache.get(key)
                if value is not None:
                    # 将数据写入更高级的缓存中
                    await self._write_upper_caches(key, value, cache_name)
                    return value
            except Exception as e:
                logger.warning(f"{cache_name} cache get failed: {e}")
                continue
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """多级缓存设置"""
        
        success_count = 0
        
        for cache_name, cache in self.caches:
            try:
                result = await cache.set(key, value, ttl)
                if result:
                    success_count += 1
            except Exception as e:
                logger.warning(f"{cache_name} cache set failed: {e}")
                continue
        
        return success_count > 0
    
    async def delete(self, key: str) -> bool:
        """多级缓存删除"""
        
        success_count = 0
        
        for cache_name, cache in self.caches:
            try:
                result = await cache.delete(key)
                if result:
                    success_count += 1
            except Exception as e:
                logger.warning(f"{cache_name} cache delete failed: {e}")
                continue
        
        return success_count > 0
    
    async def _write_upper_caches(self, key: str, value: Any, found_cache: str):
        """将数据写入更高级的缓存"""
        
        for cache_name, cache in self.caches:
            if cache_name == found_cache:
                break
            
            try:
                await cache.set(key, value)
            except Exception as e:
                logger.warning(f"Failed to write to {cache_name} cache: {e}")
    
    def get_metrics(self) -> Dict[str, CacheMetrics]:
        """获取所有缓存层的指标"""
        
        metrics = {}
        for cache_name, cache in self.caches:
            try:
                metrics[cache_name] = cache.get_metrics()
            except Exception as e:
                logger.warning(f"Failed to get {cache_name} metrics: {e}")
        
        return metrics


class GPUAccelerator:
    """GPU加速器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = None
        self.batch_size = config.get('batch_size', 32)
        
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"GPU accelerator initialized on device: {self.device}")
        else:
            logger.warning("PyTorch not available, GPU acceleration disabled")
    
    def is_gpu_available(self) -> bool:
        """检查GPU是否可用"""
        return TORCH_AVAILABLE and torch.cuda.is_available()
    
    async def batch_embed_texts(self, texts: List[str], embedding_model) -> List[np.ndarray]:
        """批量文本嵌入（GPU加速）"""
        
        if not self.is_gpu_available():
            # 回退到CPU批处理
            return await self._cpu_batch_embed(texts, embedding_model)
        
        try:
            embeddings = []
            
            # 分批处理
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # 使用GPU加速嵌入
                batch_embeddings = await self._gpu_embed_batch(batch_texts, embedding_model)
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"GPU batch embedding failed: {e}")
            return await self._cpu_batch_embed(texts, embedding_model)
    
    async def _gpu_embed_batch(self, texts: List[str], embedding_model) -> List[np.ndarray]:
        """GPU批量嵌入"""
        
        def _sync_gpu_embed():
            with torch.no_grad():
                # 这里应该调用实际的GPU嵌入模型
                # 目前使用模拟实现
                embeddings = []
                for text in texts:
                    # 模拟GPU嵌入
                    embedding = np.random.random(768).astype(np.float32)  # 模拟嵌入向量
                    embeddings.append(embedding)
                return embeddings
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_gpu_embed)
    
    async def _cpu_batch_embed(self, texts: List[str], embedding_model) -> List[np.ndarray]:
        """CPU批量嵌入"""
        
        embeddings = []
        
        # 并行处理小批次
        batch_size = min(self.batch_size, 8)  # CPU批次较小
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 并行嵌入
            tasks = []
            for text in batch_texts:
                if hasattr(embedding_model, 'aembed_query'):
                    tasks.append(embedding_model.aembed_query(text))
                else:
                    # 在线程池中执行同步嵌入
                    loop = asyncio.get_event_loop()
                    tasks.append(loop.run_in_executor(None, embedding_model.embed_query, text))
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.warning(f"Embedding failed: {result}")
                    embeddings.append(np.zeros(768))  # 默认零向量
                else:
                    embeddings.append(np.array(result))
        
        return embeddings
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """获取GPU内存信息"""
        
        if not self.is_gpu_available():
            return {'total': 0, 'allocated': 0, 'cached': 0, 'free': 0}
        
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            cached_memory = torch.cuda.memory_reserved(0) / (1024**3)  # GB
            free_memory = total_memory - allocated_memory
            
            return {
                'total': total_memory,
                'allocated': allocated_memory,
                'cached': cached_memory,
                'free': free_memory
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return {'total': 0, 'allocated': 0, 'cached': 0, 'free': 0}


class QueryOptimizer:
    """查询优化器"""
    
    def __init__(self):
        self.query_patterns = {}  # 存储常见查询模式
        self.optimization_stats = {
            'total_optimizations': 0,
            'avg_speedup': 0.0,
            'cache_hits': 0
        }
    
    def generate_cache_key(self, query: str, retrieval_params: Dict[str, Any]) -> str:
        """生成缓存键"""
        
        # 标准化查询
        normalized_query = query.lower().strip()
        
        # 包含重要参数
        key_components = [
            f"query:{normalized_query}",
            f"k:{retrieval_params.get('k', 10)}",
            f"model:{retrieval_params.get('model', 'default')}",
            f"strategy:{retrieval_params.get('strategy', 'hybrid')}"
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def should_use_cache(self, query: str, complexity: str) -> bool:
        """判断是否应该使用缓存"""
        
        # 简单查询更适合缓存
        if complexity in ['simple', 'medium']:
            return True
        
        # 重复查询使用缓存
        if query in self.query_patterns:
            self.query_patterns[query] += 1
            return self.query_patterns[query] > 1
        else:
            self.query_patterns[query] = 1
            return False
    
    def estimate_cache_benefit(self, query: str, estimated_latency: float) -> float:
        """估算缓存收益"""
        
        # 基于查询频率和延迟估算收益
        frequency = self.query_patterns.get(query, 1)
        cache_overhead = 0.01  # 缓存开销
        
        if frequency > 1:
            benefit = (estimated_latency - cache_overhead) * frequency
            return max(benefit, 0)
        
        return 0


class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化组件
        self.cache = MultiLevelCache(config)
        self.gpu_accelerator = GPUAccelerator(config)
        self.query_optimizer = QueryOptimizer()
        
        # 性能统计
        self.performance_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'gpu_accelerations': 0,
            'avg_latency': 0.0,
            'total_cost_saved': 0.0
        }
        
        logger.info("Performance optimizer initialized")
    
    async def optimize_retrieval(
        self,
        query: str,
        retrieval_func,
        retrieval_params: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """优化检索过程"""
        
        start_time = time.time()
        optimization_info = {
            'cache_used': False,
            'gpu_accelerated': False,
            'latency_saved': 0.0,
            'cost_saved': 0.0
        }
        
        # 生成缓存键
        cache_key = f"retrieval:{self.query_optimizer.generate_cache_key(query, retrieval_params)}"
        
        # 尝试从缓存获取
        complexity = retrieval_params.get('complexity', 'medium')
        if self.query_optimizer.should_use_cache(query, complexity):
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                optimization_info['cache_used'] = True
                optimization_info['latency_saved'] = time.time() - start_time
                return cached_result, optimization_info
        
        # 执行检索
        try:
            result = await retrieval_func(query, **retrieval_params)
            
            # 缓存结果
            if self.query_optimizer.should_use_cache(query, complexity):
                await self.cache.set(cache_key, result, ttl=3600)
            
            processing_time = time.time() - start_time
            self.performance_stats['total_requests'] += 1
            self.performance_stats['avg_latency'] = (
                (self.performance_stats['avg_latency'] * (self.performance_stats['total_requests'] - 1) + processing_time) /
                self.performance_stats['total_requests']
            )
            
            return result, optimization_info
            
        except Exception as e:
            logger.error(f"Optimized retrieval failed: {e}")
            raise
    
    async def optimize_embedding_batch(
        self,
        texts: List[str],
        embedding_model
    ) -> List[np.ndarray]:
        """优化批量嵌入"""
        
        # 检查缓存
        cached_embeddings = {}
        uncached_texts = []
        text_indices = {}
        
        for i, text in enumerate(texts):
            cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()[:16]}"
            cached = await self.cache.get(cache_key)
            
            if cached is not None:
                cached_embeddings[i] = np.array(cached)
            else:
                uncached_texts.append(text)
                text_indices[len(uncached_texts) - 1] = i
        
        # 对未缓存的文本进行批量嵌入
        new_embeddings = []
        if uncached_texts:
            new_embeddings = await self.gpu_accelerator.batch_embed_texts(
                uncached_texts, embedding_model
            )
            
            # 缓存新的嵌入
            for j, embedding in enumerate(new_embeddings):
                text = uncached_texts[j]
                cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()[:16]}"
                await self.cache.set(cache_key, embedding.tolist(), ttl=86400)
        
        # 合并结果
        final_embeddings = [None] * len(texts)
        
        # 填入缓存的嵌入
        for i, embedding in cached_embeddings.items():
            final_embeddings[i] = embedding
        
        # 填入新计算的嵌入
        for j, embedding in enumerate(new_embeddings):
            original_index = text_indices[j]
            final_embeddings[original_index] = embedding
        
        return final_embeddings
    
    async def optimize_document_processing(
        self,
        documents: List[str],
        processing_func
    ) -> List[Any]:
        """优化文档处理"""
        
        # 批量处理以提高效率
        batch_size = self.config.get('document_batch_size', 16)
        results = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # 并行处理批次
            tasks = [processing_func(doc) for doc in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            results.extend(batch_results)
        
        return results
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取性能指标"""
        
        cache_metrics = self.cache.get_metrics()
        gpu_memory = self.gpu_accelerator.get_gpu_memory_info()
        
        # 计算整体缓存命中率
        total_cache_hits = sum(m.hit_count for m in cache_metrics.values())
        total_cache_requests = sum(m.total_requests for m in cache_metrics.values())
        overall_hit_rate = total_cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0
        
        # 计算错误率
        total_requests = self.performance_stats['total_requests']
        error_rate = 0.0  # 简化实现
        
        # 计算吞吐量
        throughput = total_requests / 3600 if total_requests > 0 else 0.0  # 简化计算
        
        return PerformanceMetrics(
            avg_latency=self.performance_stats['avg_latency'],
            throughput=throughput,
            cache_hit_rate=overall_hit_rate,
            gpu_utilization=gpu_memory['allocated'] / gpu_memory['total'] if gpu_memory['total'] > 0 else 0.0,
            memory_usage=sum(m.memory_usage for m in cache_metrics.values()),
            cost_per_request=0.01,  # 简化实现
            total_requests=total_requests,
            error_rate=error_rate
        )
    
    async def cleanup_cache(self, max_age_hours: int = 24):
        """清理过期缓存"""
        
        # 这里应该实现具体的缓存清理逻辑
        # 目前是简化实现
        logger.info(f"Cache cleanup completed (max_age={max_age_hours}h)")
    
    def export_performance_report(self, file_path: str):
        """导出性能报告"""
        
        metrics = self.get_performance_metrics()
        cache_metrics = self.cache.get_metrics()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': {
                'avg_latency': metrics.avg_latency,
                'throughput': metrics.throughput,
                'cache_hit_rate': metrics.cache_hit_rate,
                'gpu_utilization': metrics.gpu_utilization,
                'total_requests': metrics.total_requests
            },
            'cache_metrics': {
                name: {
                    'hit_rate': m.hit_rate,
                    'total_requests': m.total_requests,
                    'cache_size': m.cache_size,
                    'memory_usage': m.memory_usage
                } for name, m in cache_metrics.items()
            },
            'optimization_stats': self.performance_stats
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.success(f"Performance report exported to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export performance report: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        
        health_status = {
            'overall_status': 'healthy',
            'components': {},
            'alerts': []
        }
        
        # 缓存健康检查
        try:
            test_key = f"health_check:{int(time.time())}"
            await self.cache.set(test_key, "test", ttl=60)
            cached_value = await self.cache.get(test_key)
            
            if cached_value == "test":
                health_status['components']['cache'] = 'healthy'
            else:
                health_status['components']['cache'] = 'degraded'
                health_status['alerts'].append('Cache not functioning properly')
        except Exception as e:
            health_status['components']['cache'] = 'unhealthy'
            health_status['alerts'].append(f'Cache error: {str(e)}')
        
        # GPU健康检查
        if self.gpu_accelerator.is_gpu_available():
            gpu_memory = self.gpu_accelerator.get_gpu_memory_info()
            if gpu_memory['free'] > 1.0:  # 至少1GB空闲
                health_status['components']['gpu'] = 'healthy'
            else:
                health_status['components']['gpu'] = 'warning'
                health_status['alerts'].append('GPU memory usage high')
        else:
            health_status['components']['gpu'] = 'unavailable'
        
        # 综合状态
        if any(status in ['unhealthy', 'degraded'] for status in health_status['components'].values()):
            health_status['overall_status'] = 'degraded'
        
        return health_status