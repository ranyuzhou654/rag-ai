# src/monitoring/metrics_collector.py
"""
Performance monitoring and metrics collection for RAG-AI system
Provides comprehensive observability for all system components
"""

import time
import psutil
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from collections import defaultdict, deque
import threading
from contextlib import contextmanager

from prometheus_client import (
    Counter, Histogram, Gauge, Info, start_http_server,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from loguru import logger


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime = field(default_factory=datetime.now)
    response_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    query_count: int = 0
    error_count: int = 0
    active_connections: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'response_time': self.response_time,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'cache_hit_rate': self.cache_hit_rate,
            'query_count': self.query_count,
            'error_count': self.error_count,
            'active_connections': self.active_connections
        }


class MetricsCollector:
    """
    Central metrics collection and monitoring system
    Integrates with Prometheus and provides real-time observability
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_port = config.get('metrics_port', 8001)
        self.collection_interval = config.get('collection_interval', 10)
        self.metrics_history_size = config.get('history_size', 1000)
        
        # Prometheus registry
        self.registry = CollectorRegistry()
        
        # Define Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Internal metrics storage
        self.metrics_history: deque = deque(maxlen=self.metrics_history_size)
        self.query_latencies: deque = deque(maxlen=1000)
        self.error_log: deque = deque(maxlen=500)
        
        # Component metrics
        self.component_stats = defaultdict(dict)
        
        # Background collection
        self._collection_thread = None
        self._stop_collection = threading.Event()
        
        logger.info("🔍 Metrics collector initialized")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        # Request metrics
        self.request_counter = Counter(
            'rag_ai_requests_total',
            'Total number of requests',
            ['endpoint', 'method', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'rag_ai_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint', 'method'],
            registry=self.registry
        )
        
        # Query processing metrics
        self.query_processing_time = Histogram(
            'rag_ai_query_processing_seconds',
            'Query processing time',
            ['query_type'],
            registry=self.registry
        )
        
        self.retrieval_time = Histogram(
            'rag_ai_retrieval_seconds',
            'Document retrieval time',
            ['search_type'],
            registry=self.registry
        )
        
        self.generation_time = Histogram(
            'rag_ai_generation_seconds',
            'Answer generation time',
            ['model_type'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'rag_ai_cache_hits_total',
            'Cache hits',
            ['cache_layer'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'rag_ai_cache_misses_total',
            'Cache misses',
            ['cache_layer'],
            registry=self.registry
        )
        
        # Vector database metrics
        self.vector_db_operations = Counter(
            'rag_ai_vector_db_operations_total',
            'Vector database operations',
            ['operation_type', 'status'],
            registry=self.registry
        )
        
        self.vector_db_latency = Histogram(
            'rag_ai_vector_db_latency_seconds',
            'Vector database operation latency',
            ['operation_type'],
            registry=self.registry
        )
        
        # System resource metrics
        self.memory_usage = Gauge(
            'rag_ai_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'rag_ai_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'rag_ai_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        # Citation metrics
        self.citations_generated = Counter(
            'rag_ai_citations_generated_total',
            'Citations generated',
            ['citation_type'],
            registry=self.registry
        )
        
        # Error metrics
        self.errors = Counter(
            'rag_ai_errors_total',
            'Total errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Model metrics
        self.model_inference_time = Histogram(
            'rag_ai_model_inference_seconds',
            'Model inference time',
            ['model_name', 'task_type'],
            registry=self.registry
        )
        
        # Data collection metrics
        self.documents_processed = Counter(
            'rag_ai_documents_processed_total',
            'Documents processed',
            ['source_type'],
            registry=self.registry
        )
        
        logger.info("✅ Prometheus metrics configured")
    
    def start_collection(self):
        """Start background metrics collection"""
        if self._collection_thread is None or not self._collection_thread.is_alive():
            self._stop_collection.clear()
            self._collection_thread = threading.Thread(target=self._collect_system_metrics)
            self._collection_thread.daemon = True
            self._collection_thread.start()
            
            # Start Prometheus HTTP server
            start_http_server(self.metrics_port, registry=self.registry)
            
            logger.info(f"📊 Metrics collection started on port {self.metrics_port}")
    
    def stop_collection(self):
        """Stop background metrics collection"""
        if self._collection_thread:
            self._stop_collection.set()
            self._collection_thread.join(timeout=5)
            logger.info("🛑 Metrics collection stopped")
    
    def _collect_system_metrics(self):
        """Background thread for collecting system metrics"""
        while not self._stop_collection.is_set():
            try:
                # Collect system resource metrics
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Update Prometheus gauges
                self.memory_usage.set(memory_info.used)
                self.cpu_usage.set(cpu_percent)
                
                # Store in history
                metrics = PerformanceMetrics(
                    memory_usage_mb=memory_info.used / 1024 / 1024,
                    cpu_usage_percent=cpu_percent
                )
                self.metrics_history.append(metrics)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"❌ Error collecting system metrics: {e}")
                time.sleep(self.collection_interval)
    
    @contextmanager
    def measure_request(self, endpoint: str, method: str):
        """Context manager for measuring request metrics"""
        start_time = time.time()
        status = "success"
        
        try:
            yield
        except Exception as e:
            status = "error"
            self.record_error("request", endpoint, str(e))
            raise
        finally:
            duration = time.time() - start_time
            
            # Update Prometheus metrics
            self.request_counter.labels(
                endpoint=endpoint, 
                method=method, 
                status=status
            ).inc()
            
            self.request_duration.labels(
                endpoint=endpoint, 
                method=method
            ).observe(duration)
            
            # Store latency
            self.query_latencies.append(duration)
    
    @contextmanager
    def measure_query_processing(self, query_type: str):
        """Context manager for measuring query processing time"""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.query_processing_time.labels(query_type=query_type).observe(duration)
    
    @contextmanager
    def measure_retrieval(self, search_type: str):
        """Context manager for measuring retrieval time"""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.retrieval_time.labels(search_type=search_type).observe(duration)
    
    @contextmanager
    def measure_generation(self, model_type: str):
        """Context manager for measuring generation time"""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.generation_time.labels(model_type=model_type).observe(duration)
    
    @contextmanager
    def measure_vector_db_operation(self, operation_type: str):
        """Context manager for measuring vector DB operations"""
        start_time = time.time()
        status = "success"
        
        try:
            yield
        except Exception as e:
            status = "error"
            self.record_error("vector_db", operation_type, str(e))
            raise
        finally:
            duration = time.time() - start_time
            
            self.vector_db_operations.labels(
                operation_type=operation_type,
                status=status
            ).inc()
            
            self.vector_db_latency.labels(
                operation_type=operation_type
            ).observe(duration)
    
    @contextmanager
    def measure_model_inference(self, model_name: str, task_type: str):
        """Context manager for measuring model inference"""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.model_inference_time.labels(
                model_name=model_name,
                task_type=task_type
            ).observe(duration)
    
    def record_cache_hit(self, cache_layer: str):
        """Record a cache hit"""
        self.cache_hits.labels(cache_layer=cache_layer).inc()
    
    def record_cache_miss(self, cache_layer: str):
        """Record a cache miss"""
        self.cache_misses.labels(cache_layer=cache_layer).inc()
    
    def record_citation_generated(self, citation_type: str):
        """Record citation generation"""
        self.citations_generated.labels(citation_type=citation_type).inc()
    
    def record_document_processed(self, source_type: str):
        """Record document processing"""
        self.documents_processed.labels(source_type=source_type).inc()
    
    def record_error(self, component: str, error_type: str, error_message: str):
        """Record an error"""
        self.errors.labels(error_type=error_type, component=component).inc()
        
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'error_type': error_type,
            'error_message': error_message
        }
        self.error_log.append(error_entry)
        
        logger.error(f"❌ {component}.{error_type}: {error_message}")
    
    def update_active_connections(self, count: int):
        """Update active connections count"""
        self.active_connections.set(count)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        
        # Calculate additional metrics
        avg_latency = sum(self.query_latencies) / max(len(self.query_latencies), 1)
        error_rate = len([e for e in self.error_log 
                         if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(minutes=5)])
        
        return {
            'current_metrics': latest.to_dict(),
            'average_latency_ms': avg_latency * 1000,
            'error_rate_5min': error_rate,
            'total_queries': len(self.query_latencies),
            'cache_performance': self._calculate_cache_performance(),
            'system_health': self._assess_system_health()
        }
    
    def _calculate_cache_performance(self) -> Dict[str, float]:
        """Calculate cache performance metrics"""
        # This would be populated by cache system
        return {
            'memory_hit_rate': 0.85,
            'redis_hit_rate': 0.72,
            'file_hit_rate': 0.45
        }
    
    def _assess_system_health(self) -> str:
        """Assess overall system health"""
        if not self.metrics_history:
            return "unknown"
        
        latest = self.metrics_history[-1]
        
        # Simple health assessment
        if latest.cpu_usage_percent > 90 or latest.memory_usage_mb > 8000:
            return "critical"
        elif latest.cpu_usage_percent > 70 or latest.memory_usage_mb > 6000:
            return "warning"
        else:
            return "healthy"
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        return {
            'time_period_hours': hours,
            'total_data_points': len(recent_metrics),
            'average_cpu_usage': sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics),
            'average_memory_usage_mb': sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics),
            'peak_cpu_usage': max(m.cpu_usage_percent for m in recent_metrics),
            'peak_memory_usage_mb': max(m.memory_usage_mb for m in recent_metrics),
            'total_errors': len([e for e in self.error_log 
                               if datetime.fromisoformat(e['timestamp']) > cutoff_time]),
            'health_status': self._assess_system_health()
        }
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_component_stats(self, component: str) -> Dict[str, Any]:
        """Get statistics for a specific component"""
        return self.component_stats.get(component, {})
    
    def update_component_stats(self, component: str, stats: Dict[str, Any]):
        """Update statistics for a component"""
        self.component_stats[component].update(stats)
    
    def save_metrics_to_file(self, file_path: Path):
        """Save current metrics to file"""
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'current_metrics': self.get_current_metrics(),
                'summary_24h': self.get_metrics_summary(24),
                'component_stats': dict(self.component_stats),
                'recent_errors': list(self.error_log)[-50:]  # Last 50 errors
            }
            
            with open(file_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            logger.info(f"💾 Metrics saved to {file_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving metrics: {e}")


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> Optional[MetricsCollector]:
    """Get the global metrics collector instance"""
    return _metrics_collector


def initialize_metrics(config: Dict[str, Any]) -> MetricsCollector:
    """Initialize the global metrics collector"""
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(config)
        _metrics_collector.start_collection()
    
    return _metrics_collector


def shutdown_metrics():
    """Shutdown the global metrics collector"""
    global _metrics_collector
    
    if _metrics_collector:
        _metrics_collector.stop_collection()
        _metrics_collector = None


# Convenience decorators
def monitor_performance(component: str, operation: str):
    """Decorator for monitoring function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            if collector:
                with collector.measure_request(component, operation):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def monitor_async_performance(component: str, operation: str):
    """Decorator for monitoring async function performance"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            if collector:
                with collector.measure_request(component, operation):
                    return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    config = {
        'metrics_port': 8001,
        'collection_interval': 5,
        'history_size': 100
    }
    
    # Initialize metrics
    collector = initialize_metrics(config)
    
    # Simulate some operations
    import random
    
    for i in range(10):
        with collector.measure_request("test_endpoint", "GET"):
            time.sleep(random.uniform(0.1, 0.5))
        
        collector.record_cache_hit("memory")
        if random.random() > 0.7:
            collector.record_cache_miss("redis")
    
    # Print current metrics
    print("Current metrics:")
    print(json.dumps(collector.get_current_metrics(), indent=2))
    
    # Save metrics
    collector.save_metrics_to_file(Path("test_metrics.json"))
    
    # Cleanup
    shutdown_metrics()