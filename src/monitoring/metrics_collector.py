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
        
        error_entry = {\n            'timestamp': datetime.now().isoformat(),\n            'component': component,\n            'error_type': error_type,\n            'error_message': error_message\n        }\n        self.error_log.append(error_entry)\n        \n        logger.error(f\"❌ {component}.{error_type}: {error_message}\")\n    \n    def update_active_connections(self, count: int):\n        \"\"\"Update active connections count\"\"\"\n        self.active_connections.set(count)\n    \n    def get_current_metrics(self) -> Dict[str, Any]:\n        \"\"\"Get current system metrics\"\"\"\n        if not self.metrics_history:\n            return {}\n        \n        latest = self.metrics_history[-1]\n        \n        # Calculate additional metrics\n        avg_latency = sum(self.query_latencies) / max(len(self.query_latencies), 1)\n        error_rate = len([e for e in self.error_log \n                         if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(minutes=5)])\n        \n        return {\n            'current_metrics': latest.to_dict(),\n            'average_latency_ms': avg_latency * 1000,\n            'error_rate_5min': error_rate,\n            'total_queries': len(self.query_latencies),\n            'cache_performance': self._calculate_cache_performance(),\n            'system_health': self._assess_system_health()\n        }\n    \n    def _calculate_cache_performance(self) -> Dict[str, float]:\n        \"\"\"Calculate cache performance metrics\"\"\"\n        # This would be populated by cache system\n        return {\n            'memory_hit_rate': 0.85,\n            'redis_hit_rate': 0.72,\n            'file_hit_rate': 0.45\n        }\n    \n    def _assess_system_health(self) -> str:\n        \"\"\"Assess overall system health\"\"\"\n        if not self.metrics_history:\n            return \"unknown\"\n        \n        latest = self.metrics_history[-1]\n        \n        # Simple health assessment\n        if latest.cpu_usage_percent > 90 or latest.memory_usage_mb > 8000:\n            return \"critical\"\n        elif latest.cpu_usage_percent > 70 or latest.memory_usage_mb > 6000:\n            return \"warning\"\n        else:\n            return \"healthy\"\n    \n    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:\n        \"\"\"Get metrics summary for the specified time period\"\"\"\n        cutoff_time = datetime.now() - timedelta(hours=hours)\n        \n        recent_metrics = [\n            m for m in self.metrics_history \n            if m.timestamp > cutoff_time\n        ]\n        \n        if not recent_metrics:\n            return {}\n        \n        return {\n            'time_period_hours': hours,\n            'total_data_points': len(recent_metrics),\n            'average_cpu_usage': sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics),\n            'average_memory_usage_mb': sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics),\n            'peak_cpu_usage': max(m.cpu_usage_percent for m in recent_metrics),\n            'peak_memory_usage_mb': max(m.memory_usage_mb for m in recent_metrics),\n            'total_errors': len([e for e in self.error_log \n                               if datetime.fromisoformat(e['timestamp']) > cutoff_time]),\n            'health_status': self._assess_system_health()\n        }\n    \n    def export_metrics(self) -> str:\n        \"\"\"Export metrics in Prometheus format\"\"\"\n        return generate_latest(self.registry).decode('utf-8')\n    \n    def get_component_stats(self, component: str) -> Dict[str, Any]:\n        \"\"\"Get statistics for a specific component\"\"\"\n        return self.component_stats.get(component, {})\n    \n    def update_component_stats(self, component: str, stats: Dict[str, Any]):\n        \"\"\"Update statistics for a component\"\"\"\n        self.component_stats[component].update(stats)\n    \n    def save_metrics_to_file(self, file_path: Path):\n        \"\"\"Save current metrics to file\"\"\"\n        try:\n            metrics_data = {\n                'timestamp': datetime.now().isoformat(),\n                'current_metrics': self.get_current_metrics(),\n                'summary_24h': self.get_metrics_summary(24),\n                'component_stats': dict(self.component_stats),\n                'recent_errors': list(self.error_log)[-50:]  # Last 50 errors\n            }\n            \n            with open(file_path, 'w') as f:\n                json.dump(metrics_data, f, indent=2, default=str)\n            \n            logger.info(f\"💾 Metrics saved to {file_path}\")\n            \n        except Exception as e:\n            logger.error(f\"❌ Error saving metrics: {e}\")\n\n\n# Global metrics collector instance\n_metrics_collector: Optional[MetricsCollector] = None\n\n\ndef get_metrics_collector() -> Optional[MetricsCollector]:\n    \"\"\"Get the global metrics collector instance\"\"\"\n    return _metrics_collector\n\n\ndef initialize_metrics(config: Dict[str, Any]) -> MetricsCollector:\n    \"\"\"Initialize the global metrics collector\"\"\"\n    global _metrics_collector\n    \n    if _metrics_collector is None:\n        _metrics_collector = MetricsCollector(config)\n        _metrics_collector.start_collection()\n    \n    return _metrics_collector\n\n\ndef shutdown_metrics():\n    \"\"\"Shutdown the global metrics collector\"\"\"\n    global _metrics_collector\n    \n    if _metrics_collector:\n        _metrics_collector.stop_collection()\n        _metrics_collector = None\n\n\n# Convenience decorators\ndef monitor_performance(component: str, operation: str):\n    \"\"\"Decorator for monitoring function performance\"\"\"\n    def decorator(func):\n        def wrapper(*args, **kwargs):\n            collector = get_metrics_collector()\n            if collector:\n                with collector.measure_request(component, operation):\n                    return func(*args, **kwargs)\n            else:\n                return func(*args, **kwargs)\n        return wrapper\n    return decorator\n\n\ndef monitor_async_performance(component: str, operation: str):\n    \"\"\"Decorator for monitoring async function performance\"\"\"\n    def decorator(func):\n        async def wrapper(*args, **kwargs):\n            collector = get_metrics_collector()\n            if collector:\n                with collector.measure_request(component, operation):\n                    return await func(*args, **kwargs)\n            else:\n                return await func(*args, **kwargs)\n        return wrapper\n    return decorator\n\n\n# Example usage and testing\nif __name__ == \"__main__\":\n    # Test configuration\n    config = {\n        'metrics_port': 8001,\n        'collection_interval': 5,\n        'history_size': 100\n    }\n    \n    # Initialize metrics\n    collector = initialize_metrics(config)\n    \n    # Simulate some operations\n    import random\n    \n    for i in range(10):\n        with collector.measure_request(\"test_endpoint\", \"GET\"):\n            time.sleep(random.uniform(0.1, 0.5))\n        \n        collector.record_cache_hit(\"memory\")\n        if random.random() > 0.7:\n            collector.record_cache_miss(\"redis\")\n    \n    # Print current metrics\n    print(\"Current metrics:\")\n    print(json.dumps(collector.get_current_metrics(), indent=2))\n    \n    # Save metrics\n    collector.save_metrics_to_file(Path(\"test_metrics.json\"))\n    \n    # Cleanup\n    shutdown_metrics()"