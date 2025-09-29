# src/storage/__init__.py
"""
存储优化模块
- 多层存储管理
- 智能缓存策略
- 数据生命周期管理
- 存储监控和分析
"""

from .storage_optimizer import StorageOptimizer, StorageTier, StoragePolicy
from .usage_analytics import UsageAnalytics, AccessPattern, StorageMetrics
from .data_lifecycle import DataLifecycleManager, LifecyclePolicy, LifecycleAction

__all__ = [
    'StorageOptimizer', 'StorageTier', 'StoragePolicy',
    'UsageAnalytics', 'AccessPattern', 'StorageMetrics',
    'DataLifecycleManager', 'LifecyclePolicy', 'LifecycleAction'
]