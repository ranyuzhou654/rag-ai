# src/optimization/__init__.py
"""
系统优化模块
- 模型注册和管理
- 配置验证
- 性能优化
"""

from .model_registry import ModelRegistry
from .config_validator import ConfigValidator, ValidationResult

__all__ = [
    'ModelRegistry',
    'ConfigValidator', 'ValidationResult'
]