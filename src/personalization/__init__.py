# src/personalization/__init__.py
"""
用户个性化模块
- 用户画像管理
- 兴趣追踪
- 个性化推荐
"""

from .user_profiler import UserProfiler, UserProfile, InteractionType
from .recommendation_engine import RecommendationEngine, RecommendationRequest, RecommendationResult
from .preference_tracker import PreferenceTracker, UserInteraction

__all__ = [
    'UserProfiler', 'UserProfile', 'InteractionType',
    'RecommendationEngine', 'RecommendationRequest', 'RecommendationResult',
    'PreferenceTracker', 'UserInteraction'
]