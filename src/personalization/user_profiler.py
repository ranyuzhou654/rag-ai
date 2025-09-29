# src/personalization/user_profiler.py
import asyncio
import json
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from loguru import logger
from enum import Enum
import hashlib
from collections import defaultdict, Counter
import uuid

class InteractionType(Enum):
    """用户交互类型"""
    QUERY = "query"
    DOCUMENT_VIEW = "document_view"
    DOCUMENT_CLICK = "document_click"
    FEEDBACK_POSITIVE = "feedback_positive"
    FEEDBACK_NEGATIVE = "feedback_negative"
    BOOKMARK = "bookmark"
    SHARE = "share"
    SEARCH_REFINEMENT = "search_refinement"
    RECOMMENDATION_CLICK = "recommendation_click"
    EXPORT_CITATION = "export_citation"

@dataclass
class UserInteraction:
    """用户交互记录"""
    interaction_id: str
    user_id: str
    session_id: str
    interaction_type: InteractionType
    timestamp: datetime
    
    # 内容相关
    query_text: Optional[str] = None
    document_id: Optional[str] = None
    document_title: Optional[str] = None
    document_abstract: Optional[str] = None
    document_authors: Optional[List[str]] = None
    document_categories: Optional[List[str]] = None
    document_keywords: Optional[List[str]] = None
    
    # 行为相关
    duration_seconds: Optional[float] = None
    scroll_depth: Optional[float] = None  # 0.0-1.0
    click_position: Optional[int] = None  # 点击位置
    rating: Optional[float] = None  # 1-5评分
    
    # 上下文信息
    search_context: Optional[Dict] = None
    recommendation_context: Optional[Dict] = None
    user_agent: Optional[str] = None
    location: Optional[str] = None

@dataclass 
class ResearchInterest:
    """研究兴趣领域"""
    category: str  # 如 "machine_learning", "computer_vision"
    keywords: Set[str]
    confidence_score: float  # 0.0-1.0
    last_updated: datetime
    interaction_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'category': self.category,
            'keywords': list(self.keywords),
            'confidence_score': self.confidence_score,
            'last_updated': self.last_updated.isoformat(),
            'interaction_count': self.interaction_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ResearchInterest':
        return cls(
            category=data['category'],
            keywords=set(data['keywords']),
            confidence_score=data['confidence_score'],
            last_updated=datetime.fromisoformat(data['last_updated']),
            interaction_count=data.get('interaction_count', 0)
        )

@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    created_at: datetime
    last_active: datetime
    
    # 兴趣建模
    research_interests: Dict[str, ResearchInterest] = field(default_factory=dict)
    preferred_authors: Dict[str, float] = field(default_factory=dict)  # author -> weight
    preferred_sources: Dict[str, float] = field(default_factory=dict)  # source -> weight
    preferred_languages: Dict[str, float] = field(default_factory=dict)  # lang -> weight
    
    # 行为模式
    activity_level: str = "medium"  # low, medium, high
    query_patterns: List[str] = field(default_factory=list)  # 常见查询模式
    reading_depth: float = 0.5  # 阅读深度 0.0-1.0
    exploration_vs_exploitation: float = 0.5  # 探索vs利用倾向 0.0-1.0
    
    # 交互统计
    total_queries: int = 0
    total_documents_viewed: int = 0
    total_positive_feedback: int = 0
    total_negative_feedback: int = 0
    average_session_duration: float = 0.0
    
    # 时间偏好
    active_hours: Set[int] = field(default_factory=set)  # 活跃小时
    active_days: Set[int] = field(default_factory=set)  # 活跃星期
    
    # 推荐历史
    recommendation_history: List[str] = field(default_factory=list)  # 已推荐文档ID
    recommendation_clicks: int = 0
    recommendation_success_rate: float = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_active': self.last_active.isoformat(),
            'research_interests': {k: v.to_dict() for k, v in self.research_interests.items()},
            'preferred_authors': self.preferred_authors,
            'preferred_sources': self.preferred_sources,
            'preferred_languages': self.preferred_languages,
            'activity_level': self.activity_level,
            'query_patterns': self.query_patterns,
            'reading_depth': self.reading_depth,
            'exploration_vs_exploitation': self.exploration_vs_exploitation,
            'total_queries': self.total_queries,
            'total_documents_viewed': self.total_documents_viewed,
            'total_positive_feedback': self.total_positive_feedback,
            'total_negative_feedback': self.total_negative_feedback,
            'average_session_duration': self.average_session_duration,
            'active_hours': list(self.active_hours),
            'active_days': list(self.active_days),
            'recommendation_history': self.recommendation_history,
            'recommendation_clicks': self.recommendation_clicks,
            'recommendation_success_rate': self.recommendation_success_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        """从字典创建用户画像"""
        research_interests = {}
        if 'research_interests' in data:
            research_interests = {
                k: ResearchInterest.from_dict(v) 
                for k, v in data['research_interests'].items()
            }
        
        return cls(
            user_id=data['user_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_active=datetime.fromisoformat(data['last_active']),
            research_interests=research_interests,
            preferred_authors=data.get('preferred_authors', {}),
            preferred_sources=data.get('preferred_sources', {}),
            preferred_languages=data.get('preferred_languages', {}),
            activity_level=data.get('activity_level', 'medium'),
            query_patterns=data.get('query_patterns', []),
            reading_depth=data.get('reading_depth', 0.5),
            exploration_vs_exploitation=data.get('exploration_vs_exploitation', 0.5),
            total_queries=data.get('total_queries', 0),
            total_documents_viewed=data.get('total_documents_viewed', 0),
            total_positive_feedback=data.get('total_positive_feedback', 0),
            total_negative_feedback=data.get('total_negative_feedback', 0),
            average_session_duration=data.get('average_session_duration', 0.0),
            active_hours=set(data.get('active_hours', [])),
            active_days=set(data.get('active_days', [])),
            recommendation_history=data.get('recommendation_history', []),
            recommendation_clicks=data.get('recommendation_clicks', 0),
            recommendation_success_rate=data.get('recommendation_success_rate', 0.0)
        )

class UserProfiler:
    """用户画像管理器"""
    
    def __init__(self, db_path: Path, storage_path: Path):
        self.db_path = db_path
        self.storage_path = storage_path
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        # 研究领域分类
        self.research_categories = {
            'machine_learning': ['ml', 'machine learning', 'neural network', 'deep learning', 'supervised', 'unsupervised'],
            'computer_vision': ['cv', 'computer vision', 'image', 'visual', 'cnn', 'object detection', 'segmentation'],
            'natural_language_processing': ['nlp', 'language model', 'transformer', 'bert', 'gpt', 'text', 'language'],
            'robotics': ['robot', 'robotics', 'autonomous', 'control', 'manipulation', 'navigation'],
            'reinforcement_learning': ['rl', 'reinforcement', 'policy', 'reward', 'agent', 'environment'],
            'recommendation_systems': ['recommendation', 'recommender', 'collaborative filtering', 'content-based'],
            'graph_neural_networks': ['gnn', 'graph neural', 'graph convolution', 'node embedding'],
            'multimodal': ['multimodal', 'vision-language', 'clip', 'cross-modal'],
            'generative_models': ['gan', 'vae', 'diffusion', 'generative', 'synthesis'],
            'optimization': ['optimization', 'gradient', 'sgd', 'adam', 'convergence']
        }
        
        self._init_database()
        
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            # 用户画像表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_data TEXT,  -- JSON格式的画像数据
                    created_at TEXT,
                    last_updated TEXT
                )
            ''')
            
            # 用户交互表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_interactions (
                    interaction_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    session_id TEXT,
                    interaction_type TEXT,
                    timestamp TEXT,
                    query_text TEXT,
                    document_id TEXT,
                    document_title TEXT,
                    document_abstract TEXT,
                    document_authors TEXT,  -- JSON array
                    document_categories TEXT,  -- JSON array
                    document_keywords TEXT,  -- JSON array
                    duration_seconds REAL,
                    scroll_depth REAL,
                    click_position INTEGER,
                    rating REAL,
                    search_context TEXT,  -- JSON
                    recommendation_context TEXT,  -- JSON
                    user_agent TEXT,
                    location TEXT,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            ''')
            
            # 用户会话表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    total_interactions INTEGER DEFAULT 0,
                    session_duration REAL DEFAULT 0.0,
                    query_count INTEGER DEFAULT 0,
                    document_views INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            ''')
            
            # 创建索引
            conn.execute('CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON user_interactions(user_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON user_interactions(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_interactions_type ON user_interactions(interaction_type)')
            
            conn.commit()
        
        logger.info(f"User profiler database initialized at {self.db_path}")
    
    def get_or_create_user_profile(self, user_id: str) -> UserProfile:
        """获取或创建用户画像"""
        profile = self.get_user_profile(user_id)
        if profile is None:
            profile = self.create_user_profile(user_id)
        return profile
    
    def create_user_profile(self, user_id: str) -> UserProfile:
        """创建新用户画像"""
        now = datetime.now(timezone.utc)
        profile = UserProfile(
            user_id=user_id,
            created_at=now,
            last_active=now
        )
        
        self.save_user_profile(profile)
        logger.info(f"Created new user profile: {user_id}")
        return profile
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT profile_data FROM user_profiles WHERE user_id = ?', 
                (user_id,)
            )
            row = cursor.fetchone()
            
            if row:
                try:
                    profile_data = json.loads(row[0])
                    return UserProfile.from_dict(profile_data)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Error loading user profile {user_id}: {e}")
                    return None
            
            return None
    
    def save_user_profile(self, profile: UserProfile):
        """保存用户画像"""
        profile.last_active = datetime.now(timezone.utc)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO user_profiles 
                (user_id, profile_data, created_at, last_updated)
                VALUES (?, ?, ?, ?)
            ''', (
                profile.user_id,
                json.dumps(profile.to_dict(), ensure_ascii=False),
                profile.created_at.isoformat(),
                profile.last_active.isoformat()
            ))
            conn.commit()
    
    def record_interaction(self, interaction: UserInteraction):
        """记录用户交互"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO user_interactions 
                (interaction_id, user_id, session_id, interaction_type, timestamp,
                 query_text, document_id, document_title, document_abstract,
                 document_authors, document_categories, document_keywords,
                 duration_seconds, scroll_depth, click_position, rating,
                 search_context, recommendation_context, user_agent, location)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                interaction.interaction_id,
                interaction.user_id,
                interaction.session_id,
                interaction.interaction_type.value,
                interaction.timestamp.isoformat(),
                interaction.query_text,
                interaction.document_id,
                interaction.document_title,
                interaction.document_abstract,
                json.dumps(interaction.document_authors) if interaction.document_authors else None,
                json.dumps(interaction.document_categories) if interaction.document_categories else None,
                json.dumps(interaction.document_keywords) if interaction.document_keywords else None,
                interaction.duration_seconds,
                interaction.scroll_depth,
                interaction.click_position,
                interaction.rating,
                json.dumps(interaction.search_context) if interaction.search_context else None,
                json.dumps(interaction.recommendation_context) if interaction.recommendation_context else None,
                interaction.user_agent,
                interaction.location
            ))
            conn.commit()
        
        # 更新用户画像
        self.update_profile_from_interaction(interaction)
    
    def update_profile_from_interaction(self, interaction: UserInteraction):
        """基于交互更新用户画像"""
        profile = self.get_or_create_user_profile(interaction.user_id)
        
        # 更新活跃时间
        hour = interaction.timestamp.hour
        day = interaction.timestamp.weekday()
        profile.active_hours.add(hour)
        profile.active_days.add(day)
        
        # 更新交互统计
        if interaction.interaction_type == InteractionType.QUERY:
            profile.total_queries += 1
            
            # 分析查询模式
            if interaction.query_text:
                self._update_query_patterns(profile, interaction.query_text)
                self._update_research_interests(profile, interaction.query_text)
        
        elif interaction.interaction_type == InteractionType.DOCUMENT_VIEW:
            profile.total_documents_viewed += 1
            
            # 更新偏好作者
            if interaction.document_authors:
                for author in interaction.document_authors:
                    profile.preferred_authors[author] = profile.preferred_authors.get(author, 0) + 1
            
            # 更新研究兴趣
            if interaction.document_title:
                self._update_research_interests(profile, interaction.document_title)
            if interaction.document_abstract:
                self._update_research_interests(profile, interaction.document_abstract)
            if interaction.document_keywords:
                for keyword in interaction.document_keywords:
                    self._update_research_interests(profile, keyword)
        
        elif interaction.interaction_type == InteractionType.FEEDBACK_POSITIVE:
            profile.total_positive_feedback += 1
        
        elif interaction.interaction_type == InteractionType.FEEDBACK_NEGATIVE:
            profile.total_negative_feedback += 1
        
        elif interaction.interaction_type == InteractionType.RECOMMENDATION_CLICK:
            profile.recommendation_clicks += 1
        
        # 更新阅读深度
        if interaction.scroll_depth:
            profile.reading_depth = (profile.reading_depth + interaction.scroll_depth) / 2
        
        # 计算活跃度级别
        profile.activity_level = self._calculate_activity_level(profile)
        
        # 计算推荐成功率
        if profile.recommendation_history:
            profile.recommendation_success_rate = profile.recommendation_clicks / len(profile.recommendation_history)
        
        self.save_user_profile(profile)
    
    def _update_query_patterns(self, profile: UserProfile, query_text: str):
        """更新查询模式"""
        query_lower = query_text.lower()
        
        # 简单的模式检测
        patterns = []
        if any(word in query_lower for word in ['how', 'how to', '如何', '怎么']):
            patterns.append('how_to')
        if any(word in query_lower for word in ['what', 'what is', '什么是', '什么']):
            patterns.append('definition')
        if any(word in query_lower for word in ['compare', 'vs', 'difference', '比较', '区别']):
            patterns.append('comparison')
        if any(word in query_lower for word in ['example', 'examples', '例子', '示例']):
            patterns.append('examples')
        if any(word in query_lower for word in ['paper', 'research', 'study', '论文', '研究']):
            patterns.append('research')
        
        for pattern in patterns:
            if pattern not in profile.query_patterns:
                profile.query_patterns.append(pattern)
    
    def _update_research_interests(self, profile: UserProfile, text: str):
        """更新研究兴趣"""
        text_lower = text.lower()
        
        for category, keywords in self.research_categories.items():
            matched_keywords = set()
            for keyword in keywords:
                if keyword in text_lower:
                    matched_keywords.add(keyword)
            
            if matched_keywords:
                if category in profile.research_interests:
                    interest = profile.research_interests[category]
                    interest.keywords.update(matched_keywords)
                    interest.interaction_count += 1
                    interest.confidence_score = min(1.0, interest.confidence_score + 0.1)
                    interest.last_updated = datetime.now(timezone.utc)
                else:
                    profile.research_interests[category] = ResearchInterest(
                        category=category,
                        keywords=matched_keywords,
                        confidence_score=0.3,
                        last_updated=datetime.now(timezone.utc),
                        interaction_count=1
                    )
    
    def _calculate_activity_level(self, profile: UserProfile) -> str:
        """计算用户活跃度级别"""
        total_interactions = (
            profile.total_queries + 
            profile.total_documents_viewed + 
            profile.total_positive_feedback + 
            profile.total_negative_feedback
        )
        
        if total_interactions < 10:
            return "low"
        elif total_interactions < 50:
            return "medium"
        else:
            return "high"
    
    def get_user_interests_vector(self, user_id: str) -> Dict[str, float]:
        """获取用户兴趣向量"""
        profile = self.get_user_profile(user_id)
        if not profile or not profile.research_interests:
            return {}
        
        interests_vector = {}
        for category, interest in profile.research_interests.items():
            # 综合考虑置信度和交互次数
            score = interest.confidence_score * (1 + np.log1p(interest.interaction_count))
            interests_vector[category] = float(score)
        
        # 归一化
        if interests_vector:
            max_score = max(interests_vector.values())
            if max_score > 0:
                interests_vector = {k: v/max_score for k, v in interests_vector.items()}
        
        return interests_vector
    
    def get_similar_users(self, user_id: str, limit: int = 10) -> List[Tuple[str, float]]:
        """寻找相似用户（基于兴趣）"""
        target_profile = self.get_user_profile(user_id)
        if not target_profile or not target_profile.research_interests:
            return []
        
        target_vector = self.get_user_interests_vector(user_id)
        if not target_vector:
            return []
        
        similar_users = []
        
        # 获取所有用户
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT user_id FROM user_profiles WHERE user_id != ?', (user_id,))
            other_users = [row[0] for row in cursor.fetchall()]
        
        for other_user_id in other_users:
            other_vector = self.get_user_interests_vector(other_user_id)
            if not other_vector:
                continue
            
            # 计算余弦相似度
            similarity = self._calculate_cosine_similarity(target_vector, other_vector)
            if similarity > 0.1:  # 过滤掉太低的相似度
                similar_users.append((other_user_id, similarity))
        
        # 按相似度排序
        similar_users.sort(key=lambda x: x[1], reverse=True)
        return similar_users[:limit]
    
    def _calculate_cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """计算余弦相似度"""
        # 获取所有维度
        all_dims = set(vec1.keys()) | set(vec2.keys())
        
        if not all_dims:
            return 0.0
        
        # 构建向量
        v1 = np.array([vec1.get(dim, 0.0) for dim in all_dims])
        v2 = np.array([vec2.get(dim, 0.0) for dim in all_dims])
        
        # 计算余弦相似度
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取用户统计信息"""
        profile = self.get_user_profile(user_id)
        if not profile:
            return {}
        
        # 获取最近活动
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 最近30天的交互统计
            cursor.execute('''
                SELECT interaction_type, COUNT(*) 
                FROM user_interactions 
                WHERE user_id = ? AND timestamp >= datetime('now', '-30 days')
                GROUP BY interaction_type
            ''', (user_id,))
            
            recent_interactions = {row[0]: row[1] for row in cursor.fetchall()}
            
            # 最常查询的主题
            cursor.execute('''
                SELECT query_text, COUNT(*) as freq
                FROM user_interactions 
                WHERE user_id = ? AND interaction_type = 'query' 
                AND timestamp >= datetime('now', '-30 days')
                GROUP BY query_text
                ORDER BY freq DESC
                LIMIT 10
            ''', (user_id,))
            
            top_queries = [{'query': row[0], 'frequency': row[1]} for row in cursor.fetchall()]
        
        return {
            'profile': profile.to_dict(),
            'interests_vector': self.get_user_interests_vector(user_id),
            'recent_interactions': recent_interactions,
            'top_queries': top_queries,
            'similar_users': [user for user, _ in self.get_similar_users(user_id, 5)]
        }
    
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """清理旧数据"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        
        with sqlite3.connect(self.db_path) as conn:
            # 清理旧交互记录
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM user_interactions WHERE timestamp < ?',
                (cutoff_date.isoformat(),)
            )
            deleted_interactions = cursor.rowcount
            
            # 清理空的用户画像
            cursor.execute('''
                DELETE FROM user_profiles 
                WHERE user_id NOT IN (
                    SELECT DISTINCT user_id FROM user_interactions
                )
            ''')
            deleted_profiles = cursor.rowcount
            
            conn.commit()
        
        logger.info(f"Cleaned up old data: {deleted_interactions} interactions, {deleted_profiles} profiles")

# 使用示例
async def main():
    """测试用户画像系统"""
    db_path = Path("data/user_profiles/profiles.db")
    storage_path = Path("data/user_profiles")
    
    profiler = UserProfiler(db_path, storage_path)
    
    # 创建测试用户
    user_id = "test_user_001"
    
    # 模拟交互
    interactions = [
        UserInteraction(
            interaction_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id="session_001",
            interaction_type=InteractionType.QUERY,
            timestamp=datetime.now(timezone.utc),
            query_text="What are the latest developments in transformer models?",
            duration_seconds=2.5
        ),
        UserInteraction(
            interaction_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id="session_001",
            interaction_type=InteractionType.DOCUMENT_VIEW,
            timestamp=datetime.now(timezone.utc),
            document_id="doc_001",
            document_title="Attention Is All You Need",
            document_authors=["Vaswani", "Shazeer"],
            document_categories=["cs.AI", "cs.CL"],
            document_keywords=["transformer", "attention", "neural network"],
            duration_seconds=120.0,
            scroll_depth=0.8
        )
    ]
    
    # 记录交互
    for interaction in interactions:
        profiler.record_interaction(interaction)
    
    # 获取用户统计
    stats = profiler.get_user_statistics(user_id)
    print("用户统计信息:")
    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__":
    asyncio.run(main())