# src/personalization/recommendation_engine.py
import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from loguru import logger
from collections import defaultdict, Counter
import random
import math

from .user_profiler import UserProfiler, UserProfile, InteractionType
from ..retrieval.vector_database import VectorDatabaseManager

@dataclass
class RecommendationRequest:
    """推荐请求"""
    user_id: str
    recommendation_type: str = "daily"  # daily, trending, personalized, similar_users
    limit: int = 10
    diversity_factor: float = 0.3  # 多样性因子
    freshness_factor: float = 0.2  # 新鲜度因子
    exclude_seen: bool = True  # 排除已看过的文档
    category_filter: Optional[List[str]] = None
    time_range_days: int = 30  # 时间范围

@dataclass  
class RecommendationResult:
    """推荐结果"""
    document_id: str
    title: str
    authors: List[str]
    abstract: str
    url: str
    published_date: Optional[datetime]
    categories: List[str]
    keywords: List[str]
    
    # 推荐相关
    recommendation_score: float
    recommendation_reason: str
    algorithm_used: str
    
    # 评分组成
    content_score: float = 0.0
    collaborative_score: float = 0.0
    popularity_score: float = 0.0
    freshness_score: float = 0.0
    diversity_penalty: float = 0.0

class RecommendationEngine:
    """推荐引擎"""
    
    def __init__(self, user_profiler: UserProfiler, vector_db: VectorDatabaseManager, 
                 storage_path: Path):
        self.user_profiler = user_profiler
        self.vector_db = vector_db
        self.storage_path = storage_path
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        # 推荐算法权重
        self.algorithm_weights = {
            'content_based': 0.4,
            'collaborative_filtering': 0.3,
            'popularity_based': 0.2,
            'trending': 0.1
        }
        
        # 缓存
        self.popularity_cache = {}
        self.trending_cache = {}
        self.cache_ttl = timedelta(hours=1)
        self.last_cache_update = None
        
    async def generate_daily_recommendations(self, user_id: str, limit: int = 10) -> List[RecommendationResult]:
        """生成每日推荐"""
        request = RecommendationRequest(
            user_id=user_id,
            recommendation_type="daily",
            limit=limit,
            diversity_factor=0.4,
            freshness_factor=0.3
        )
        
        return await self.generate_recommendations(request)
    
    async def generate_recommendations(self, request: RecommendationRequest) -> List[RecommendationResult]:
        """生成推荐列表"""
        logger.info(f"Generating {request.recommendation_type} recommendations for user {request.user_id}")
        
        # 获取用户画像
        user_profile = self.user_profiler.get_user_profile(request.user_id)
        if not user_profile:
            # 新用户，返回热门和趋势推荐
            return await self._generate_cold_start_recommendations(request)
        
        # 获取候选文档
        candidates = await self._get_candidate_documents(user_profile, request)
        
        if not candidates:
            logger.warning(f"No candidates found for user {request.user_id}")
            return []
        
        # 生成推荐得分
        recommendations = []
        seen_documents = set(user_profile.recommendation_history) if request.exclude_seen else set()
        
        for doc in candidates:
            if doc['id'] in seen_documents:
                continue
                
            # 计算综合推荐得分
            rec_result = await self._calculate_recommendation_score(
                user_profile, doc, request
            )
            
            if rec_result and rec_result.recommendation_score > 0.1:  # 过滤低分推荐
                recommendations.append(rec_result)
        
        # 多样性处理
        if request.diversity_factor > 0:
            recommendations = self._apply_diversity_filtering(
                recommendations, request.diversity_factor
            )
        
        # 排序并返回
        recommendations.sort(key=lambda x: x.recommendation_score, reverse=True)
        return recommendations[:request.limit]
    
    async def _generate_cold_start_recommendations(self, request: RecommendationRequest) -> List[RecommendationResult]:
        """为新用户生成冷启动推荐"""
        logger.info(f"Generating cold start recommendations for new user {request.user_id}")
        
        # 获取热门文档
        trending_docs = self.vector_db.get_trending_papers(days=7, limit=request.limit * 2)
        
        recommendations = []
        for doc in trending_docs[:request.limit]:
            rec_result = RecommendationResult(
                document_id=doc['id'],
                title=doc.get('title', ''),
                authors=doc.get('authors', []),
                abstract=doc.get('abstract', ''),
                url=doc.get('url', ''),
                published_date=doc.get('published_date'),
                categories=doc.get('categories', []),
                keywords=doc.get('keywords', []),
                recommendation_score=0.8,
                recommendation_reason="热门论文推荐",
                algorithm_used="popularity_based",
                popularity_score=0.8
            )
            recommendations.append(rec_result)
        
        return recommendations
    
    async def _get_candidate_documents(self, user_profile: UserProfile, 
                                     request: RecommendationRequest) -> List[Dict]:
        """获取候选文档"""
        candidates = []
        
        # 1. 基于兴趣的内容检索
        if user_profile.research_interests:
            content_candidates = await self._get_content_based_candidates(
                user_profile, request.limit * 3
            )
            candidates.extend(content_candidates)
        
        # 2. 协同过滤候选
        collaborative_candidates = await self._get_collaborative_candidates(
            user_profile, request.limit * 2
        )
        candidates.extend(collaborative_candidates)
        
        # 3. 热门和趋势文档
        trending_candidates = self.vector_db.get_trending_papers(
            days=request.time_range_days, 
            limit=request.limit
        )
        candidates.extend(trending_candidates)
        
        # 4. 偏好作者的新文档
        if user_profile.preferred_authors:
            author_candidates = await self._get_preferred_author_candidates(
                user_profile, request.limit
            )
            candidates.extend(author_candidates)
        
        # 去重
        seen_ids = set()
        unique_candidates = []
        for doc in candidates:
            doc_id = doc.get('id') or doc.get('document_id')
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_candidates.append(doc)
        
        logger.info(f"Found {len(unique_candidates)} candidate documents")
        return unique_candidates
    
    async def _get_content_based_candidates(self, user_profile: UserProfile, 
                                          limit: int) -> List[Dict]:
        """基于内容的候选推荐"""
        candidates = []
        
        # 构建兴趣查询
        interest_queries = []
        for category, interest in user_profile.research_interests.items():
            if interest.confidence_score > 0.3:
                # 取置信度最高的几个关键词
                top_keywords = sorted(interest.keywords, 
                                    key=lambda x: len(x), reverse=True)[:3]
                query = " ".join(top_keywords)
                interest_queries.append((query, interest.confidence_score))
        
        # 对每个兴趣查询检索相关文档
        for query, confidence in interest_queries[:5]:  # 限制查询数量
            try:
                # 使用embedding模型编码查询
                from sentence_transformers import SentenceTransformer
                embedder = SentenceTransformer('BAAI/bge-m3')
                query_vector = embedder.encode([query], convert_to_numpy=True)[0]
                
                # 搜索相关文档
                search_results = self.vector_db.search(
                    query_vector=query_vector,
                    query_text=query,
                    top_k=limit // len(interest_queries) + 5,
                    search_type="hybrid"
                )
                
                for result in search_results:
                    result['content_relevance'] = confidence
                    candidates.append(result)
                    
            except Exception as e:
                logger.error(f"Error in content-based search for query '{query}': {e}")
        
        return candidates
    
    async def _get_collaborative_candidates(self, user_profile: UserProfile, 
                                          limit: int) -> List[Dict]:
        """基于协同过滤的候选推荐"""
        candidates = []
        
        # 寻找相似用户
        similar_users = self.user_profiler.get_similar_users(
            user_profile.user_id, limit=10
        )
        
        if not similar_users:
            return candidates
        
        # 获取相似用户喜欢的文档
        similar_user_docs = defaultdict(float)
        
        for similar_user_id, similarity in similar_users:
            similar_profile = self.user_profiler.get_user_profile(similar_user_id)
            if not similar_profile:
                continue
            
            # 从该用户的正向反馈中获取文档
            user_interactions = self._get_user_positive_interactions(similar_user_id)
            
            for doc_id in user_interactions:
                similar_user_docs[doc_id] += similarity
        
        # 按得分排序，取前N个文档
        top_docs = sorted(similar_user_docs.items(), 
                         key=lambda x: x[1], reverse=True)[:limit]
        
        # 获取文档详情
        for doc_id, score in top_docs:
            doc_info = await self._get_document_info(doc_id)
            if doc_info:
                doc_info['collaborative_score'] = score
                candidates.append(doc_info)
        
        return candidates
    
    def _get_user_positive_interactions(self, user_id: str) -> List[str]:
        """获取用户的正向交互文档"""
        import sqlite3
        
        doc_ids = []
        with sqlite3.connect(self.user_profiler.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT document_id 
                FROM user_interactions 
                WHERE user_id = ? 
                AND interaction_type IN ('feedback_positive', 'document_view', 'bookmark')
                AND document_id IS NOT NULL
                AND duration_seconds > 30  -- 浏览超过30秒
            ''', (user_id,))
            
            doc_ids = [row[0] for row in cursor.fetchall()]
        
        return doc_ids
    
    async def _get_preferred_author_candidates(self, user_profile: UserProfile, 
                                             limit: int) -> List[Dict]:
        """获取偏好作者的新文档"""
        candidates = []
        
        # 按偏好权重排序作者
        top_authors = sorted(user_profile.preferred_authors.items(), 
                           key=lambda x: x[1], reverse=True)[:5]
        
        for author, weight in top_authors:
            author_papers = self.vector_db.get_papers_by_author(
                author_name=author, 
                limit=limit // len(top_authors) + 2
            )
            
            for paper in author_papers:
                paper['author_preference_score'] = weight
                candidates.append(paper)
        
        return candidates
    
    async def _get_document_info(self, document_id: str) -> Optional[Dict]:
        """获取文档信息"""
        # 这里应该从向量数据库或文档存储中获取文档信息
        # 简化实现，返回基本信息
        try:
            # 从Qdrant中查询文档
            from qdrant_client.http import models as qdrant_models
            
            search_result = self.vector_db.db.client.scroll(
                collection_name=self.vector_db.db.collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="source_id",
                            match=qdrant_models.MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            if search_result[0]:
                point = search_result[0][0]
                payload = point.payload
                
                return {
                    'id': document_id,
                    'title': payload.get('metadata', {}).get('title', ''),
                    'authors': payload.get('metadata', {}).get('authors', []),
                    'abstract': payload.get('metadata', {}).get('abstract', ''),
                    'url': payload.get('metadata', {}).get('url', ''),
                    'published_date': payload.get('metadata', {}).get('published_date'),
                    'categories': payload.get('metadata', {}).get('categories', []),
                    'keywords': payload.get('metadata', {}).get('keywords', []),
                    'content': payload.get('content', '')
                }
        except Exception as e:
            logger.error(f"Error getting document info for {document_id}: {e}")
        
        return None
    
    async def _calculate_recommendation_score(self, user_profile: UserProfile, 
                                            document: Dict, 
                                            request: RecommendationRequest) -> Optional[RecommendationResult]:
        """计算推荐得分"""
        try:
            # 内容相似度得分
            content_score = self._calculate_content_score(user_profile, document)
            
            # 协同过滤得分
            collaborative_score = document.get('collaborative_score', 0.0)
            
            # 流行度得分
            popularity_score = await self._calculate_popularity_score(document)
            
            # 新鲜度得分
            freshness_score = self._calculate_freshness_score(document)
            
            # 作者偏好得分
            author_score = self._calculate_author_preference_score(user_profile, document)
            
            # 综合得分计算
            final_score = (
                self.algorithm_weights['content_based'] * content_score +
                self.algorithm_weights['collaborative_filtering'] * collaborative_score +
                self.algorithm_weights['popularity_based'] * popularity_score +
                self.algorithm_weights['trending'] * freshness_score +
                0.1 * author_score  # 作者偏好加权
            )
            
            # 应用多样性和新鲜度因子
            final_score *= (1 + request.freshness_factor * freshness_score)
            
            # 确定推荐理由
            reason_parts = []
            if content_score > 0.5:
                reason_parts.append("与您的研究兴趣匹配")
            if collaborative_score > 0.3:
                reason_parts.append("相似用户喜欢")
            if popularity_score > 0.7:
                reason_parts.append("热门论文")
            if freshness_score > 0.8:
                reason_parts.append("最新发布")
            if author_score > 0.5:
                reason_parts.append("您关注的作者")
            
            recommendation_reason = "、".join(reason_parts) if reason_parts else "系统推荐"
            
            return RecommendationResult(
                document_id=document.get('id', ''),
                title=document.get('title', ''),
                authors=document.get('authors', []),
                abstract=document.get('abstract', ''),
                url=document.get('url', ''),
                published_date=document.get('published_date'),
                categories=document.get('categories', []),
                keywords=document.get('keywords', []),
                recommendation_score=final_score,
                recommendation_reason=recommendation_reason,
                algorithm_used="hybrid",
                content_score=content_score,
                collaborative_score=collaborative_score,
                popularity_score=popularity_score,
                freshness_score=freshness_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating recommendation score: {e}")
            return None
    
    def _calculate_content_score(self, user_profile: UserProfile, document: Dict) -> float:
        """计算内容相似度得分"""
        if not user_profile.research_interests:
            return 0.0
        
        doc_text = " ".join([
            document.get('title', ''),
            document.get('abstract', ''),
            " ".join(document.get('keywords', []))
        ]).lower()
        
        total_score = 0.0
        total_weight = 0.0
        
        for category, interest in user_profile.research_interests.items():
            category_score = 0.0
            
            # 检查关键词匹配
            for keyword in interest.keywords:
                if keyword in doc_text:
                    category_score += 1.0
            
            # 归一化类别得分
            if interest.keywords:
                category_score /= len(interest.keywords)
            
            # 按兴趣置信度加权
            weight = interest.confidence_score
            total_score += category_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    async def _calculate_popularity_score(self, document: Dict) -> float:
        """计算流行度得分"""
        # 简化实现：基于文档在系统中的交互频率
        doc_id = document.get('id', '')
        if not doc_id:
            return 0.0
        
        # 从缓存中获取或计算
        if (self.last_cache_update and 
            datetime.now() - self.last_cache_update < self.cache_ttl and 
            doc_id in self.popularity_cache):
            return self.popularity_cache[doc_id]
        
        # 查询文档的交互统计
        import sqlite3
        
        interaction_count = 0
        with sqlite3.connect(self.user_profiler.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) 
                FROM user_interactions 
                WHERE document_id = ? 
                AND timestamp >= datetime('now', '-30 days')
            ''', (doc_id,))
            
            result = cursor.fetchone()
            if result:
                interaction_count = result[0]
        
        # 归一化得分（假设最大交互数为100）
        popularity_score = min(1.0, interaction_count / 100.0)
        
        # 缓存结果
        self.popularity_cache[doc_id] = popularity_score
        return popularity_score
    
    def _calculate_freshness_score(self, document: Dict) -> float:
        """计算新鲜度得分"""
        published_date = document.get('published_date')
        if not published_date:
            return 0.5  # 默认中等新鲜度
        
        if isinstance(published_date, str):
            try:
                published_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            except:
                return 0.5
        
        # 计算天数差
        now = datetime.now(timezone.utc)
        if published_date.tzinfo is None:
            published_date = published_date.replace(tzinfo=timezone.utc)
        
        days_old = (now - published_date).days
        
        # 新鲜度得分：最近7天得分1.0，之后指数衰减
        if days_old <= 7:
            return 1.0
        elif days_old <= 30:
            return math.exp(-(days_old - 7) / 15)  # 30天后约0.14
        else:
            return 0.1
    
    def _calculate_author_preference_score(self, user_profile: UserProfile, document: Dict) -> float:
        """计算作者偏好得分"""
        doc_authors = document.get('authors', [])
        if not doc_authors or not user_profile.preferred_authors:
            return 0.0
        
        max_preference = max(user_profile.preferred_authors.values()) if user_profile.preferred_authors else 1.0
        
        total_score = 0.0
        for author in doc_authors:
            if author in user_profile.preferred_authors:
                # 归一化作者偏好
                score = user_profile.preferred_authors[author] / max_preference
                total_score += score
        
        # 平均得分
        return min(1.0, total_score / len(doc_authors))
    
    def _apply_diversity_filtering(self, recommendations: List[RecommendationResult], 
                                 diversity_factor: float) -> List[RecommendationResult]:
        """应用多样性过滤"""
        if diversity_factor <= 0 or len(recommendations) <= 1:
            return recommendations
        
        selected = []
        remaining = recommendations.copy()
        
        # 选择最高分的作为第一个
        selected.append(remaining.pop(0))
        
        while remaining and len(selected) < len(recommendations):
            best_candidate = None
            best_score = -1
            
            for i, candidate in enumerate(remaining):
                # 计算与已选择推荐的多样性
                diversity_score = self._calculate_diversity_score(candidate, selected)
                
                # 综合得分 = 原始得分 × (1 + 多样性因子 × 多样性得分)
                combined_score = candidate.recommendation_score * (1 + diversity_factor * diversity_score)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = i
            
            if best_candidate is not None:
                selected.append(remaining.pop(best_candidate))
            else:
                break
        
        return selected
    
    def _calculate_diversity_score(self, candidate: RecommendationResult, 
                                 selected: List[RecommendationResult]) -> float:
        """计算多样性得分"""
        if not selected:
            return 1.0
        
        # 比较类别多样性
        candidate_categories = set(candidate.categories)
        
        diversity_scores = []
        for selected_rec in selected:
            selected_categories = set(selected_rec.categories)
            
            # Jaccard距离作为多样性指标
            intersection = len(candidate_categories & selected_categories)
            union = len(candidate_categories | selected_categories)
            
            if union == 0:
                diversity = 1.0
            else:
                diversity = 1.0 - (intersection / union)
            
            diversity_scores.append(diversity)
        
        # 返回平均多样性
        return sum(diversity_scores) / len(diversity_scores)
    
    async def update_recommendation_feedback(self, user_id: str, document_id: str, 
                                           action: str, context: Optional[Dict] = None):
        """更新推荐反馈"""
        from .user_profiler import UserInteraction
        import uuid
        
        # 记录用户交互
        interaction_type = InteractionType.RECOMMENDATION_CLICK
        if action == "click":
            interaction_type = InteractionType.RECOMMENDATION_CLICK
        elif action == "positive_feedback":
            interaction_type = InteractionType.FEEDBACK_POSITIVE
        elif action == "negative_feedback":
            interaction_type = InteractionType.FEEDBACK_NEGATIVE
        
        interaction = UserInteraction(
            interaction_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=context.get('session_id', 'unknown') if context else 'unknown',
            interaction_type=interaction_type,
            timestamp=datetime.now(timezone.utc),
            document_id=document_id,
            recommendation_context=context
        )
        
        self.user_profiler.record_interaction(interaction)
        
        # 更新用户画像中的推荐历史
        user_profile = self.user_profiler.get_user_profile(user_id)
        if user_profile:
            if document_id not in user_profile.recommendation_history:
                user_profile.recommendation_history.append(document_id)
            
            if action == "click":
                user_profile.recommendation_clicks += 1
            
            self.user_profiler.save_user_profile(user_profile)
        
        logger.info(f"Updated recommendation feedback: user={user_id}, doc={document_id}, action={action}")

# 使用示例
async def main():
    """测试推荐引擎"""
    from pathlib import Path
    from ..retrieval.vector_database import VectorDatabaseManager
    
    # 初始化组件
    user_profiler = UserProfiler(
        db_path=Path("data/user_profiles/profiles.db"),
        storage_path=Path("data/user_profiles")
    )
    
    vector_db_config = {
        'qdrant_host': 'localhost',
        'qdrant_port': 6333,
        'collection_name': 'ai_papers'
    }
    vector_db = VectorDatabaseManager(vector_db_config)
    
    rec_engine = RecommendationEngine(
        user_profiler=user_profiler,
        vector_db=vector_db,
        storage_path=Path("data/recommendations")
    )
    
    # 生成推荐
    user_id = "test_user_001"
    recommendations = await rec_engine.generate_daily_recommendations(user_id, limit=5)
    
    print("每日推荐结果:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.title}")
        print(f"   得分: {rec.recommendation_score:.3f}")
        print(f"   理由: {rec.recommendation_reason}")
        print(f"   算法: {rec.algorithm_used}")
        print()

if __name__ == "__main__":
    asyncio.run(main())