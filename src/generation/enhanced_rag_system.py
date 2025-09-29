# src/generation/enhanced_rag_system.py
import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

# 现有组件
from .ultimate_rag_system import UltimateRAGSystem
from ..retrieval.vector_database import VectorDatabaseManager
from ..feedback.feedback_system import FeedbackCollector

# 新的个性化组件
from ..personalization.user_profiler import UserProfiler, UserInteraction, InteractionType
from ..personalization.recommendation_engine import RecommendationEngine, RecommendationRequest, RecommendationResult
from ..personalization.preference_tracker import PreferenceTracker
from ..storage.usage_analytics import UsageAnalytics
from ..storage.storage_optimizer import StorageOptimizer
from ..storage.data_lifecycle import DataLifecycleManager

@dataclass
class EnhancedRAGResponse:
    """增强的RAG响应"""
    # 基础RAG响应
    answer: str
    source_chunks: List[Dict]
    confidence: float
    generation_time: float
    token_count: int
    
    # 个性化扩展
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    recommendations: List[RecommendationResult] = field(default_factory=list)
    personalization_score: float = 0.0
    user_interests_matched: List[str] = field(default_factory=list)
    
    # 系统优化信息
    cache_hit: bool = False
    storage_tier_used: str = "unknown"
    retrieval_optimization: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PersonalizedQuery:
    """个性化查询"""
    original_query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[Dict] = None
    
    # 个性化参数
    enable_recommendations: bool = True
    recommendation_limit: int = 5
    personalization_weight: float = 0.3
    diversity_factor: float = 0.2
    
    # 系统参数
    use_cache: bool = True
    preferred_storage_tier: Optional[str] = None
    max_retrieval_time: float = 10.0

class EnhancedRAGSystem:
    """增强的RAG系统 - 集成个性化推荐和存储优化"""
    
    def __init__(self, config: Dict[str, Any], storage_root: Path):
        self.config = config
        self.storage_root = storage_root
        
        # 初始化核心组件
        self.vector_db = VectorDatabaseManager(config)
        self.feedback_collector = FeedbackCollector(storage_root / "feedback" / "feedback.db")
        
        # 初始化个性化组件
        self.user_profiler = UserProfiler(
            db_path=storage_root / "user_profiles" / "profiles.db",
            storage_path=storage_root / "user_profiles"
        )
        
        self.preference_tracker = PreferenceTracker(
            user_profiler=self.user_profiler,
            storage_path=storage_root / "preferences"
        )
        
        # 初始化存储优化组件
        self.usage_analytics = UsageAnalytics(
            db_path=storage_root / "storage" / "usage_analytics.db",
            storage_root=storage_root / "storage"
        )
        
        self.storage_optimizer = StorageOptimizer(
            storage_root=storage_root / "storage",
            usage_analytics=self.usage_analytics
        )
        
        self.data_lifecycle = DataLifecycleManager(
            usage_analytics=self.usage_analytics,
            storage_optimizer=self.storage_optimizer
        )
        
        # 初始化推荐引擎
        self.recommendation_engine = RecommendationEngine(
            user_profiler=self.user_profiler,
            vector_db=self.vector_db,
            storage_path=storage_root / "recommendations"
        )
        
        # 初始化基础RAG系统
        self.base_rag = UltimateRAGSystem(config)
        if hasattr(self.base_rag, 'set_database'):
            self.base_rag.set_database(self.vector_db)
        
        # 系统状态
        self.is_initialized = False
        self.background_tasks = []
        
        # 性能统计
        self.system_stats = {
            'total_queries': 0,
            'personalized_queries': 0,
            'recommendations_generated': 0,
            'cache_hits': 0,
            'avg_response_time': 0.0,
            'user_satisfaction': 0.0
        }
    
    async def initialize(self):
        """初始化系统"""
        if self.is_initialized:
            return
        
        logger.info("Initializing Enhanced RAG System...")
        
        try:
            # 启动个性化组件
            await self.preference_tracker.start_tracking()
            
            # 启动存储优化组件
            await self.usage_analytics.start_analytics()
            await self.storage_optimizer.start_optimizer()
            await self.data_lifecycle.start_lifecycle_management()
            
            # 启动推荐引擎
            # (推荐引擎没有需要启动的后台任务)
            
            self.is_initialized = True
            logger.info("Enhanced RAG System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced RAG System: {e}")
            raise
    
    async def shutdown(self):
        """关闭系统"""
        if not self.is_initialized:
            return
        
        logger.info("Shutting down Enhanced RAG System...")
        
        try:
            # 停止后台任务
            for task in self.background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # 停止个性化组件
            await self.preference_tracker.stop_tracking()
            
            # 停止存储优化组件
            await self.usage_analytics.stop_analytics()
            await self.storage_optimizer.stop_optimizer()
            await self.data_lifecycle.stop_lifecycle_management()
            
            self.is_initialized = False
            logger.info("Enhanced RAG System shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def generate_enhanced_answer(self, query: PersonalizedQuery) -> EnhancedRAGResponse:
        """生成增强的个性化回答"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        session_id = query.session_id or str(uuid.uuid4())
        
        # 记录查询开始
        await self._track_query_start(query, session_id)
        
        try:
            # 1. 获取用户画像和偏好
            user_profile = None
            if query.user_id:
                user_profile = self.user_profiler.get_user_profile(query.user_id)
            
            # 2. 个性化查询增强
            enhanced_query = await self._enhance_query_with_personalization(
                query, user_profile
            )
            
            # 3. 执行基础RAG检索
            base_response = await self._execute_base_rag(enhanced_query)
            
            # 4. 生成个性化推荐
            recommendations = []
            if query.enable_recommendations and query.user_id:
                recommendations = await self._generate_recommendations(
                    query, user_profile, base_response
                )
            
            # 5. 计算个性化得分
            personalization_score = await self._calculate_personalization_score(
                query, user_profile, base_response
            )
            
            # 6. 构建增强响应
            response = EnhancedRAGResponse(
                answer=base_response.answer,
                source_chunks=base_response.source_chunks,
                confidence=base_response.confidence,
                generation_time=time.time() - start_time,
                token_count=getattr(base_response, 'token_count', 0),
                user_id=query.user_id,
                session_id=session_id,
                recommendations=recommendations,
                personalization_score=personalization_score,
                user_interests_matched=await self._identify_matched_interests(
                    query, user_profile
                ),
                cache_hit=False,  # 从base_response获取
                storage_tier_used="unknown",  # 从存储分析获取
                retrieval_optimization={}
            )
            
            # 7. 记录用户交互和访问
            await self._track_response_generated(query, response, session_id)
            
            # 8. 更新系统统计
            self._update_system_stats(query, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating enhanced answer: {e}")
            # 回退到基础RAG
            try:
                base_response = await self.base_rag.generate_answer(query.original_query)
                return EnhancedRAGResponse(
                    answer=base_response.answer,
                    source_chunks=getattr(base_response, 'source_chunks', []),
                    confidence=getattr(base_response, 'confidence', 0.5),
                    generation_time=time.time() - start_time,
                    token_count=getattr(base_response, 'token_count', 0),
                    user_id=query.user_id,
                    session_id=session_id
                )
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise
    
    async def _track_query_start(self, query: PersonalizedQuery, session_id: str):
        """跟踪查询开始"""
        if query.user_id:
            await self.preference_tracker.track_search(
                user_id=query.user_id,
                query_text=query.original_query,
                session_id=session_id,
                search_context=query.context
            )
    
    async def _enhance_query_with_personalization(self, query: PersonalizedQuery, 
                                                user_profile) -> str:
        """使用个性化信息增强查询"""
        enhanced_query = query.original_query
        
        if user_profile and user_profile.research_interests:
            # 基于用户兴趣添加上下文
            top_interests = sorted(
                user_profile.research_interests.items(),
                key=lambda x: x[1].confidence_score,
                reverse=True
            )[:3]
            
            # 添加隐含的兴趣上下文（不直接修改查询文本）
            # 这将在检索时使用
            query.context = query.context or {}
            query.context['user_interests'] = [interest[0] for interest in top_interests]
            query.context['user_keywords'] = []
            
            for _, interest in top_interests:
                query.context['user_keywords'].extend(list(interest.keywords)[:5])
        
        return enhanced_query
    
    async def _execute_base_rag(self, query: PersonalizedQuery):
        """执行基础RAG检索"""
        # 这里应该调用现有的RAG系统
        # 简化实现，直接使用ultimate_rag_system
        try:
            if hasattr(self.base_rag, 'generate_answer'):
                return await self.base_rag.generate_answer(
                    query.original_query,
                    top_k=10,
                    context_chunks=5
                )
            else:
                # 手动构造响应结构
                from dataclasses import dataclass
                
                @dataclass
                class BasicResponse:
                    answer: str
                    source_chunks: List[Dict]
                    confidence: float
                    token_count: int = 0
                
                # 使用vector_db进行检索
                from sentence_transformers import SentenceTransformer
                embedder = SentenceTransformer('BAAI/bge-m3')
                query_vector = embedder.encode([query.original_query], convert_to_numpy=True)[0]
                
                search_results = self.vector_db.search(
                    query_vector=query_vector,
                    query_text=query.original_query,
                    top_k=5,
                    search_type="hybrid"
                )
                
                # 简单拼接答案
                answer = f"基于检索结果，关于'{query.original_query}'的信息如下：\n\n"
                for i, result in enumerate(search_results[:3], 1):
                    content = result.get('content', '')[:200]
                    answer += f"{i}. {content}...\n\n"
                
                return BasicResponse(
                    answer=answer,
                    source_chunks=search_results,
                    confidence=0.8,
                    token_count=len(answer.split())
                )
                
        except Exception as e:
            logger.error(f"Error in base RAG execution: {e}")
            raise
    
    async def _generate_recommendations(self, query: PersonalizedQuery, 
                                      user_profile, base_response) -> List[RecommendationResult]:
        """生成个性化推荐"""
        if not query.user_id:
            return []
        
        try:
            recommendations = await self.recommendation_engine.generate_daily_recommendations(
                user_id=query.user_id,
                limit=query.recommendation_limit
            )
            
            # 更新推荐反馈
            for rec in recommendations:
                await self.recommendation_engine.update_recommendation_feedback(
                    user_id=query.user_id,
                    document_id=rec.document_id,
                    action="view",
                    context={
                        'query': query.original_query,
                        'session_id': query.session_id
                    }
                )
            
            self.system_stats['recommendations_generated'] += len(recommendations)
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    async def _calculate_personalization_score(self, query: PersonalizedQuery, 
                                             user_profile, base_response) -> float:
        """计算个性化得分"""
        if not user_profile:
            return 0.0
        
        score = 0.0
        
        # 基于用户兴趣匹配
        if user_profile.research_interests:
            query_lower = query.original_query.lower()
            for category, interest in user_profile.research_interests.items():
                for keyword in interest.keywords:
                    if keyword.lower() in query_lower:
                        score += interest.confidence_score * 0.3
        
        # 基于查询模式匹配
        if user_profile.query_patterns:
            for pattern in user_profile.query_patterns:
                if pattern in query.original_query.lower():
                    score += 0.2
        
        # 归一化得分
        return min(1.0, score)
    
    async def _identify_matched_interests(self, query: PersonalizedQuery, 
                                        user_profile) -> List[str]:
        """识别匹配的用户兴趣"""
        if not user_profile or not user_profile.research_interests:
            return []
        
        matched_interests = []
        query_lower = query.original_query.lower()
        
        for category, interest in user_profile.research_interests.items():
            for keyword in interest.keywords:
                if keyword.lower() in query_lower:
                    matched_interests.append(category)
                    break
        
        return matched_interests
    
    async def _track_response_generated(self, query: PersonalizedQuery, 
                                      response: EnhancedRAGResponse, 
                                      session_id: str):
        """跟踪响应生成"""
        if not query.user_id:
            return
        
        # 记录文档访问
        for source_chunk in response.source_chunks:
            document_id = source_chunk.get('source_id') or source_chunk.get('document_id')
            if document_id:
                await self.usage_analytics.log_document_access(
                    document_id=document_id,
                    user_id=query.user_id,
                    access_type="search_result",
                    load_time_ms=response.generation_time * 1000,
                    cache_hit=response.cache_hit,
                    source_tier=response.storage_tier_used
                )
                
                # 跟踪页面浏览
                await self.preference_tracker.track_page_view(
                    user_id=query.user_id,
                    document_id=document_id,
                    session_id=session_id,
                    page_context={
                        'query': query.original_query,
                        'result_position': response.source_chunks.index(source_chunk),
                        'confidence': response.confidence
                    }
                )
    
    def _update_system_stats(self, query: PersonalizedQuery, response: EnhancedRAGResponse):
        """更新系统统计"""
        self.system_stats['total_queries'] += 1
        
        if query.user_id:
            self.system_stats['personalized_queries'] += 1
        
        if response.cache_hit:
            self.system_stats['cache_hits'] += 1
        
        # 更新平均响应时间
        current_avg = self.system_stats['avg_response_time']
        total_queries = self.system_stats['total_queries']
        new_avg = (current_avg * (total_queries - 1) + response.generation_time) / total_queries
        self.system_stats['avg_response_time'] = new_avg
    
    async def submit_feedback(self, user_id: str, query: str, response: EnhancedRAGResponse, 
                            feedback_type: str, feedback_value: Any) -> str:
        """提交用户反馈"""
        # 记录反馈到现有系统
        feedback_id = self.feedback_collector.collect_thumbs_feedback(
            query=query,
            answer=response.answer,
            is_positive=(feedback_type == "positive" or 
                        (feedback_type == "rating" and feedback_value > 3)),
            source_chunks=response.source_chunks
        )
        
        # 更新个性化系统
        if user_id and response.session_id:
            await self.preference_tracker.track_feedback(
                user_id=user_id,
                document_id="response",  # 整体响应反馈
                rating=5.0 if feedback_type == "positive" else 1.0,
                session_id=response.session_id
            )
        
        # 更新推荐反馈
        if response.recommendations:
            for rec in response.recommendations:
                await self.recommendation_engine.update_recommendation_feedback(
                    user_id=user_id,
                    document_id=rec.document_id,
                    action="positive_feedback" if feedback_type == "positive" else "negative_feedback",
                    context={'original_query': query, 'feedback_value': feedback_value}
                )
        
        return feedback_id
    
    async def get_user_dashboard(self, user_id: str) -> Dict[str, Any]:
        """获取用户仪表板数据"""
        dashboard = {
            'user_profile': None,
            'recent_activity': {},
            'recommendations': [],
            'interests': {},
            'system_usage': {},
            'personalization_score': 0.0
        }
        
        try:
            # 用户画像
            user_profile = self.user_profiler.get_user_profile(user_id)
            if user_profile:
                dashboard['user_profile'] = user_profile.to_dict()
                
                # 兴趣向量
                dashboard['interests'] = self.user_profiler.get_user_interests_vector(user_id)
                
                # 个性化得分
                dashboard['personalization_score'] = min(1.0, sum(dashboard['interests'].values()) / 5)
            
            # 最新推荐
            dashboard['recommendations'] = await self.recommendation_engine.generate_daily_recommendations(
                user_id=user_id, limit=10
            )
            
            # 用户行为分析
            dashboard['recent_activity'] = await self.preference_tracker.get_user_behavior_insights(
                user_id=user_id, days=30
            )
            
            # 系统使用统计
            dashboard['system_usage'] = {
                'total_queries': user_profile.total_queries if user_profile else 0,
                'documents_viewed': user_profile.total_documents_viewed if user_profile else 0,
                'positive_feedback': user_profile.total_positive_feedback if user_profile else 0,
                'activity_level': user_profile.activity_level if user_profile else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error getting user dashboard: {e}")
        
        return dashboard
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """获取系统概览"""
        overview = {
            'system_stats': self.system_stats.copy(),
            'storage_metrics': {},
            'recommendation_performance': {},
            'user_engagement': {},
            'optimization_status': {}
        }
        
        try:
            # 存储指标
            storage_metrics = await self.usage_analytics.calculate_storage_metrics()
            overview['storage_metrics'] = storage_metrics.to_dict()
            
            # 推荐性能
            overview['recommendation_performance'] = {
                'total_recommendations': self.system_stats['recommendations_generated'],
                'recommendation_click_rate': 0.0,  # 需要计算
                'user_satisfaction': self.system_stats['user_satisfaction']
            }
            
            # 优化状态
            overview['optimization_status'] = {
                'storage_optimizer': self.storage_optimizer.get_migration_status(),
                'lifecycle_management': self.data_lifecycle.get_lifecycle_summary(),
                'analytics_summary': self.usage_analytics.get_analytics_summary()
            }
            
        except Exception as e:
            logger.error(f"Error getting system overview: {e}")
        
        return overview

# 使用示例
async def main():
    """测试增强RAG系统"""
    config = {
        'embedding_model': 'BAAI/bge-m3',
        'llm_model': 'Qwen/Qwen2-7B-Instruct',
        'device': 'cpu',
        'qdrant_host': 'localhost',
        'qdrant_port': 6333,
        'collection_name': 'ai_papers'
    }
    
    storage_root = Path("data")
    
    # 初始化系统
    rag_system = EnhancedRAGSystem(config, storage_root)
    await rag_system.initialize()
    
    # 创建个性化查询
    query = PersonalizedQuery(
        original_query="What are the latest developments in transformer models?",
        user_id="test_user_001",
        session_id="session_001",
        enable_recommendations=True,
        recommendation_limit=5
    )
    
    # 生成增强回答
    response = await rag_system.generate_enhanced_answer(query)
    
    print("增强RAG响应:")
    print(f"回答: {response.answer[:200]}...")
    print(f"置信度: {response.confidence:.3f}")
    print(f"个性化得分: {response.personalization_score:.3f}")
    print(f"匹配兴趣: {response.user_interests_matched}")
    print(f"推荐数量: {len(response.recommendations)}")
    
    # 获取用户仪表板
    dashboard = await rag_system.get_user_dashboard("test_user_001")
    print("\\n用户仪表板:")
    print(f"个性化得分: {dashboard['personalization_score']:.3f}")
    print(f"推荐数量: {len(dashboard['recommendations'])}")
    
    # 关闭系统
    await rag_system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())