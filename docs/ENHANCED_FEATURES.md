# Enhanced RAG-AI Features Guide

本文档详细介绍了RAG-AI系统中新增的高级功能，包括个性化推荐、存储优化、用户画像等核心特性。

## 📋 目录

- [个性化系统](#个性化系统)
- [存储优化系统](#存储优化系统)
- [增强RAG引擎](#增强rag引擎)
- [用户界面增强](#用户界面增强)
- [API端点详解](#api端点详解)
- [配置和部署](#配置和部署)

## 🎯 个性化系统

### 用户画像管理

**核心文件**: `src/personalization/user_profiler.py`

系统通过智能分析用户交互行为，构建精准的用户画像：

```python
@dataclass
class UserProfile:
    """用户画像数据结构"""
    user_id: str
    research_interests: List[ResearchInterest]
    interaction_history: List[UserInteraction]
    preferences: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    
    # 统计信息
    total_queries: int = 0
    avg_session_duration: float = 0.0
    preferred_response_length: str = "medium"
    favorite_topics: List[str] = field(default_factory=list)
```

**主要功能**:

1. **自动兴趣提取**
```python
async def extract_research_interests(self, query: str, response_content: str) -> List[ResearchInterest]:
    """从查询和响应中提取研究兴趣"""
    # 使用NLP技术提取关键概念
    keywords = await self._extract_keywords(query, response_content)
    concepts = await self._extract_concepts(response_content)
    
    interests = []
    for keyword in keywords:
        interest = ResearchInterest(
            topic=keyword,
            weight=self._calculate_weight(keyword, concepts),
            last_interaction=datetime.now(timezone.utc)
        )
        interests.append(interest)
    
    return interests
```

2. **用户偏好学习**
```python
async def update_profile_from_interaction(self, user_id: str, interaction: UserInteraction):
    """根据用户交互更新画像"""
    profile = await self.get_or_create_user_profile(user_id)
    
    # 更新研究兴趣
    new_interests = await self.extract_research_interests(
        interaction.query, interaction.response_content
    )
    
    # 合并和权重调整
    profile.research_interests = self._merge_interests(
        profile.research_interests, new_interests
    )
    
    # 更新偏好设置
    profile.preferences.update({
        'response_length': self._infer_preferred_length(interaction),
        'detail_level': self._infer_detail_preference(interaction),
        'source_preference': self._analyze_source_usage(interaction)
    })
```

### 推荐引擎

**核心文件**: `src/personalization/recommendation_engine.py`

实现混合推荐算法，结合内容过滤和协同过滤：

```python
class RecommendationEngine:
    """智能推荐引擎"""
    
    async def generate_daily_recommendations(
        self, 
        user_id: str, 
        limit: int = 10,
        days_back: int = 7
    ) -> List[RecommendationItem]:
        """生成每日个性化推荐"""
        
        # 获取用户画像
        profile = await self.user_profiler.get_user_profile(user_id)
        if not profile:
            return await self._generate_default_recommendations(limit)
        
        # 多策略推荐
        content_based = await self._content_based_recommendations(profile, limit // 2)
        collaborative = await self._collaborative_filtering(profile, limit // 2)
        trending = await self._trending_recommendations(limit // 4)
        
        # 混合和排序
        all_recommendations = content_based + collaborative + trending
        scored_items = []
        
        for item in all_recommendations:
            score = await self._calculate_recommendation_score(item, profile)
            scored_items.append((score, item))
        
        # 返回按分数排序的推荐
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_items[:limit]]
```

**推荐算法详解**:

1. **内容过滤**
```python
async def _content_based_recommendations(self, profile: UserProfile, limit: int):
    """基于内容的推荐"""
    recommendations = []
    
    for interest in profile.research_interests[:5]:  # Top 5 interests
        # 查找相关文档
        similar_docs = await self.vector_db.similarity_search(
            query_vector=interest.embedding,
            filter_params={
                'published_date': {'gte': datetime.now() - timedelta(days=30)},
                'quality_score': {'gte': 0.7}
            },
            limit=limit // 5
        )
        
        for doc in similar_docs:
            score = self._calculate_content_similarity(interest, doc)
            rec_item = RecommendationItem(
                document_id=doc.id,
                title=doc.title,
                summary=doc.summary,
                recommendation_reason=f"基于您对'{interest.topic}'的兴趣",
                score=score,
                recommendation_type="content_based"
            )
            recommendations.append(rec_item)
    
    return recommendations
```

2. **协同过滤**
```python
async def _collaborative_filtering(self, profile: UserProfile, limit: int):
    """协同过滤推荐"""
    # 找到相似用户
    similar_users = await self._find_similar_users(profile)
    recommendations = []
    
    for similar_user_id, similarity_score in similar_users[:10]:
        # 获取相似用户最近的高评分交互
        recent_interactions = await self.storage.get_user_interactions(
            similar_user_id,
            days_back=14,
            min_rating=4.0
        )
        
        for interaction in recent_interactions:
            if interaction.document_id not in profile.viewed_documents:
                score = similarity_score * interaction.rating
                rec_item = RecommendationItem(
                    document_id=interaction.document_id,
                    recommendation_reason="相似用户也对此感兴趣",
                    score=score,
                    recommendation_type="collaborative"
                )
                recommendations.append(rec_item)
    
    return recommendations[:limit]
```

## 💾 存储优化系统

### 多层存储架构

**核心文件**: `src/storage/storage_optimizer.py`

实现智能的多层存储系统，根据访问模式自动优化数据分布：

```python
class StorageTier(Enum):
    """存储层级定义"""
    HOT = "hot"        # 高频访问，SSD存储
    WARM = "warm"      # 中频访问，混合存储
    COLD = "cold"      # 低频访问，HDD存储
    ARCHIVED = "archived"  # 归档，压缩存储

class StorageOptimizer:
    """存储优化器"""
    
    async def optimize_storage(
        self,
        target_hot_ratio: float = 0.1,
        target_warm_ratio: float = 0.3,
        target_cold_ratio: float = 0.5
    ) -> OptimizationResult:
        """执行存储优化"""
        
        # 分析当前存储分布
        current_distribution = await self._analyze_current_distribution()
        
        # 获取访问模式数据
        access_patterns = await self.usage_analytics.analyze_access_patterns(
            days=30
        )
        
        # 生成迁移计划
        migration_plan = self._generate_migration_plan(
            current_distribution,
            access_patterns,
            target_hot_ratio,
            target_warm_ratio,
            target_cold_ratio
        )
        
        # 执行迁移
        migration_results = []
        for migration in migration_plan:
            result = await self._execute_migration(migration)
            migration_results.append(result)
        
        return OptimizationResult(
            total_documents_moved=len(migration_results),
            storage_saved=sum(r.storage_saved for r in migration_results),
            performance_impact=self._calculate_performance_impact(migration_results),
            completed_at=datetime.now(timezone.utc)
        )
```

### 访问模式分析

**核心文件**: `src/storage/usage_analytics.py`

智能分析文档访问模式，为存储优化提供数据支持：

```python
class UsageAnalytics:
    """使用分析器"""
    
    async def analyze_access_patterns(self, days: int = 30) -> List[AccessPattern]:
        """分析访问模式"""
        
        # 获取访问日志
        access_logs = await self.storage.get_access_logs(days_back=days)
        
        patterns = []
        document_stats = {}
        
        # 统计每个文档的访问模式
        for log in access_logs:
            doc_id = log.document_id
            if doc_id not in document_stats:
                document_stats[doc_id] = {
                    'total_accesses': 0,
                    'unique_users': set(),
                    'access_times': [],
                    'access_types': [],
                    'response_times': []
                }
            
            stats = document_stats[doc_id]
            stats['total_accesses'] += 1
            stats['unique_users'].add(log.user_id)
            stats['access_times'].append(log.access_time)
            stats['access_types'].append(log.access_type)
            stats['response_times'].append(log.response_time)
        
        # 生成访问模式
        for doc_id, stats in document_stats.items():
            pattern = AccessPattern(
                document_id=doc_id,
                total_accesses=stats['total_accesses'],
                unique_users=len(stats['unique_users']),
                avg_daily_accesses=stats['total_accesses'] / days,
                access_frequency=self._calculate_frequency(stats['access_times']),
                user_diversity=self._calculate_diversity(stats['unique_users']),
                avg_response_time=sum(stats['response_times']) / len(stats['response_times']),
                recommended_tier=self._recommend_storage_tier(stats),
                last_accessed=max(stats['access_times'])
            )
            patterns.append(pattern)
        
        return patterns
```

### 数据生命周期管理

**核心文件**: `src/storage/data_lifecycle.py`

自动化的数据生命周期管理，确保系统性能和存储效率：

```python
class DataLifecycleManager:
    """数据生命周期管理器"""
    
    async def run_lifecycle_policies(self) -> LifecycleResult:
        """执行生命周期策略"""
        
        policies = await self._load_lifecycle_policies()
        results = []
        
        for policy in policies:
            try:
                if policy.policy_type == "aging":
                    result = await self._process_aging_policy(policy)
                elif policy.policy_type == "cleanup":
                    result = await self._process_cleanup_policy(policy)
                elif policy.policy_type == "migration":
                    result = await self._process_migration_policy(policy)
                elif policy.policy_type == "compression":
                    result = await self._process_compression_policy(policy)
                
                results.append(result)
                logger.info(f"Lifecycle policy {policy.name} executed successfully")
                
            except Exception as e:
                logger.error(f"Failed to execute policy {policy.name}: {e}")
                results.append(PolicyResult(
                    policy_name=policy.name,
                    success=False,
                    error_message=str(e)
                ))
        
        return LifecycleResult(
            policies_executed=len(results),
            successful_policies=sum(1 for r in results if r.success),
            total_data_processed=sum(r.data_processed for r in results if r.success),
            storage_reclaimed=sum(r.storage_saved for r in results if r.success)
        )
```

## 🚀 增强RAG引擎

### 个性化查询处理

**核心文件**: `src/generation/enhanced_rag_system.py`

整合个性化功能的增强RAG系统：

```python
class EnhancedRAGSystem:
    """增强的RAG系统"""
    
    async def generate_enhanced_answer(
        self,
        query: PersonalizedQuery
    ) -> EnhancedRAGResponse:
        """生成个性化增强回答"""
        
        # 获取用户画像
        user_profile = None
        if query.user_id:
            user_profile = await self.user_profiler.get_user_profile(query.user_id)
        
        # 个性化查询重写
        enhanced_query = await self._personalize_query(query, user_profile)
        
        # 智能检索
        retrieval_results = await self._intelligent_retrieval(
            enhanced_query, user_profile
        )
        
        # 个性化生成
        answer = await self._personalized_generation(
            enhanced_query, retrieval_results, user_profile
        )
        
        # 记录交互用于学习
        if query.user_id:
            interaction = UserInteraction(
                user_id=query.user_id,
                query=query.query,
                response_content=answer.answer,
                sources=answer.sources,
                timestamp=datetime.now(timezone.utc),
                rating=None  # 待用户反馈
            )
            await self.user_profiler.record_interaction(interaction)
        
        return answer
```

### 智能检索增强

```python
async def _intelligent_retrieval(
    self,
    query: PersonalizedQuery,
    user_profile: Optional[UserProfile]
) -> List[RetrievalResult]:
    """智能检索，结合用户偏好"""
    
    # 基础向量检索
    base_results = await self.vector_db.similarity_search(
        query_vector=query.embedding,
        limit=query.initial_retrieve * 2
    )
    
    # 用户偏好过滤
    if user_profile:
        filtered_results = []
        for result in base_results:
            # 计算与用户兴趣的相关性
            interest_score = self._calculate_interest_alignment(
                result, user_profile.research_interests
            )
            
            # 应用用户偏好权重
            result.score = result.score * (1 + interest_score * 0.3)
            
            # 过滤用户不感兴趣的内容
            if interest_score > -0.5:  # 阈值过滤
                filtered_results.append(result)
        
        base_results = filtered_results
    
    # 重排序和多样性优化
    final_results = await self.reranker.rerank_with_diversity(
        base_results,
        query.query,
        target_count=query.top_k,
        diversity_weight=0.3
    )
    
    return final_results
```

## 🎨 用户界面增强

### 个性化Streamlit界面

**核心文件**: `enhanced_app.py`

增强的Streamlit界面，集成个性化功能：

```python
def render_personalized_chat_message(
    role: str, 
    content: str, 
    sources: list = None, 
    metrics: dict = None,
    recommendations: list = None
):
    """渲染个性化聊天消息"""
    
    if role == "assistant":
        # 基础回答显示
        st.markdown(f'<div class="assistant-message">{content}</div>', 
                   unsafe_allow_html=True)
        
        # 性能指标
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("置信度", f"{metrics.get('confidence', 0):.2f}")
            with col2:
                st.metric("响应时间", f"{metrics.get('generation_time', 0):.2f}s")
            with col3:
                st.metric("Token数量", metrics.get('token_count', 0))
            with col4:
                st.metric("个性化分数", f"{metrics.get('personalization_score', 0):.2f}")
        
        # 相关推荐
        if recommendations:
            with st.expander("🎯 为您推荐", expanded=False):
                for rec in recommendations[:3]:
                    st.markdown(f"""
                    **{rec.title}**  
                    {rec.summary[:200]}...  
                    *推荐理由: {rec.recommendation_reason}*
                    """)
```

### 用户仪表板

```python
def render_user_dashboard(user_id: str):
    """渲染用户个人仪表板"""
    
    # 获取用户数据
    profile = st.session_state.enhanced_rag.user_profiler.get_user_profile(user_id)
    if not profile:
        st.warning("未找到用户画像，请先进行一些查询来建立您的个性化档案。")
        return
    
    st.header(f"📊 {user_id} 的个人仪表板")
    
    # 基础统计
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总查询数", profile.total_queries)
    with col2:
        st.metric("兴趣主题", len(profile.research_interests))
    with col3:
        st.metric("平均会话时长", f"{profile.avg_session_duration:.1f}分钟")
    with col4:
        st.metric("偏好响应长度", profile.preferred_response_length)
    
    # 研究兴趣分析
    st.subheader("🔬 研究兴趣分布")
    if profile.research_interests:
        interests_data = {
            'topic': [interest.topic for interest in profile.research_interests[:10]],
            'weight': [interest.weight for interest in profile.research_interests[:10]]
        }
        
        fig = px.bar(interests_data, x='topic', y='weight', 
                    title="研究兴趣权重分布")
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # 每日推荐
    st.subheader("📰 今日推荐")
    recommendations = asyncio.run(
        st.session_state.enhanced_rag.recommendation_engine.generate_daily_recommendations(
            user_id, limit=5
        )
    )
    
    for i, rec in enumerate(recommendations):
        with st.expander(f"{i+1}. {rec.title}"):
            st.markdown(f"**摘要**: {rec.summary}")
            st.markdown(f"**推荐理由**: {rec.recommendation_reason}")
            st.markdown(f"**推荐分数**: {rec.score:.3f}")
            
            if st.button(f"查看详情", key=f"rec_{i}"):
                # 触发详细查看
                st.session_state.selected_document = rec.document_id
                st.rerun()
```

## 🔗 API端点详解

### 个性化查询端点

**核心文件**: `api/enhanced_main.py`

```python
@app.post("/api/v2/ask", response_model=EnhancedRAGResponse)
async def enhanced_ask_question(
    request: PersonalizedQuestionRequest,
    background_tasks: BackgroundTasks
):
    """增强的问答端点，支持个性化"""
    
    try:
        # 构建个性化查询
        personalized_query = PersonalizedQuery(
            query=request.query,
            user_id=request.user_id,
            context=request.context,
            preferences=request.preferences,
            max_results=request.max_results,
            include_recommendations=request.include_recommendations
        )
        
        # 生成个性化回答
        response = await enhanced_rag_system.generate_enhanced_answer(personalized_query)
        
        # 异步更新用户画像
        if request.user_id:
            background_tasks.add_task(
                update_user_profile_async,
                request.user_id,
                request.query,
                response.answer
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Enhanced question processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/recommendations/{user_id}")
async def get_user_recommendations(
    user_id: str,
    limit: int = Query(10, ge=1, le=50),
    days_back: int = Query(7, ge=1, le=30)
):
    """获取用户个性化推荐"""
    
    try:
        recommendations = await enhanced_rag_system.recommendation_engine.generate_daily_recommendations(
            user_id=user_id,
            limit=limit,
            days_back=days_back
        )
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "total_count": len(recommendations),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/user/{user_id}/dashboard")
async def get_user_dashboard(user_id: str):
    """获取用户仪表板数据"""
    
    try:
        # 获取用户画像
        profile = await enhanced_rag_system.user_profiler.get_user_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # 获取使用统计
        usage_stats = await enhanced_rag_system.storage_optimizer.get_user_usage_stats(user_id)
        
        # 获取最近推荐
        recent_recommendations = await enhanced_rag_system.recommendation_engine.get_recent_recommendations(
            user_id, limit=5
        )
        
        return {
            "user_profile": profile,
            "usage_statistics": usage_stats,
            "recent_recommendations": recent_recommendations,
            "dashboard_generated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Dashboard data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### 存储优化端点

```python
@app.post("/api/v2/storage/optimize")
async def optimize_storage(
    request: StorageOptimizationRequest,
    background_tasks: BackgroundTasks
):
    """触发存储优化"""
    
    try:
        # 异步执行存储优化
        background_tasks.add_task(
            run_storage_optimization,
            request.target_hot_ratio,
            request.target_warm_ratio,
            request.target_cold_ratio,
            request.dry_run
        )
        
        return {
            "message": "Storage optimization started",
            "job_id": str(uuid.uuid4()),
            "estimated_duration": "5-10 minutes"
        }
        
    except Exception as e:
        logger.error(f"Storage optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/storage/analytics")
async def get_storage_analytics(
    days: int = Query(30, ge=1, le=90)
):
    """获取存储分析数据"""
    
    try:
        analytics = await enhanced_rag_system.usage_analytics.get_comprehensive_analytics(days)
        
        return {
            "analytics_period_days": days,
            "storage_distribution": analytics.storage_distribution,
            "access_patterns": analytics.access_patterns,
            "optimization_recommendations": analytics.optimization_recommendations,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Storage analytics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

## ⚙️ 配置和部署

### 环境变量配置

增强功能需要以下额外的环境变量：

```bash
# 个性化功能
ENABLE_PERSONALIZATION=true
USER_PROFILE_RETENTION_DAYS=365
RECOMMENDATION_REFRESH_HOURS=24

# 存储优化
ENABLE_STORAGE_OPTIMIZATION=true
STORAGE_OPTIMIZATION_SCHEDULE="0 2 * * *"  # 每天凌晨2点
DEFAULT_HOT_RATIO=0.1
DEFAULT_WARM_RATIO=0.3
DEFAULT_COLD_RATIO=0.5

# 缓存配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
CACHE_TTL_HOURS=24

# 分析和监控
ENABLE_USAGE_ANALYTICS=true
ANALYTICS_RETENTION_DAYS=90
METRICS_COLLECTION_INTERVAL=300  # 5分钟
```

### Docker部署配置

增强的Docker Compose配置：

```yaml
version: '3.8'

services:
  enhanced-api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - QDRANT_HOST=qdrant
      - ENABLE_PERSONALIZATION=true
      - ENABLE_STORAGE_OPTIMIZATION=true
    depends_on:
      - redis
      - qdrant
    volumes:
      - ./project_data:/app/project_data
      - ./logs:/app/logs

  enhanced-frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://enhanced-api:8000
    depends_on:
      - enhanced-api

  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  qdrant:
    image: qdrant/qdrant:v1.7.0
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  redis_data:
  qdrant_data:
```

### 系统监控

个性化功能的监控指标：

```python
# Prometheus指标定义
from prometheus_client import Counter, Histogram, Gauge

# 用户交互指标
user_queries_total = Counter(
    'rag_user_queries_total',
    'Total number of user queries',
    ['user_id', 'query_type']
)

recommendation_generation_time = Histogram(
    'rag_recommendation_generation_seconds',
    'Time spent generating recommendations',
    ['recommendation_type']
)

storage_tier_distribution = Gauge(
    'rag_storage_tier_documents',
    'Number of documents in each storage tier',
    ['tier']
)

personalization_score = Histogram(
    'rag_personalization_score',
    'Personalization effectiveness score',
    ['user_segment']
)
```

## 📈 性能优化建议

### 1. 用户画像缓存
```python
# 使用Redis缓存用户画像
@cache_with_ttl(ttl=3600)  # 1小时缓存
async def get_user_profile_cached(user_id: str):
    return await user_profiler.get_user_profile(user_id)
```

### 2. 推荐预计算
```python
# 每日预计算推荐
@scheduled_task("0 1 * * *")  # 每天凌晨1点
async def precompute_daily_recommendations():
    active_users = await get_active_users(days=7)
    for user_id in active_users:
        recommendations = await recommendation_engine.generate_daily_recommendations(user_id)
        await cache.set(f"daily_recs:{user_id}", recommendations, ttl=86400)
```

### 3. 存储优化调度
```python
# 智能调度存储优化
@scheduled_task("0 2 * * *")  # 每天凌晨2点
async def smart_storage_optimization():
    # 检查系统负载
    if await system_monitor.get_cpu_usage() < 0.3:
        await storage_optimizer.optimize_storage()
    else:
        logger.info("Skipping storage optimization due to high system load")
```

通过这些增强功能，RAG-AI系统现在提供了完整的个性化体验，智能的存储管理，以及强大的用户分析能力。系统能够学习用户偏好，自动优化性能，并提供精准的个性化推荐。