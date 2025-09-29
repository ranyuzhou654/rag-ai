# api/enhanced_main.py
import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# 引入增强的RAG系统
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.generation.enhanced_rag_system import EnhancedRAGSystem, PersonalizedQuery, EnhancedRAGResponse
from src.personalization.user_profiler import UserProfiler, InteractionType, UserInteraction
from src.personalization.recommendation_engine import RecommendationEngine, RecommendationRequest
from src.storage.usage_analytics import UsageAnalytics
from configs.config import config

# API模型定义
class QueryRequest(BaseModel):
    """查询请求"""
    query: str = Field(..., description="用户查询文本")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    enable_recommendations: bool = Field(True, description="是否启用推荐")
    recommendation_limit: int = Field(5, ge=1, le=20, description="推荐数量限制")
    personalization_weight: float = Field(0.3, ge=0.0, le=1.0, description="个性化权重")
    max_retrieval_time: float = Field(10.0, ge=1.0, le=30.0, description="最大检索时间")
    context: Optional[Dict[str, Any]] = Field(None, description="查询上下文")

class FeedbackRequest(BaseModel):
    """反馈请求"""
    user_id: str = Field(..., description="用户ID")
    query: str = Field(..., description="原始查询")
    response_id: str = Field(..., description="响应ID")
    feedback_type: str = Field(..., description="反馈类型：positive, negative, rating")
    feedback_value: Optional[Any] = Field(None, description="反馈值")
    comment: Optional[str] = Field(None, description="反馈评论")

class RecommendationRequestModel(BaseModel):
    """推荐请求"""
    user_id: str = Field(..., description="用户ID")
    recommendation_type: str = Field("daily", description="推荐类型")
    limit: int = Field(10, ge=1, le=50, description="推荐数量")
    diversity_factor: float = Field(0.3, ge=0.0, le=1.0, description="多样性因子")
    category_filter: Optional[List[str]] = Field(None, description="类别过滤")

class UserInteractionModel(BaseModel):
    """用户交互模型"""
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
    interaction_type: str = Field(..., description="交互类型")
    document_id: Optional[str] = Field(None, description="文档ID")
    query_text: Optional[str] = Field(None, description="查询文本")
    duration_seconds: Optional[float] = Field(None, description="持续时间（秒）")
    scroll_depth: Optional[float] = Field(None, description="滚动深度")
    rating: Optional[float] = Field(None, description="评分")

class DocumentAccessModel(BaseModel):
    """文档访问模型"""
    document_id: str = Field(..., description="文档ID")
    user_id: str = Field(..., description="用户ID")
    access_type: str = Field("view", description="访问类型")
    duration: Optional[float] = Field(None, description="访问时长")
    source_context: Optional[Dict[str, Any]] = Field(None, description="来源上下文")

# 全局变量
enhanced_rag_system: Optional[EnhancedRAGSystem] = None
app = FastAPI(
    title="Enhanced RAG API",
    description="个性化推荐增强的RAG问答系统API",
    version="2.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """启动事件"""
    global enhanced_rag_system
    
    try:
        # 初始化增强RAG系统
        storage_root = Path(config.STORAGE_ROOT)
        rag_config = {
            'embedding_model': config.EMBEDDING_MODEL,
            'llm_model': config.LLM_MODEL,
            'device': config.DEVICE,
            'qdrant_host': config.QDRANT_HOST,
            'qdrant_port': config.QDRANT_PORT,
            'collection_name': config.COLLECTION_NAME,
            'HUGGING_FACE_TOKEN': config.HUGGING_FACE_TOKEN
        }
        
        enhanced_rag_system = EnhancedRAGSystem(rag_config, storage_root)
        await enhanced_rag_system.initialize()
        
        print("✅ Enhanced RAG API started successfully")
        
    except Exception as e:
        print(f"❌ Failed to start Enhanced RAG API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    global enhanced_rag_system
    
    if enhanced_rag_system:
        await enhanced_rag_system.shutdown()
        print("✅ Enhanced RAG API shut down successfully")

# 依赖注入
async def get_rag_system() -> EnhancedRAGSystem:
    """获取RAG系统实例"""
    if enhanced_rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return enhanced_rag_system

# 核心API端点
@app.post("/api/v2/ask")
async def ask_question(
    request: QueryRequest,
    rag_system: EnhancedRAGSystem = Depends(get_rag_system)
) -> Dict[str, Any]:
    """增强问答接口"""
    try:
        # 创建个性化查询
        personalized_query = PersonalizedQuery(
            original_query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            context=request.context,
            enable_recommendations=request.enable_recommendations,
            recommendation_limit=request.recommendation_limit,
            personalization_weight=request.personalization_weight,
            max_retrieval_time=request.max_retrieval_time
        )
        
        # 生成增强回答
        response = await rag_system.generate_enhanced_answer(personalized_query)
        
        # 构建API响应
        return {
            "response_id": str(uuid.uuid4()),
            "answer": response.answer,
            "confidence": response.confidence,
            "generation_time": response.generation_time,
            "personalization_score": response.personalization_score,
            "user_interests_matched": response.user_interests_matched,
            "sources": [
                {
                    "content": chunk.get("content", ""),
                    "metadata": chunk.get("metadata", {}),
                    "scores": chunk.get("scores", {}),
                    "semantic_type": chunk.get("semantic_type", "content")
                }
                for chunk in response.source_chunks
            ],
            "recommendations": [
                {
                    "document_id": rec.document_id,
                    "title": rec.title,
                    "authors": rec.authors,
                    "abstract": rec.abstract[:200] + "..." if len(rec.abstract) > 200 else rec.abstract,
                    "url": rec.url,
                    "recommendation_score": rec.recommendation_score,
                    "recommendation_reason": rec.recommendation_reason,
                    "published_date": rec.published_date.isoformat() if rec.published_date else None
                }
                for rec in response.recommendations
            ],
            "system_info": {
                "cache_hit": response.cache_hit,
                "storage_tier": response.storage_tier_used,
                "token_count": response.token_count
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/v2/ask/stream")
async def ask_question_stream(
    request: QueryRequest,
    rag_system: EnhancedRAGSystem = Depends(get_rag_system)
):
    """流式问答接口"""
    async def generate_stream():
        try:
            # 先返回开始标记
            yield f"data: {json.dumps({'type': 'start', 'message': 'Processing query...'})}\n\n"
            
            # 创建个性化查询
            personalized_query = PersonalizedQuery(
                original_query=request.query,
                user_id=request.user_id,
                session_id=request.session_id,
                context=request.context,
                enable_recommendations=request.enable_recommendations,
                recommendation_limit=request.recommendation_limit
            )
            
            # 生成回答
            response = await rag_system.generate_enhanced_answer(personalized_query)
            
            # 分段发送答案
            answer_parts = response.answer.split('\n\n')
            for i, part in enumerate(answer_parts):
                if part.strip():
                    yield f"data: {json.dumps({'type': 'content', 'content': part.strip(), 'index': i})}\n\n"
                    await asyncio.sleep(0.1)  # 模拟流式输出
            
            # 发送推荐
            if response.recommendations:
                yield f"data: {json.dumps({'type': 'recommendations', 'recommendations': [{'title': rec.title, 'reason': rec.recommendation_reason} for rec in response.recommendations[:3]]})}\n\n"
            
            # 发送完成标记
            yield f"data: {json.dumps({'type': 'complete', 'metadata': {'confidence': response.confidence, 'personalization_score': response.personalization_score}})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/plain")

@app.post("/api/v2/recommendations")
async def get_recommendations(
    request: RecommendationRequestModel,
    rag_system: EnhancedRAGSystem = Depends(get_rag_system)
) -> Dict[str, Any]:
    """获取个性化推荐"""
    try:
        recommendations = await rag_system.recommendation_engine.generate_recommendations(
            RecommendationRequest(
                user_id=request.user_id,
                recommendation_type=request.recommendation_type,
                limit=request.limit,
                diversity_factor=request.diversity_factor,
                category_filter=request.category_filter
            )
        )
        
        return {
            "user_id": request.user_id,
            "recommendation_type": request.recommendation_type,
            "total_recommendations": len(recommendations),
            "recommendations": [
                {
                    "document_id": rec.document_id,
                    "title": rec.title,
                    "authors": rec.authors,
                    "abstract": rec.abstract[:300] + "..." if len(rec.abstract) > 300 else rec.abstract,
                    "url": rec.url,
                    "published_date": rec.published_date.isoformat() if rec.published_date else None,
                    "categories": rec.categories,
                    "recommendation_score": rec.recommendation_score,
                    "recommendation_reason": rec.recommendation_reason,
                    "algorithm_used": rec.algorithm_used,
                    "scores": {
                        "content_score": rec.content_score,
                        "collaborative_score": rec.collaborative_score,
                        "popularity_score": rec.popularity_score,
                        "freshness_score": rec.freshness_score
                    }
                }
                for rec in recommendations
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.post("/api/v2/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    rag_system: EnhancedRAGSystem = Depends(get_rag_system)
) -> Dict[str, str]:
    """提交用户反馈"""
    try:
        # 这里需要从某处获取原始响应，简化实现
        from src.generation.enhanced_rag_system import EnhancedRAGResponse
        
        mock_response = EnhancedRAGResponse(
            answer="mock answer",
            source_chunks=[],
            confidence=0.8,
            generation_time=1.0,
            token_count=100,
            user_id=request.user_id,
            session_id="mock_session",
            recommendations=[]
        )
        
        feedback_id = await rag_system.submit_feedback(
            user_id=request.user_id,
            query=request.query,
            response=mock_response,
            feedback_type=request.feedback_type,
            feedback_value=request.feedback_value
        )
        
        return {
            "feedback_id": feedback_id,
            "status": "success",
            "message": "Feedback submitted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

@app.post("/api/v2/track/interaction")
async def track_user_interaction(
    request: UserInteractionModel,
    rag_system: EnhancedRAGSystem = Depends(get_rag_system)
) -> Dict[str, str]:
    """跟踪用户交互"""
    try:
        # 转换交互类型
        interaction_type_map = {
            "page_view": "page_view",
            "search": "search", 
            "click": "click",
            "dwell": "dwell",
            "scroll": "scroll",
            "feedback": "feedback"
        }
        
        # 根据交互类型调用相应的跟踪方法
        if request.interaction_type == "search":
            await rag_system.preference_tracker.track_search(
                user_id=request.user_id,
                query_text=request.query_text or "",
                session_id=request.session_id
            )
        elif request.interaction_type == "page_view" and request.document_id:
            await rag_system.preference_tracker.track_page_view(
                user_id=request.user_id,
                document_id=request.document_id,
                session_id=request.session_id
            )
        elif request.interaction_type == "dwell" and request.document_id and request.duration_seconds:
            await rag_system.preference_tracker.track_dwell_time(
                user_id=request.user_id,
                document_id=request.document_id,
                duration=request.duration_seconds,
                session_id=request.session_id
            )
        elif request.interaction_type == "scroll" and request.document_id and request.scroll_depth:
            await rag_system.preference_tracker.track_scroll_behavior(
                user_id=request.user_id,
                document_id=request.document_id,
                scroll_depth=request.scroll_depth,
                session_id=request.session_id
            )
        elif request.interaction_type == "feedback" and request.document_id and request.rating:
            await rag_system.preference_tracker.track_feedback(
                user_id=request.user_id,
                document_id=request.document_id,
                rating=request.rating,
                session_id=request.session_id
            )
        
        return {
            "status": "success",
            "message": "Interaction tracked successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tracking interaction: {str(e)}")

@app.post("/api/v2/track/document-access")
async def track_document_access(
    request: DocumentAccessModel,
    rag_system: EnhancedRAGSystem = Depends(get_rag_system)
) -> Dict[str, str]:
    """跟踪文档访问"""
    try:
        await rag_system.usage_analytics.log_document_access(
            document_id=request.document_id,
            user_id=request.user_id,
            access_type=request.access_type,
            duration=request.duration
        )
        
        return {
            "status": "success",
            "message": "Document access tracked successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tracking document access: {str(e)}")

@app.get("/api/v2/user/{user_id}/dashboard")
async def get_user_dashboard(
    user_id: str,
    rag_system: EnhancedRAGSystem = Depends(get_rag_system)
) -> Dict[str, Any]:
    """获取用户仪表板"""
    try:
        dashboard = await rag_system.get_user_dashboard(user_id)
        return dashboard
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting user dashboard: {str(e)}")

@app.get("/api/v2/user/{user_id}/profile")
async def get_user_profile(
    user_id: str,
    rag_system: EnhancedRAGSystem = Depends(get_rag_system)
) -> Dict[str, Any]:
    """获取用户画像"""
    try:
        profile = rag_system.user_profiler.get_user_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        return {
            "user_profile": profile.to_dict(),
            "interests_vector": rag_system.user_profiler.get_user_interests_vector(user_id),
            "similar_users": rag_system.user_profiler.get_similar_users(user_id, 5),
            "statistics": rag_system.user_profiler.get_user_statistics(user_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting user profile: {str(e)}")

@app.get("/api/v2/system/overview")
async def get_system_overview(
    rag_system: EnhancedRAGSystem = Depends(get_rag_system)
) -> Dict[str, Any]:
    """获取系统概览"""
    try:
        overview = await rag_system.get_system_overview()
        return overview
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system overview: {str(e)}")

@app.get("/api/v2/storage/metrics")
async def get_storage_metrics(
    rag_system: EnhancedRAGSystem = Depends(get_rag_system)
) -> Dict[str, Any]:
    """获取存储指标"""
    try:
        metrics = await rag_system.usage_analytics.calculate_storage_metrics()
        return metrics.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting storage metrics: {str(e)}")

@app.get("/api/v2/storage/optimization")
async def get_optimization_recommendations(
    rag_system: EnhancedRAGSystem = Depends(get_rag_system)
) -> Dict[str, Any]:
    """获取存储优化建议"""
    try:
        recommendations = await rag_system.usage_analytics.get_optimization_recommendations()
        return recommendations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting optimization recommendations: {str(e)}")

@app.post("/api/v2/storage/optimize")
async def trigger_storage_optimization(
    background_tasks: BackgroundTasks,
    rag_system: EnhancedRAGSystem = Depends(get_rag_system)
) -> Dict[str, str]:
    """触发存储优化"""
    try:
        # 在后台执行优化
        background_tasks.add_task(rag_system.storage_optimizer.optimize_storage)
        
        return {
            "status": "started",
            "message": "Storage optimization started in background"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting storage optimization: {str(e)}")

@app.get("/api/v2/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "system_initialized": enhanced_rag_system is not None and enhanced_rag_system.is_initialized
    }

# 用于直接运行
if __name__ == "__main__":
    uvicorn.run(
        "enhanced_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=True
    )