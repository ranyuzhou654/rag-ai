# enhanced_app.py - 增强的Streamlit前端应用
import streamlit as st
import asyncio
import sys
import time
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 导入配置和模块
try:
    from configs.config import config
    from src.generation.enhanced_rag_system import EnhancedRAGSystem, PersonalizedQuery
    from src.personalization.recommendation_engine import RecommendationRequest
except ImportError as e:
    st.error(f"导入模块失败: {e}")
    st.stop()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 页面配置
st.set_page_config(
    page_title="🚀 个性化AI智能问答系统",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .recommendation-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .personalization-score {
        background: linear-gradient(90deg, #007bff, #28a745);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .interest-tag {
        background-color: #e7f3ff;
        color: #0066cc;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8em;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #388e3c;
    }
    
    .metric-card {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .dashboard-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_enhanced_rag_system():
    """初始化增强RAG系统（带缓存）"""
    with st.spinner("🔄 正在初始化个性化AI智能问答系统..."):
        try:
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
            
            enhanced_rag = EnhancedRAGSystem(rag_config, storage_root)
            
            # 运行异步初始化
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(enhanced_rag.initialize())
            loop.close()
            
            st.success("✅ 个性化系统初始化成功!")
            return enhanced_rag
            
        except Exception as e:
            st.error(f"❌ 系统初始化失败: {type(e).__name__}: {e}")
            return None

def render_personalized_chat_message(role: str, content: str, 
                                   personalization_info: Optional[Dict] = None,
                                   recommendations: Optional[List] = None,
                                   sources: Optional[List] = None):
    """渲染个性化聊天消息"""
    
    if role == "user":
        st.markdown(f'''
        <div class="chat-message user-message">
            <strong>🤔 您的问题:</strong><br>
            {content}
        </div>
        ''', unsafe_allow_html=True)
        
    else:  # assistant
        st.markdown(f'''
        <div class="chat-message assistant-message">
            <strong>🚀 AI回答:</strong><br>
            {content}
        </div>
        ''', unsafe_allow_html=True)
        
        # 显示个性化信息
        if personalization_info:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'''
                <div class="metric-card">
                    <strong>个性化得分</strong><br>
                    <span class="personalization-score">{personalization_info.get('personalization_score', 0):.2f}</span>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'''
                <div class="metric-card">
                    <strong>置信度</strong><br>
                    <span style="color: #1976d2; font-size: 1.5em;">{personalization_info.get('confidence', 0):.2f}</span>
                </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'''
                <div class="metric-card">
                    <strong>响应时间</strong><br>
                    <span style="color: #388e3c; font-size: 1.2em;">{personalization_info.get('generation_time', 0):.2f}s</span>
                </div>
                ''', unsafe_allow_html=True)
                
            with col4:
                st.markdown(f'''
                <div class="metric-card">
                    <strong>匹配兴趣</strong><br>
                    <span style="color: #f57c00; font-size: 1.2em;">{len(personalization_info.get('user_interests_matched', []))}</span>
                </div>
                ''', unsafe_allow_html=True)
        
        # 显示匹配的用户兴趣
        if personalization_info and personalization_info.get('user_interests_matched'):
            st.markdown("**🎯 匹配的研究兴趣:**")
            interests_html = ""
            for interest in personalization_info['user_interests_matched']:
                interests_html += f'<span class="interest-tag">{interest.replace("_", " ").title()}</span>'
            st.markdown(interests_html, unsafe_allow_html=True)
        
        # 显示推荐
        if recommendations:
            with st.expander(f"📚 为您推荐 ({len(recommendations)}篇相关论文)", expanded=False):
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f'''
                    <div class="recommendation-card">
                        <strong>📄 {rec.get('title', f'推荐文档 {i}')}</strong>
                        <span style="float: right; color: #666;">
                            得分: {rec.get('recommendation_score', 0):.3f}
                        </span><br>
                        <small style="color: #666;">
                            {rec.get('recommendation_reason', '基于您的兴趣推荐')}
                        </small><br>
                        <small style="color: #888;">
                            作者: {', '.join(rec.get('authors', [])[:3])}
                        </small>
                    </div>
                    ''', unsafe_allow_html=True)
        
        # 显示参考来源
        if sources:
            with st.expander(f"📑 参考来源 ({len(sources)}个)", expanded=False):
                for i, source in enumerate(sources, 1):
                    source_title = source.get('metadata', {}).get('title', f"来源 {i}")
                    scores = source.get('scores', {})
                    source_score = scores.get('hybrid_score', 0.0)
                    
                    st.markdown(f'''
                    <div style="border-left: 3px solid #ff9800; padding-left: 1rem; margin-bottom: 0.5rem;">
                        <strong>📄 {source_title[:80]}...</strong>
                        <span style="float: right; color: #666;">
                            Score: {source_score:.3f}
                        </span><br>
                        <small style="color: #666;">
                            {source.get('content', '')[:200]}...
                        </small>
                    </div>
                    ''', unsafe_allow_html=True)

def render_user_dashboard(enhanced_rag, user_id: str):
    """渲染用户仪表板"""
    st.header("👤 个人仪表板")
    
    try:
        # 异步获取仪表板数据
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        dashboard_data = loop.run_until_complete(enhanced_rag.get_user_dashboard(user_id))
        loop.close()
        
        if not dashboard_data.get('user_profile'):
            st.info("🆕 欢迎新用户！开始使用系统后，这里将显示您的个性化信息。")
            return
        
        # 用户基本信息
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div class="dashboard-card">
                <h4>🎯 个性化得分</h4>
                <h2 style="color: #007bff;">{dashboard_data.get("personalization_score", 0):.2f}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            usage = dashboard_data.get('system_usage', {})
            st.markdown(f'''
            <div class="dashboard-card">
                <h4>📊 查询次数</h4>
                <h2 style="color: #28a745;">{usage.get("total_queries", 0)}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="dashboard-card">
                <h4>📖 浏览文档</h4>
                <h2 style="color: #ffc107;">{usage.get("documents_viewed", 0)}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'''
            <div class="dashboard-card">
                <h4>👍 积极反馈</h4>
                <h2 style="color: #dc3545;">{usage.get("positive_feedback", 0)}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        # 研究兴趣
        interests = dashboard_data.get('interests', {})
        if interests:
            st.subheader("🧠 您的研究兴趣")
            
            # 创建兴趣雷达图
            categories = list(interests.keys())
            values = list(interests.values())
            
            if len(categories) >= 3:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='研究兴趣强度',
                    line_color='rgb(0, 123, 255)'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # 如果类别太少，使用条形图
                if categories:
                    fig = px.bar(
                        x=categories, y=values,
                        title="研究兴趣强度",
                        labels={'x': '研究领域', 'y': '兴趣强度'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # 最新推荐
        recommendations = dashboard_data.get('recommendations', [])
        if recommendations:
            st.subheader("📚 为您推荐的最新论文")
            
            for rec in recommendations[:5]:
                st.markdown(f'''
                <div class="recommendation-card">
                    <strong>📄 {rec.title}</strong>
                    <span style="float: right; color: #666;">
                        {rec.recommendation_score:.3f}
                    </span><br>
                    <small style="color: #666; display: block; margin: 0.5rem 0;">
                        {rec.recommendation_reason}
                    </small>
                    <small style="color: #888;">
                        作者: {', '.join(rec.authors[:3])} | 发布: {rec.published_date or 'Unknown'}
                    </small>
                </div>
                ''', unsafe_allow_html=True)
                
                if st.button(f"查看详情", key=f"rec_{rec.document_id}"):
                    st.info(f"📖 **摘要**: {rec.abstract}")
                    if rec.url:
                        st.markdown(f"🔗 [原文链接]({rec.url})")
        
        # 活动模式分析
        activity = dashboard_data.get('recent_activity', {})
        if activity:
            st.subheader("📈 活动模式分析")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("活跃级别", activity.get('engagement_level', 'Unknown').title())
                st.metric("平均会话时长", f"{activity.get('avg_session_duration', 0):.1f} 分钟")
            
            with col2:
                st.metric("总会话数", activity.get('total_sessions', 0))
                st.metric("总使用时长", f"{activity.get('total_duration', 0) / 60:.1f} 分钟")
    
    except Exception as e:
        st.error(f"获取仪表板数据失败: {e}")

def render_system_analytics(enhanced_rag):
    """渲染系统分析界面"""
    st.header("📊 系统分析")
    
    try:
        # 获取系统概览
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        overview = loop.run_until_complete(enhanced_rag.get_system_overview())
        loop.close()
        
        # 系统统计
        system_stats = overview.get('system_stats', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总查询数", system_stats.get('total_queries', 0))
        
        with col2:
            st.metric("个性化查询", system_stats.get('personalized_queries', 0))
        
        with col3:
            st.metric("生成推荐数", system_stats.get('recommendations_generated', 0))
        
        with col4:
            cache_rate = (system_stats.get('cache_hits', 0) / max(system_stats.get('total_queries', 1), 1)) * 100
            st.metric("缓存命中率", f"{cache_rate:.1f}%")
        
        # 存储指标
        storage_metrics = overview.get('storage_metrics', {})
        if storage_metrics:
            st.subheader("💾 存储使用情况")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 存储分布饼图
                if 'documents_by_tier' in storage_metrics:
                    tier_data = storage_metrics['documents_by_tier']
                    if tier_data:
                        fig = px.pie(
                            values=list(tier_data.values()),
                            names=list(tier_data.keys()),
                            title="文档分层分布"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 存储成本
                st.metric("预估月度成本", f"${storage_metrics.get('estimated_storage_cost', 0):.2f}")
                st.metric("总存储大小", f"{storage_metrics.get('total_storage_gb', 0):.2f} GB")
                st.metric("文档总数", storage_metrics.get('total_documents', 0))
        
        # 优化状态
        optimization = overview.get('optimization_status', {})
        if optimization:
            st.subheader("⚡ 系统优化状态")
            
            storage_opt = optimization.get('storage_optimizer', {})
            if storage_opt:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("队列中的迁移", storage_opt.get('queue_size', 0))
                
                with col2:
                    st.metric("活跃迁移", storage_opt.get('active_migrations', 0))
                
                with col3:
                    opt_stats = storage_opt.get('optimization_stats', {})
                    st.metric("成功迁移", opt_stats.get('successful_migrations', 0))
    
    except Exception as e:
        st.error(f"获取系统分析数据失败: {e}")

def main():
    """主函数"""
    
    st.title("🚀 个性化AI智能问答系统")
    st.markdown("""
    **基于个性化推荐的智能AI助手** - 根据您的研究兴趣和使用习惯，提供量身定制的答案和推荐。
    """)
    
    # 初始化系统
    if 'enhanced_rag_system' not in st.session_state:
        enhanced_rag_system = initialize_enhanced_rag_system()
        if enhanced_rag_system is None:
            st.stop()
        st.session_state.enhanced_rag_system = enhanced_rag_system
        st.session_state.chat_history = []
    
    # 用户ID输入
    with st.sidebar:
        st.header("👤 用户设置")
        
        # 用户ID输入
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        
        user_id_input = st.text_input(
            "用户ID", 
            value=st.session_state.user_id or "",
            help="输入您的用户ID以获得个性化体验"
        )
        
        if user_id_input and user_id_input != st.session_state.user_id:
            st.session_state.user_id = user_id_input
            st.session_state.session_id = str(uuid.uuid4())
            st.success(f"✅ 已设置用户ID: {user_id_input}")
        
        # 个性化设置
        st.subheader("⚙️ 个性化设置")
        enable_recommendations = st.checkbox("启用智能推荐", value=True)
        recommendation_limit = st.slider("推荐数量", 1, 10, 5)
        personalization_weight = st.slider("个性化权重", 0.0, 1.0, 0.3, 0.1)
        
        # 查询设置
        st.subheader("🔍 查询设置")
        max_retrieval_time = st.slider("最大检索时间(秒)", 1.0, 30.0, 10.0)
        diversity_factor = st.slider("结果多样性", 0.0, 1.0, 0.2, 0.1)
    
    # 主界面标签页
    tab1, tab2, tab3, tab4 = st.tabs(["💬 智能问答", "👤 个人仪表板", "📊 系统分析", "ℹ️ 帮助"])
    
    with tab1:
        st.markdown("### 💬 与个性化AI助手对话")
        
        # 显示聊天历史
        for message in st.session_state.chat_history:
            render_personalized_chat_message(
                role=message['role'],
                content=message['content'],
                personalization_info=message.get('personalization_info'),
                recommendations=message.get('recommendations'),
                sources=message.get('sources')
            )
        
        # 用户输入
        user_query = st.chat_input("请输入您的问题（支持中英文）:")
        
        if user_query:
            # 添加用户消息到历史
            st.session_state.chat_history.append({'role': 'user', 'content': user_query})
            
            with st.chat_message("user"):
                st.markdown(user_query)
            
            with st.chat_message("assistant"):
                with st.spinner("🚀 AI正在为您个性化思考中..."):
                    try:
                        # 创建个性化查询
                        personalized_query = PersonalizedQuery(
                            original_query=user_query,
                            user_id=st.session_state.user_id,
                            session_id=getattr(st.session_state, 'session_id', str(uuid.uuid4())),
                            enable_recommendations=enable_recommendations,
                            recommendation_limit=recommendation_limit,
                            personalization_weight=personalization_weight,
                            max_retrieval_time=max_retrieval_time
                        )
                        
                        # 生成增强回答
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        response = loop.run_until_complete(
                            st.session_state.enhanced_rag_system.generate_enhanced_answer(personalized_query)
                        )
                        loop.close()
                        
                        # 准备个性化信息
                        personalization_info = {
                            'personalization_score': response.personalization_score,
                            'confidence': response.confidence,
                            'generation_time': response.generation_time,
                            'user_interests_matched': response.user_interests_matched
                        }
                        
                        # 准备推荐信息
                        recommendations = []
                        if response.recommendations:
                            recommendations = [
                                {
                                    'title': rec.title,
                                    'authors': rec.authors,
                                    'abstract': rec.abstract,
                                    'url': rec.url,
                                    'recommendation_score': rec.recommendation_score,
                                    'recommendation_reason': rec.recommendation_reason,
                                    'published_date': rec.published_date.isoformat() if rec.published_date else None
                                }
                                for rec in response.recommendations
                            ]
                        
                        # 渲染回答
                        render_personalized_chat_message(
                            "assistant", 
                            response.answer,
                            personalization_info=personalization_info,
                            recommendations=recommendations,
                            sources=response.source_chunks
                        )
                        
                        # 添加到聊天历史
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response.answer,
                            'personalization_info': personalization_info,
                            'recommendations': recommendations,
                            'sources': response.source_chunks
                        })
                        
                        # 反馈按钮
                        col1, col2, col3 = st.columns([1, 1, 4])
                        with col1:
                            if st.button("👍 有帮助", key=f"pos_{len(st.session_state.chat_history)}"):
                                st.success("感谢您的反馈！")
                        with col2:
                            if st.button("👎 没帮助", key=f"neg_{len(st.session_state.chat_history)}"):
                                st.info("我们会继续改进！")
                        
                    except Exception as e:
                        st.error(f"❌ 生成回答时出错: {e}")
    
    with tab2:
        if st.session_state.user_id:
            render_user_dashboard(st.session_state.enhanced_rag_system, st.session_state.user_id)
        else:
            st.info("👈 请在侧边栏输入用户ID以查看个人仪表板")
    
    with tab3:
        render_system_analytics(st.session_state.enhanced_rag_system)
    
    with tab4:
        st.markdown("""
        ### 📖 使用指南
        
        #### 🚀 个性化功能
        - **用户画像**: 系统会根据您的查询和行为建立个人画像
        - **智能推荐**: 基于您的兴趣推荐相关论文和内容
        - **个性化回答**: 答案会根据您的研究背景进行定制
        
        #### 💡 使用技巧
        1. 设置用户ID以启用个性化功能
        2. 多次使用系统以建立更准确的用户画像
        3. 通过反馈帮助系统学习您的偏好
        4. 查看个人仪表板了解您的研究兴趣分析
        
        #### 🔧 系统架构
        - **检索模型**: BAAI/bge-m3
        - **生成模型**: Qwen2-7B-Instruct  
        - **向量数据库**: Qdrant
        - **推荐引擎**: 混合协同过滤
        - **存储优化**: 多层存储自动管理
        
        #### 📊 功能特色
        - ✅ 个性化问答
        - ✅ 智能推荐系统
        - ✅ 用户行为分析
        - ✅ 存储自动优化
        - ✅ 实时反馈学习
        """)

if __name__ == "__main__":
    main()