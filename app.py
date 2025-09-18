# app.py - Streamlit前端应用
# app.py - Streamlit前端应用
import streamlit as st
import asyncio
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 仅加载配置，不再需要在这里登录
try:
    from configs.config import config
except ImportError as e:
    st.error(f"导入配置失败: {e}")
    st.stop()

# 导入自定义模块
try:
    from src.generation.rag_generator import RAGSystem
    from src.retrieval.vector_database import VectorDatabaseManager
    from src.retrieval.reranker import AdvancedReranker
    from src.frontend.discovery_interface import DiscoveryInterface
except ImportError as e:
    st.error(f"导入模块失败: {e}")
    st.stop()

import streamlit as st
import asyncio
import time
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os


# 页面配置
st.set_page_config(
    page_title="🤖 AI技术资讯智能问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stAlert > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
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
    
    .source-card {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #ff9800;
        margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """初始化RAG系统（带缓存）"""
    with st.spinner("🔄 正在初始化AI智能问答系统..."):
        try:
            # 配置
            rag_config = {
                'embedding_model': config.EMBEDDING_MODEL,
                'llm_model': config.LLM_MODEL,
                'device': config.DEVICE,
                'max_context_length': 3000,
                'qdrant_host': config.QDRANT_HOST,
                'qdrant_port': config.QDRANT_PORT,
                'collection_name': config.COLLECTION_NAME,
                'vector_size': 1024,
                'HUGGING_FACE_TOKEN': config.HUGGING_FACE_TOKEN # 修改: 将token添加到配置字典
            }
            
            db_manager = VectorDatabaseManager(rag_config)
            
            reranker = None
            reranker_available = False
            try:
                reranker = AdvancedReranker()
                reranker_available = True
            except Exception as e:
                st.warning(f"重排序器初始化失败: {e}")
            
            # 检查知识库状态
            stats = db_manager.db.get_collection_stats()
            if stats.get('total_points', 0) == 0:
                st.error("⚠️ 知识库为空！请先运行数据收集和处理脚本。")
                return None, None, None
            
            # 注入依赖项来初始化主系统
            rag_system = RAGSystem(config=rag_config, db_manager=db_manager, reranker=reranker)
            st.success(f"✅ 系统初始化成功! 知识库包含 {stats.get('total_points', 0)} 个文档片段。")
            return rag_system, db_manager, reranker_available
            
        except Exception as e:
            # 捕获并显示更详细的错误信息
            st.error(f"❌ 系统初始化失败: {type(e).__name__}: {e}")
            return None, None, None


def render_chat_message(role: str, content: str, sources: list = None, metrics: dict = None):
    """渲染聊天消息"""
    
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
            <strong>🤖 AI回答:</strong><br>
            {content}
        </div>
        ''', unsafe_allow_html=True)
        
        # 显示指标
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'''
                <div class="metric-card">
                    <strong>置信度</strong><br>
                    <span style="color: #1976d2; font-size: 1.5em;">{metrics.get('confidence', 0):.2f}</span>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'''
                <div class="metric-card">
                    <strong>生成时间</strong><br>
                    <span style="color: #388e3c; font-size: 1.2em;">{metrics.get('generation_time', 0):.2f}s</span>
                </div>
                ''', unsafe_allow_html=True)
                
            with col3:
                st.markdown(f'''
                <div class="metric-card">
                    <strong>Token数量</strong><br>
                    <span style="color: #f57c00; font-size: 1.2em;">{metrics.get('token_count', 0)}</span>
                </div>
                ''', unsafe_allow_html=True)
                
            with col4:
                st.markdown(f'''
                <div class="metric-card">
                    <strong>参考源数量</strong><br>
                    <span style="color: #7b1fa2; font-size: 1.2em;">{len(sources) if sources else 0}</span>
                </div>
                ''', unsafe_allow_html=True)
        
        # 显示参考来源
        if sources:
            with st.expander(f"📚 参考来源 ({len(sources)}个)", expanded=False):
                for i, source in enumerate(sources, 1):
                    source_title = source.get('metadata', {}).get('title', f"来源 {i}")
                    source_type = source.get('semantic_type', 'content')
                    scores = source.get('scores', {})
                    source_score = scores.get('rerank_score', scores.get('hybrid_score', 0.0))
                    
                    st.markdown(f'''
                    <div class="source-card">
                        <strong>📄 {source_title[:80]}...</strong>
                        <span style="float: right; color: #666;">
                            [{source_type}] Score: {source_score:.3f}
                        </span><br>
                        <small style="color: #666;">
                            {source['content'][:200]}...
                        </small>
                    </div>
                    ''', unsafe_allow_html=True)

def render_knowledge_base_stats(db_manager):
    """渲染知识库统计信息"""
    
    try:
        stats = db_manager.db.get_collection_stats()
        
        st.markdown("### 📊 知识库统计")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("总文档片段", stats.get('total_points', 0))
        
        with col2:
            st.metric("向量维度", stats.get('vector_size', 0))
            
        with col3:
            st.metric("距离度量", stats.get('distance_metric', 'N/A'))
        
        if st.checkbox("显示详细统计", key="show_detailed_stats"):
            sample_data = {
                '数据源': ['ArXiv论文', 'AI博客', 'Hugging Face', '会议论文'],
                '文档数量': [450, 230, 180, 140],
            }
            df = pd.DataFrame(sample_data)
            fig = px.pie(df, values='文档数量', names='数据源', title="数据源分布")
            st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"获取统计信息失败: {e}")

def main():
    """主函数"""
    
    st.title("🤖 AI技术资讯智能问答系统")
    st.markdown("""
    **基于RAG技术的智能AI资讯助手** 实时获取和分析最新的AI论文、技术博客和开源项目，为您提供准确、及时的技术解答。
    """)
    
    with st.sidebar:
        st.header("⚙️ 系统配置")
        st.subheader("🔍 查询参数")
        
        initial_retrieve = st.slider("初始检索数量", 5, 30, 15, help="增加数量可提高召回率，但会增加处理时间")
        top_k = st.slider("最终结果数量", 3, 10, 5, help="经过MMR多样性选择后，返回给用户的最终文档数量")
        context_chunks = st.slider("上下文片段数量", 2, 5, 3, help="提供给模型的上下文片段数量")
        
        with st.expander("🔧 高级选项"):
            enable_reranker = st.checkbox("启用重排序", value=True, help="使用Advanced Reranker提升检索精度和多样性")
            lambda_mult = st.slider("MMR 多样性 (Lambda)", 0.0, 1.0, 0.5, 0.1, help="控制相关性与多样性的平衡。1.0完全相关，0.0完全多样。")
            show_debug_info = st.checkbox("显示调试信息", value=False)
            temperature = st.slider("生成温度", 0.0, 1.0, 0.1, 0.1, help="控制生成的随机性")
    
    if 'rag_system' not in st.session_state:
        rag_system, db_manager, reranker_available = initialize_rag_system()
        if rag_system is None:
            st.stop()
        st.session_state.rag_system = rag_system
        st.session_state.db_manager = db_manager
        st.session_state.reranker_available = reranker_available
        st.session_state.chat_history = []
        
        # 初始化内容发现界面
        discovery_interface = DiscoveryInterface()
        discovery_interface.initialize_components(
            vector_db_manager=db_manager,
            rag_system=rag_system,
            llm_generator=rag_system.llm_generator if hasattr(rag_system, 'llm_generator') else None
        )
        st.session_state.discovery_interface = discovery_interface
    
    tab1, tab2, tab3, tab4 = st.tabs(["💬 智能问答", "🔍 内容发现", "📊 知识库概览", "ℹ️ 系统信息"])
    
    with tab1:
        st.markdown("### 💬 与AI助手对话")
        
        for message in st.session_state.chat_history:
            render_chat_message(role=message['role'], content=message['content'], sources=message.get('sources'), metrics=message.get('metrics'))
        
        user_query = st.chat_input("请输入您的问题（支持中英文）:")
        
        if user_query:
            st.session_state.chat_history.append({'role': 'user', 'content': user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("🤔 AI正在思考中..."):
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        query_params = {
                            'initial_retrieve': initial_retrieve,
                            'top_k': top_k,
                            'context_chunks': context_chunks,
                            'lambda_mult': lambda_mult,
                            'use_reranker': enable_reranker and st.session_state.reranker_available
                        }
                        
                        result = loop.run_until_complete(
                            st.session_state.rag_system.generate_answer(user_query, **query_params)
                        )
                        
                        metrics = {'confidence': result.confidence, 'generation_time': result.generation_time, 'token_count': result.token_count}
                        render_chat_message("assistant", result.answer, sources=result.source_chunks, metrics=metrics)
                        
                        st.session_state.chat_history.append({'role': 'assistant', 'content': result.answer, 'sources': result.source_chunks, 'metrics': metrics})

                        if show_debug_info:
                            with st.expander("🔍 调试信息"):
                                st.json({'params': query_params})
                        
                    except Exception as e:
                        st.error(f"❌ 生成回答时出错: {e}")
                    finally:
                        loop.close()
    
    with tab2:
        st.session_state.discovery_interface.render_discovery_tab()
    
    with tab3:
        render_knowledge_base_stats(st.session_state.db_manager)
        st.markdown("### 💡 示例查询")
        example_queries = ["什么是Transformer架构的核心创新？", "如何提高大语言模型的效率？", "RAG技术有什么优势和局限性？"]
        for query in example_queries:
            if st.button(f"📝 {query}"):
                st.session_state.user_input = query
                st.rerun()
    
    with tab4:
        st.markdown("### ℹ️ 系统架构信息")
        system_info = {
            "🔍 检索模型": "BAAI/bge-m3", "🧠 生成模型": "Qwen2-7B-Instruct", "🗄️ 向量数据库": "Qdrant",
            "🔄 重排序器": "BAAI/bge-reranker-base + MMR", "🌐 前端框架": "Streamlit",
            "📊 内容分析": "TF-IDF + LDA主题建模", "🎯 智能发现": "主题感知RAG + 交互式大纲"
        }
        for component, description in system_info.items():
            st.markdown(f"**{component}**: {description}")

if __name__ == "__main__":
    main()

