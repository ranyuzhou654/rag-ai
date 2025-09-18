# src/frontend/discovery_interface.py
"""
内容发现界面组件 - 用于Streamlit前端的主题发现和交互
"""

import streamlit as st
import asyncio
import json
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime
from loguru import logger

from ..analysis.content_analyzer import ContentAnalyzer, ContentAnalysis
from ..generation.content_summarizer import ContentSummarizer
from ..generation.topic_outline_generator import TopicOutlineGenerator, TopicOutline
from ..generation.topic_aware_rag import TopicAwareRAGEngine, TopicContext

class DiscoveryInterface:
    """内容发现界面管理器"""
    
    def __init__(self):
        self.content_analyzer = None
        self.content_summarizer = None
        self.outline_generator = None
        self.topic_aware_rag = None
        
        # 界面状态
        if 'discovery_state' not in st.session_state:
            st.session_state.discovery_state = {
                'current_outline': None,
                'selected_topic': None,
                'analysis_in_progress': False,
                'last_analysis_time': None,
                'topic_chat_history': []
            }
    
    def initialize_components(self, vector_db_manager, rag_system, llm_generator=None):
        """初始化内容发现组件"""
        
        try:
            self.content_analyzer = ContentAnalyzer(vector_db_manager, llm_generator)
            self.content_summarizer = ContentSummarizer(llm_generator)
            self.outline_generator = TopicOutlineGenerator(self.content_summarizer)
            
            # 如果有现有大纲，初始化主题感知RAG
            if st.session_state.discovery_state['current_outline']:
                self.topic_aware_rag = TopicAwareRAGEngine(
                    rag_system, 
                    st.session_state.discovery_state['current_outline']
                )
            
            logger.info("Discovery interface components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize discovery components: {e}")
            st.error(f"初始化内容发现组件失败: {e}")
            return False
    
    def render_discovery_tab(self):
        """渲染内容发现标签页"""
        
        st.markdown("### 🔍 智能内容发现")
        st.markdown("""
        **发现知识库中的热点话题** - 系统将自动分析现有文档，识别技术趋势和研究热点，为您提供结构化的主题大纲。
        """)
        
        # 控制面板
        self._render_control_panel()
        
        # 主要内容区域
        if st.session_state.discovery_state['current_outline']:
            self._render_outline_content()
        else:
            self._render_welcome_screen()
    
    def _render_control_panel(self):
        """渲染控制面板"""
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            analysis_days = st.slider(
                "分析时间范围（天）", 
                min_value=1, 
                max_value=30, 
                value=7, 
                help="分析最近N天的文档内容"
            )
        
        with col2:
            max_topics = st.selectbox(
                "最大主题数", 
                options=[10, 15, 20, 25], 
                index=1,
                help="生成的主题数量上限"
            )
        
        with col3:
            outline_template = st.selectbox(
                "大纲模板", 
                options=["comprehensive", "focused", "detailed"],
                index=0,
                format_func=lambda x: {
                    "comprehensive": "全面分析",
                    "focused": "聚焦核心", 
                    "detailed": "详细展开"
                }[x]
            )
        
        with col4:
            st.write("")  # 占位符
            st.write("")  # 占位符
            if st.button(
                "🚀 开始分析", 
                type="primary",
                disabled=st.session_state.discovery_state['analysis_in_progress'],
                use_container_width=True
            ):
                self._start_content_analysis(analysis_days, max_topics, outline_template)
        
        # 分析状态显示
        if st.session_state.discovery_state['analysis_in_progress']:
            st.info("🔄 正在分析内容，请稍候...")
            
        elif st.session_state.discovery_state['last_analysis_time']:
            last_time = st.session_state.discovery_state['last_analysis_time']
            st.success(f"✅ 最后分析时间: {last_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _render_welcome_screen(self):
        """渲染欢迎界面"""
        
        st.markdown("""
        ---
        
        ## 👋 欢迎使用智能内容发现
        
        点击上方的"开始分析"按钮，系统将：
        
        1. **📊 分析现有知识库** - 扫描最近的文档和论文
        2. **🏷️ 识别热点话题** - 自动提取技术趋势和研究重点
        3. **📋 生成主题大纲** - 创建结构化的主题导航
        4. **💡 智能问答** - 基于选定主题进行深度对话
        
        ### 💫 功能特色
        
        - **自动主题发现**: 无需手动搜索，系统智能识别热点
        - **分层组织**: 按重要性和类别组织主题
        - **深度问答**: 基于选定主题的精准RAG对话
        - **实时更新**: 支持定期更新分析结果
        """)
        
        # 显示一些统计信息
        self._render_knowledge_base_preview()
    
    def _render_knowledge_base_preview(self):
        """渲染知识库预览"""
        
        st.markdown("### 📚 知识库概览")
        
        try:
            if hasattr(st.session_state, 'db_manager') and st.session_state.db_manager:
                stats = st.session_state.db_manager.db.get_collection_stats()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("文档总数", f"{stats.get('total_points', 0):,}")
                
                with col2:
                    st.metric("向量维度", stats.get('vector_size', 'N/A'))
                
                with col3:
                    st.metric("距离算法", stats.get('distance_metric', 'N/A'))
                
                # 模拟数据源分布
                if stats.get('total_points', 0) > 0:
                    source_data = {
                        '数据源': ['ArXiv论文', 'AI博客', 'HuggingFace', '会议论文', '技术文档'],
                        '文档数量': [450, 280, 180, 140, 120],
                        '最近更新': ['2天前', '1天前', '3天前', '5天前', '1周前']
                    }
                    
                    df = pd.DataFrame(source_data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(
                            df, 
                            values='文档数量', 
                            names='数据源',
                            title="数据源分布"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**数据源详情**")
                        st.dataframe(df, use_container_width=True)
        
        except Exception as e:
            st.warning(f"无法获取知识库统计信息: {e}")
    
    def _start_content_analysis(self, days: int, max_topics: int, template: str):
        """开始内容分析"""
        
        if not self.content_analyzer:
            st.error("❌ 内容分析器未初始化")
            return
        
        st.session_state.discovery_state['analysis_in_progress'] = True
        
        # 创建进度占位符
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        try:
            with progress_placeholder.container():
                progress_bar = st.progress(0)
                
                # 步骤1: 内容分析
                status_placeholder.info("🔍 正在分析文档内容...")
                progress_bar.progress(20)
                
                # 运行异步分析
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                content_analysis = loop.run_until_complete(
                    self.content_analyzer.analyze_content(
                        days=days,
                        max_topics=max_topics,
                        min_doc_count=2
                    )
                )
                
                progress_bar.progress(50)
                
                # 步骤2: 生成大纲
                status_placeholder.info("📋 正在生成主题大纲...")
                
                outline = loop.run_until_complete(
                    self.outline_generator.generate_outline(
                        content_analysis,
                        template=template,
                        include_summaries=True
                    )
                )
                
                progress_bar.progress(80)
                
                # 步骤3: 初始化主题感知RAG
                status_placeholder.info("🤖 正在初始化智能问答...")
                
                if hasattr(st.session_state, 'rag_system'):
                    self.topic_aware_rag = TopicAwareRAGEngine(
                        st.session_state.rag_system,
                        outline
                    )
                
                progress_bar.progress(100)
                
                # 保存结果
                st.session_state.discovery_state['current_outline'] = outline
                st.session_state.discovery_state['last_analysis_time'] = datetime.now()
                
                loop.close()
                
                # 清理进度显示
                progress_placeholder.empty()
                status_placeholder.success("✅ 分析完成！")
                
                time.sleep(1)
                status_placeholder.empty()
                
                # 刷新页面以显示结果
                st.rerun()
                
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            progress_placeholder.empty()
            status_placeholder.error(f"❌ 分析失败: {str(e)}")
            
        finally:
            st.session_state.discovery_state['analysis_in_progress'] = False
    
    def _render_outline_content(self):
        """渲染大纲内容"""
        
        outline = st.session_state.discovery_state['current_outline']
        
        st.markdown("---")
        
        # 大纲标题和概览
        st.markdown(f"## {outline.title}")
        
        # 概览信息
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**概览**: {outline.overview}")
        
        with col2:
            self._render_outline_statistics(outline)
        
        # 主要内容
        tab1, tab2, tab3 = st.tabs(["📋 主题大纲", "💬 智能问答", "📊 分析统计"])
        
        with tab1:
            self._render_topic_outline(outline)
        
        with tab2:
            self._render_topic_chat(outline)
        
        with tab3:
            self._render_analysis_statistics(outline)
    
    def _render_outline_statistics(self, outline: TopicOutline):
        """渲染大纲统计信息"""
        
        stats = outline.statistics
        
        st.markdown("**统计信息**")
        
        metrics_data = [
            ("总主题数", stats.get('total_topics', 0)),
            ("分析文档", stats.get('total_documents_analyzed', 0)),
            ("章节数", stats.get('sections_count', 0))
        ]
        
        for label, value in metrics_data:
            st.metric(label, value)
    
    def _render_topic_outline(self, outline: TopicOutline):
        """渲染主题大纲"""
        
        st.markdown("### 📚 主题导航")
        
        for i, section in enumerate(outline.sections):
            with st.expander(f"{section.title} ({len(section.topics)}个主题)", expanded=(i == 0)):
                st.markdown(f"**{section.subtitle}**")
                st.markdown(section.description)
                
                # 主题列表
                for j, topic in enumerate(section.topics):
                    self._render_topic_card(topic, section_idx=i, topic_idx=j)
    
    def _render_topic_card(self, topic, section_idx: int, topic_idx: int):
        """渲染主题卡片"""
        
        # 创建唯一的按钮key
        button_key = f"topic_btn_{section_idx}_{topic_idx}"
        
        # 主题卡片容器
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**🎯 {topic.title}**")
                st.markdown(f"*{topic.summary[:150]}...*" if len(topic.summary) > 150 else topic.summary)
                
                # 关键词标签
                if topic.keywords:
                    keywords_display = " ".join([f"`{kw}`" for kw in topic.keywords[:5]])
                    st.markdown(f"关键词: {keywords_display}")
                
                # 统计信息
                st.caption(f"📄 {topic.doc_count}篇文档 | ⭐ 重要性: {topic.importance_score:.2f}")
            
            with col2:
                # 选择主题按钮
                if st.button(
                    "💬 基于此主题提问",
                    key=button_key,
                    help="选择此主题进行深度问答",
                    use_container_width=True
                ):
                    st.session_state.discovery_state['selected_topic'] = topic
                    st.success(f"✅ 已选择主题: {topic.title}")
                    st.rerun()
        
        st.markdown("---")
    
    def _render_topic_chat(self, outline: TopicOutline):
        """渲染主题聊天界面"""
        
        selected_topic = st.session_state.discovery_state.get('selected_topic')
        
        if not selected_topic:
            st.info("💡 请先从主题大纲中选择一个感兴趣的主题开始对话")
            return
        
        # 显示当前选择的主题
        st.markdown(f"### 🎯 当前主题: {selected_topic.title}")
        
        with st.expander("主题详情", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**摘要**: {selected_topic.summary}")
                st.markdown(f"**文档数量**: {selected_topic.doc_count}")
                st.markdown(f"**重要性评分**: {selected_topic.importance_score:.3f}")
            
            with col2:
                st.markdown("**关键词**:")
                for keyword in selected_topic.keywords[:8]:
                    st.write(f"- {keyword}")
        
        # 聊天历史
        chat_history = st.session_state.discovery_state['topic_chat_history']
        
        for message in chat_history:
            if message['role'] == 'user':
                st.chat_message("user").write(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])
                    
                    # 显示主题相关性
                    if 'topic_relevance' in message:
                        relevance = message['topic_relevance']
                        st.caption(f"🎯 主题相关性: {relevance:.1%}")
        
        # 聊天输入
        user_input = st.chat_input("基于选定主题提问...")
        
        if user_input:
            # 添加用户消息
            chat_history.append({'role': 'user', 'content': user_input})
            st.chat_message("user").write(user_input)
            
            # 生成回答
            with st.chat_message("assistant"):
                with st.spinner("🤔 AI正在基于主题背景思考..."):
                    try:
                        response = self._generate_topic_aware_response(user_input, selected_topic)
                        
                        st.write(response['answer'])
                        
                        # 显示主题相关性
                        if 'topic_relevance' in response:
                            st.caption(f"🎯 主题相关性: {response['topic_relevance']:.1%}")
                        
                        # 相关主题建议
                        if response.get('related_suggestions'):
                            with st.expander("🔗 相关主题建议"):
                                for suggestion in response['related_suggestions']:
                                    st.write(f"- {suggestion}")
                        
                        # 添加助手消息
                        chat_history.append({
                            'role': 'assistant', 
                            'content': response['answer'],
                            'topic_relevance': response.get('topic_relevance', 0)
                        })
                        
                    except Exception as e:
                        error_msg = f"抱歉，生成回答时出现错误: {e}"
                        st.error(error_msg)
                        chat_history.append({'role': 'assistant', 'content': error_msg})
        
        # 清除话题按钮
        if st.button("🔄 更换主题", help="清除当前主题选择和聊天历史"):
            st.session_state.discovery_state['selected_topic'] = None
            st.session_state.discovery_state['topic_chat_history'] = []
            st.rerun()
    
    def _generate_topic_aware_response(self, user_input: str, selected_topic) -> Dict[str, Any]:
        """生成主题感知的回答"""
        
        if not self.topic_aware_rag:
            # 回退到基础RAG
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                st.session_state.rag_system.generate_answer(user_input)
            )
            
            loop.close()
            
            return {
                'answer': result.answer,
                'topic_relevance': 0.0,
                'related_suggestions': []
            }
        
        # 使用主题感知RAG
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                self.topic_aware_rag.query_with_topic_context(
                    query=user_input,
                    selected_topic=selected_topic,
                    retrieval_strategy="topic_focused"
                )
            )
            
            return {
                'answer': result.answer,
                'topic_relevance': result.topic_relevance_score,
                'related_suggestions': result.related_topic_suggestions,
                'generation_time': result.generation_time
            }
            
        except Exception as e:
            logger.error(f"Topic-aware response generation failed: {e}")
            raise
        
        finally:
            loop.close()
    
    def _render_analysis_statistics(self, outline: TopicOutline):
        """渲染分析统计信息"""
        
        st.markdown("### 📊 详细分析统计")
        
        stats = outline.statistics
        
        # 基本统计
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总主题数", stats.get('total_topics', 0))
        
        with col2:
            st.metric("分析文档", stats.get('total_documents_analyzed', 0))
        
        with col3:
            st.metric("分析耗时", f"{stats.get('analysis_duration', 0):.2f}s")
        
        with col4:
            st.metric("覆盖领域", stats.get('coverage_areas', 0))
        
        # 主题分布图表
        if 'topic_distribution' in stats:
            st.markdown("#### 📈 主题分布")
            
            distribution_data = stats['topic_distribution']
            
            df = pd.DataFrame([
                {'章节': section, '主题数量': count}
                for section, count in distribution_data.items()
            ])
            
            if not df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        df, 
                        x='章节', 
                        y='主题数量',
                        title="各章节主题数量分布"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.pie(
                        df, 
                        values='主题数量', 
                        names='章节',
                        title="主题分布比例"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # 热门关键词
        if 'top_keywords' in stats and stats['top_keywords']:
            st.markdown("#### 🏷️ 热门关键词")
            
            keywords_df = pd.DataFrame(
                stats['top_keywords'], 
                columns=['关键词', '出现频次']
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.dataframe(keywords_df, use_container_width=True)
            
            with col2:
                if len(keywords_df) > 0:
                    fig = px.bar(
                        keywords_df.head(8), 
                        x='出现频次', 
                        y='关键词',
                        orientation='h',
                        title="关键词频次分布"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # 分析元数据
        st.markdown("#### ⚙️ 分析配置")
        
        analysis_metadata = outline.analysis_period
        
        metadata_info = {
            "分析时间段": analysis_metadata,
            "生成时间": outline.generated_time.strftime("%Y-%m-%d %H:%M:%S"),
            "主题提取方法": "混合算法",
            "摘要生成": "基于LLM" if self.content_summarizer else "基于模板"
        }
        
        for key, value in metadata_info.items():
            st.write(f"- **{key}**: {value}")

# 主要接口函数
def render_discovery_interface(vector_db_manager, rag_system, llm_generator=None):
    """渲染内容发现界面的主要入口函数"""
    
    if 'discovery_interface' not in st.session_state:
        st.session_state.discovery_interface = DiscoveryInterface()
    
    interface = st.session_state.discovery_interface
    
    # 初始化组件（如果还未初始化）
    if not interface.content_analyzer:
        success = interface.initialize_components(vector_db_manager, rag_system, llm_generator)
        if not success:
            return
    
    # 渲染界面
    interface.render_discovery_tab()