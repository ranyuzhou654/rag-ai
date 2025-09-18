# src/generation/topic_outline_generator.py
"""
交互式主题大纲生成器 - 将分析结果组织成用户友好的结构化大纲
"""

import asyncio
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger

from ..analysis.content_analyzer import ContentAnalysis, TopicItem
from .content_summarizer import ContentSummarizer, SummaryRequest

@dataclass
class OutlineSection:
    """大纲章节"""
    title: str
    subtitle: str
    description: str
    topics: List[TopicItem]
    metadata: Dict[str, Any] = field(default_factory=dict)
    display_priority: int = 1  # 1=高, 2=中, 3=低

@dataclass
class TopicOutline:
    """完整的主题大纲"""
    title: str
    overview: str
    sections: List[OutlineSection]
    trending_topics: List[TopicItem]
    featured_topics: List[TopicItem]
    topic_relationships: Dict[str, List[str]]
    statistics: Dict[str, Any]
    generated_time: datetime = field(default_factory=datetime.now)
    analysis_period: str = ""

@dataclass
class InteractiveElement:
    """交互式元素"""
    element_type: str  # button, card, tree_node, modal
    element_id: str
    title: str
    content: Any
    actions: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class TopicOutlineGenerator:
    """交互式主题大纲生成器"""
    
    def __init__(self, content_summarizer: Optional[ContentSummarizer] = None):
        self.summarizer = content_summarizer or ContentSummarizer()
        
        # 大纲模板配置
        self.outline_templates = {
            "comprehensive": {
                "sections": [
                    {
                        "title": "🔥 热门话题",
                        "subtitle": "当前最受关注的技术主题",
                        "key": "hot_topics",
                        "max_items": 5,
                        "priority": 1
                    },
                    {
                        "title": "📈 趋势技术",
                        "subtitle": "快速发展的新兴技术领域",
                        "key": "trending_topics", 
                        "max_items": 8,
                        "priority": 2
                    },
                    {
                        "title": "🚀 技术突破",
                        "subtitle": "重要的研究进展和创新",
                        "key": "recent_breakthroughs",
                        "max_items": 5,
                        "priority": 1
                    }
                ]
            },
            
            "focused": {
                "sections": [
                    {
                        "title": "💡 核心主题",
                        "subtitle": "最重要的技术焦点",
                        "key": "hot_topics",
                        "max_items": 3,
                        "priority": 1
                    },
                    {
                        "title": "🔍 相关领域",
                        "subtitle": "扩展的技术应用",
                        "key": "trending_topics",
                        "max_items": 5,
                        "priority": 2
                    }
                ]
            },
            
            "detailed": {
                "sections": [
                    {
                        "title": "🎯 重点领域",
                        "subtitle": "当前研究的核心方向",
                        "key": "hot_topics",
                        "max_items": 6,
                        "priority": 1
                    },
                    {
                        "title": "📊 发展趋势",
                        "subtitle": "技术发展的新动向",
                        "key": "trending_topics",
                        "max_items": 10,
                        "priority": 2
                    },
                    {
                        "title": "💫 创新突破",
                        "subtitle": "最新的技术创新",
                        "key": "recent_breakthroughs",
                        "max_items": 6,
                        "priority": 1
                    }
                ]
            }
        }
        
        logger.info("Topic outline generator initialized")
    
    async def generate_outline(
        self,
        content_analysis: ContentAnalysis,
        template: str = "comprehensive",
        include_summaries: bool = True,
        max_topics_per_section: Optional[int] = None
    ) -> TopicOutline:
        """生成交互式主题大纲"""
        
        logger.info(f"Generating outline with template '{template}'")
        start_time = datetime.now()
        
        try:
            # 1. 选择大纲模板
            template_config = self.outline_templates.get(template, self.outline_templates["comprehensive"])
            
            # 2. 生成大纲概述
            overview = await self._generate_outline_overview(content_analysis)
            
            # 3. 构建大纲章节
            sections = await self._build_outline_sections(
                content_analysis, 
                template_config,
                include_summaries,
                max_topics_per_section
            )
            
            # 4. 选择特色主题
            featured_topics = self._select_featured_topics(content_analysis)
            
            # 5. 提取趋势主题
            trending_topics = self._extract_trending_topics(content_analysis)
            
            # 6. 计算统计信息
            statistics = self._calculate_outline_statistics(content_analysis, sections)
            
            # 7. 构建完整大纲
            outline = TopicOutline(
                title=self._generate_outline_title(content_analysis, template),
                overview=overview,
                sections=sections,
                trending_topics=trending_topics,
                featured_topics=featured_topics,
                topic_relationships=content_analysis.topic_relationships,
                statistics=statistics,
                analysis_period=content_analysis.analysis_period
            )
            
            generation_time = (datetime.now() - start_time).total_seconds()
            logger.success(f"Outline generated in {generation_time:.2f}s with {len(sections)} sections")
            
            return outline
            
        except Exception as e:
            logger.error(f"Failed to generate outline: {e}")
            return self._create_fallback_outline(content_analysis)
    
    async def _generate_outline_overview(self, content_analysis: ContentAnalysis) -> str:
        """生成大纲概述"""
        
        total_topics = (len(content_analysis.hot_topics) + 
                       len(content_analysis.trending_topics) + 
                       len(content_analysis.recent_breakthroughs))
        
        if total_topics == 0:
            return "暂未发现显著的技术主题趋势。"
        
        # 识别主要技术领域
        main_areas = self._identify_main_technical_areas(content_analysis)
        
        if self.summarizer.llm_generator:
            try:
                # 使用LLM生成概述
                prompt = f"""基于以下技术主题分析结果，生成一个100字左右的概述：

分析期间: {content_analysis.analysis_period}
热门主题数: {len(content_analysis.hot_topics)}
趋势主题数: {len(content_analysis.trending_topics)}
技术突破数: {len(content_analysis.recent_breakthroughs)}

主要技术领域: {', '.join(main_areas)}

请生成一个简洁的概述，描述当前AI技术的主要发展趋势和研究热点："""

                overview = await self.summarizer.llm_generator.generate_text(
                    prompt,
                    max_length=150,
                    temperature=0.3
                )
                
                return overview.strip()
                
            except Exception as e:
                logger.warning(f"Failed to generate LLM overview: {e}")
        
        # 回退到模板概述
        return self._generate_template_overview(content_analysis, main_areas)
    
    def _identify_main_technical_areas(self, content_analysis: ContentAnalysis) -> List[str]:
        """识别主要技术领域"""
        
        area_keywords = {
            "深度学习": ["deep learning", "neural", "transformer", "attention", "深度学习"],
            "自然语言处理": ["nlp", "language", "text", "自然语言", "语言模型"],
            "计算机视觉": ["vision", "image", "visual", "detection", "计算机视觉"],
            "机器学习": ["machine learning", "ml", "algorithm", "机器学习", "算法"],
            "人工智能系统": ["ai system", "artificial intelligence", "人工智能", "智能系统"]
        }
        
        identified_areas = []
        all_topics = (content_analysis.hot_topics + 
                     content_analysis.trending_topics + 
                     content_analysis.recent_breakthroughs)
        
        for area, keywords in area_keywords.items():
            for topic in all_topics:
                topic_text = f"{topic.title} {' '.join(topic.keywords)}".lower()
                if any(keyword in topic_text for keyword in keywords):
                    if area not in identified_areas:
                        identified_areas.append(area)
                    break
        
        return identified_areas[:4]  # 最多4个主要领域
    
    def _generate_template_overview(self, content_analysis: ContentAnalysis, main_areas: List[str]) -> str:
        """生成模板概述"""
        
        total_topics = (len(content_analysis.hot_topics) + 
                       len(content_analysis.trending_topics) + 
                       len(content_analysis.recent_breakthroughs))
        
        areas_text = "、".join(main_areas) if main_areas else "多个技术领域"
        
        return f"在{content_analysis.analysis_period}的分析中，共发现{total_topics}个重要技术主题，" \
               f"主要涵盖{areas_text}等方向。这些主题展现了当前AI技术发展的核心趋势和创新重点，" \
               f"为技术研究和应用提供了重要参考。"
    
    async def _build_outline_sections(
        self,
        content_analysis: ContentAnalysis,
        template_config: Dict,
        include_summaries: bool,
        max_topics_per_section: Optional[int]
    ) -> List[OutlineSection]:
        """构建大纲章节"""
        
        sections = []
        
        for section_config in template_config["sections"]:
            # 获取对应的主题列表
            topics = self._get_topics_for_section(content_analysis, section_config)
            
            if not topics:
                continue
            
            # 限制主题数量
            max_items = max_topics_per_section or section_config.get("max_items", 10)
            topics = topics[:max_items]
            
            # 为主题生成摘要（如果需要）
            if include_summaries:
                topics = await self._add_summaries_to_topics(topics)
            
            # 生成章节描述
            description = await self._generate_section_description(
                section_config, topics
            )
            
            section = OutlineSection(
                title=section_config["title"],
                subtitle=section_config["subtitle"],
                description=description,
                topics=topics,
                display_priority=section_config.get("priority", 2),
                metadata={
                    "section_key": section_config["key"],
                    "topic_count": len(topics),
                    "has_summaries": include_summaries
                }
            )
            
            sections.append(section)
        
        return sections
    
    def _get_topics_for_section(self, content_analysis: ContentAnalysis, section_config: Dict) -> List[TopicItem]:
        """获取章节对应的主题"""
        
        section_key = section_config["key"]
        
        if section_key == "hot_topics":
            return content_analysis.hot_topics
        elif section_key == "trending_topics":
            return content_analysis.trending_topics
        elif section_key == "recent_breakthroughs":
            return content_analysis.recent_breakthroughs
        else:
            return []
    
    async def _add_summaries_to_topics(self, topics: List[TopicItem]) -> List[TopicItem]:
        """为主题添加摘要"""
        
        enhanced_topics = []
        
        for topic in topics:
            try:
                # 为每个主题生成简短摘要
                if not topic.summary or len(topic.summary) < 50:
                    # 准备摘要请求
                    summary_request = SummaryRequest(
                        topic_title=topic.title,
                        documents=[{"doc": {"content": "", "metadata": {"title": paper}}} for paper in topic.key_papers[:3]],
                        keywords=topic.keywords,
                        summary_type="brief",
                        max_length=100,
                        target_audience="general"
                    )
                    
                    # 生成摘要
                    summary_result = await self.summarizer.generate_summary(summary_request)
                    topic.summary = summary_result.summary
                
                enhanced_topics.append(topic)
                
            except Exception as e:
                logger.warning(f"Failed to enhance topic '{topic.title}': {e}")
                enhanced_topics.append(topic)
        
        return enhanced_topics
    
    async def _generate_section_description(self, section_config: Dict, topics: List[TopicItem]) -> str:
        """生成章节描述"""
        
        if not topics:
            return "暂无相关主题。"
        
        topic_count = len(topics)
        section_title = section_config["title"]
        
        # 提取主要关键词
        all_keywords = []
        for topic in topics[:3]:  # 只从前3个主题提取关键词
            all_keywords.extend(topic.keywords[:2])
        
        main_keywords = list(set(all_keywords))[:5]
        keywords_text = "、".join(main_keywords) if main_keywords else "相关技术"
        
        description_templates = {
            "🔥 热门话题": f"发现{topic_count}个热门技术主题，主要涉及{keywords_text}等领域，展现了当前技术发展的核心方向。",
            "📈 趋势技术": f"识别出{topic_count}个技术趋势，包括{keywords_text}等新兴技术，代表了未来发展的重要方向。",
            "🚀 技术突破": f"汇总了{topic_count}个重要技术突破，涵盖{keywords_text}等创新领域，标志着技术进步的新里程碑。",
            "💡 核心主题": f"聚焦{topic_count}个核心技术主题，重点关注{keywords_text}等关键技术的最新进展。",
            "🔍 相关领域": f"探索{topic_count}个相关技术领域，深入{keywords_text}等应用方向的发展现状。",
            "🎯 重点领域": f"分析{topic_count}个重点技术领域，全面覆盖{keywords_text}等核心技术的研究动态。"
        }
        
        return description_templates.get(section_title, f"包含{topic_count}个相关技术主题。")
    
    def _select_featured_topics(self, content_analysis: ContentAnalysis) -> List[TopicItem]:
        """选择特色主题"""
        
        all_topics = (content_analysis.hot_topics + 
                     content_analysis.trending_topics + 
                     content_analysis.recent_breakthroughs)
        
        # 按重要性评分排序
        all_topics.sort(key=lambda x: x.importance_score, reverse=True)
        
        # 选择前3个作为特色主题
        featured = []
        for topic in all_topics[:5]:
            if topic.importance_score > 0.6 and topic.doc_count >= 3:
                featured.append(topic)
            if len(featured) >= 3:
                break
        
        return featured
    
    def _extract_trending_topics(self, content_analysis: ContentAnalysis) -> List[TopicItem]:
        """提取趋势主题"""
        
        # 获取所有趋势主题，按文档数量排序
        trending = content_analysis.trending_topics.copy()
        trending.sort(key=lambda x: x.doc_count, reverse=True)
        
        return trending[:6]  # 最多6个趋势主题
    
    def _calculate_outline_statistics(
        self, 
        content_analysis: ContentAnalysis, 
        sections: List[OutlineSection]
    ) -> Dict[str, Any]:
        """计算大纲统计信息"""
        
        total_topics = sum(len(section.topics) for section in sections)
        total_documents = content_analysis.analysis_metadata.get('total_documents_analyzed', 0)
        
        # 计算主题分布
        topic_distribution = {}
        for section in sections:
            topic_distribution[section.title] = len(section.topics)
        
        # 计算关键词频率
        all_keywords = []
        for section in sections:
            for topic in section.topics:
                all_keywords.extend(topic.keywords[:3])
        
        keyword_freq = {}
        for keyword in set(all_keywords):
            keyword_freq[keyword] = all_keywords.count(keyword)
        
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_topics": total_topics,
            "total_documents_analyzed": total_documents,
            "sections_count": len(sections),
            "topic_distribution": topic_distribution,
            "top_keywords": top_keywords,
            "analysis_duration": content_analysis.analysis_metadata.get('analysis_duration', 0),
            "coverage_areas": len(set(all_keywords))
        }
    
    def _generate_outline_title(self, content_analysis: ContentAnalysis, template: str) -> str:
        """生成大纲标题"""
        
        period = content_analysis.analysis_period
        total_topics = (len(content_analysis.hot_topics) + 
                       len(content_analysis.trending_topics) + 
                       len(content_analysis.recent_breakthroughs))
        
        template_titles = {
            "comprehensive": f"AI技术热点全景图 - {period}",
            "focused": f"核心技术焦点 - {period}",
            "detailed": f"技术发展详细分析 - {period}"
        }
        
        base_title = template_titles.get(template, f"技术主题概览 - {period}")
        
        if total_topics > 0:
            return f"{base_title} ({total_topics}个主题)"
        else:
            return base_title
    
    def _create_fallback_outline(self, content_analysis: ContentAnalysis) -> TopicOutline:
        """创建回退大纲"""
        
        return TopicOutline(
            title=f"技术主题概览 - {content_analysis.analysis_period}",
            overview="当前暂无足够数据生成详细的技术主题分析。",
            sections=[],
            trending_topics=[],
            featured_topics=[],
            topic_relationships={},
            statistics={
                "total_topics": 0,
                "total_documents_analyzed": 0,
                "sections_count": 0
            },
            analysis_period=content_analysis.analysis_period
        )
    
    def generate_interactive_elements(self, outline: TopicOutline) -> List[InteractiveElement]:
        """生成交互式元素"""
        
        elements = []
        
        # 1. 概览卡片
        overview_element = InteractiveElement(
            element_type="card",
            element_id="outline_overview",
            title="技术概览",
            content={
                "overview": outline.overview,
                "statistics": outline.statistics
            },
            actions=[
                {"type": "expand", "label": "查看详细统计"},
                {"type": "refresh", "label": "刷新分析"}
            ]
        )
        elements.append(overview_element)
        
        # 2. 章节导航
        for i, section in enumerate(outline.sections):
            section_element = InteractiveElement(
                element_type="tree_node",
                element_id=f"section_{i}",
                title=section.title,
                content={
                    "subtitle": section.subtitle,
                    "description": section.description,
                    "topic_count": len(section.topics)
                },
                actions=[
                    {"type": "expand", "label": "展开主题"},
                    {"type": "explore", "label": "深入探索"}
                ],
                metadata={
                    "priority": section.display_priority,
                    "section_index": i
                }
            )
            elements.append(section_element)
            
            # 3. 主题按钮
            for j, topic in enumerate(section.topics):
                topic_element = InteractiveElement(
                    element_type="button",
                    element_id=f"topic_{i}_{j}",
                    title=topic.title,
                    content={
                        "summary": topic.summary,
                        "keywords": topic.keywords[:5],
                        "doc_count": topic.doc_count,
                        "importance_score": topic.importance_score
                    },
                    actions=[
                        {"type": "ask_question", "label": "基于此主题提问"},
                        {"type": "view_details", "label": "查看详情"},
                        {"type": "related_topics", "label": "相关主题"}
                    ],
                    metadata={
                        "section_index": i,
                        "topic_index": j,
                        "document_ids": topic.document_ids
                    }
                )
                elements.append(topic_element)
        
        # 4. 特色主题卡片
        if outline.featured_topics:
            featured_element = InteractiveElement(
                element_type="modal",
                element_id="featured_topics",
                title="精选主题",
                content={
                    "topics": [
                        {
                            "title": topic.title,
                            "summary": topic.summary,
                            "importance": topic.importance_score
                        }
                        for topic in outline.featured_topics
                    ]
                },
                actions=[
                    {"type": "select_topic", "label": "选择主题"},
                    {"type": "compare_topics", "label": "主题对比"}
                ]
            )
            elements.append(featured_element)
        
        return elements
    
    def format_outline_for_display(self, outline: TopicOutline) -> Dict[str, Any]:
        """格式化大纲用于显示"""
        
        formatted = {
            "title": outline.title,
            "overview": outline.overview,
            "generated_time": outline.generated_time.isoformat(),
            "analysis_period": outline.analysis_period,
            "statistics": outline.statistics,
            "sections": []
        }
        
        for section in outline.sections:
            section_data = {
                "title": section.title,
                "subtitle": section.subtitle,
                "description": section.description,
                "priority": section.display_priority,
                "topics": []
            }
            
            for topic in section.topics:
                topic_data = {
                    "title": topic.title,
                    "summary": topic.summary,
                    "keywords": topic.keywords,
                    "doc_count": topic.doc_count,
                    "importance_score": round(topic.importance_score, 3),
                    "key_papers": topic.key_papers[:3],
                    "document_ids": topic.document_ids
                }
                section_data["topics"].append(topic_data)
            
            formatted["sections"].append(section_data)
        
        # 添加特色主题
        if outline.featured_topics:
            formatted["featured_topics"] = [
                {
                    "title": topic.title,
                    "summary": topic.summary,
                    "importance_score": round(topic.importance_score, 3),
                    "document_ids": topic.document_ids
                }
                for topic in outline.featured_topics
            ]
        
        # 添加主题关系
        formatted["topic_relationships"] = outline.topic_relationships
        
        return formatted