# src/generation/topic_aware_rag.py
"""
主题感知RAG引擎 - 基于用户选择的主题进行优化检索和生成
"""

import asyncio
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from ..analysis.content_analyzer import TopicItem
from ..generation.topic_outline_generator import TopicOutline
from .ultimate_rag_system import UltimateRAGSystem, UltimateGenerationResult

@dataclass
class TopicContext:
    """主题上下文"""
    selected_topic: TopicItem
    related_topics: List[TopicItem]
    topic_keywords: List[str]
    document_ids: List[str]
    context_weight: float = 1.0

@dataclass
class TopicAwareQuery:
    """主题感知查询"""
    original_query: str
    topic_context: Optional[TopicContext]
    enhanced_query: str
    retrieval_strategy: str = "topic_focused"
    context_expansion: bool = True

@dataclass
class TopicAwareResult(UltimateGenerationResult):
    """主题感知生成结果"""
    topic_context_used: Optional[TopicContext] = None
    topic_relevance_score: float = 0.0
    context_enhancement_applied: bool = False
    related_topic_suggestions: List[str] = field(default_factory=list)

class TopicAwareRAGEngine:
    """主题感知RAG引擎"""
    
    def __init__(
        self, 
        base_rag_system: UltimateRAGSystem,
        topic_outline: Optional[TopicOutline] = None
    ):
        self.base_rag = base_rag_system
        self.topic_outline = topic_outline
        self.topic_index = self._build_topic_index() if topic_outline else {}
        
        # 主题检索策略配置
        self.retrieval_strategies = {
            "topic_focused": {
                "topic_weight": 0.7,
                "semantic_weight": 0.3,
                "expand_keywords": True,
                "filter_by_topic": True
            },
            "topic_expanded": {
                "topic_weight": 0.5,
                "semantic_weight": 0.5,
                "expand_keywords": True,
                "filter_by_topic": False
            },
            "semantic_priority": {
                "topic_weight": 0.3,
                "semantic_weight": 0.7,
                "expand_keywords": False,
                "filter_by_topic": False
            }
        }
        
        logger.info("Topic-aware RAG engine initialized")
    
    def _build_topic_index(self) -> Dict[str, Any]:
        """构建主题索引"""
        
        if not self.topic_outline:
            return {}
        
        topic_index = {
            "topics_by_id": {},
            "topics_by_keyword": {},
            "document_to_topics": {},
            "topic_relationships": self.topic_outline.topic_relationships
        }
        
        # 建立主题索引
        all_topics = []
        for section in self.topic_outline.sections:
            all_topics.extend(section.topics)
        
        for topic in all_topics:
            # 按ID索引
            topic_index["topics_by_id"][topic.title] = topic
            
            # 按关键词索引
            for keyword in topic.keywords:
                if keyword not in topic_index["topics_by_keyword"]:
                    topic_index["topics_by_keyword"][keyword] = []
                topic_index["topics_by_keyword"][keyword].append(topic.title)
            
            # 文档到主题的映射
            for doc_id in topic.document_ids:
                if doc_id not in topic_index["document_to_topics"]:
                    topic_index["document_to_topics"][doc_id] = []
                topic_index["document_to_topics"][doc_id].append(topic.title)
        
        logger.info(f"Built topic index with {len(all_topics)} topics")
        return topic_index
    
    async def query_with_topic_context(
        self,
        query: str,
        selected_topic: Optional[TopicItem] = None,
        retrieval_strategy: str = "topic_focused",
        **kwargs
    ) -> TopicAwareResult:
        """基于主题上下文的查询"""
        
        logger.info(f"Processing topic-aware query: '{query[:50]}...'")
        start_time = datetime.now()
        
        try:
            # 1. 构建主题上下文
            topic_context = None
            if selected_topic:
                topic_context = await self._build_topic_context(selected_topic)
            
            # 2. 增强查询
            enhanced_query_info = await self._enhance_query_with_topic(
                query, topic_context, retrieval_strategy
            )
            
            # 3. 执行主题感知检索
            retrieval_results = await self._topic_aware_retrieval(
                enhanced_query_info, topic_context, **kwargs
            )
            
            # 4. 主题感知生成
            generation_result = await self._topic_aware_generation(
                enhanced_query_info, retrieval_results, topic_context
            )
            
            # 5. 生成相关主题建议
            related_suggestions = self._generate_related_topic_suggestions(
                query, topic_context
            )
            
            # 6. 构建增强结果
            topic_aware_result = TopicAwareResult(
                **generation_result.__dict__,
                topic_context_used=topic_context,
                topic_relevance_score=self._calculate_topic_relevance(
                    generation_result.answer, topic_context
                ),
                context_enhancement_applied=topic_context is not None,
                related_topic_suggestions=related_suggestions
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.success(f"Topic-aware query completed in {processing_time:.2f}s")
            
            return topic_aware_result
            
        except Exception as e:
            logger.error(f"Topic-aware query failed: {e}")
            # 回退到基础RAG
            base_result = await self.base_rag.generate_answer(query, **kwargs)
            return TopicAwareResult(**base_result.__dict__)
    
    async def _build_topic_context(self, selected_topic: TopicItem) -> TopicContext:
        """构建主题上下文"""
        
        # 找到相关主题
        related_topics = []
        if selected_topic.title in self.topic_index.get("topic_relationships", {}):
            related_topic_titles = self.topic_index["topic_relationships"][selected_topic.title]
            for title in related_topic_titles[:3]:  # 最多3个相关主题
                if title in self.topic_index["topics_by_id"]:
                    related_topics.append(self.topic_index["topics_by_id"][title])
        
        # 收集主题关键词
        topic_keywords = selected_topic.keywords.copy()
        for related_topic in related_topics:
            # 添加相关主题的核心关键词
            topic_keywords.extend(related_topic.keywords[:3])
        
        # 去重并保持顺序
        unique_keywords = []
        seen = set()
        for keyword in topic_keywords:
            if keyword not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword)
        
        # 收集相关文档ID
        document_ids = selected_topic.document_ids.copy()
        for related_topic in related_topics:
            document_ids.extend(related_topic.document_ids[:5])  # 每个相关主题最多5个文档
        
        context = TopicContext(
            selected_topic=selected_topic,
            related_topics=related_topics,
            topic_keywords=unique_keywords[:15],  # 最多15个关键词
            document_ids=list(set(document_ids)),  # 去重
            context_weight=1.0
        )
        
        logger.debug(f"Built topic context with {len(context.topic_keywords)} keywords, "
                    f"{len(context.document_ids)} documents")
        
        return context
    
    async def _enhance_query_with_topic(
        self,
        query: str,
        topic_context: Optional[TopicContext],
        strategy: str
    ) -> TopicAwareQuery:
        """使用主题上下文增强查询"""
        
        if not topic_context:
            return TopicAwareQuery(
                original_query=query,
                topic_context=None,
                enhanced_query=query,
                retrieval_strategy=strategy
            )
        
        strategy_config = self.retrieval_strategies.get(strategy, self.retrieval_strategies["topic_focused"])
        
        enhanced_query = query
        
        # 根据策略增强查询
        if strategy_config["expand_keywords"]:
            # 添加主题关键词到查询中
            relevant_keywords = self._select_relevant_keywords(
                query, topic_context.topic_keywords
            )
            
            if relevant_keywords:
                keyword_expansion = " ".join(relevant_keywords)
                enhanced_query = f"{query} {keyword_expansion}"
        
        # 如果有LLM，可以进行更智能的查询重写
        if hasattr(self.base_rag, 'llm_generator') and self.base_rag.llm_generator:
            try:
                enhanced_query = await self._llm_enhanced_query_rewrite(
                    query, topic_context, enhanced_query
                )
            except Exception as e:
                logger.warning(f"LLM query enhancement failed: {e}")
        
        return TopicAwareQuery(
            original_query=query,
            topic_context=topic_context,
            enhanced_query=enhanced_query,
            retrieval_strategy=strategy,
            context_expansion=strategy_config["expand_keywords"]
        )
    
    def _select_relevant_keywords(self, query: str, topic_keywords: List[str]) -> List[str]:
        """选择与查询相关的主题关键词"""
        
        query_lower = query.lower()
        relevant_keywords = []
        
        for keyword in topic_keywords:
            keyword_lower = keyword.lower()
            
            # 如果关键词已经在查询中，跳过
            if keyword_lower in query_lower:
                continue
            
            # 检查关键词是否与查询语义相关
            if self._is_keyword_relevant(query_lower, keyword_lower):
                relevant_keywords.append(keyword)
            
            # 限制关键词数量
            if len(relevant_keywords) >= 5:
                break
        
        return relevant_keywords
    
    def _is_keyword_relevant(self, query: str, keyword: str) -> bool:
        """检查关键词是否与查询相关"""
        
        # 简单的相关性检查
        # 在实际应用中，可以使用更复杂的语义相似度计算
        
        query_words = set(query.split())
        keyword_words = set(keyword.split())
        
        # 检查词汇重叠
        if query_words & keyword_words:
            return True
        
        # 检查技术领域相关性
        tech_domains = {
            'learning': ['train', 'model', 'algorithm', 'optimization'],
            'neural': ['network', 'deep', 'layer', 'neuron'],
            'vision': ['image', 'visual', 'detection', 'recognition'],
            'language': ['text', 'nlp', 'linguistic', 'semantic']
        }
        
        for domain, related_words in tech_domains.items():
            if domain in keyword and any(word in query for word in related_words):
                return True
        
        return False
    
    async def _llm_enhanced_query_rewrite(
        self,
        original_query: str,
        topic_context: TopicContext,
        current_enhanced_query: str
    ) -> str:
        """使用LLM增强查询重写"""
        
        prompt = f"""基于以下技术主题上下文，优化用户查询以获得更精准的检索结果：

原始查询: {original_query}
主题: {topic_context.selected_topic.title}
主题关键词: {', '.join(topic_context.topic_keywords[:8])}
相关主题: {', '.join([t.title for t in topic_context.related_topics])}

请重写查询，使其：
1. 保持原始查询的核心意图
2. 融入相关的技术术语和概念
3. 提高检索的精准度和相关性

重写后的查询："""

        try:
            enhanced_query = await self.base_rag.llm_generator.generate_text(
                prompt,
                max_length=100,
                temperature=0.3
            )
            
            # 清理生成的查询
            enhanced_query = enhanced_query.strip()
            if enhanced_query and len(enhanced_query) > 10:
                return enhanced_query
            
        except Exception as e:
            logger.warning(f"LLM query rewrite failed: {e}")
        
        return current_enhanced_query
    
    async def _topic_aware_retrieval(
        self,
        query_info: TopicAwareQuery,
        topic_context: Optional[TopicContext],
        **kwargs
    ) -> Any:
        """主题感知检索"""
        
        if not topic_context:
            # 没有主题上下文，使用基础检索
            return await self.base_rag._enhanced_retrieval(query_info.enhanced_query, **kwargs)
        
        strategy_config = self.retrieval_strategies[query_info.retrieval_strategy]
        
        # 1. 基础语义检索
        base_results = await self.base_rag._enhanced_retrieval(query_info.enhanced_query, **kwargs)
        
        # 2. 主题相关文档检索
        topic_results = []
        if strategy_config["filter_by_topic"]:
            topic_results = await self._retrieve_topic_documents(
                query_info.enhanced_query,
                topic_context,
                kwargs.get('top_k', 10)
            )
        
        # 3. 融合检索结果
        combined_results = self._combine_retrieval_results(
            base_results,
            topic_results,
            strategy_config,
            topic_context
        )
        
        return combined_results
    
    async def _retrieve_topic_documents(
        self,
        query: str,
        topic_context: TopicContext,
        top_k: int
    ) -> List[Dict]:
        """检索主题相关文档"""
        
        # 这里需要与向量数据库接口，根据文档ID检索具体文档
        # 简化实现：返回主题相关的文档ID
        topic_doc_results = []
        
        for doc_id in topic_context.document_ids[:top_k]:
            # 在实际实现中，这里应该从向量数据库获取具体文档
            topic_doc_results.append({
                'id': doc_id,
                'score': 0.8,  # 给主题相关文档较高的基础分数
                'source': 'topic_context',
                'metadata': {'topic_selected': True}
            })
        
        return topic_doc_results
    
    def _combine_retrieval_results(
        self,
        base_results: List[Dict],
        topic_results: List[Dict],
        strategy_config: Dict,
        topic_context: TopicContext
    ) -> List[Dict]:
        """融合检索结果"""
        
        combined_results = []
        seen_ids = set()
        
        # 合并结果并重新评分
        all_results = []
        
        # 添加基础检索结果
        for result in base_results:
            if result.get('id') not in seen_ids:
                result['combined_score'] = (
                    result.get('score', 0) * strategy_config['semantic_weight']
                )
                result['source_type'] = 'semantic'
                all_results.append(result)
                seen_ids.add(result.get('id'))
        
        # 添加主题检索结果
        for result in topic_results:
            if result.get('id') not in seen_ids:
                result['combined_score'] = (
                    result.get('score', 0) * strategy_config['topic_weight']
                )
                result['source_type'] = 'topic'
                all_results.append(result)
                seen_ids.add(result.get('id'))
            else:
                # 如果文档已存在，增强其分数
                for existing in all_results:
                    if existing.get('id') == result.get('id'):
                        existing['combined_score'] += (
                            result.get('score', 0) * strategy_config['topic_weight']
                        )
                        existing['source_type'] = 'both'
                        break
        
        # 按融合分数排序
        all_results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        return all_results
    
    async def _topic_aware_generation(
        self,
        query_info: TopicAwareQuery,
        retrieval_results: List[Dict],
        topic_context: Optional[TopicContext]
    ) -> UltimateGenerationResult:
        """主题感知生成"""
        
        if not topic_context:
            # 没有主题上下文，使用基础生成
            return await self.base_rag._generate_answer_from_context(
                query_info.enhanced_query,
                retrieval_results
            )
        
        # 增强系统提示，包含主题上下文
        enhanced_system_prompt = self._build_topic_aware_system_prompt(topic_context)
        
        # 增强用户查询，包含主题信息
        enhanced_user_query = self._build_topic_aware_user_query(
            query_info,
            topic_context
        )
        
        # 使用增强的提示进行生成
        try:
            # 这里需要修改base_rag的生成方法以支持自定义系统提示
            result = await self.base_rag._generate_answer_from_context(
                enhanced_user_query,
                retrieval_results,
                system_prompt=enhanced_system_prompt
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Topic-aware generation failed: {e}")
            # 回退到基础生成
            return await self.base_rag._generate_answer_from_context(
                query_info.enhanced_query,
                retrieval_results
            )
    
    def _build_topic_aware_system_prompt(self, topic_context: TopicContext) -> str:
        """构建主题感知的系统提示"""
        
        base_prompt = """你是一个专业的AI技术助手，专注于提供准确、深入的技术解答。"""
        
        topic_prompt = f"""
当前对话聚焦于"{topic_context.selected_topic.title}"这一技术主题。

主题背景：
- 核心技术领域：{topic_context.selected_topic.title}
- 关键技术概念：{', '.join(topic_context.topic_keywords[:8])}
- 相关技术方向：{', '.join([t.title for t in topic_context.related_topics])}

请在回答时：
1. 重点关注与该主题相关的技术内容
2. 使用专业术语，但确保解释清晰
3. 如果适当，可以提及相关的技术发展和应用场景
4. 保持回答的准确性和权威性

基于以上主题背景和提供的参考资料，请回答用户的问题。
"""
        
        return base_prompt + topic_prompt
    
    def _build_topic_aware_user_query(
        self,
        query_info: TopicAwareQuery,
        topic_context: TopicContext
    ) -> str:
        """构建主题感知的用户查询"""
        
        base_query = query_info.original_query
        
        # 如果查询较短或缺乏上下文，添加主题信息
        if len(base_query.split()) < 5:
            context_addition = f"（基于{topic_context.selected_topic.title}主题背景）"
            return f"{base_query} {context_addition}"
        
        return base_query
    
    def _calculate_topic_relevance(
        self,
        answer: str,
        topic_context: Optional[TopicContext]
    ) -> float:
        """计算答案与主题的相关性"""
        
        if not topic_context:
            return 0.0
        
        answer_lower = answer.lower()
        relevance_score = 0.0
        
        # 检查主题关键词在答案中的出现
        total_keywords = len(topic_context.topic_keywords)
        if total_keywords == 0:
            return 0.0
        
        matched_keywords = 0
        for keyword in topic_context.topic_keywords:
            if keyword.lower() in answer_lower:
                matched_keywords += 1
        
        # 基础相关性分数
        keyword_relevance = matched_keywords / total_keywords
        
        # 检查主题标题的出现
        if topic_context.selected_topic.title.lower() in answer_lower:
            title_relevance = 0.3
        else:
            title_relevance = 0.0
        
        # 检查相关主题的出现
        related_relevance = 0.0
        for related_topic in topic_context.related_topics:
            if related_topic.title.lower() in answer_lower:
                related_relevance += 0.1
        
        relevance_score = min(keyword_relevance * 0.6 + title_relevance + related_relevance, 1.0)
        
        return relevance_score
    
    def _generate_related_topic_suggestions(
        self,
        query: str,
        topic_context: Optional[TopicContext]
    ) -> List[str]:
        """生成相关主题建议"""
        
        suggestions = []
        
        if not topic_context:
            return suggestions
        
        # 添加相关主题
        for related_topic in topic_context.related_topics:
            suggestions.append(related_topic.title)
        
        # 基于关键词查找其他相关主题
        query_keywords = set(query.lower().split())
        
        for keyword in topic_context.topic_keywords:
            if keyword in self.topic_index.get("topics_by_keyword", {}):
                related_topic_titles = self.topic_index["topics_by_keyword"][keyword]
                for title in related_topic_titles:
                    if (title != topic_context.selected_topic.title and 
                        title not in suggestions and
                        len(suggestions) < 5):
                        suggestions.append(title)
        
        return suggestions[:5]  # 最多5个建议
    
    def update_topic_outline(self, new_outline: TopicOutline):
        """更新主题大纲"""
        
        self.topic_outline = new_outline
        self.topic_index = self._build_topic_index()
        logger.info("Topic outline updated")
    
    def get_available_topics(self) -> List[Dict[str, Any]]:
        """获取可用的主题列表"""
        
        if not self.topic_outline:
            return []
        
        topics = []
        for section in self.topic_outline.sections:
            for topic in section.topics:
                topics.append({
                    "title": topic.title,
                    "summary": topic.summary,
                    "keywords": topic.keywords[:5],
                    "doc_count": topic.doc_count,
                    "importance_score": topic.importance_score,
                    "section": section.title
                })
        
        return sorted(topics, key=lambda x: x["importance_score"], reverse=True)