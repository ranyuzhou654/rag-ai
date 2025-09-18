# src/analysis/content_analyzer.py
"""
内容分析器 - 从现有知识库中发现热点话题和趋势
"""

import asyncio
import json
import re
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import numpy as np
from loguru import logger

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using simplified topic extraction")

@dataclass
class TopicItem:
    """主题项数据结构"""
    title: str
    summary: str
    importance_score: float
    doc_count: int
    key_papers: List[str]
    related_topics: List[str]
    document_ids: List[str]  # 用于后续RAG检索
    keywords: List[str] = field(default_factory=list)
    source_distribution: Dict[str, int] = field(default_factory=dict)
    time_trend: List[Tuple[str, int]] = field(default_factory=list)

@dataclass
class ContentAnalysis:
    """内容分析结果"""
    hot_topics: List[TopicItem]
    trending_topics: List[TopicItem]
    recent_breakthroughs: List[TopicItem]
    topic_relationships: Dict[str, List[str]]
    analysis_metadata: Dict[str, Any]
    generated_time: datetime = field(default_factory=datetime.now)
    analysis_period: str = ""

class ContentAnalyzer:
    """智能内容分析器"""
    
    def __init__(self, vector_db_manager, llm_generator=None):
        self.vector_db = vector_db_manager
        self.llm_generator = llm_generator
        
        # 技术关键词词典
        self.tech_keywords = {
            'deep_learning': ['深度学习', 'deep learning', 'neural network', 'transformer', 'attention'],
            'machine_learning': ['机器学习', 'machine learning', 'ML', 'algorithm', 'model'],
            'nlp': ['自然语言处理', 'NLP', 'language model', 'LLM', 'text processing'],
            'computer_vision': ['计算机视觉', 'computer vision', 'CV', 'image recognition', 'object detection'],
            'ai_systems': ['人工智能', 'artificial intelligence', 'AI system', 'AGI', 'automation'],
            'research_methods': ['研究方法', 'methodology', 'experiment', 'evaluation', 'benchmark']
        }
        
        # 重要性评分权重
        self.importance_weights = {
            'recency': 0.3,      # 时间新近性
            'frequency': 0.25,   # 出现频率
            'diversity': 0.2,    # 来源多样性
            'complexity': 0.15,  # 技术复杂度
            'relevance': 0.1     # 相关性分数
        }
        
        logger.info("Content analyzer initialized")
    
    async def analyze_content(
        self, 
        days: int = 7, 
        max_topics: int = 20,
        min_doc_count: int = 3
    ) -> ContentAnalysis:
        """分析现有知识库内容"""
        
        logger.info(f"Starting content analysis for last {days} days")
        start_time = datetime.now()
        
        try:
            # 1. 获取最近的文档
            recent_docs = await self._get_recent_documents(days)
            logger.info(f"Retrieved {len(recent_docs)} recent documents")
            
            if len(recent_docs) < min_doc_count:
                logger.warning(f"Insufficient documents ({len(recent_docs)}) for analysis")
                return self._create_empty_analysis(days)
            
            # 2. 提取主题和关键词
            topics_data = await self._extract_topics_from_documents(recent_docs)
            
            # 3. 计算主题重要性评分
            scored_topics = await self._score_topics(topics_data, recent_docs)
            
            # 4. 分类主题
            categorized_topics = self._categorize_topics(scored_topics, max_topics)
            
            # 5. 识别主题关系
            topic_relationships = self._identify_topic_relationships(scored_topics)
            
            # 6. 构建分析结果
            analysis = ContentAnalysis(
                hot_topics=categorized_topics['hot'],
                trending_topics=categorized_topics['trending'],
                recent_breakthroughs=categorized_topics['breakthroughs'],
                topic_relationships=topic_relationships,
                analysis_metadata={
                    'total_documents_analyzed': len(recent_docs),
                    'analysis_duration': (datetime.now() - start_time).total_seconds(),
                    'topic_extraction_method': 'hybrid' if SKLEARN_AVAILABLE else 'keyword_based',
                    'categories_found': len(scored_topics)
                },
                analysis_period=f"Last {days} days"
            )
            
            logger.success(f"Content analysis completed in {analysis.analysis_metadata['analysis_duration']:.2f}s")
            return analysis
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return self._create_empty_analysis(days)
    
    async def _get_recent_documents(self, days: int) -> List[Dict]:
        """从向量数据库获取最近的文档"""
        
        try:
            # 获取数据库中的所有点
            # 这里需要根据实际的vector_db接口进行调整
            all_points = await self._fetch_all_points()
            
            # 筛选最近的文档
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_docs = []
            
            for point in all_points:
                # 尝试从metadata中获取时间信息
                metadata = point.get('metadata', {})
                doc_date = self._extract_document_date(metadata)
                
                if doc_date and doc_date > cutoff_date:
                    recent_docs.append({
                        'id': point.get('id'),
                        'content': point.get('content', ''),
                        'metadata': metadata,
                        'date': doc_date
                    })
            
            # 按时间排序
            recent_docs.sort(key=lambda x: x['date'], reverse=True)
            
            return recent_docs
            
        except Exception as e:
            logger.error(f"Failed to get recent documents: {e}")
            return []
    
    async def _fetch_all_points(self) -> List[Dict]:
        """获取向量数据库中的所有数据点"""
        
        try:
            # 使用scroll方法获取所有点
            all_points = []
            scroll_result = self.vector_db.db.client.scroll(
                collection_name=self.vector_db.collection_name,
                limit=10000,  # 根据实际情况调整
                with_payload=True,
                with_vectors=False  # 不需要向量数据，只要内容和元数据
            )
            
            if scroll_result[0]:  # 检查是否有结果
                for point in scroll_result[0]:
                    all_points.append({
                        'id': point.id,
                        'content': point.payload.get('content', ''),
                        'metadata': point.payload
                    })
            
            return all_points
            
        except Exception as e:
            logger.error(f"Failed to fetch points from vector database: {e}")
            return []
    
    def _extract_document_date(self, metadata: Dict) -> Optional[datetime]:
        """从文档元数据中提取日期"""
        
        # 尝试多种日期字段名
        date_fields = ['published_date', 'date', 'created_at', 'timestamp', 'pub_date']
        
        for field in date_fields:
            if field in metadata:
                date_value = metadata[field]
                try:
                    if isinstance(date_value, str):
                        # 尝试解析ISO格式日期
                        return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                    elif isinstance(date_value, datetime):
                        return date_value
                except:
                    continue
        
        # 如果没有找到日期，返回None（将被过滤掉）
        return None
    
    async def _extract_topics_from_documents(self, documents: List[Dict]) -> Dict[str, Any]:
        """从文档中提取主题"""
        
        if SKLEARN_AVAILABLE:
            return await self._extract_topics_ml(documents)
        else:
            return await self._extract_topics_keyword_based(documents)
    
    async def _extract_topics_ml(self, documents: List[Dict]) -> Dict[str, Any]:
        """使用机器学习方法提取主题"""
        
        # 准备文本数据
        texts = []
        doc_metadata = []
        
        for doc in documents:
            content = doc['content']
            title = doc['metadata'].get('title', '')
            combined_text = f"{title} {content}"
            texts.append(combined_text)
            doc_metadata.append(doc)
        
        if len(texts) < 3:
            return await self._extract_topics_keyword_based(documents)
        
        try:
            # TF-IDF向量化
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # LDA主题建模
            n_topics = min(10, len(texts) // 2)  # 动态确定主题数
            if n_topics < 2:
                n_topics = 2
            
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            
            lda.fit(tfidf_matrix)
            
            # 提取主题
            topics = {}
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                
                # 为主题生成描述性标题
                topic_title = self._generate_topic_title(top_words)
                
                topics[topic_title] = {
                    'keywords': top_words,
                    'documents': [],
                    'importance': topic.max(),
                    'topic_idx': topic_idx
                }
            
            # 将文档分配给主题
            doc_topic_matrix = lda.transform(tfidf_matrix)
            for doc_idx, doc in enumerate(doc_metadata):
                best_topic_idx = doc_topic_matrix[doc_idx].argmax()
                topic_prob = doc_topic_matrix[doc_idx][best_topic_idx]
                
                if topic_prob > 0.3:  # 只分配高置信度的文档
                    for title, topic_data in topics.items():
                        if topic_data['topic_idx'] == best_topic_idx:
                            topic_data['documents'].append({
                                'doc': doc,
                                'probability': topic_prob
                            })
                            break
            
            return {
                'method': 'ml',
                'topics': topics,
                'feature_names': feature_names,
                'total_docs': len(documents)
            }
            
        except Exception as e:
            logger.error(f"ML topic extraction failed: {e}")
            return await self._extract_topics_keyword_based(documents)
    
    async def _extract_topics_keyword_based(self, documents: List[Dict]) -> Dict[str, Any]:
        """基于关键词的主题提取"""
        
        topics = {}
        
        # 统计关键词出现频率
        keyword_doc_map = defaultdict(list)
        keyword_counts = Counter()
        
        for doc in documents:
            content_text = f"{doc['metadata'].get('title', '')} {doc['content']}".lower()
            doc_keywords = set()
            
            # 检查技术关键词
            for category, keywords in self.tech_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in content_text:
                        keyword_counts[keyword] += 1
                        doc_keywords.add(keyword)
                        keyword_doc_map[keyword].append(doc)
            
            # 提取高频词汇
            words = re.findall(r'\b[a-zA-Z]{4,}\b', content_text)
            word_freq = Counter(words)
            
            # 选择频率最高的词作为潜在关键词
            for word, freq in word_freq.most_common(5):
                if freq > 1 and word not in {'this', 'that', 'with', 'from', 'they', 'have', 'been', 'will'}:
                    keyword_counts[word] += freq
                    if word not in doc_keywords:
                        doc_keywords.add(word)
                        keyword_doc_map[word].append(doc)
        
        # 基于关键词聚类形成主题
        processed_keywords = set()
        
        for keyword, count in keyword_counts.most_common(50):
            if keyword in processed_keywords or count < 2:
                continue
            
            # 寻找相关关键词
            related_keywords = [keyword]
            related_docs = keyword_doc_map[keyword].copy()
            
            for other_keyword, other_count in keyword_counts.items():
                if (other_keyword != keyword and 
                    other_keyword not in processed_keywords and
                    len(set(keyword_doc_map[keyword]) & set(keyword_doc_map[other_keyword])) > 0):
                    
                    related_keywords.append(other_keyword)
                    related_docs.extend(keyword_doc_map[other_keyword])
                    processed_keywords.add(other_keyword)
            
            processed_keywords.add(keyword)
            
            # 去重文档
            unique_docs = {}
            for doc in related_docs:
                unique_docs[doc['id']] = doc
            
            if len(unique_docs) >= 2:  # 至少2个文档才构成主题
                topic_title = self._generate_topic_title_from_keywords(related_keywords)
                topics[topic_title] = {
                    'keywords': related_keywords[:10],
                    'documents': [{'doc': doc, 'probability': 1.0} for doc in unique_docs.values()],
                    'importance': count / len(documents),
                    'topic_idx': len(topics)
                }
        
        return {
            'method': 'keyword_based',
            'topics': topics,
            'total_docs': len(documents)
        }
    
    def _generate_topic_title(self, keywords: List[str]) -> str:
        """根据关键词生成主题标题"""
        
        # 优先选择技术术语
        tech_terms = []
        for keyword in keywords[:5]:
            if any(keyword.lower() in tech_keywords for tech_keywords in self.tech_keywords.values()):
                tech_terms.append(keyword)
        
        if tech_terms:
            return ' & '.join(tech_terms[:2]).title()
        else:
            # 使用前两个最重要的关键词
            return ' & '.join(keywords[:2]).title()
    
    def _generate_topic_title_from_keywords(self, keywords: List[str]) -> str:
        """从关键词列表生成主题标题"""
        
        # 优先处理技术术语
        primary_keywords = []
        for keyword in keywords:
            if len(keyword) > 3 and keyword.lower() not in {'paper', 'model', 'method', 'system', 'approach'}:
                primary_keywords.append(keyword)
        
        if not primary_keywords:
            primary_keywords = keywords
        
        # 取前2个关键词组成标题
        title_keywords = primary_keywords[:2]
        return ' & '.join([kw.title() for kw in title_keywords])
    
    async def _score_topics(self, topics_data: Dict, documents: List[Dict]) -> List[TopicItem]:
        """计算主题重要性评分"""
        
        scored_topics = []
        total_docs = len(documents)
        
        for topic_title, topic_info in topics_data['topics'].items():
            topic_docs = topic_info['documents']
            
            if len(topic_docs) == 0:
                continue
            
            # 计算各项评分
            recency_score = self._calculate_recency_score(topic_docs)
            frequency_score = len(topic_docs) / total_docs
            diversity_score = self._calculate_diversity_score(topic_docs)
            complexity_score = self._calculate_complexity_score(topic_info['keywords'])
            relevance_score = topic_info.get('importance', 0.5)
            
            # 加权计算总分
            importance_score = (
                self.importance_weights['recency'] * recency_score +
                self.importance_weights['frequency'] * frequency_score +
                self.importance_weights['diversity'] * diversity_score +
                self.importance_weights['complexity'] * complexity_score +
                self.importance_weights['relevance'] * relevance_score
            )
            
            # 生成智能摘要
            summary = await self._generate_topic_summary(topic_title, topic_docs, topic_info['keywords'])
            
            # 提取关键论文
            key_papers = self._extract_key_papers(topic_docs)
            
            # 计算来源分布
            source_distribution = self._calculate_source_distribution(topic_docs)
            
            topic_item = TopicItem(
                title=topic_title,
                summary=summary,
                importance_score=importance_score,
                doc_count=len(topic_docs),
                key_papers=key_papers,
                related_topics=[],  # 后续填充
                document_ids=[doc_info['doc']['id'] for doc_info in topic_docs],
                keywords=topic_info['keywords'],
                source_distribution=source_distribution
            )
            
            scored_topics.append(topic_item)
        
        # 按重要性排序
        scored_topics.sort(key=lambda x: x.importance_score, reverse=True)
        
        return scored_topics
    
    def _calculate_recency_score(self, topic_docs: List[Dict]) -> float:
        """计算时间新近性评分"""
        
        if not topic_docs:
            return 0.0
        
        now = datetime.now()
        total_score = 0.0
        
        for doc_info in topic_docs:
            doc_date = doc_info['doc'].get('date')
            if doc_date:
                days_ago = (now - doc_date).days
                # 使用指数衰减函数，最近的文档得分更高
                recency = np.exp(-days_ago / 7.0)  # 7天衰减因子
                total_score += recency
        
        return total_score / len(topic_docs)
    
    def _calculate_diversity_score(self, topic_docs: List[Dict]) -> float:
        """计算来源多样性评分"""
        
        sources = set()
        for doc_info in topic_docs:
            source = doc_info['doc']['metadata'].get('source', 'unknown')
            sources.add(source)
        
        # 多样性评分：不同来源数 / 总文档数的平方根
        diversity = len(sources) / np.sqrt(len(topic_docs))
        return min(diversity, 1.0)
    
    def _calculate_complexity_score(self, keywords: List[str]) -> float:
        """计算技术复杂度评分"""
        
        # 基于关键词的技术复杂度
        tech_term_count = 0
        for keyword in keywords:
            if any(keyword.lower() in tech_list for tech_list in self.tech_keywords.values()):
                tech_term_count += 1
        
        complexity = tech_term_count / len(keywords) if keywords else 0
        return min(complexity, 1.0)
    
    async def _generate_topic_summary(self, title: str, topic_docs: List[Dict], keywords: List[str]) -> str:
        """生成主题摘要"""
        
        if self.llm_generator:
            try:
                # 准备上下文
                doc_samples = []
                for doc_info in topic_docs[:3]:  # 只取前3个文档作为样本
                    doc = doc_info['doc']
                    doc_title = doc['metadata'].get('title', 'Untitled')
                    doc_content = doc['content'][:200]  # 只取前200字符
                    doc_samples.append(f"标题: {doc_title}\n内容: {doc_content}...")
                
                context = "\n\n".join(doc_samples)
                keywords_str = ", ".join(keywords[:5])
                
                prompt = f"""请为以下技术主题生成一个简洁的摘要（50-100字）：

主题: {title}
关键词: {keywords_str}

相关文档示例:
{context}

请用中文生成摘要，突出这个主题的核心内容和技术要点："""

                summary = await self.llm_generator.generate_text(prompt, max_length=150)
                return summary.strip()
                
            except Exception as e:
                logger.warning(f"Failed to generate LLM summary for {title}: {e}")
        
        # 回退到基于关键词的简单摘要
        if keywords:
            return f"关于{title}的研究，主要涉及{', '.join(keywords[:3])}等技术领域，共发现{len(topic_docs)}篇相关文档。"
        else:
            return f"关于{title}的技术主题，包含{len(topic_docs)}篇相关研究文档。"
    
    def _extract_key_papers(self, topic_docs: List[Dict]) -> List[str]:
        """提取关键论文"""
        
        papers = []
        for doc_info in topic_docs[:5]:  # 最多5篇
            doc = doc_info['doc']
            title = doc['metadata'].get('title', 'Untitled Paper')
            papers.append(title)
        
        return papers
    
    def _calculate_source_distribution(self, topic_docs: List[Dict]) -> Dict[str, int]:
        """计算来源分布"""
        
        source_counts = Counter()
        for doc_info in topic_docs:
            source = doc_info['doc']['metadata'].get('source', 'unknown')
            source_counts[source] += 1
        
        return dict(source_counts)
    
    def _categorize_topics(self, scored_topics: List[TopicItem], max_topics: int) -> Dict[str, List[TopicItem]]:
        """将主题分类"""
        
        # 限制总主题数量
        limited_topics = scored_topics[:max_topics]
        
        # 按重要性和时间特征分类
        hot_topics = []
        trending_topics = []
        breakthroughs = []
        
        for topic in limited_topics:
            if topic.importance_score > 0.7:
                hot_topics.append(topic)
            elif topic.doc_count >= 3 and topic.importance_score > 0.5:
                trending_topics.append(topic)
            elif any('breakthrough' in kw.lower() or 'novel' in kw.lower() or 'new' in kw.lower() 
                    for kw in topic.keywords):
                breakthroughs.append(topic)
            else:
                trending_topics.append(topic)
        
        # 确保各类别有内容
        if not hot_topics and limited_topics:
            hot_topics = limited_topics[:3]
        
        if not trending_topics and len(limited_topics) > 3:
            trending_topics = limited_topics[3:8]
        
        if not breakthroughs and len(limited_topics) > 8:
            breakthroughs = limited_topics[8:]
        
        return {
            'hot': hot_topics[:5],
            'trending': trending_topics[:8],
            'breakthroughs': breakthroughs[:5]
        }
    
    def _identify_topic_relationships(self, topics: List[TopicItem]) -> Dict[str, List[str]]:
        """识别主题间关系"""
        
        relationships = {}
        
        for i, topic in enumerate(topics):
            related = []
            
            for j, other_topic in enumerate(topics):
                if i != j:
                    # 计算关键词重叠度
                    common_keywords = set(topic.keywords) & set(other_topic.keywords)
                    if len(common_keywords) >= 2:
                        related.append(other_topic.title)
                    
                    # 计算文档重叠度
                    common_docs = set(topic.document_ids) & set(other_topic.document_ids)
                    if len(common_docs) > 0:
                        if other_topic.title not in related:
                            related.append(other_topic.title)
            
            if related:
                relationships[topic.title] = related[:3]  # 最多3个相关主题
        
        return relationships
    
    def _create_empty_analysis(self, days: int) -> ContentAnalysis:
        """创建空的分析结果"""
        
        return ContentAnalysis(
            hot_topics=[],
            trending_topics=[],
            recent_breakthroughs=[],
            topic_relationships={},
            analysis_metadata={
                'total_documents_analyzed': 0,
                'analysis_duration': 0,
                'topic_extraction_method': 'none',
                'categories_found': 0
            },
            analysis_period=f"Last {days} days"
        )