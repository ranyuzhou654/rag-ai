# src/analysis/topic_extractor.py
"""
主题提取器 - 高级主题建模和分析工具
"""

import re
import asyncio
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import math
from loguru import logger

@dataclass
class TopicAnalysis:
    """主题分析结果"""
    topics: List[Dict]
    keyword_importance: Dict[str, float]
    topic_coherence: Dict[str, float]
    topic_hierarchy: Dict[str, List[str]]

class TopicExtractor:
    """高级主题提取器"""
    
    def __init__(self):
        # AI/ML领域专业词汇库
        self.domain_vocabulary = {
            'algorithms': {
                'transformer', 'attention', 'bert', 'gpt', 'lstm', 'cnn', 'rnn',
                'reinforcement learning', 'supervised learning', 'unsupervised learning',
                'gradient descent', 'backpropagation', 'optimization'
            },
            'applications': {
                'computer vision', 'natural language processing', 'nlp', 'speech recognition',
                'machine translation', 'image recognition', 'object detection', 'classification',
                'generation', 'summarization', 'question answering'
            },
            'techniques': {
                'fine-tuning', 'pre-training', 'transfer learning', 'few-shot learning',
                'zero-shot learning', 'multi-modal', 'federated learning', 'meta-learning',
                'adversarial training', 'contrastive learning'
            },
            'architectures': {
                'neural network', 'deep learning', 'convolutional', 'recurrent',
                'generative adversarial', 'autoencoder', 'variational', 'diffusion',
                'graph neural network', 'vision transformer'
            },
            'evaluation': {
                'benchmark', 'dataset', 'metric', 'evaluation', 'performance',
                'accuracy', 'precision', 'recall', 'f1-score', 'bleu', 'rouge'
            }
        }
        
        # 停用词
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
            'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'this', 'that',
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'paper', 'study', 'research',
            'method', 'approach', 'technique', 'algorithm', 'model', 'system', 'framework'
        }
        
    async def extract_topics_advanced(
        self, 
        documents: List[Dict], 
        num_topics: int = 10,
        min_topic_size: int = 2
    ) -> TopicAnalysis:
        """高级主题提取"""
        
        logger.info(f"Starting advanced topic extraction for {len(documents)} documents")
        
        # 1. 文本预处理和特征提取
        processed_docs = await self._preprocess_documents(documents)
        
        # 2. 提取关键词和短语
        keywords = await self._extract_keywords(processed_docs)
        
        # 3. 计算关键词重要性
        keyword_importance = self._calculate_keyword_importance(keywords, processed_docs)
        
        # 4. 基于关键词聚类形成主题
        topics = await self._cluster_keywords_to_topics(
            keyword_importance, processed_docs, num_topics, min_topic_size
        )
        
        # 5. 计算主题连贯性
        topic_coherence = self._calculate_topic_coherence(topics, processed_docs)
        
        # 6. 构建主题层次结构
        topic_hierarchy = self._build_topic_hierarchy(topics, keyword_importance)
        
        return TopicAnalysis(
            topics=topics,
            keyword_importance=keyword_importance,
            topic_coherence=topic_coherence,
            topic_hierarchy=topic_hierarchy
        )
    
    async def _preprocess_documents(self, documents: List[Dict]) -> List[Dict]:
        """文档预处理"""
        
        processed = []
        
        for doc in documents:
            title = doc['metadata'].get('title', '')
            content = doc['content']
            
            # 合并标题和内容
            full_text = f"{title} {content}"
            
            # 文本清理
            cleaned_text = self._clean_text(full_text)
            
            # 分词和过滤
            tokens = self._tokenize_and_filter(cleaned_text)
            
            # 提取专业术语
            technical_terms = self._extract_technical_terms(cleaned_text)
            
            processed.append({
                'id': doc['id'],
                'original_doc': doc,
                'cleaned_text': cleaned_text,
                'tokens': tokens,
                'technical_terms': technical_terms,
                'word_count': len(tokens)
            })
        
        return processed
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符，保留字母、数字和空格
        text = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', text)
        
        # 处理多个空格
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _tokenize_and_filter(self, text: str) -> List[str]:
        """分词和过滤"""
        
        # 基本分词
        words = text.split()
        
        # 过滤停用词和短词
        filtered_words = []
        for word in words:
            if (len(word) >= 3 and 
                word not in self.stopwords and
                not word.isdigit()):
                filtered_words.append(word)
        
        return filtered_words
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """提取专业术语"""
        
        technical_terms = []
        
        for category, terms in self.domain_vocabulary.items():
            for term in terms:
                if term in text:
                    technical_terms.append(term)
        
        # 提取多词技术术语
        bigrams = self._extract_bigrams(text)
        trigrams = self._extract_trigrams(text)
        
        for ngram in bigrams + trigrams:
            if self._is_technical_term(ngram):
                technical_terms.append(ngram)
        
        return list(set(technical_terms))
    
    def _extract_bigrams(self, text: str) -> List[str]:
        """提取二元短语"""
        
        words = text.split()
        bigrams = []
        
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if self._is_meaningful_phrase(bigram):
                bigrams.append(bigram)
        
        return bigrams
    
    def _extract_trigrams(self, text: str) -> List[str]:
        """提取三元短语"""
        
        words = text.split()
        trigrams = []
        
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            if self._is_meaningful_phrase(trigram):
                trigrams.append(trigram)
        
        return trigrams
    
    def _is_technical_term(self, term: str) -> bool:
        """判断是否为技术术语"""
        
        # 检查是否包含技术关键词
        tech_indicators = [
            'learning', 'network', 'model', 'algorithm', 'detection',
            'recognition', 'processing', 'generation', 'optimization',
            'training', 'neural', 'deep', 'machine', 'artificial'
        ]
        
        return any(indicator in term for indicator in tech_indicators)
    
    def _is_meaningful_phrase(self, phrase: str) -> bool:
        """判断短语是否有意义"""
        
        words = phrase.split()
        
        # 过滤包含停用词的短语
        if any(word in self.stopwords for word in words):
            return False
        
        # 确保所有词都足够长
        if any(len(word) < 3 for word in words):
            return False
        
        return True
    
    async def _extract_keywords(self, processed_docs: List[Dict]) -> Dict[str, List[str]]:
        """提取关键词"""
        
        keywords = {
            'single_words': [],
            'technical_terms': [],
            'phrases': []
        }
        
        for doc in processed_docs:
            # 单词关键词
            keywords['single_words'].extend(doc['tokens'])
            
            # 技术术语
            keywords['technical_terms'].extend(doc['technical_terms'])
            
            # 从清理后的文本中提取短语
            phrases = self._extract_bigrams(doc['cleaned_text'])
            keywords['phrases'].extend(phrases)
        
        return keywords
    
    def _calculate_keyword_importance(
        self, 
        keywords: Dict[str, List[str]], 
        processed_docs: List[Dict]
    ) -> Dict[str, float]:
        """计算关键词重要性"""
        
        importance_scores = {}
        total_docs = len(processed_docs)
        
        # 合并所有关键词
        all_keywords = []
        for keyword_list in keywords.values():
            all_keywords.extend(keyword_list)
        
        # 计算词频
        keyword_freq = Counter(all_keywords)
        
        # 计算文档频率
        keyword_doc_freq = defaultdict(int)
        for doc in processed_docs:
            doc_text = doc['cleaned_text']
            doc_keywords = set()
            
            for keyword in keyword_freq.keys():
                if keyword in doc_text:
                    doc_keywords.add(keyword)
            
            for keyword in doc_keywords:
                keyword_doc_freq[keyword] += 1
        
        # 计算TF-IDF分数
        for keyword, tf in keyword_freq.items():
            df = keyword_doc_freq[keyword]
            if df > 0:
                # TF-IDF计算
                tf_score = tf / sum(keyword_freq.values())
                idf_score = math.log(total_docs / df)
                tfidf_score = tf_score * idf_score
                
                # 技术术语加权
                if keyword in sum(self.domain_vocabulary.values(), set()):
                    tfidf_score *= 2.0
                
                # 短语加权
                if ' ' in keyword:
                    tfidf_score *= 1.5
                
                importance_scores[keyword] = tfidf_score
        
        return importance_scores
    
    async def _cluster_keywords_to_topics(
        self,
        keyword_importance: Dict[str, float],
        processed_docs: List[Dict],
        num_topics: int,
        min_topic_size: int
    ) -> List[Dict]:
        """将关键词聚类成主题"""
        
        # 选择最重要的关键词作为种子
        top_keywords = sorted(
            keyword_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:num_topics * 5]  # 取更多关键词用于聚类
        
        # 基于共现关系聚类
        topics = []
        used_keywords = set()
        
        for seed_keyword, seed_score in top_keywords:
            if seed_keyword in used_keywords:
                continue
            
            # 找到与种子关键词共现的关键词
            related_keywords = self._find_cooccurring_keywords(
                seed_keyword, keyword_importance, processed_docs
            )
            
            # 过滤已使用的关键词
            related_keywords = [kw for kw in related_keywords if kw not in used_keywords]
            
            if len(related_keywords) >= min_topic_size:
                # 构建主题
                topic_keywords = [seed_keyword] + related_keywords[:9]  # 最多10个关键词
                topic_title = self._generate_topic_title(topic_keywords)
                
                # 找到属于这个主题的文档
                topic_docs = self._find_topic_documents(topic_keywords, processed_docs)
                
                topic = {
                    'title': topic_title,
                    'keywords': topic_keywords,
                    'documents': topic_docs,
                    'size': len(topic_docs),
                    'coherence_score': 0.0  # 后续计算
                }
                
                topics.append(topic)
                used_keywords.update(topic_keywords)
        
        # 限制主题数量
        topics = sorted(topics, key=lambda x: x['size'], reverse=True)[:num_topics]
        
        return topics
    
    def _find_cooccurring_keywords(
        self,
        seed_keyword: str,
        keyword_importance: Dict[str, float],
        processed_docs: List[Dict]
    ) -> List[str]:
        """找到与种子关键词共现的关键词"""
        
        cooccurrence_counts = Counter()
        
        # 在包含种子关键词的文档中统计其他关键词
        for doc in processed_docs:
            if seed_keyword in doc['cleaned_text']:
                for keyword in keyword_importance.keys():
                    if keyword != seed_keyword and keyword in doc['cleaned_text']:
                        cooccurrence_counts[keyword] += 1
        
        # 按共现频率排序
        cooccurring_keywords = [
            keyword for keyword, count in cooccurrence_counts.most_common(15)
            if count >= 2  # 至少在2个文档中共现
        ]
        
        return cooccurring_keywords
    
    def _generate_topic_title(self, keywords: List[str]) -> str:
        """生成主题标题"""
        
        # 优先选择技术术语
        tech_terms = []
        for keyword in keywords:
            if (keyword in sum(self.domain_vocabulary.values(), set()) or
                ' ' in keyword):  # 短语通常更有描述性
                tech_terms.append(keyword)
        
        if tech_terms:
            # 选择最具代表性的1-2个术语
            selected_terms = tech_terms[:2]
            return ' & '.join([term.title() for term in selected_terms])
        else:
            # 使用前两个最重要的单词
            return ' & '.join([kw.title() for kw in keywords[:2]])
    
    def _find_topic_documents(
        self, 
        topic_keywords: List[str], 
        processed_docs: List[Dict]
    ) -> List[Dict]:
        """找到属于主题的文档"""
        
        topic_docs = []
        
        for doc in processed_docs:
            # 计算文档与主题的匹配度
            match_count = 0
            for keyword in topic_keywords:
                if keyword in doc['cleaned_text']:
                    match_count += 1
            
            # 如果匹配度足够高，则归入该主题
            match_ratio = match_count / len(topic_keywords)
            if match_ratio >= 0.3:  # 至少匹配30%的关键词
                topic_docs.append({
                    'doc': doc['original_doc'],
                    'match_ratio': match_ratio,
                    'matched_keywords': [
                        kw for kw in topic_keywords 
                        if kw in doc['cleaned_text']
                    ]
                })
        
        # 按匹配度排序
        topic_docs.sort(key=lambda x: x['match_ratio'], reverse=True)
        
        return topic_docs
    
    def _calculate_topic_coherence(
        self, 
        topics: List[Dict], 
        processed_docs: List[Dict]
    ) -> Dict[str, float]:
        """计算主题连贯性"""
        
        coherence_scores = {}
        
        for topic in topics:
            keywords = topic['keywords']
            
            # 计算关键词间的平均共现率
            cooccurrence_sum = 0
            pair_count = 0
            
            for i, kw1 in enumerate(keywords):
                for kw2 in keywords[i+1:]:
                    # 计算两个关键词的共现文档数
                    cooccur_docs = 0
                    for doc in processed_docs:
                        if kw1 in doc['cleaned_text'] and kw2 in doc['cleaned_text']:
                            cooccur_docs += 1
                    
                    if cooccur_docs > 0:
                        cooccurrence_sum += cooccur_docs
                    pair_count += 1
            
            # 计算平均连贯性
            if pair_count > 0:
                coherence = cooccurrence_sum / (pair_count * len(processed_docs))
            else:
                coherence = 0.0
            
            coherence_scores[topic['title']] = coherence
            topic['coherence_score'] = coherence
        
        return coherence_scores
    
    def _build_topic_hierarchy(
        self, 
        topics: List[Dict], 
        keyword_importance: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """构建主题层次结构"""
        
        hierarchy = {}
        
        for topic in topics:
            related_topics = []
            
            for other_topic in topics:
                if topic['title'] != other_topic['title']:
                    # 计算主题间的关键词重叠
                    common_keywords = set(topic['keywords']) & set(other_topic['keywords'])
                    
                    if len(common_keywords) >= 2:  # 至少2个共同关键词
                        related_topics.append(other_topic['title'])
            
            if related_topics:
                hierarchy[topic['title']] = related_topics
        
        return hierarchy