# src/retrieval/advanced_reranking.py
"""
高级重排序系统
集成Cohere Rerank 3、NVIDIA NeMo和自定义重排序策略
实现多信号融合和上下文感知重排序
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
import time
import numpy as np
from loguru import logger

# LangChain组件
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.document_transformers import LongContextReorder

# 第三方重排序模型
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    logger.warning("Cohere not available, using fallback rerankers")

# 本地组件
from .hybrid_retriever import EnhancedDocument, RetrievalMetadata


@dataclass
class RerankingSignal:
    """重排序信号"""
    name: str
    score: float
    weight: float
    metadata: Dict[str, Any]


@dataclass
class RerankingResult:
    """重排序结果"""
    document: EnhancedDocument
    original_rank: int
    new_rank: int
    combined_score: float
    individual_signals: List[RerankingSignal]
    improvement_score: float  # 排名提升分数


@dataclass
class RerankingMetrics:
    """重排序指标"""
    total_documents: int
    processing_time: float
    reranker_used: str
    avg_score_change: float
    top_k_stability: float  # 前K个结果的稳定性
    diversity_score: float   # 结果多样性


class BaseReranker(ABC):
    """重排序器基类"""
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[EnhancedDocument],
        top_k: Optional[int] = None
    ) -> List[RerankingResult]:
        """重排序文档"""
        pass
    
    @abstractmethod
    def get_reranker_name(self) -> str:
        """获取重排序器名称"""
        pass


class CohereReranker(BaseReranker):
    """Cohere重排序器"""
    
    def __init__(self, api_key: str, model: str = "rerank-multilingual-v3.0", top_n: int = 100):
        if not COHERE_AVAILABLE:
            raise ImportError("Cohere package not available")
        
        self.client = cohere.Client(api_key)
        self.model = model
        self.top_n = top_n
        logger.info(f"Cohere reranker initialized with model: {model}")
    
    async def rerank(
        self,
        query: str,
        documents: List[EnhancedDocument],
        top_k: Optional[int] = None
    ) -> List[RerankingResult]:
        """使用Cohere重排序"""
        
        if not documents:
            return []
        
        start_time = time.time()
        
        try:
            # 准备文档文本
            document_texts = [doc.page_content for doc in documents]
            
            # 调用Cohere API
            # 注意：Cohere API通常是同步的，在线程池中执行
            def _sync_rerank():
                return self.client.rerank(
                    model=self.model,
                    query=query,
                    documents=document_texts,
                    top_n=min(self.top_n, len(documents))
                )
            
            loop = asyncio.get_event_loop()
            rerank_response = await loop.run_in_executor(None, _sync_rerank)
            
            # 构建重排序结果
            results = []
            for new_rank, result in enumerate(rerank_response.results):
                original_doc = documents[result.index]
                
                # 创建重排序信号
                cohere_signal = RerankingSignal(
                    name="cohere_relevance",
                    score=float(result.relevance_score),
                    weight=1.0,
                    metadata={"model": self.model}
                )
                
                reranking_result = RerankingResult(
                    document=original_doc,
                    original_rank=result.index,
                    new_rank=new_rank,
                    combined_score=float(result.relevance_score),
                    individual_signals=[cohere_signal],
                    improvement_score=result.index - new_rank  # 正值表示排名提升
                )
                results.append(reranking_result)
            
            processing_time = time.time() - start_time
            logger.debug(f"Cohere reranking completed in {processing_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            # 返回原始排序
            return self._create_fallback_results(documents)
    
    def get_reranker_name(self) -> str:
        return f"cohere_{self.model}"
    
    def _create_fallback_results(self, documents: List[EnhancedDocument]) -> List[RerankingResult]:
        """创建备用结果（保持原始顺序）"""
        results = []
        for i, doc in enumerate(documents):
            fallback_signal = RerankingSignal(
                name="original_order",
                score=1.0 - (i / len(documents)),
                weight=1.0,
                metadata={"fallback": True}
            )
            
            result = RerankingResult(
                document=doc,
                original_rank=i,
                new_rank=i,
                combined_score=fallback_signal.score,
                individual_signals=[fallback_signal],
                improvement_score=0.0
            )
            results.append(result)
        
        return results


class MultiSignalReranker(BaseReranker):
    """多信号重排序器 - 融合多种排序信号"""
    
    def __init__(self, embeddings: Optional[Embeddings] = None, config: Dict[str, Any] = None):
        self.embeddings = embeddings
        self.config = config or {}
        
        # 信号权重配置（基于improvements.md的建议）
        self.signal_weights = {
            'semantic_similarity': self.config.get('semantic_weight', 0.4),
            'keyword_matching': self.config.get('keyword_weight', 0.2),
            'chunk_quality': self.config.get('quality_weight', 0.15),
            'source_authority': self.config.get('authority_weight', 0.15),
            'user_feedback': self.config.get('feedback_weight', 0.05),
            'recency': self.config.get('recency_weight', 0.05)
        }
        
        # 归一化权重
        total_weight = sum(self.signal_weights.values())
        if total_weight > 0:
            self.signal_weights = {k: v / total_weight for k, v in self.signal_weights.items()}
        
        logger.info(f"Multi-signal reranker initialized with weights: {self.signal_weights}")
    
    async def rerank(
        self,
        query: str,
        documents: List[EnhancedDocument],
        top_k: Optional[int] = None
    ) -> List[RerankingResult]:
        """多信号重排序"""
        
        if not documents:
            return []
        
        start_time = time.time()
        
        # 并行计算所有信号
        tasks = []
        
        # 1. 语义相似度
        if self.embeddings and self.signal_weights.get('semantic_similarity', 0) > 0:
            tasks.append(self._compute_semantic_similarity(query, documents))
        
        # 2. 关键词匹配
        if self.signal_weights.get('keyword_matching', 0) > 0:
            tasks.append(self._compute_keyword_matching(query, documents))
        
        # 3. 块质量评估
        if self.signal_weights.get('chunk_quality', 0) > 0:
            tasks.append(self._compute_chunk_quality(documents))
        
        # 4. 来源权威性
        if self.signal_weights.get('source_authority', 0) > 0:
            tasks.append(self._compute_source_authority(documents))
        
        # 5. 用户反馈
        if self.signal_weights.get('user_feedback', 0) > 0:
            tasks.append(self._compute_user_feedback(documents))
        
        # 6. 时效性
        if self.signal_weights.get('recency', 0) > 0:
            tasks.append(self._compute_recency(documents))
        
        # 并行执行信号计算
        signal_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集有效信号
        all_signals = {}
        for result in signal_results:
            if isinstance(result, dict):
                all_signals.update(result)
        
        # 融合信号并重排序
        reranked_results = self._fuse_signals_and_rerank(documents, all_signals)
        
        # 应用top_k限制
        if top_k:
            reranked_results = reranked_results[:top_k]
        
        processing_time = time.time() - start_time
        logger.debug(f"Multi-signal reranking completed in {processing_time:.3f}s")
        
        return reranked_results
    
    async def _compute_semantic_similarity(self, query: str, documents: List[EnhancedDocument]) -> Dict[str, List[RerankingSignal]]:
        """计算语义相似度信号"""
        try:
            # 获取查询嵌入
            query_embedding = await self.embeddings.aembed_query(query)
            query_vec = np.array(query_embedding)
            
            # 计算文档嵌入（如果没有缓存）
            doc_embeddings = []
            for doc in documents:
                if hasattr(doc, 'embedding') and doc.embedding is not None:
                    doc_embeddings.append(doc.embedding)
                else:
                    doc_embedding = await self.embeddings.aembed_query(doc.page_content)
                    doc_embeddings.append(np.array(doc_embedding))
            
            # 计算相似度
            signals = {}
            for i, doc_vec in enumerate(doc_embeddings):
                similarity = float(np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)))
                
                signal = RerankingSignal(
                    name="semantic_similarity",
                    score=max(0, similarity),  # 确保非负
                    weight=self.signal_weights['semantic_similarity'],
                    metadata={"embedding_dim": len(query_vec)}
                )
                signals[f"doc_{i}"] = [signal]
            
            return signals
            
        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")
            return {}
    
    async def _compute_keyword_matching(self, query: str, documents: List[EnhancedDocument]) -> Dict[str, List[RerankingSignal]]:
        """计算关键词匹配信号"""
        try:
            # 简单的关键词匹配逻辑
            query_terms = set(query.lower().split())
            
            signals = {}
            for i, doc in enumerate(documents):
                doc_terms = set(doc.page_content.lower().split())
                
                # 计算Jaccard相似度
                intersection = len(query_terms.intersection(doc_terms))
                union = len(query_terms.union(doc_terms))
                jaccard_score = intersection / union if union > 0 else 0
                
                # 计算词频匹配
                term_matches = sum(1 for term in query_terms if term in doc_terms)
                coverage_score = term_matches / len(query_terms) if query_terms else 0
                
                # 综合分数
                combined_score = (jaccard_score + coverage_score) / 2
                
                signal = RerankingSignal(
                    name="keyword_matching",
                    score=combined_score,
                    weight=self.signal_weights['keyword_matching'],
                    metadata={
                        "jaccard_score": jaccard_score,
                        "coverage_score": coverage_score,
                        "matched_terms": intersection
                    }
                )
                signals[f"doc_{i}"] = [signal]
            
            return signals
            
        except Exception as e:
            logger.warning(f"Keyword matching computation failed: {e}")
            return {}
    
    async def _compute_chunk_quality(self, documents: List[EnhancedDocument]) -> Dict[str, List[RerankingSignal]]:
        """计算块质量信号"""
        try:
            signals = {}
            for i, doc in enumerate(documents):
                # 使用现有的质量评估逻辑
                if doc.retrieval_metadata:
                    quality_score = doc.retrieval_metadata.chunk_quality_score
                else:
                    # 简单质量评估
                    quality_score = self._assess_chunk_quality(doc.page_content)
                
                signal = RerankingSignal(
                    name="chunk_quality",
                    score=quality_score,
                    weight=self.signal_weights['chunk_quality'],
                    metadata={"content_length": len(doc.page_content)}
                )
                signals[f"doc_{i}"] = [signal]
            
            return signals
            
        except Exception as e:
            logger.warning(f"Chunk quality computation failed: {e}")
            return {}
    
    async def _compute_source_authority(self, documents: List[EnhancedDocument]) -> Dict[str, List[RerankingSignal]]:
        """计算来源权威性信号"""
        try:
            signals = {}
            for i, doc in enumerate(documents):
                # 使用现有的权威性评估逻辑
                if doc.retrieval_metadata:
                    authority_score = doc.retrieval_metadata.source_authority
                else:
                    # 简单权威性评估
                    authority_score = self._assess_source_authority(doc.metadata)
                
                signal = RerankingSignal(
                    name="source_authority",
                    score=authority_score,
                    weight=self.signal_weights['source_authority'],
                    metadata={"source_type": doc.metadata.get('source_type', 'unknown')}
                )
                signals[f"doc_{i}"] = [signal]
            
            return signals
            
        except Exception as e:
            logger.warning(f"Source authority computation failed: {e}")
            return {}
    
    async def _compute_user_feedback(self, documents: List[EnhancedDocument]) -> Dict[str, List[RerankingSignal]]:
        """计算用户反馈信号"""
        try:
            # 这里可以集成实际的用户反馈数据
            signals = {}
            for i, doc in enumerate(documents):
                # 从文档元数据中获取用户反馈
                feedback_score = doc.metadata.get('user_rating', 0.5)  # 默认中性评分
                thumbs_up = doc.metadata.get('thumbs_up', 0)
                thumbs_down = doc.metadata.get('thumbs_down', 0)
                
                # 计算综合反馈分数
                if thumbs_up + thumbs_down > 0:
                    feedback_ratio = thumbs_up / (thumbs_up + thumbs_down)
                else:
                    feedback_ratio = 0.5
                
                combined_feedback = (feedback_score + feedback_ratio) / 2
                
                signal = RerankingSignal(
                    name="user_feedback",
                    score=combined_feedback,
                    weight=self.signal_weights['user_feedback'],
                    metadata={
                        "thumbs_up": thumbs_up,
                        "thumbs_down": thumbs_down,
                        "rating": feedback_score
                    }
                )
                signals[f"doc_{i}"] = [signal]
            
            return signals
            
        except Exception as e:
            logger.warning(f"User feedback computation failed: {e}")
            return {}
    
    async def _compute_recency(self, documents: List[EnhancedDocument]) -> Dict[str, List[RerankingSignal]]:
        """计算时效性信号"""
        try:
            from datetime import datetime, timedelta
            
            signals = {}
            now = datetime.now()
            
            for i, doc in enumerate(documents):
                # 获取文档时间
                doc_time_str = doc.metadata.get('created_time') or doc.metadata.get('loaded_at')
                
                if doc_time_str:
                    try:
                        doc_time = datetime.fromisoformat(doc_time_str.replace('Z', '+00:00'))
                        time_diff = now - doc_time
                        
                        # 计算新鲜度分数（30天内为1.0，线性衰减）
                        days_old = time_diff.days
                        if days_old <= 30:
                            recency_score = 1.0 - (days_old / 30 * 0.5)  # 最多减少50%
                        else:
                            recency_score = 0.5  # 超过30天的基础分数
                    except:
                        recency_score = 0.5  # 时间解析失败
                else:
                    recency_score = 0.5  # 没有时间信息
                
                signal = RerankingSignal(
                    name="recency",
                    score=recency_score,
                    weight=self.signal_weights['recency'],
                    metadata={"time_info": doc_time_str or "unknown"}
                )
                signals[f"doc_{i}"] = [signal]
            
            return signals
            
        except Exception as e:
            logger.warning(f"Recency computation failed: {e}")
            return {}
    
    def _fuse_signals_and_rerank(
        self,
        documents: List[EnhancedDocument],
        all_signals: Dict[str, List[RerankingSignal]]
    ) -> List[RerankingResult]:
        """融合信号并重排序"""
        
        results = []
        
        for i, doc in enumerate(documents):
            doc_key = f"doc_{i}"
            doc_signals = all_signals.get(doc_key, [])
            
            # 计算加权综合分数
            combined_score = 0.0
            for signal in doc_signals:
                combined_score += signal.score * signal.weight
            
            # 创建重排序结果
            result = RerankingResult(
                document=doc,
                original_rank=i,
                new_rank=i,  # 稍后会更新
                combined_score=combined_score,
                individual_signals=doc_signals,
                improvement_score=0.0  # 稍后会更新
            )
            results.append(result)
        
        # 按综合分数排序
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # 更新新排名和改进分数
        for new_rank, result in enumerate(results):
            result.new_rank = new_rank
            result.improvement_score = result.original_rank - new_rank
        
        return results
    
    def _assess_chunk_quality(self, content: str) -> float:
        """评估块质量（简化版）"""
        quality = 0.5
        
        # 长度评估
        if 100 <= len(content) <= 1000:
            quality += 0.2
        
        # 结构化程度
        if any(punct in content for punct in ['。', '，', '：', '；', '.', ',']):
            quality += 0.2
        
        # 专业词汇密度
        technical_terms = ['算法', '模型', '网络', '学习', '训练', '数据', '系统']
        if any(term in content for term in technical_terms):
            quality += 0.1
        
        return min(quality, 1.0)
    
    def _assess_source_authority(self, metadata: Dict) -> float:
        """评估来源权威性（简化版）"""
        authority = 0.5
        
        source_type = metadata.get('source_type', '')
        if source_type == 'academic_paper':
            authority += 0.3
        elif source_type == 'official_doc':
            authority += 0.2
        elif source_type == 'blog':
            authority += 0.1
        
        return min(authority, 1.0)
    
    def get_reranker_name(self) -> str:
        return "multi_signal"


class ContextualReorderReranker(BaseReranker):
    """上下文重排序器 - 实现最优上下文排列"""
    
    def __init__(self):
        self.reorderer = LongContextReorder()
        logger.info("Contextual reorder reranker initialized")
    
    async def rerank(
        self,
        query: str,
        documents: List[EnhancedDocument],
        top_k: Optional[int] = None
    ) -> List[RerankingResult]:
        """上下文感知重排序"""
        
        if not documents:
            return []
        
        try:
            # 转换为标准Document格式
            standard_docs = [
                Document(page_content=doc.page_content, metadata=doc.metadata)
                for doc in documents
            ]
            
            # 执行重排序
            def _sync_reorder():
                return self.reorderer.transform_documents(standard_docs)
            
            loop = asyncio.get_event_loop()
            reordered_docs = await loop.run_in_executor(None, _sync_reorder)
            
            # 构建重排序结果
            results = []
            for new_rank, reordered_doc in enumerate(reordered_docs):
                # 找到原始位置
                original_rank = 0
                original_enhanced_doc = documents[0]
                
                for i, orig_doc in enumerate(documents):
                    if orig_doc.page_content == reordered_doc.page_content:
                        original_rank = i
                        original_enhanced_doc = orig_doc
                        break
                
                signal = RerankingSignal(
                    name="contextual_reorder",
                    score=1.0 - (new_rank / len(documents)),
                    weight=1.0,
                    metadata={"reorder_strategy": "lost_in_middle"}
                )
                
                result = RerankingResult(
                    document=original_enhanced_doc,
                    original_rank=original_rank,
                    new_rank=new_rank,
                    combined_score=signal.score,
                    individual_signals=[signal],
                    improvement_score=original_rank - new_rank
                )
                results.append(result)
            
            if top_k:
                results = results[:top_k]
            
            return results
            
        except Exception as e:
            logger.error(f"Contextual reordering failed: {e}")
            # 返回原始顺序
            return [
                RerankingResult(
                    document=doc,
                    original_rank=i,
                    new_rank=i,
                    combined_score=1.0 - (i / len(documents)),
                    individual_signals=[],
                    improvement_score=0.0
                )
                for i, doc in enumerate(documents)
            ]
    
    def get_reranker_name(self) -> str:
        return "contextual_reorder"


class AdvancedRerankingOrchestrator:
    """高级重排序协调器 - 协调多个重排序策略"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rerankers = {}
        
        # 初始化可用的重排序器
        self._initialize_rerankers()
        
        # 默认策略
        self.default_strategy = config.get('default_reranking_strategy', 'multi_signal')
        
        logger.info(f"Advanced reranking orchestrator initialized with {len(self.rerankers)} rerankers")
    
    def _initialize_rerankers(self):
        """初始化重排序器"""
        
        # 1. Cohere重排序器
        if self.config.get('cohere_api_key') and COHERE_AVAILABLE:
            try:
                self.rerankers['cohere'] = CohereReranker(
                    api_key=self.config['cohere_api_key'],
                    model=self.config.get('cohere_model', 'rerank-multilingual-v3.0')
                )
                logger.info("✅ Cohere reranker initialized")
            except Exception as e:
                logger.warning(f"Cohere reranker initialization failed: {e}")
        
        # 2. 多信号重排序器
        try:
            embeddings = self.config.get('embeddings_model')
            self.rerankers['multi_signal'] = MultiSignalReranker(embeddings, self.config)
            logger.info("✅ Multi-signal reranker initialized")
        except Exception as e:
            logger.warning(f"Multi-signal reranker initialization failed: {e}")
        
        # 3. 上下文重排序器
        try:
            self.rerankers['contextual_reorder'] = ContextualReorderReranker()
            logger.info("✅ Contextual reorder reranker initialized")
        except Exception as e:
            logger.warning(f"Contextual reorder reranker initialization failed: {e}")
    
    async def rerank(
        self,
        query: str,
        documents: List[EnhancedDocument],
        strategy: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Tuple[List[RerankingResult], RerankingMetrics]:
        """执行重排序"""
        
        if not documents:
            return [], RerankingMetrics(
                total_documents=0,
                processing_time=0.0,
                reranker_used="none",
                avg_score_change=0.0,
                top_k_stability=1.0,
                diversity_score=0.0
            )
        
        start_time = time.time()
        selected_strategy = strategy or self.default_strategy
        
        # 选择重排序器
        if selected_strategy not in self.rerankers:
            logger.warning(f"Strategy {selected_strategy} not available, using multi_signal")
            selected_strategy = 'multi_signal'
        
        reranker = self.rerankers.get(selected_strategy)
        if not reranker:
            # 创建默认结果
            results = [
                RerankingResult(
                    document=doc,
                    original_rank=i,
                    new_rank=i,
                    combined_score=1.0 - (i / len(documents)),
                    individual_signals=[],
                    improvement_score=0.0
                )
                for i, doc in enumerate(documents)
            ]
            processing_time = time.time() - start_time
            metrics = self._calculate_metrics(results, processing_time, "none")
            return results, metrics
        
        # 执行重排序
        try:
            results = await reranker.rerank(query, documents, top_k)
            processing_time = time.time() - start_time
            
            # 计算指标
            metrics = self._calculate_metrics(results, processing_time, reranker.get_reranker_name())
            
            logger.info(f"Reranking completed: {len(documents)} docs reranked in {processing_time:.3f}s "
                       f"using {reranker.get_reranker_name()}")
            
            return results, metrics
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            
            # 返回原始顺序
            results = [
                RerankingResult(
                    document=doc,
                    original_rank=i,
                    new_rank=i,
                    combined_score=1.0 - (i / len(documents)),
                    individual_signals=[],
                    improvement_score=0.0
                )
                for i, doc in enumerate(documents)
            ]
            processing_time = time.time() - start_time
            metrics = self._calculate_metrics(results, processing_time, "fallback")
            
            return results, metrics
    
    def _calculate_metrics(self, results: List[RerankingResult], processing_time: float, reranker_name: str) -> RerankingMetrics:
        """计算重排序指标"""
        
        if not results:
            return RerankingMetrics(
                total_documents=0,
                processing_time=processing_time,
                reranker_used=reranker_name,
                avg_score_change=0.0,
                top_k_stability=1.0,
                diversity_score=0.0
            )
        
        # 平均分数变化
        score_changes = [result.improvement_score for result in results]
        avg_score_change = sum(score_changes) / len(score_changes)
        
        # Top-K稳定性（前5个结果中有多少保持在前5）
        top_k = min(5, len(results))
        original_top_k = set(range(top_k))
        new_top_k = set(result.original_rank for result in results[:top_k])
        top_k_stability = len(original_top_k.intersection(new_top_k)) / top_k
        
        # 多样性分数（简化版：不同来源的占比）
        sources = set()
        for result in results[:10]:  # 前10个结果
            source = result.document.metadata.get('source', 'unknown')
            sources.add(source)
        diversity_score = len(sources) / min(10, len(results))
        
        return RerankingMetrics(
            total_documents=len(results),
            processing_time=processing_time,
            reranker_used=reranker_name,
            avg_score_change=avg_score_change,
            top_k_stability=top_k_stability,
            diversity_score=diversity_score
        )
    
    def get_available_strategies(self) -> List[str]:
        """获取可用的重排序策略"""
        return list(self.rerankers.keys())
    
    def get_strategy_info(self, strategy: str) -> Dict[str, Any]:
        """获取策略信息"""
        reranker = self.rerankers.get(strategy)
        if not reranker:
            return {"available": False, "name": strategy}
        
        return {
            "available": True,
            "name": reranker.get_reranker_name(),
            "type": type(reranker).__name__
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置并重新初始化"""
        self.config.update(new_config)
        self._initialize_rerankers()
        self.default_strategy = self.config.get('default_reranking_strategy', 'multi_signal')
        
        logger.info("Reranking orchestrator config updated")