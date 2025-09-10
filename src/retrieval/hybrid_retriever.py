# src/retrieval/hybrid_retriever.py
"""
混合检索器 - 整合Dense Vector、Sparse Vector和Knowledge Graph检索
实现2024-2025年最新的RAG检索优化技术
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio
from loguru import logger

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

# 本地组件导入
from .vector_database import VectorDatabaseManager
from src.knowledge_graph.kg_retriever import KnowledgeGraphRetriever


@dataclass
class RetrievalMetadata:
    """检索元数据，用于追踪检索来源和质量"""
    retrieval_method: str  # 'vector', 'bm25', 'knowledge_graph', 'hybrid'
    confidence_score: float
    chunk_quality_score: float
    source_authority: float
    retrieval_time: float
    kg_entities: Optional[List[str]] = None
    kg_relations: Optional[List[Dict]] = None


class EnhancedDocument(Document):
    """增强的文档类，包含检索元数据"""
    
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None, retrieval_metadata: RetrievalMetadata = None):
        super().__init__(page_content=page_content, metadata=metadata or {})
        self.retrieval_metadata = retrieval_metadata


class DenseVectorRetriever(BaseRetriever):
    """Dense Vector检索器 - 基于语义相似度"""
    
    def __init__(self, vector_store: QdrantVectorStore, k: int = 10):
        super().__init__()
        self.vector_store = vector_store
        self.k = k
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """同步检索文档"""
        return asyncio.run(self._aget_relevant_documents(query))
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步检索文档"""
        import time
        start_time = time.time()
        
        try:
            # 执行向量相似度搜索
            results = await self.vector_store.asimilarity_search_with_score(
                query, k=self.k
            )
            
            retrieval_time = time.time() - start_time
            
            enhanced_docs = []
            for doc, score in results:
                # 创建检索元数据
                metadata = RetrievalMetadata(
                    retrieval_method='vector',
                    confidence_score=float(score),
                    chunk_quality_score=self._assess_chunk_quality(doc.page_content),
                    source_authority=self._assess_source_authority(doc.metadata),
                    retrieval_time=retrieval_time
                )
                
                # 创建增强文档
                enhanced_doc = EnhancedDocument(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, 'vector_score': score},
                    retrieval_metadata=metadata
                )
                enhanced_docs.append(enhanced_doc)
            
            logger.debug(f"Dense vector retrieval: {len(enhanced_docs)} docs in {retrieval_time:.3f}s")
            return enhanced_docs
            
        except Exception as e:
            logger.error(f"Dense vector retrieval failed: {e}")
            return []
    
    def _assess_chunk_quality(self, content: str) -> float:
        """评估文档块质量"""
        # 简单的质量评估逻辑
        quality = 0.5  # 基础分
        
        # 长度评估
        if 100 <= len(content) <= 1000:
            quality += 0.2
        
        # 结构化程度（包含标点符号等）
        if any(punct in content for punct in ['。', '，', '：', '；']):
            quality += 0.2
        
        # 专业词汇密度（简化版）
        technical_terms = ['算法', '模型', '网络', '学习', '训练', '数据']
        if any(term in content for term in technical_terms):
            quality += 0.1
        
        return min(quality, 1.0)
    
    def _assess_source_authority(self, metadata: Dict) -> float:
        """评估来源权威性"""
        authority = 0.5  # 基础分
        
        source_type = metadata.get('source_type', '')
        if source_type == 'academic_paper':
            authority += 0.3
        elif source_type == 'official_doc':
            authority += 0.2
        elif source_type == 'blog':
            authority += 0.1
        
        return min(authority, 1.0)


class SparseVectorRetriever(BaseRetriever):
    """Sparse Vector检索器 - 基于关键词匹配(BM25)"""
    
    def __init__(self, documents: List[Document], k: int = 10):
        super().__init__()
        self.k = k
        self.documents = documents
        
        # 构建BM25索引
        corpus = [doc.page_content for doc in documents]
        self.bm25 = BM25Okapi([text.split() for text in corpus])
        
        logger.info(f"BM25 index built with {len(documents)} documents")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """同步检索文档"""
        return asyncio.run(self._aget_relevant_documents(query))
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步检索文档"""
        import time
        start_time = time.time()
        
        try:
            # BM25评分
            query_tokens = query.split()
            scores = self.bm25.get_scores(query_tokens)
            
            # 获取top-k结果
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.k]
            
            retrieval_time = time.time() - start_time
            
            enhanced_docs = []
            for idx in top_indices:
                if scores[idx] > 0:  # 只返回有相关性的结果
                    doc = self.documents[idx]
                    
                    # 创建检索元数据
                    metadata = RetrievalMetadata(
                        retrieval_method='bm25',
                        confidence_score=float(scores[idx]),
                        chunk_quality_score=self._assess_chunk_quality(doc.page_content),
                        source_authority=self._assess_source_authority(doc.metadata),
                        retrieval_time=retrieval_time
                    )
                    
                    # 创建增强文档
                    enhanced_doc = EnhancedDocument(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, 'bm25_score': scores[idx]},
                        retrieval_metadata=metadata
                    )
                    enhanced_docs.append(enhanced_doc)
            
            logger.debug(f"BM25 retrieval: {len(enhanced_docs)} docs in {retrieval_time:.3f}s")
            return enhanced_docs
            
        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
            return []
    
    def _assess_chunk_quality(self, content: str) -> float:
        """评估文档块质量（与DenseVectorRetriever相同）"""
        quality = 0.5
        if 100 <= len(content) <= 1000:
            quality += 0.2
        if any(punct in content for punct in ['。', '，', '：', '；']):
            quality += 0.2
        technical_terms = ['算法', '模型', '网络', '学习', '训练', '数据']
        if any(term in content for term in technical_terms):
            quality += 0.1
        return min(quality, 1.0)
    
    def _assess_source_authority(self, metadata: Dict) -> float:
        """评估来源权威性（与DenseVectorRetriever相同）"""
        authority = 0.5
        source_type = metadata.get('source_type', '')
        if source_type == 'academic_paper':
            authority += 0.3
        elif source_type == 'official_doc':
            authority += 0.2
        elif source_type == 'blog':
            authority += 0.1
        return min(authority, 1.0)


class KnowledgeGraphRetrieverWrapper(BaseRetriever):
    """知识图谱检索器包装器"""
    
    def __init__(self, kg_retriever: KnowledgeGraphRetriever, k: int = 5):
        super().__init__()
        self.kg_retriever = kg_retriever
        self.k = k
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """同步检索文档"""
        return asyncio.run(self._aget_relevant_documents(query))
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步检索文档"""
        import time
        start_time = time.time()
        
        try:
            # 使用知识图谱检索
            kg_results = await self.kg_retriever.retrieve_enhanced_context(query, top_k=self.k)
            
            retrieval_time = time.time() - start_time
            
            enhanced_docs = []
            for result in kg_results:
                # 从KG结果构建文档
                content = result.get('enhanced_content', result.get('content', ''))
                entities = result.get('entities', [])
                relations = result.get('relations', [])
                
                if content:  # 只有在有内容时才创建文档
                    # 创建检索元数据
                    metadata = RetrievalMetadata(
                        retrieval_method='knowledge_graph',
                        confidence_score=result.get('confidence', 0.7),
                        chunk_quality_score=0.8,  # KG增强的内容通常质量较高
                        source_authority=0.9,  # 结构化知识通常权威性较高
                        retrieval_time=retrieval_time,
                        kg_entities=entities,
                        kg_relations=relations
                    )
                    
                    # 创建增强文档
                    enhanced_doc = EnhancedDocument(
                        page_content=content,
                        metadata={
                            **result.get('metadata', {}),
                            'kg_entities': entities,
                            'kg_relations': relations
                        },
                        retrieval_metadata=metadata
                    )
                    enhanced_docs.append(enhanced_doc)
            
            logger.debug(f"KG retrieval: {len(enhanced_docs)} docs in {retrieval_time:.3f}s")
            return enhanced_docs
            
        except Exception as e:
            logger.error(f"Knowledge graph retrieval failed: {e}")
            return []


class HybridRetriever(BaseRetriever):
    """混合检索器 - 整合多种检索策略"""
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        documents: List[Document],
        kg_retriever: Optional[KnowledgeGraphRetriever] = None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.3,
        kg_weight: float = 0.2,
        k: int = 20
    ):
        super().__init__()
        self.k = k
        
        # 初始化各个检索器
        self.dense_retriever = DenseVectorRetriever(vector_store, k=k//2)
        self.sparse_retriever = SparseVectorRetriever(documents, k=k//2)
        
        self.kg_retriever = None
        if kg_retriever:
            self.kg_retriever = KnowledgeGraphRetrieverWrapper(kg_retriever, k=k//4)
        
        # 权重配置
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.kg_weight = kg_weight if kg_retriever else 0
        
        # 重新标准化权重
        total_weight = self.vector_weight + self.bm25_weight + self.kg_weight
        if total_weight > 0:
            self.vector_weight /= total_weight
            self.bm25_weight /= total_weight
            self.kg_weight /= total_weight
        
        logger.info(f"Hybrid retriever initialized with weights: Vector={self.vector_weight:.2f}, BM25={self.bm25_weight:.2f}, KG={self.kg_weight:.2f}")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """同步检索文档"""
        return asyncio.run(self._aget_relevant_documents(query))
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步混合检索"""
        import time
        start_time = time.time()
        
        # 并行执行所有检索策略
        tasks = []
        
        # Dense vector检索
        tasks.append(self.dense_retriever._aget_relevant_documents(query))
        
        # BM25检索
        tasks.append(self.sparse_retriever._aget_relevant_documents(query))
        
        # Knowledge Graph检索（如果启用）
        if self.kg_retriever:
            tasks.append(self.kg_retriever._aget_relevant_documents(query))
        
        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        vector_docs = results[0] if not isinstance(results[0], Exception) else []
        bm25_docs = results[1] if not isinstance(results[1], Exception) else []
        kg_docs = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else []
        
        # 融合结果
        fused_docs = self._fuse_results(vector_docs, bm25_docs, kg_docs, query)
        
        total_time = time.time() - start_time
        logger.info(f"Hybrid retrieval completed: {len(vector_docs)} vector + {len(bm25_docs)} BM25 + {len(kg_docs)} KG → {len(fused_docs)} final docs in {total_time:.3f}s")
        
        return fused_docs[:self.k]
    
    def _fuse_results(self, vector_docs: List[EnhancedDocument], bm25_docs: List[EnhancedDocument], kg_docs: List[EnhancedDocument], query: str) -> List[EnhancedDocument]:
        """融合多个检索器的结果"""
        
        # 收集所有文档并计算综合分数
        doc_scores = {}
        all_docs = {}
        
        # 处理向量检索结果
        for doc in vector_docs:
            doc_id = self._get_doc_id(doc)
            base_score = doc.retrieval_metadata.confidence_score
            weighted_score = base_score * self.vector_weight
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + weighted_score
            all_docs[doc_id] = doc
        
        # 处理BM25检索结果
        for doc in bm25_docs:
            doc_id = self._get_doc_id(doc)
            base_score = doc.retrieval_metadata.confidence_score
            # BM25分数需要归一化
            normalized_score = min(base_score / 10.0, 1.0)  # 简单的归一化
            weighted_score = normalized_score * self.bm25_weight
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + weighted_score
            
            if doc_id not in all_docs:
                all_docs[doc_id] = doc
        
        # 处理知识图谱检索结果
        for doc in kg_docs:
            doc_id = self._get_doc_id(doc)
            base_score = doc.retrieval_metadata.confidence_score
            weighted_score = base_score * self.kg_weight
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + weighted_score
            
            if doc_id not in all_docs:
                all_docs[doc_id] = doc
        
        # 按分数排序
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        # 构建最终结果，更新元数据
        fused_docs = []
        for doc_id in sorted_doc_ids:
            doc = all_docs[doc_id]
            
            # 更新检索元数据
            doc.retrieval_metadata.retrieval_method = 'hybrid'
            doc.retrieval_metadata.confidence_score = doc_scores[doc_id]
            
            # 在文档元数据中添加融合分数
            doc.metadata['fusion_score'] = doc_scores[doc_id]
            doc.metadata['retrieval_methods'] = self._get_retrieval_methods(doc_id, vector_docs, bm25_docs, kg_docs)
            
            fused_docs.append(doc)
        
        return fused_docs
    
    def _get_doc_id(self, doc: EnhancedDocument) -> str:
        """生成文档ID用于去重"""
        # 使用内容的哈希作为ID
        import hashlib
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        return content_hash
    
    def _get_retrieval_methods(self, doc_id: str, vector_docs: List[EnhancedDocument], bm25_docs: List[EnhancedDocument], kg_docs: List[EnhancedDocument]) -> List[str]:
        """获取文档被哪些检索方法检索到"""
        methods = []
        
        if any(self._get_doc_id(doc) == doc_id for doc in vector_docs):
            methods.append('vector')
        if any(self._get_doc_id(doc) == doc_id for doc in bm25_docs):
            methods.append('bm25')
        if any(self._get_doc_id(doc) == doc_id for doc in kg_docs):
            methods.append('knowledge_graph')
        
        return methods