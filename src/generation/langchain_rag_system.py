# src/generation/langchain_rag_system.py
"""
基于LangChain LCEL的现代化RAG系统
实现2024-2025年最新的RAG架构和优化技术
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio
import time
import json
from loguru import logger

# LangChain核心组件
from langchain_core.runnables import (
    RunnablePassthrough, 
    RunnableParallel, 
    RunnableLambda,
    RunnableSequence
)
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_qdrant import QdrantVectorStore

# RAG专用组件
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# 本地组件导入
from .hybrid_retriever import HybridRetriever, EnhancedDocument, RetrievalMetadata
from src.retrieval.query_intelligence import QueryIntelligenceEngine
from src.knowledge_graph.kg_retriever import KnowledgeGraphRetriever
from configs.config import config


@dataclass
class LangChainRAGResult:
    """LangChain RAG系统结果"""
    answer: str
    source_documents: List[EnhancedDocument]
    retrieval_metadata: Dict[str, Any]
    generation_metadata: Dict[str, Any]
    total_time: float
    confidence_score: float


class QueryComplexityAnalyzer:
    """查询复杂度分析器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.complexity_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个查询复杂度分析专家。请分析用户查询的复杂度并返回JSON格式结果。

复杂度等级：
- SIMPLE: 简单事实查询，直接答案
- MEDIUM: 需要一定推理的查询
- COMPLEX: 多步推理、比较分析等
- CRITICAL: 需要深度分析、创意思考的查询

请返回以下JSON格式：
{{
    "complexity": "SIMPLE|MEDIUM|COMPLEX|CRITICAL",
    "reasoning": "分析理由",
    "keywords": ["关键词1", "关键词2"],
    "query_type": "factual|comparative|explanatory|procedural|creative"
}}"""),
            ("human", "查询: {query}")
        ])
        
        self.analysis_chain = (
            self.complexity_prompt 
            | self.llm 
            | JsonOutputParser()
        )
    
    async def analyze(self, query: str) -> Dict[str, Any]:
        """分析查询复杂度"""
        try:
            result = await self.analysis_chain.ainvoke({"query": query})
            return result
        except Exception as e:
            logger.warning(f"Query complexity analysis failed: {e}")
            return {
                "complexity": "MEDIUM",
                "reasoning": "分析失败，使用默认复杂度",
                "keywords": query.split()[:5],
                "query_type": "factual"
            }


class ContextualRetriever:
    """上下文感知检索器"""
    
    def __init__(self, hybrid_retriever: HybridRetriever, reranker=None):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        
        # 如果有重排序器，创建压缩检索器
        if reranker:
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=hybrid_retriever
            )
        else:
            self.compression_retriever = hybrid_retriever
    
    async def retrieve(self, query: str, complexity_info: Dict) -> List[EnhancedDocument]:
        """基于查询复杂度的自适应检索"""
        
        # 根据复杂度调整检索参数
        k = self._get_k_by_complexity(complexity_info["complexity"])
        self.hybrid_retriever.k = k
        
        # 执行检索
        docs = await self.compression_retriever._aget_relevant_documents(query)
        
        # 如果是复杂查询，可能需要更多上下文
        if complexity_info["complexity"] in ["COMPLEX", "CRITICAL"] and len(docs) < k:
            # 尝试使用更宽泛的搜索
            expanded_query = f"{query} {' '.join(complexity_info.get('keywords', []))}"
            additional_docs = await self.hybrid_retriever._aget_relevant_documents(expanded_query)
            
            # 合并去重
            seen_ids = {self._get_doc_id(doc) for doc in docs}
            for doc in additional_docs:
                if self._get_doc_id(doc) not in seen_ids:
                    docs.append(doc)
                    if len(docs) >= k:
                        break
        
        return docs[:k]
    
    def _get_k_by_complexity(self, complexity: str) -> int:
        """根据复杂度确定检索数量"""
        k_map = {
            "SIMPLE": 5,
            "MEDIUM": 10,
            "COMPLEX": 15,
            "CRITICAL": 20
        }
        return k_map.get(complexity, 10)
    
    def _get_doc_id(self, doc: EnhancedDocument) -> str:
        """获取文档ID"""
        import hashlib
        return hashlib.md5(doc.page_content.encode()).hexdigest()


class ResponseGenerator:
    """响应生成器"""
    
    def __init__(self, llm, local_llm=None):
        self.llm = llm  # 主要LLM（通常是API模型）
        self.local_llm = local_llm  # 本地LLM
        
        # 生成提示模板
        self.generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的AI助手，专门回答技术相关问题。请基于提供的上下文信息回答用户问题。

要求：
1. 答案要准确、详细且有条理
2. 优先使用提供的上下文信息
3. 如果上下文不足，明确说明
4. 引用相关的文档来源
5. 保持专业性和客观性

上下文信息：
{context}

检索元数据：
{retrieval_info}"""),
            ("human", "问题：{query}")
        ])
        
        # 自我评估提示
        self.self_eval_prompt = ChatPromptTemplate.from_messages([
            ("system", """请评估以下回答的质量。返回JSON格式：
{{
    "relevance_score": 0.8,
    "completeness_score": 0.9,
    "accuracy_score": 0.85,
    "overall_confidence": 0.85,
    "reasoning": "评估理由"
}}

问题: {query}
回答: {answer}
参考上下文: {context}"""),
            ("human", "请评估上述回答的质量")
        ])
    
    async def generate(self, query: str, docs: List[EnhancedDocument], complexity_info: Dict) -> LangChainRAGResult:
        """生成回答"""
        start_time = time.time()
        
        # 选择合适的模型
        selected_llm = self._select_model(complexity_info["complexity"])
        
        # 准备上下文
        context = self._format_context(docs)
        retrieval_info = self._format_retrieval_info(docs)
        
        # 生成链
        generation_chain = (
            self.generation_prompt 
            | selected_llm 
            | StrOutputParser()
        )
        
        # 生成答案
        answer = await generation_chain.ainvoke({
            "query": query,
            "context": context,
            "retrieval_info": retrieval_info
        })
        
        # 自我评估
        eval_chain = (
            self.self_eval_prompt 
            | selected_llm 
            | JsonOutputParser()
        )
        
        try:
            evaluation = await eval_chain.ainvoke({
                "query": query,
                "answer": answer,
                "context": context[:1000]  # 限制长度避免超出限制
            })
            confidence_score = evaluation.get("overall_confidence", 0.7)
        except Exception as e:
            logger.warning(f"Self-evaluation failed: {e}")
            confidence_score = 0.7  # 默认置信度
            evaluation = {"reasoning": "自我评估失败"}
        
        total_time = time.time() - start_time
        
        # 构建结果
        result = LangChainRAGResult(
            answer=answer,
            source_documents=docs,
            retrieval_metadata={
                "num_sources": len(docs),
                "retrieval_methods": self._get_retrieval_methods(docs),
                "avg_relevance": self._calculate_avg_relevance(docs)
            },
            generation_metadata={
                "model_used": selected_llm.__class__.__name__,
                "complexity": complexity_info["complexity"],
                "evaluation": evaluation
            },
            total_time=total_time,
            confidence_score=confidence_score
        )
        
        return result
    
    def _select_model(self, complexity: str):
        """根据复杂度选择模型"""
        if complexity in ["SIMPLE", "MEDIUM"] and self.local_llm:
            return self.local_llm
        return self.llm
    
    def _format_context(self, docs: List[EnhancedDocument]) -> str:
        """格式化上下文"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('title', doc.metadata.get('source', f'文档{i}'))
            content = doc.page_content
            context_parts.append(f"[来源{i}: {source}]\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _format_retrieval_info(self, docs: List[EnhancedDocument]) -> str:
        """格式化检索信息"""
        info_parts = []
        for i, doc in enumerate(docs, 1):
            if doc.retrieval_metadata:
                method = doc.retrieval_metadata.retrieval_method
                score = doc.retrieval_metadata.confidence_score
                info_parts.append(f"文档{i}: {method}检索, 相关性={score:.3f}")
        
        return "; ".join(info_parts)
    
    def _get_retrieval_methods(self, docs: List[EnhancedDocument]) -> List[str]:
        """获取使用的检索方法"""
        methods = set()
        for doc in docs:
            if doc.retrieval_metadata:
                methods.add(doc.retrieval_metadata.retrieval_method)
        return list(methods)
    
    def _calculate_avg_relevance(self, docs: List[EnhancedDocument]) -> float:
        """计算平均相关性"""
        if not docs:
            return 0.0
        
        scores = []
        for doc in docs:
            if doc.retrieval_metadata:
                scores.append(doc.retrieval_metadata.confidence_score)
        
        return sum(scores) / len(scores) if scores else 0.0


class LangChainRAGSystem:
    """基于LangChain LCEL的现代RAG系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("🚀 Initializing LangChain RAG System...")
        
        # 初始化模型
        self.llm = self._init_llm()
        self.local_llm = self._init_local_llm()
        self.embeddings = self._init_embeddings()
        
        # 初始化向量存储
        self.vector_store = self._init_vector_store()
        
        # 初始化知识图谱（如果配置）
        self.kg_retriever = self._init_kg_retriever()
        
        # 初始化重排序器
        self.reranker = self._init_reranker()
        
        # 初始化核心组件
        self.query_analyzer = QueryComplexityAnalyzer(self.llm)
        self.query_intelligence = QueryIntelligenceEngine(config)
        
        # 初始化混合检索器（需要文档列表，稍后设置）
        self.hybrid_retriever = None
        self.contextual_retriever = None
        self.response_generator = ResponseGenerator(self.llm, self.local_llm)
        
        # 构建LCEL链
        self._build_rag_chain()
        
        logger.success("✅ LangChain RAG System initialized!")
    
    def _init_llm(self):
        """初始化主要LLM"""
        if self.config.get('openai_api_key'):
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                api_key=self.config['openai_api_key']
            )
        else:
            # 使用本地模型作为备选
            return self._init_local_llm()
    
    def _init_local_llm(self):
        """初始化本地LLM"""
        try:
            # 这里可以集成本地模型，如Qwen2
            # 暂时返回None，表示未配置
            return None
        except Exception as e:
            logger.warning(f"Local LLM initialization failed: {e}")
            return None
    
    def _init_embeddings(self):
        """初始化嵌入模型"""
        if self.config.get('openai_api_key'):
            return OpenAIEmbeddings(api_key=self.config['openai_api_key'])
        else:
            # 使用本地嵌入模型
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={'device': 'cpu'}
            )
    
    def _init_vector_store(self):
        """初始化向量存储"""
        try:
            from qdrant_client import QdrantClient
            
            client = QdrantClient(
                host=self.config.get('qdrant_host', 'localhost'),
                port=self.config.get('qdrant_port', 6333)
            )
            
            return QdrantVectorStore(
                client=client,
                collection_name=self.config.get('collection_name', 'documents'),
                embeddings=self.embeddings
            )
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            return None
    
    def _init_kg_retriever(self):
        """初始化知识图谱检索器"""
        if self.config.get('enable_knowledge_graph', False):
            try:
                return KnowledgeGraphRetriever(self.config)
            except Exception as e:
                logger.warning(f"Knowledge graph retriever initialization failed: {e}")
        return None
    
    def _init_reranker(self):
        """初始化重排序器"""
        try:
            if self.config.get('cohere_api_key'):
                return CohereRerank(
                    cohere_api_key=self.config['cohere_api_key'],
                    top_n=10
                )
        except Exception as e:
            logger.warning(f"Reranker initialization failed: {e}")
        return None
    
    def _build_rag_chain(self):
        """构建LCEL RAG链"""
        
        # 查询分析和增强步骤
        query_processing_chain = RunnableParallel({
            "original_query": RunnablePassthrough(),
            "complexity_analysis": RunnableLambda(self._analyze_query_complexity),
            "enhanced_queries": RunnableLambda(self._enhance_query)
        })
        
        # 检索步骤
        retrieval_chain = RunnableLambda(self._contextual_retrieve)
        
        # 生成步骤
        generation_chain = RunnableLambda(self._generate_response)
        
        # 完整的RAG链
        self.rag_chain = (
            query_processing_chain
            | retrieval_chain
            | generation_chain
        )
        
        logger.info("✅ LCEL RAG chain built successfully")
    
    async def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """分析查询复杂度"""
        return await self.query_analyzer.analyze(query)
    
    async def _enhance_query(self, query: str) -> Dict[str, Any]:
        """增强查询"""
        try:
            return await self.query_intelligence.process_query(query)
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}")
            return {"enhanced_query": query, "sub_questions": [], "hyde_document": None}
    
    async def _contextual_retrieve(self, processed_input: Dict) -> Dict:
        """上下文检索"""
        query = processed_input["original_query"]
        complexity = processed_input["complexity_analysis"]
        
        # 确保混合检索器已初始化
        if not self.hybrid_retriever:
            await self._init_hybrid_retriever()
        
        if not self.contextual_retriever:
            return processed_input  # 返回原输入，避免链中断
        
        # 执行检索
        docs = await self.contextual_retriever.retrieve(query, complexity)
        
        return {
            **processed_input,
            "retrieved_docs": docs
        }
    
    async def _generate_response(self, retrieval_result: Dict) -> LangChainRAGResult:
        """生成响应"""
        query = retrieval_result["original_query"]
        docs = retrieval_result.get("retrieved_docs", [])
        complexity = retrieval_result["complexity_analysis"]
        
        return await self.response_generator.generate(query, docs, complexity)
    
    async def _init_hybrid_retriever(self):
        """初始化混合检索器"""
        try:
            # 获取文档列表用于BM25
            # 这里需要根据实际的向量存储实现来获取文档
            documents = await self._get_documents_from_vector_store()
            
            self.hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                documents=documents,
                kg_retriever=self.kg_retriever,
                vector_weight=0.5,
                bm25_weight=0.3,
                kg_weight=0.2
            )
            
            self.contextual_retriever = ContextualRetriever(
                self.hybrid_retriever,
                self.reranker
            )
            
            logger.info("✅ Hybrid retriever initialized")
            
        except Exception as e:
            logger.error(f"Hybrid retriever initialization failed: {e}")
            self.hybrid_retriever = None
            self.contextual_retriever = None
    
    async def _get_documents_from_vector_store(self) -> List[Document]:
        """从向量存储获取文档（用于BM25索引）"""
        try:
            # 这是一个简化的实现，实际应该从向量存储中获取所有文档
            # 由于Qdrant的限制，这里可能需要分批获取
            
            # 临时返回空列表，实际使用时需要实现具体逻辑
            return []
        except Exception as e:
            logger.warning(f"Failed to get documents from vector store: {e}")
            return []
    
    async def query(self, user_query: str, **kwargs) -> LangChainRAGResult:
        """执行RAG查询"""
        logger.info(f"🔍 Processing query: {user_query[:100]}...")
        
        try:
            start_time = time.time()
            
            # 执行RAG链
            result = await self.rag_chain.ainvoke(user_query)
            
            # 更新总时间
            result.total_time = time.time() - start_time
            
            logger.success(f"✅ Query completed in {result.total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            
            # 返回错误结果
            return LangChainRAGResult(
                answer=f"抱歉，处理查询时出现错误：{str(e)}",
                source_documents=[],
                retrieval_metadata={},
                generation_metadata={},
                total_time=time.time() - start_time if 'start_time' in locals() else 0,
                confidence_score=0.0
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "components_initialized": {
                "vector_store": self.vector_store is not None,
                "hybrid_retriever": self.hybrid_retriever is not None,
                "kg_retriever": self.kg_retriever is not None,
                "reranker": self.reranker is not None,
                "local_llm": self.local_llm is not None
            },
            "config": {
                "enable_kg": self.config.get('enable_knowledge_graph', False),
                "enable_reranker": self.reranker is not None
            }
        }