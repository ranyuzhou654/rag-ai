# src/retrieval/enhanced_agentic_rag.py
"""
增强的智能体RAG系统
基于improvements.md的建议，实现细化评估维度和具象化改进建议
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
import json
from loguru import logger

# LangChain组件
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

# 本地组件
from .hybrid_retriever import HybridRetriever, EnhancedDocument


class AgenticDecision(Enum):
    """智能体决策类型"""
    PROCEED = "proceed"           # 继续使用当前结果
    RETRY = "retry"              # 重新检索
    EXPAND_QUERY = "expand_query"  # 扩展查询
    SEEK_MORE = "seek_more"      # 寻找更多信息
    REFINE_QUERY = "refine_query" # 精炼查询


@dataclass
class DetailedEvaluation:
    """详细评估结果"""
    # 四个核心维度
    relevance_score: float      # 相关性 (0-1)
    completeness_score: float   # 完整性 (0-1)
    novelty_score: float        # 新颖性 (0-1)
    authority_score: float      # 权威性 (0-1)
    
    # 综合评分
    overall_score: float
    
    # 评估详情
    evaluation_reasoning: str
    specific_issues: List[str]
    
    # 具象化改进建议
    improvement_suggestions: Dict[str, str]
    
    # 决策
    recommended_action: AgenticDecision
    confidence_in_decision: float


@dataclass
class RetrievalIteration:
    """检索迭代记录"""
    iteration_number: int
    query_used: str
    documents_retrieved: List[EnhancedDocument]
    evaluation: DetailedEvaluation
    action_taken: AgenticDecision
    processing_time: float


@dataclass
class AgenticRetrievalResult:
    """智能体检索结果"""
    final_documents: List[EnhancedDocument]
    all_iterations: List[RetrievalIteration]
    total_iterations: int
    success: bool
    final_evaluation: DetailedEvaluation
    total_time: float
    
    # 额外信息
    query_evolution: List[str]
    strategies_used: List[str]
    performance_metrics: Dict[str, Any]


class DetailedEvaluator:
    """详细评估器 - 实现四维评估体系"""
    
    def __init__(self, llm, config: Dict[str, Any]):
        self.llm = llm
        self.config = config
        
        # 评估提示模板
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的检索质量评估专家。请对检索结果进行详细评估，并返回JSON格式结果。

评估维度说明：
1. 相关性(Relevance): 检索到的文档与用户查询的相关程度
2. 完整性(Completeness): 检索结果是否足以完整回答用户问题
3. 新颖性(Novelty): 检索结果是否包含多样化的信息，避免重复
4. 权威性(Authority): 检索结果的来源可信度和专业性

决策类型说明：
- proceed: 当前结果满足需求，可以继续
- retry: 结果质量较差，需要重新检索
- expand_query: 结果不够完整，需要扩展查询范围
- seek_more: 需要寻找更多补充信息
- refine_query: 查询过于宽泛，需要精炼

请返回以下JSON格式：
{{
    "relevance_score": 0.8,
    "completeness_score": 0.7,
    "novelty_score": 0.6,
    "authority_score": 0.9,
    "overall_score": 0.75,
    "evaluation_reasoning": "详细的评估理由",
    "specific_issues": ["具体问题1", "具体问题2"],
    "improvement_suggestions": {{
        "query_modification": "具体的查询修改建议",
        "search_strategy": "搜索策略建议",
        "additional_keywords": "建议添加的关键词",
        "time_constraint": "时间范围建议",
        "domain_focus": "领域聚焦建议"
    }},
    "recommended_action": "proceed|retry|expand_query|seek_more|refine_query",
    "confidence_in_decision": 0.85
}}

原始查询：{query}
检索历史：{retrieval_history}"""),
            ("human", """请评估以下检索结果：

检索到的文档数量：{num_docs}
文档信息：
{document_info}

检索策略：{retrieval_methods}
平均相关性分数：{avg_relevance}
""")
        ])
        
        self.evaluation_chain = (
            self.evaluation_prompt 
            | self.llm 
            | JsonOutputParser()
        )
    
    async def evaluate_retrieval(
        self,
        query: str,
        documents: List[EnhancedDocument],
        retrieval_history: List[RetrievalIteration],
        retrieval_methods: List[str]
    ) -> DetailedEvaluation:
        """执行详细评估"""
        
        try:
            # 准备文档信息
            document_info = self._format_document_info(documents)
            avg_relevance = self._calculate_average_relevance(documents)
            
            # 格式化检索历史
            history_summary = self._format_retrieval_history(retrieval_history)
            
            # 执行评估
            evaluation_result = await self.evaluation_chain.ainvoke({
                "query": query,
                "num_docs": len(documents),
                "document_info": document_info,
                "retrieval_methods": retrieval_methods,
                "avg_relevance": avg_relevance,
                "retrieval_history": history_summary
            })
            
            # 构建详细评估对象
            return DetailedEvaluation(
                relevance_score=evaluation_result.get("relevance_score", 0.0),
                completeness_score=evaluation_result.get("completeness_score", 0.0),
                novelty_score=evaluation_result.get("novelty_score", 0.0),
                authority_score=evaluation_result.get("authority_score", 0.0),
                overall_score=evaluation_result.get("overall_score", 0.0),
                evaluation_reasoning=evaluation_result.get("evaluation_reasoning", ""),
                specific_issues=evaluation_result.get("specific_issues", []),
                improvement_suggestions=evaluation_result.get("improvement_suggestions", {}),
                recommended_action=AgenticDecision(evaluation_result.get("recommended_action", "proceed")),
                confidence_in_decision=evaluation_result.get("confidence_in_decision", 0.5)
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            
            # 返回默认评估
            return self._create_fallback_evaluation(documents)
    
    def _format_document_info(self, documents: List[EnhancedDocument]) -> str:
        """格式化文档信息"""
        doc_info = []
        
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get('title', f'文档{i}')
            source = doc.metadata.get('source', '未知来源')
            length = len(doc.page_content)
            
            retrieval_method = "未知"
            confidence = 0.0
            if doc.retrieval_metadata:
                retrieval_method = doc.retrieval_metadata.retrieval_method
                confidence = doc.retrieval_metadata.confidence_score
            
            doc_info.append(
                f"文档{i}: {title} | 来源: {source} | 长度: {length}字符 | "
                f"检索方式: {retrieval_method} | 置信度: {confidence:.3f}"
            )
        
        return "\n".join(doc_info)
    
    def _calculate_average_relevance(self, documents: List[EnhancedDocument]) -> float:
        """计算平均相关性"""
        if not documents:
            return 0.0
        
        scores = []
        for doc in documents:
            if doc.retrieval_metadata:
                scores.append(doc.retrieval_metadata.confidence_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _format_retrieval_history(self, history: List[RetrievalIteration]) -> str:
        """格式化检索历史"""
        if not history:
            return "无历史记录"
        
        history_items = []
        for iteration in history:
            history_items.append(
                f"第{iteration.iteration_number}轮: 查询='{iteration.query_used}', "
                f"结果={len(iteration.documents_retrieved)}篇, "
                f"评分={iteration.evaluation.overall_score:.2f}, "
                f"决策={iteration.action_taken.value}"
            )
        
        return " -> ".join(history_items)
    
    def _create_fallback_evaluation(self, documents: List[EnhancedDocument]) -> DetailedEvaluation:
        """创建默认评估（当LLM评估失败时）"""
        
        # 基于简单规则的评估
        num_docs = len(documents)
        avg_relevance = self._calculate_average_relevance(documents)
        
        # 简单评分逻辑
        relevance_score = min(avg_relevance, 1.0)
        completeness_score = min(num_docs / 5.0, 1.0)  # 假设5篇文档为完整
        novelty_score = 0.7 if num_docs >= 3 else 0.4  # 多样性简单评估
        authority_score = 0.6  # 默认权威性
        
        overall_score = (relevance_score + completeness_score + novelty_score + authority_score) / 4
        
        # 决策逻辑
        if overall_score >= 0.8:
            action = AgenticDecision.PROCEED
        elif overall_score >= 0.6:
            action = AgenticDecision.EXPAND_QUERY
        else:
            action = AgenticDecision.RETRY
        
        return DetailedEvaluation(
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            novelty_score=novelty_score,
            authority_score=authority_score,
            overall_score=overall_score,
            evaluation_reasoning="LLM评估失败，使用规则评估",
            specific_issues=["LLM评估不可用"],
            improvement_suggestions={
                "query_modification": "尝试添加更多关键词",
                "search_strategy": "使用不同的检索策略"
            },
            recommended_action=action,
            confidence_in_decision=0.3
        )


class QueryRefiner:
    """查询精炼器 - 基于评估结果生成优化查询"""
    
    def __init__(self, llm):
        self.llm = llm
        
        self.query_refinement_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是查询优化专家。基于评估反馈和改进建议，生成优化后的查询。

优化策略：
1. expand_query: 添加相关概念和同义词，扩大检索范围
2. refine_query: 增加限定词，缩小检索范围提高精确度
3. seek_more: 从不同角度重新表达查询
4. retry: 用完全不同的表达方式重写查询

请返回JSON格式：
{{
    "optimized_query": "优化后的查询",
    "optimization_strategy": "使用的优化策略",
    "added_keywords": ["新增关键词1", "新增关键词2"],
    "reasoning": "优化理由"
}}"""),
            ("human", """原始查询：{original_query}
评估反馈：{evaluation_feedback}
改进建议：{improvement_suggestions}
需要执行的操作：{recommended_action}

请生成优化查询。""")
        ])
        
        self.refinement_chain = (
            self.query_refinement_prompt 
            | self.llm 
            | JsonOutputParser()
        )
    
    async def refine_query(
        self,
        original_query: str,
        evaluation: DetailedEvaluation,
        iteration_count: int
    ) -> str:
        """基于评估结果精炼查询"""
        
        try:
            refinement_result = await self.refinement_chain.ainvoke({
                "original_query": original_query,
                "evaluation_feedback": evaluation.evaluation_reasoning,
                "improvement_suggestions": json.dumps(evaluation.improvement_suggestions, ensure_ascii=False, indent=2),
                "recommended_action": evaluation.recommended_action.value
            })
            
            optimized_query = refinement_result.get("optimized_query", original_query)
            
            logger.debug(f"Query refinement: '{original_query}' -> '{optimized_query}'")
            
            return optimized_query
            
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}")
            
            # 简单的备用优化逻辑
            return self._simple_query_refinement(original_query, evaluation, iteration_count)
    
    def _simple_query_refinement(
        self,
        original_query: str,
        evaluation: DetailedEvaluation,
        iteration_count: int
    ) -> str:
        """简单的查询优化逻辑"""
        
        action = evaluation.recommended_action
        
        if action == AgenticDecision.EXPAND_QUERY:
            # 添加相关词汇
            related_terms = ["相关", "类似", "对比", "分析"]
            return f"{original_query} {related_terms[iteration_count % len(related_terms)]}"
        
        elif action == AgenticDecision.REFINE_QUERY:
            # 添加限定词
            constraints = ["具体", "详细", "深入", "最新"]
            return f"{constraints[iteration_count % len(constraints)]} {original_query}"
        
        elif action == AgenticDecision.SEEK_MORE:
            # 换个角度
            perspectives = ["如何", "为什么", "什么是", "哪些"]
            return f"{perspectives[iteration_count % len(perspectives)]} {original_query}"
        
        else:
            # 默认添加一些通用词
            return f"{original_query} 详细信息"


class EnhancedAgenticRAG:
    """增强的智能体RAG系统"""
    
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        llm,
        config: Dict[str, Any]
    ):
        self.hybrid_retriever = hybrid_retriever
        self.llm = llm
        self.config = config
        
        # 初始化组件
        self.evaluator = DetailedEvaluator(llm, config)
        self.query_refiner = QueryRefiner(llm)
        
        # 配置参数
        self.max_iterations = config.get('max_agentic_iterations', 3)
        self.min_score_threshold = config.get('min_score_threshold', 0.7)
        self.early_stop_threshold = config.get('early_stop_threshold', 0.9)
        
        logger.info(f"Enhanced Agentic RAG initialized: max_iterations={self.max_iterations}, "
                   f"score_threshold={self.min_score_threshold}")
    
    async def retrieve_with_agency(self, query: str) -> AgenticRetrievalResult:
        """执行智能体检索"""
        
        start_time = time.time()
        logger.info(f"🤖 Starting agentic retrieval for: {query[:50]}...")
        
        # 初始化状态
        current_query = query
        all_iterations = []
        query_evolution = [query]
        strategies_used = set()
        
        final_documents = []
        final_evaluation = None
        success = False
        
        for iteration in range(self.max_iterations):
            logger.debug(f"🔄 Agentic iteration {iteration + 1}/{self.max_iterations}")
            
            iteration_start = time.time()
            
            try:
                # 1. 执行检索
                documents = await self.hybrid_retriever._aget_relevant_documents(current_query)
                
                # 记录使用的策略
                if documents and hasattr(documents[0], 'retrieval_metadata'):
                    strategies_used.add(documents[0].retrieval_metadata.retrieval_method)
                
                # 2. 评估检索结果
                evaluation = await self.evaluator.evaluate_retrieval(
                    query=current_query,
                    documents=documents,
                    retrieval_history=all_iterations,
                    retrieval_methods=list(strategies_used)
                )
                
                iteration_time = time.time() - iteration_start
                
                # 3. 记录迭代
                iteration_record = RetrievalIteration(
                    iteration_number=iteration + 1,
                    query_used=current_query,
                    documents_retrieved=documents,
                    evaluation=evaluation,
                    action_taken=evaluation.recommended_action,
                    processing_time=iteration_time
                )
                all_iterations.append(iteration_record)
                
                # 4. 决策
                if (evaluation.recommended_action == AgenticDecision.PROCEED or 
                    evaluation.overall_score >= self.early_stop_threshold):
                    # 成功完成
                    final_documents = documents
                    final_evaluation = evaluation
                    success = True
                    logger.success(f"✅ Agentic retrieval succeeded in {iteration + 1} iterations")
                    break
                
                elif iteration == self.max_iterations - 1:
                    # 最后一次迭代，使用当前结果
                    final_documents = documents
                    final_evaluation = evaluation
                    success = evaluation.overall_score >= self.min_score_threshold
                    if success:
                        logger.success(f"✅ Agentic retrieval completed with acceptable score")
                    else:
                        logger.warning(f"⚠️ Agentic retrieval completed with low score")
                    break
                
                else:
                    # 5. 优化查询继续迭代
                    optimized_query = await self.query_refiner.refine_query(
                        original_query=query,
                        evaluation=evaluation,
                        iteration_count=iteration
                    )
                    
                    if optimized_query != current_query:
                        current_query = optimized_query
                        query_evolution.append(optimized_query)
                        logger.debug(f"🔧 Query refined to: {optimized_query}")
                    else:
                        logger.warning("Query refinement produced no change, continuing with current query")
                
            except Exception as e:
                logger.error(f"❌ Iteration {iteration + 1} failed: {e}")
                
                # 如果有之前的结果，使用它们
                if all_iterations:
                    last_iteration = all_iterations[-1]
                    final_documents = last_iteration.documents_retrieved
                    final_evaluation = last_iteration.evaluation
                    success = False
                    break
                else:
                    # 创建空结果
                    final_documents = []
                    final_evaluation = self.evaluator._create_fallback_evaluation([])
                    success = False
                    break
        
        total_time = time.time() - start_time
        
        # 构建结果
        result = AgenticRetrievalResult(
            final_documents=final_documents,
            all_iterations=all_iterations,
            total_iterations=len(all_iterations),
            success=success,
            final_evaluation=final_evaluation,
            total_time=total_time,
            query_evolution=query_evolution,
            strategies_used=list(strategies_used),
            performance_metrics={
                "avg_iteration_time": total_time / len(all_iterations) if all_iterations else 0,
                "total_documents_evaluated": sum(len(it.documents_retrieved) for it in all_iterations),
                "final_score": final_evaluation.overall_score if final_evaluation else 0.0,
                "iterations_used": len(all_iterations),
                "max_iterations": self.max_iterations
            }
        )
        
        logger.info(f"🎯 Agentic retrieval completed: {len(final_documents)} final docs, "
                   f"score={final_evaluation.overall_score:.2f}, time={total_time:.2f}s")
        
        return result
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        self.config.update(new_config)
        self.max_iterations = self.config.get('max_agentic_iterations', 3)
        self.min_score_threshold = self.config.get('min_score_threshold', 0.7)
        self.early_stop_threshold = self.config.get('early_stop_threshold', 0.9)
        
        logger.info(f"Agentic RAG config updated: max_iter={self.max_iterations}, "
                   f"min_score={self.min_score_threshold}, early_stop={self.early_stop_threshold}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计（可以扩展为实际统计）"""
        return {
            "config": {
                "max_iterations": self.max_iterations,
                "min_score_threshold": self.min_score_threshold,
                "early_stop_threshold": self.early_stop_threshold
            },
            "components": {
                "evaluator": "DetailedEvaluator",
                "query_refiner": "QueryRefiner",
                "hybrid_retriever": type(self.hybrid_retriever).__name__
            }
        }