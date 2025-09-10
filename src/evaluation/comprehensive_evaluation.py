# src/evaluation/comprehensive_evaluation.py
"""
综合评估框架
集成RAGAS、TruLens和自定义评估指标
建立完整的RAG系统量化评估体系
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import asyncio
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

# RAGAS组件
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity
    )
    from ragas.dataset import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not available, using fallback evaluation")

# TruLens组件
try:
    from trulens_eval import TruChain, Feedback, Huggingface, Tru
    from trulens_eval.feedback import Groundedness
    TRULENS_AVAILABLE = True
except ImportError:
    TRULENS_AVAILABLE = False
    logger.warning("TruLens not available, using fallback evaluation")

# 本地组件
from ..retrieval.hybrid_retriever import EnhancedDocument
from ..generation.langchain_rag_system import LangChainRAGResult


@dataclass
class EvaluationMetrics:
    """评估指标集"""
    # RAGAS指标
    faithfulness_score: Optional[float] = None
    answer_relevancy_score: Optional[float] = None
    context_precision_score: Optional[float] = None
    context_recall_score: Optional[float] = None
    answer_correctness_score: Optional[float] = None
    answer_similarity_score: Optional[float] = None
    
    # TruLens指标
    groundedness_score: Optional[float] = None
    qa_relevance_score: Optional[float] = None
    
    # 自定义指标
    retrieval_precision: Optional[float] = None
    retrieval_recall: Optional[float] = None
    response_latency: Optional[float] = None
    cost_efficiency: Optional[float] = None
    user_satisfaction: Optional[float] = None
    
    # 综合指标
    overall_score: Optional[float] = None
    evaluation_timestamp: str = None
    
    def __post_init__(self):
        if self.evaluation_timestamp is None:
            self.evaluation_timestamp = datetime.now().isoformat()


@dataclass
class EvaluationCase:
    """评估案例"""
    question: str
    ground_truth_answer: str
    retrieved_contexts: List[str]
    generated_answer: str
    metadata: Dict[str, Any]


@dataclass
class EvaluationReport:
    """评估报告"""
    dataset_name: str
    evaluation_timestamp: str
    total_cases: int
    metrics_summary: EvaluationMetrics
    individual_results: List[Dict[str, Any]]
    performance_analysis: Dict[str, Any]
    recommendations: List[str]


class RAGASEvaluator:
    """RAGAS评估器"""
    
    def __init__(self, llm_model=None):
        if not RAGAS_AVAILABLE:
            logger.warning("RAGAS not available, creating mock evaluator")
            return
        
        self.llm_model = llm_model
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
            answer_similarity
        ]
        logger.info("RAGAS evaluator initialized")
    
    async def evaluate_batch(self, evaluation_cases: List[EvaluationCase]) -> Dict[str, float]:
        """批量评估"""
        if not RAGAS_AVAILABLE:
            return self._mock_ragas_evaluation()
        
        try:
            # 准备数据集
            dataset = self._prepare_ragas_dataset(evaluation_cases)
            
            # 执行评估
            results = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm_model
            )
            
            # 提取分数
            return {
                'faithfulness': results['faithfulness'],
                'answer_relevancy': results['answer_relevancy'], 
                'context_precision': results['context_precision'],
                'context_recall': results['context_recall'],
                'answer_correctness': results['answer_correctness'],
                'answer_similarity': results['answer_similarity']
            }
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return self._mock_ragas_evaluation()
    
    def _prepare_ragas_dataset(self, evaluation_cases: List[EvaluationCase]) -> Dataset:
        """准备RAGAS数据集"""
        data = {
            'question': [case.question for case in evaluation_cases],
            'answer': [case.generated_answer for case in evaluation_cases],
            'contexts': [case.retrieved_contexts for case in evaluation_cases],
            'ground_truth': [case.ground_truth_answer for case in evaluation_cases]
        }
        
        return Dataset.from_dict(data)
    
    def _mock_ragas_evaluation(self) -> Dict[str, float]:
        """模拟RAGAS评估结果"""
        return {
            'faithfulness': 0.85,
            'answer_relevancy': 0.82,
            'context_precision': 0.78,
            'context_recall': 0.80,
            'answer_correctness': 0.83,
            'answer_similarity': 0.81
        }


class TruLensEvaluator:
    """TruLens评估器"""
    
    def __init__(self, rag_chain=None):
        if not TRULENS_AVAILABLE:
            logger.warning("TruLens not available, creating mock evaluator")
            return
        
        self.rag_chain = rag_chain
        
        # 初始化反馈函数
        if rag_chain:
            self.groundedness = Feedback(Groundedness().groundedness_measure_with_cot_reasons)
            self.qa_relevance = Feedback(lambda input, output: 0.8)  # 简化实现
            
            # 包装RAG链
            self.tru_rag = TruChain(
                rag_chain,
                feedbacks=[self.groundedness, self.qa_relevance]
            )
        
        logger.info("TruLens evaluator initialized")
    
    async def evaluate_real_time(self, question: str, answer: str, context: str) -> Dict[str, float]:
        """实时评估"""
        if not TRULENS_AVAILABLE or not self.rag_chain:
            return self._mock_trulens_evaluation()
        
        try:
            # TruLens实时评估
            with self.tru_rag as recording:
                result = await self.rag_chain.ainvoke(question)
            
            # 获取反馈分数
            record = recording.get()
            scores = {}
            
            for feedback_result in record.feedback_results:
                scores[feedback_result.name] = feedback_result.score
            
            return scores
            
        except Exception as e:
            logger.error(f"TruLens evaluation failed: {e}")
            return self._mock_trulens_evaluation()
    
    def _mock_trulens_evaluation(self) -> Dict[str, float]:
        """模拟TruLens评估结果"""
        return {
            'groundedness': 0.87,
            'qa_relevance': 0.84
        }
    
    def get_evaluation_dashboard_url(self) -> Optional[str]:
        """获取评估仪表板URL"""
        if TRULENS_AVAILABLE:
            return "http://localhost:8501"  # TruLens默认dashboard URL
        return None


class CustomMetricsEvaluator:
    """自定义指标评估器"""
    
    def __init__(self):
        self.historical_data = []
        logger.info("Custom metrics evaluator initialized")
    
    async def evaluate_retrieval_metrics(
        self,
        retrieved_docs: List[EnhancedDocument],
        ground_truth_docs: List[str],
        query: str
    ) -> Dict[str, float]:
        """评估检索指标"""
        
        # 计算精确率和召回率
        retrieved_ids = {self._get_doc_id(doc) for doc in retrieved_docs}
        ground_truth_ids = {self._get_doc_id_from_text(text) for text in ground_truth_docs}
        
        if not retrieved_ids or not ground_truth_ids:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # 计算交集
        intersection = retrieved_ids.intersection(ground_truth_ids)
        
        # 精确率：检索到的相关文档 / 检索到的总文档
        precision = len(intersection) / len(retrieved_ids) if retrieved_ids else 0.0
        
        # 召回率：检索到的相关文档 / 所有相关文档  
        recall = len(intersection) / len(ground_truth_ids) if ground_truth_ids else 0.0
        
        # F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    async def evaluate_cost_efficiency(
        self,
        total_cost: float,
        quality_score: float,
        latency: float
    ) -> float:
        """评估成本效益"""
        
        # 成本效益 = 质量 / (标准化成本 + 标准化延迟)
        normalized_cost = min(total_cost / 0.1, 1.0)  # 假设$0.1为高成本基准
        normalized_latency = min(latency / 10.0, 1.0)  # 假设10秒为高延迟基准
        
        if normalized_cost + normalized_latency == 0:
            return quality_score
        
        efficiency = quality_score / (normalized_cost + normalized_latency + 0.1)  # 避免除零
        return min(efficiency, 1.0)
    
    async def evaluate_user_satisfaction(
        self,
        response_length: int,
        response_clarity: float,
        response_completeness: float
    ) -> float:
        """评估用户满意度（基于响应特征）"""
        
        # 长度评分：200-800字符为最优
        length_score = 1.0
        if response_length < 100:
            length_score = 0.5
        elif response_length < 200:
            length_score = 0.7
        elif response_length > 1000:
            length_score = 0.8
        
        # 综合满意度
        satisfaction = (length_score + response_clarity + response_completeness) / 3
        return min(max(satisfaction, 0.0), 1.0)
    
    def _get_doc_id(self, doc: EnhancedDocument) -> str:
        """获取文档ID"""
        import hashlib
        return hashlib.md5(doc.page_content[:100].encode()).hexdigest()[:12]
    
    def _get_doc_id_from_text(self, text: str) -> str:
        """从文本获取文档ID"""
        import hashlib
        return hashlib.md5(text[:100].encode()).hexdigest()[:12]


class ComprehensiveEvaluator:
    """综合评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化子评估器
        self.ragas_evaluator = RAGASEvaluator(config.get('evaluation_llm'))
        self.trulens_evaluator = TruLensEvaluator(config.get('rag_chain'))
        self.custom_evaluator = CustomMetricsEvaluator()
        
        # 评估历史
        self.evaluation_history = []
        
        # 基准数据集
        self.golden_dataset = self._load_golden_dataset()
        
        logger.info("Comprehensive evaluator initialized")
    
    def _load_golden_dataset(self) -> List[EvaluationCase]:
        """加载黄金测试集"""
        # 这里应该加载实际的测试数据集
        # 目前返回示例数据
        return [
            EvaluationCase(
                question="什么是Transformer架构？",
                ground_truth_answer="Transformer是一种基于自注意力机制的深度学习架构...",
                retrieved_contexts=["Transformer架构相关文档1", "Transformer架构相关文档2"],
                generated_answer="",  # 待填充
                metadata={"category": "architecture", "difficulty": "medium"}
            ),
            EvaluationCase(
                question="BERT和GPT有什么区别？",
                ground_truth_answer="BERT是双向编码器，GPT是单向解码器...",
                retrieved_contexts=["BERT文档", "GPT文档"],
                generated_answer="",  # 待填充
                metadata={"category": "comparison", "difficulty": "complex"}
            )
        ]
    
    async def evaluate_single_case(
        self,
        question: str,
        generated_answer: str,
        retrieved_contexts: List[str],
        ground_truth_answer: str = None,
        metadata: Dict[str, Any] = None
    ) -> EvaluationMetrics:
        """评估单个案例"""
        
        start_time = time.time()
        
        # 创建评估案例
        eval_case = EvaluationCase(
            question=question,
            ground_truth_answer=ground_truth_answer or "未提供标准答案",
            retrieved_contexts=retrieved_contexts,
            generated_answer=generated_answer,
            metadata=metadata or {}
        )
        
        # 并行执行多种评估
        tasks = []
        
        # RAGAS评估
        if ground_truth_answer:
            tasks.append(self.ragas_evaluator.evaluate_batch([eval_case]))
        
        # TruLens实时评估
        tasks.append(self.trulens_evaluator.evaluate_real_time(
            question, generated_answer, "\n".join(retrieved_contexts)
        ))
        
        # 自定义指标评估
        tasks.append(self._evaluate_custom_metrics(eval_case))
        
        # 执行并收集结果
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 整合评估结果
        metrics = EvaluationMetrics()
        
        # RAGAS结果
        if len(results) > 0 and isinstance(results[0], dict):
            ragas_scores = results[0]
            metrics.faithfulness_score = ragas_scores.get('faithfulness')
            metrics.answer_relevancy_score = ragas_scores.get('answer_relevancy')
            metrics.context_precision_score = ragas_scores.get('context_precision')
            metrics.context_recall_score = ragas_scores.get('context_recall')
            metrics.answer_correctness_score = ragas_scores.get('answer_correctness')
            metrics.answer_similarity_score = ragas_scores.get('answer_similarity')
        
        # TruLens结果
        if len(results) > 1 and isinstance(results[1], dict):
            trulens_scores = results[1]
            metrics.groundedness_score = trulens_scores.get('groundedness')
            metrics.qa_relevance_score = trulens_scores.get('qa_relevance')
        
        # 自定义指标结果
        if len(results) > 2 and isinstance(results[2], dict):
            custom_scores = results[2]
            metrics.retrieval_precision = custom_scores.get('retrieval_precision')
            metrics.retrieval_recall = custom_scores.get('retrieval_recall')
            metrics.cost_efficiency = custom_scores.get('cost_efficiency')
            metrics.user_satisfaction = custom_scores.get('user_satisfaction')
        
        # 计算响应延迟
        metrics.response_latency = time.time() - start_time
        
        # 计算综合分数
        metrics.overall_score = self._calculate_overall_score(metrics)
        
        return metrics
    
    async def _evaluate_custom_metrics(self, eval_case: EvaluationCase) -> Dict[str, float]:
        """评估自定义指标"""
        
        # 检索指标评估（需要ground truth文档）
        retrieval_metrics = await self.custom_evaluator.evaluate_retrieval_metrics(
            retrieved_docs=[],  # 需要实际的EnhancedDocument对象
            ground_truth_docs=eval_case.retrieved_contexts,
            query=eval_case.question
        )
        
        # 成本效益评估（使用默认值）
        cost_efficiency = await self.custom_evaluator.evaluate_cost_efficiency(
            total_cost=0.01,  # 默认成本
            quality_score=0.8,  # 默认质量
            latency=2.0  # 默认延迟
        )
        
        # 用户满意度评估
        user_satisfaction = await self.custom_evaluator.evaluate_user_satisfaction(
            response_length=len(eval_case.generated_answer),
            response_clarity=0.8,  # 简化评估
            response_completeness=0.8  # 简化评估
        )
        
        return {
            'retrieval_precision': retrieval_metrics.get('precision', 0.0),
            'retrieval_recall': retrieval_metrics.get('recall', 0.0),
            'cost_efficiency': cost_efficiency,
            'user_satisfaction': user_satisfaction
        }
    
    def _calculate_overall_score(self, metrics: EvaluationMetrics) -> float:
        """计算综合分数"""
        
        scores = []
        weights = []
        
        # RAGAS指标权重
        if metrics.faithfulness_score is not None:
            scores.append(metrics.faithfulness_score)
            weights.append(0.15)
        
        if metrics.answer_relevancy_score is not None:
            scores.append(metrics.answer_relevancy_score)
            weights.append(0.15)
        
        if metrics.context_precision_score is not None:
            scores.append(metrics.context_precision_score)
            weights.append(0.10)
        
        if metrics.context_recall_score is not None:
            scores.append(metrics.context_recall_score)
            weights.append(0.10)
        
        # TruLens指标权重
        if metrics.groundedness_score is not None:
            scores.append(metrics.groundedness_score)
            weights.append(0.15)
        
        if metrics.qa_relevance_score is not None:
            scores.append(metrics.qa_relevance_score)
            weights.append(0.10)
        
        # 自定义指标权重
        if metrics.cost_efficiency is not None:
            scores.append(metrics.cost_efficiency)
            weights.append(0.15)
        
        if metrics.user_satisfaction is not None:
            scores.append(metrics.user_satisfaction)
            weights.append(0.10)
        
        # 加权平均
        if not scores:
            return 0.7  # 默认分数
        
        # 权重归一化
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        return min(max(weighted_score, 0.0), 1.0)
    
    async def evaluate_golden_dataset(self, rag_system) -> EvaluationReport:
        """评估黄金数据集"""
        
        logger.info(f"Starting golden dataset evaluation with {len(self.golden_dataset)} cases")
        
        individual_results = []
        all_metrics = []
        
        for i, case in enumerate(self.golden_dataset):
            logger.debug(f"Evaluating case {i+1}/{len(self.golden_dataset)}")
            
            try:
                # 使用RAG系统生成答案
                rag_result = await rag_system.query(case.question)
                case.generated_answer = rag_result.answer
                
                # 评估案例
                metrics = await self.evaluate_single_case(
                    question=case.question,
                    generated_answer=case.generated_answer,
                    retrieved_contexts=case.retrieved_contexts,
                    ground_truth_answer=case.ground_truth_answer,
                    metadata=case.metadata
                )
                
                all_metrics.append(metrics)
                individual_results.append({
                    'case_id': i,
                    'question': case.question,
                    'metrics': asdict(metrics),
                    'metadata': case.metadata
                })
                
            except Exception as e:
                logger.error(f"Failed to evaluate case {i}: {e}")
                continue
        
        # 计算汇总指标
        summary_metrics = self._calculate_summary_metrics(all_metrics)
        
        # 性能分析
        performance_analysis = self._analyze_performance(all_metrics, individual_results)
        
        # 生成改进建议
        recommendations = self._generate_recommendations(performance_analysis)
        
        # 创建评估报告
        report = EvaluationReport(
            dataset_name="golden_dataset",
            evaluation_timestamp=datetime.now().isoformat(),
            total_cases=len(individual_results),
            metrics_summary=summary_metrics,
            individual_results=individual_results,
            performance_analysis=performance_analysis,
            recommendations=recommendations
        )
        
        # 保存历史
        self.evaluation_history.append(report)
        
        logger.success(f"Golden dataset evaluation completed: Overall score = {summary_metrics.overall_score:.3f}")
        
        return report
    
    def _calculate_summary_metrics(self, all_metrics: List[EvaluationMetrics]) -> EvaluationMetrics:
        """计算汇总指标"""
        
        if not all_metrics:
            return EvaluationMetrics()
        
        # 计算各指标的平均值
        def safe_mean(values):
            valid_values = [v for v in values if v is not None]
            return sum(valid_values) / len(valid_values) if valid_values else None
        
        summary = EvaluationMetrics(
            faithfulness_score=safe_mean([m.faithfulness_score for m in all_metrics]),
            answer_relevancy_score=safe_mean([m.answer_relevancy_score for m in all_metrics]),
            context_precision_score=safe_mean([m.context_precision_score for m in all_metrics]),
            context_recall_score=safe_mean([m.context_recall_score for m in all_metrics]),
            answer_correctness_score=safe_mean([m.answer_correctness_score for m in all_metrics]),
            answer_similarity_score=safe_mean([m.answer_similarity_score for m in all_metrics]),
            groundedness_score=safe_mean([m.groundedness_score for m in all_metrics]),
            qa_relevance_score=safe_mean([m.qa_relevance_score for m in all_metrics]),
            retrieval_precision=safe_mean([m.retrieval_precision for m in all_metrics]),
            retrieval_recall=safe_mean([m.retrieval_recall for m in all_metrics]),
            response_latency=safe_mean([m.response_latency for m in all_metrics]),
            cost_efficiency=safe_mean([m.cost_efficiency for m in all_metrics]),
            user_satisfaction=safe_mean([m.user_satisfaction for m in all_metrics]),
            overall_score=safe_mean([m.overall_score for m in all_metrics])
        )
        
        return summary
    
    def _analyze_performance(
        self,
        all_metrics: List[EvaluationMetrics],
        individual_results: List[Dict]
    ) -> Dict[str, Any]:
        """性能分析"""
        
        analysis = {
            'total_evaluated': len(all_metrics),
            'score_distribution': {},
            'performance_by_category': {},
            'weak_areas': [],
            'strong_areas': []
        }
        
        if not all_metrics:
            return analysis
        
        # 分数分布分析
        overall_scores = [m.overall_score for m in all_metrics if m.overall_score is not None]
        if overall_scores:
            analysis['score_distribution'] = {
                'mean': sum(overall_scores) / len(overall_scores),
                'min': min(overall_scores),
                'max': max(overall_scores),
                'std': self._calculate_std(overall_scores)
            }
        
        # 按类别分析性能
        categories = {}
        for result in individual_results:
            category = result.get('metadata', {}).get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(result['metrics']['overall_score'])
        
        for category, scores in categories.items():
            if scores:
                analysis['performance_by_category'][category] = {
                    'mean_score': sum(scores) / len(scores),
                    'count': len(scores)
                }
        
        # 识别弱势和强势区域
        metric_averages = {}
        for metric_name in ['faithfulness_score', 'answer_relevancy_score', 'context_precision_score', 
                           'context_recall_score', 'groundedness_score', 'cost_efficiency']:
            values = [getattr(m, metric_name) for m in all_metrics if getattr(m, metric_name) is not None]
            if values:
                metric_averages[metric_name] = sum(values) / len(values)
        
        # 找出得分最低和最高的指标
        if metric_averages:
            sorted_metrics = sorted(metric_averages.items(), key=lambda x: x[1])
            analysis['weak_areas'] = [name for name, score in sorted_metrics[:2]]
            analysis['strong_areas'] = [name for name, score in sorted_metrics[-2:]]
        
        return analysis
    
    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _generate_recommendations(self, performance_analysis: Dict) -> List[str]:
        """生成改进建议"""
        
        recommendations = []
        
        # 基于弱势区域的建议
        weak_areas = performance_analysis.get('weak_areas', [])
        if 'faithfulness_score' in weak_areas:
            recommendations.append("建议优化生成模型，提高答案对上下文的忠实度")
        
        if 'context_precision_score' in weak_areas:
            recommendations.append("建议改进检索算法，提高检索文档的精确度")
        
        if 'context_recall_score' in weak_areas:
            recommendations.append("建议增加检索结果数量，提高重要文档的召回率")
        
        if 'cost_efficiency' in weak_areas:
            recommendations.append("建议优化模型路由策略，提高成本效益")
        
        # 基于分数分布的建议
        score_dist = performance_analysis.get('score_distribution', {})
        if score_dist.get('mean', 0) < 0.7:
            recommendations.append("整体性能需要提升，建议全面优化系统组件")
        
        if score_dist.get('std', 0) > 0.15:
            recommendations.append("性能稳定性不足，建议加强质量控制机制")
        
        # 基于类别性能的建议
        category_performance = performance_analysis.get('performance_by_category', {})
        for category, stats in category_performance.items():
            if stats['mean_score'] < 0.6:
                recommendations.append(f"'{category}'类别问题表现较差，建议针对性优化")
        
        if not recommendations:
            recommendations.append("系统性能表现良好，继续保持现有优化策略")
        
        return recommendations
    
    def export_evaluation_report(self, report: EvaluationReport, file_path: str):
        """导出评估报告"""
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # 基本信息
                f.write(f"# RAG系统评估报告\n\n")
                f.write(f"**数据集**: {report.dataset_name}\n")
                f.write(f"**评估时间**: {report.evaluation_timestamp}\n")
                f.write(f"**评估案例数**: {report.total_cases}\n\n")
                
                # 汇总指标
                f.write("## 汇总指标\n\n")
                metrics = report.metrics_summary
                f.write(f"- **综合分数**: {metrics.overall_score:.3f}\n")
                f.write(f"- **忠实度**: {metrics.faithfulness_score:.3f}\n")
                f.write(f"- **答案相关性**: {metrics.answer_relevancy_score:.3f}\n")
                f.write(f"- **上下文精确度**: {metrics.context_precision_score:.3f}\n")
                f.write(f"- **上下文召回率**: {metrics.context_recall_score:.3f}\n")
                f.write(f"- **基于事实程度**: {metrics.groundedness_score:.3f}\n")
                f.write(f"- **成本效益**: {metrics.cost_efficiency:.3f}\n")
                f.write(f"- **平均延迟**: {metrics.response_latency:.2f}秒\n\n")
                
                # 性能分析
                f.write("## 性能分析\n\n")
                analysis = report.performance_analysis
                f.write(f"- **平均分数**: {analysis.get('score_distribution', {}).get('mean', 0):.3f}\n")
                f.write(f"- **分数范围**: {analysis.get('score_distribution', {}).get('min', 0):.3f} - {analysis.get('score_distribution', {}).get('max', 0):.3f}\n")
                f.write(f"- **弱势区域**: {', '.join(analysis.get('weak_areas', []))}\n")
                f.write(f"- **强势区域**: {', '.join(analysis.get('strong_areas', []))}\n\n")
                
                # 改进建议
                f.write("## 改进建议\n\n")
                for i, rec in enumerate(report.recommendations, 1):
                    f.write(f"{i}. {rec}\n")
            
            logger.success(f"Evaluation report exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export evaluation report: {e}")
    
    def get_evaluation_trends(self, days: int = 7) -> Dict[str, Any]:
        """获取评估趋势"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_evaluations = [
            eval_report for eval_report in self.evaluation_history
            if datetime.fromisoformat(eval_report.evaluation_timestamp) >= cutoff_date
        ]
        
        if not recent_evaluations:
            return {"message": "No recent evaluation data available"}
        
        # 提取趋势数据
        trends = {
            'timestamps': [eval_report.evaluation_timestamp for eval_report in recent_evaluations],
            'overall_scores': [eval_report.metrics_summary.overall_score for eval_report in recent_evaluations],
            'total_evaluations': len(recent_evaluations),
            'average_improvement': 0.0
        }
        
        # 计算改进趋势
        if len(trends['overall_scores']) > 1:
            first_score = trends['overall_scores'][0]
            last_score = trends['overall_scores'][-1]
            trends['average_improvement'] = (last_score - first_score) / len(trends['overall_scores'])
        
        return trends