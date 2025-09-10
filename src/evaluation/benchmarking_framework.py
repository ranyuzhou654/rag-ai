# src/evaluation/benchmarking_framework.py
"""
基准测试框架
建立完整的量化评估指标和基准测试体系
用于客观评估系统性能提升和对比不同版本
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import asyncio
import time
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

# 统计分析
import statistics
from scipy import stats
import pandas as pd

# 本地组件
from .comprehensive_evaluation import (
    ComprehensiveEvaluator, 
    EvaluationMetrics, 
    EvaluationCase,
    EvaluationReport
)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    benchmark_name: str
    system_version: str
    timestamp: str
    
    # 核心性能指标
    overall_score: float
    latency_p50: float  # 中位延迟
    latency_p95: float  # 95分位延迟
    latency_p99: float  # 99分位延迟
    throughput: float   # QPS
    
    # RAG特定指标
    retrieval_precision: float
    retrieval_recall: float
    generation_quality: float
    factual_accuracy: float
    
    # 成本效益指标
    cost_per_query: float
    cost_efficiency_score: float
    
    # 可靠性指标
    success_rate: float
    error_rate: float
    
    # 资源使用指标
    memory_usage_mb: float
    gpu_utilization: float
    
    # 详细分数分布
    score_distribution: Dict[str, float]
    
    # 统计信息
    total_queries: int
    test_duration_seconds: float


@dataclass
class ComparisonResult:
    """对比结果"""
    baseline_version: str
    new_version: str
    
    # 性能变化
    overall_improvement: float  # 百分比变化
    latency_improvement: float
    throughput_improvement: float
    
    # 质量变化
    quality_improvement: float
    accuracy_improvement: float
    
    # 成本变化
    cost_reduction: float
    efficiency_improvement: float
    
    # 统计显著性
    statistical_significance: Dict[str, bool]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # 详细对比
    metric_comparisons: Dict[str, Dict[str, float]]


class StandardBenchmarks:
    """标准基准测试集"""
    
    def __init__(self):
        self.benchmarks = {
            'qa_accuracy': self._create_qa_benchmark(),
            'retrieval_quality': self._create_retrieval_benchmark(),
            'generation_speed': self._create_speed_benchmark(),
            'cost_efficiency': self._create_cost_benchmark(),
            'stress_test': self._create_stress_benchmark()
        }
        logger.info(f"Standard benchmarks initialized: {list(self.benchmarks.keys())}")
    
    def _create_qa_benchmark(self) -> List[EvaluationCase]:
        """创建问答准确性基准"""
        return [
            EvaluationCase(
                question="什么是Transformer架构？",
                ground_truth_answer="Transformer是一种基于自注意力机制的深度学习架构，由Google在2017年提出。它完全基于注意力机制，不使用循环神经网络或卷积神经网络。",
                retrieved_contexts=[],
                generated_answer="",
                metadata={"category": "definition", "difficulty": "medium", "domain": "nlp"}
            ),
            EvaluationCase(
                question="BERT和GPT有什么区别？",
                ground_truth_answer="BERT是双向编码器表示，使用掩码语言模型进行预训练；GPT是生成式预训练变换器，使用自回归语言模型。BERT更适合理解任务，GPT更适合生成任务。",
                retrieved_contexts=[],
                generated_answer="",
                metadata={"category": "comparison", "difficulty": "complex", "domain": "nlp"}
            ),
            EvaluationCase(
                question="如何优化深度学习模型的训练速度？",
                ground_truth_answer="可以通过以下方法优化：1）使用更大的批次大小；2）选择合适的优化器；3）应用混合精度训练；4）使用数据并行或模型并行；5）优化数据加载流程。",
                retrieved_contexts=[],
                generated_answer="",
                metadata={"category": "how_to", "difficulty": "complex", "domain": "ml"}
            ),
            EvaluationCase(
                question="什么是注意力机制？",
                ground_truth_answer="注意力机制是一种让模型在处理序列时能够关注到最相关部分的技术。它通过计算查询、键和值之间的相似度来分配权重，从而帮助模型聚焦于重要信息。",
                retrieved_contexts=[],
                generated_answer="",
                metadata={"category": "definition", "difficulty": "medium", "domain": "nlp"}
            ),
            EvaluationCase(
                question="为什么需要正则化技术？",
                ground_truth_answer="正则化技术用于防止模型过拟合，提高泛化能力。通过在损失函数中添加惩罚项或使用Dropout等技术，可以约束模型复杂度，使其在未见过的数据上表现更好。",
                retrieved_contexts=[],
                generated_answer="",
                metadata={"category": "why", "difficulty": "medium", "domain": "ml"}
            )
        ]
    
    def _create_retrieval_benchmark(self) -> List[EvaluationCase]:
        """创建检索质量基准"""
        return [
            EvaluationCase(
                question="深度学习中的梯度消失问题",
                ground_truth_answer="梯度消失是指在深层网络反向传播过程中，梯度逐层递减，导致浅层参数难以更新的问题。",
                retrieved_contexts=[
                    "梯度消失问题详解文档",
                    "深层网络训练技巧",
                    "激活函数选择指南"
                ],
                generated_answer="",
                metadata={"retrieval_focus": True, "expected_doc_types": ["technical", "tutorial"]}
            ),
            # 更多检索测试用例...
        ]
    
    def _create_speed_benchmark(self) -> List[EvaluationCase]:
        """创建速度基准测试"""
        # 简单查询，用于测试速度
        return [
            EvaluationCase(
                question="AI",
                ground_truth_answer="人工智能",
                retrieved_contexts=[],
                generated_answer="",
                metadata={"benchmark_type": "speed", "complexity": "simple"}
            ),
            EvaluationCase(
                question="机器学习",
                ground_truth_answer="机器学习是人工智能的一个分支",
                retrieved_contexts=[],
                generated_answer="",
                metadata={"benchmark_type": "speed", "complexity": "simple"}
            )
        ] * 50  # 重复50次用于速度测试
    
    def _create_cost_benchmark(self) -> List[EvaluationCase]:
        """创建成本效益基准"""
        return [
            EvaluationCase(
                question="解释神经网络的工作原理",
                ground_truth_answer="神经网络通过多层神经元的连接来模拟人脑处理信息的方式",
                retrieved_contexts=[],
                generated_answer="",
                metadata={"cost_focus": True, "expected_cost": 0.01}
            )
        ] * 20
    
    def _create_stress_benchmark(self) -> List[EvaluationCase]:
        """创建压力测试基准"""
        # 并发查询测试
        base_case = EvaluationCase(
            question="什么是深度学习？",
            ground_truth_answer="深度学习是机器学习的一个分支",
            retrieved_contexts=[],
            generated_answer="",
            metadata={"benchmark_type": "stress"}
        )
        return [base_case] * 100  # 100个并发查询


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        
    async def start_profiling(self, operation: str):
        """开始性能分析"""
        self.start_times[operation] = time.time()
        
    async def end_profiling(self, operation: str) -> float:
        """结束性能分析并返回耗时"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
            
            return duration
        return 0.0
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """获取性能统计"""
        stats = {}
        
        for operation, times in self.metrics.items():
            if times:
                stats[operation] = {
                    'count': len(times),
                    'mean': statistics.mean(times),
                    'median': statistics.median(times),
                    'std': statistics.stdev(times) if len(times) > 1 else 0.0,
                    'min': min(times),
                    'max': max(times),
                    'p95': np.percentile(times, 95),
                    'p99': np.percentile(times, 99)
                }
        
        return stats
    
    def reset(self):
        """重置性能指标"""
        self.metrics.clear()
        self.start_times.clear()


class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self, rag_system, evaluator: ComprehensiveEvaluator):
        self.rag_system = rag_system
        self.evaluator = evaluator
        self.profiler = PerformanceProfiler()
        self.benchmarks = StandardBenchmarks()
        
        logger.info("Benchmark runner initialized")
    
    async def run_benchmark(
        self,
        benchmark_name: str,
        system_version: str = "current",
        config_overrides: Dict[str, Any] = None
    ) -> BenchmarkResult:
        """运行单个基准测试"""
        
        logger.info(f"Running benchmark: {benchmark_name}")
        
        if benchmark_name not in self.benchmarks.benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        # 获取测试用例
        test_cases = self.benchmarks.benchmarks[benchmark_name]
        
        # 重置性能分析器
        self.profiler.reset()
        
        start_time = time.time()
        results = []
        errors = 0
        total_cost = 0.0
        
        # 执行测试用例
        if benchmark_name == 'stress_test':
            # 并发执行压力测试
            results, errors, total_cost = await self._run_concurrent_test(test_cases)
        else:
            # 顺序执行常规测试
            for i, case in enumerate(test_cases):
                try:
                    await self.profiler.start_profiling('single_query')
                    
                    # 执行查询
                    rag_result = await self.rag_system.query(case.question)
                    case.generated_answer = rag_result.answer
                    case.retrieved_contexts = [doc.page_content for doc in rag_result.source_documents]
                    
                    # 评估结果
                    metrics = await self.evaluator.evaluate_single_case(
                        question=case.question,
                        generated_answer=case.generated_answer,
                        retrieved_contexts=case.retrieved_contexts,
                        ground_truth_answer=case.ground_truth_answer,
                        metadata=case.metadata
                    )
                    
                    results.append(metrics)
                    total_cost += getattr(rag_result, 'cost', 0.01)  # 默认成本
                    
                    await self.profiler.end_profiling('single_query')
                    
                except Exception as e:
                    logger.error(f"Test case {i} failed: {e}")
                    errors += 1
        
        total_time = time.time() - start_time
        
        # 计算综合指标
        benchmark_result = self._calculate_benchmark_metrics(
            benchmark_name=benchmark_name,
            system_version=system_version,
            results=results,
            total_time=total_time,
            total_cost=total_cost,
            errors=errors,
            total_queries=len(test_cases)
        )
        
        logger.success(f"Benchmark {benchmark_name} completed: overall_score={benchmark_result.overall_score:.3f}")
        
        return benchmark_result
    
    async def _run_concurrent_test(
        self,
        test_cases: List[EvaluationCase]
    ) -> Tuple[List[EvaluationMetrics], int, float]:
        """运行并发测试"""
        
        async def execute_single_case(case):
            try:
                await self.profiler.start_profiling('concurrent_query')
                
                rag_result = await self.rag_system.query(case.question)
                case.generated_answer = rag_result.answer
                
                metrics = await self.evaluator.evaluate_single_case(
                    question=case.question,
                    generated_answer=case.generated_answer,
                    retrieved_contexts=[],
                    ground_truth_answer=case.ground_truth_answer
                )
                
                await self.profiler.end_profiling('concurrent_query')
                
                return metrics, getattr(rag_result, 'cost', 0.01)
                
            except Exception as e:
                logger.error(f"Concurrent test case failed: {e}")
                return None, 0.0
        
        # 并发执行
        tasks = [execute_single_case(case) for case in test_cases]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集结果
        results = []
        total_cost = 0.0
        errors = 0
        
        for task_result in task_results:
            if isinstance(task_result, Exception):
                errors += 1
            elif task_result[0] is not None:
                results.append(task_result[0])
                total_cost += task_result[1]
            else:
                errors += 1
        
        return results, errors, total_cost
    
    def _calculate_benchmark_metrics(
        self,
        benchmark_name: str,
        system_version: str,
        results: List[EvaluationMetrics],
        total_time: float,
        total_cost: float,
        errors: int,
        total_queries: int
    ) -> BenchmarkResult:
        """计算基准测试指标"""
        
        # 性能统计
        perf_stats = self.profiler.get_statistics()
        query_times = perf_stats.get('single_query', {}) or perf_stats.get('concurrent_query', {})
        
        # 计算延迟指标
        latency_p50 = query_times.get('median', 0.0)
        latency_p95 = query_times.get('p95', 0.0)
        latency_p99 = query_times.get('p99', 0.0)
        
        # 计算吞吐量
        throughput = total_queries / total_time if total_time > 0 else 0.0
        
        # 计算质量指标
        if results:
            overall_scores = [r.overall_score for r in results if r.overall_score is not None]
            overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
            
            # 检索指标
            retrieval_precisions = [r.retrieval_precision for r in results if r.retrieval_precision is not None]
            retrieval_precision = sum(retrieval_precisions) / len(retrieval_precisions) if retrieval_precisions else 0.0
            
            retrieval_recalls = [r.retrieval_recall for r in results if r.retrieval_recall is not None]
            retrieval_recall = sum(retrieval_recalls) / len(retrieval_recalls) if retrieval_recalls else 0.0
            
            # 生成质量指标
            faithfulness_scores = [r.faithfulness_score for r in results if r.faithfulness_score is not None]
            generation_quality = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
            
            # 事实准确性
            answer_correctness = [r.answer_correctness_score for r in results if r.answer_correctness_score is not None]
            factual_accuracy = sum(answer_correctness) / len(answer_correctness) if answer_correctness else 0.0
            
        else:
            overall_score = 0.0
            retrieval_precision = 0.0
            retrieval_recall = 0.0
            generation_quality = 0.0
            factual_accuracy = 0.0
        
        # 成本效益
        cost_per_query = total_cost / total_queries if total_queries > 0 else 0.0
        cost_efficiency_score = overall_score / (cost_per_query + 0.001) if overall_score > 0 else 0.0
        
        # 可靠性指标
        success_rate = (total_queries - errors) / total_queries if total_queries > 0 else 0.0
        error_rate = errors / total_queries if total_queries > 0 else 0.0
        
        # 分数分布
        score_distribution = {}
        if results:
            for metric_name in ['overall_score', 'faithfulness_score', 'answer_relevancy_score']:
                scores = [getattr(r, metric_name) for r in results if getattr(r, metric_name) is not None]
                if scores:
                    score_distribution[metric_name] = {
                        'mean': statistics.mean(scores),
                        'std': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                        'min': min(scores),
                        'max': max(scores)
                    }
        
        return BenchmarkResult(
            benchmark_name=benchmark_name,
            system_version=system_version,
            timestamp=datetime.now().isoformat(),
            overall_score=overall_score,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            throughput=throughput,
            retrieval_precision=retrieval_precision,
            retrieval_recall=retrieval_recall,
            generation_quality=generation_quality,
            factual_accuracy=factual_accuracy,
            cost_per_query=cost_per_query,
            cost_efficiency_score=cost_efficiency_score,
            success_rate=success_rate,
            error_rate=error_rate,
            memory_usage_mb=0.0,  # 简化实现
            gpu_utilization=0.0,  # 简化实现
            score_distribution=score_distribution,
            total_queries=total_queries,
            test_duration_seconds=total_time
        )
    
    async def run_full_benchmark_suite(
        self,
        system_version: str = "current"
    ) -> Dict[str, BenchmarkResult]:
        """运行完整基准测试套件"""
        
        logger.info("Running full benchmark suite")
        
        results = {}
        
        for benchmark_name in self.benchmarks.benchmarks.keys():
            try:
                result = await self.run_benchmark(benchmark_name, system_version)
                results[benchmark_name] = result
            except Exception as e:
                logger.error(f"Benchmark {benchmark_name} failed: {e}")
        
        logger.success(f"Full benchmark suite completed: {len(results)} benchmarks")
        
        return results
    
    def compare_benchmarks(
        self,
        baseline_results: Dict[str, BenchmarkResult],
        new_results: Dict[str, BenchmarkResult]
    ) -> ComparisonResult:
        """对比基准测试结果"""
        
        baseline_version = list(baseline_results.values())[0].system_version if baseline_results else "unknown"
        new_version = list(new_results.values())[0].system_version if new_results else "unknown"
        
        # 计算整体改进
        baseline_overall = statistics.mean([r.overall_score for r in baseline_results.values()])
        new_overall = statistics.mean([r.overall_score for r in new_results.values()])
        overall_improvement = ((new_overall - baseline_overall) / baseline_overall) * 100 if baseline_overall > 0 else 0
        
        # 计算延迟改进
        baseline_latency = statistics.mean([r.latency_p50 for r in baseline_results.values()])
        new_latency = statistics.mean([r.latency_p50 for r in new_results.values()])
        latency_improvement = ((baseline_latency - new_latency) / baseline_latency) * 100 if baseline_latency > 0 else 0
        
        # 计算成本降低
        baseline_cost = statistics.mean([r.cost_per_query for r in baseline_results.values()])
        new_cost = statistics.mean([r.cost_per_query for r in new_results.values()])
        cost_reduction = ((baseline_cost - new_cost) / baseline_cost) * 100 if baseline_cost > 0 else 0
        
        # 详细指标对比
        metric_comparisons = {}
        common_benchmarks = set(baseline_results.keys()) & set(new_results.keys())
        
        for benchmark in common_benchmarks:
            baseline = baseline_results[benchmark]
            new = new_results[benchmark]
            
            metric_comparisons[benchmark] = {
                'overall_score_change': new.overall_score - baseline.overall_score,
                'latency_change': new.latency_p50 - baseline.latency_p50,
                'throughput_change': new.throughput - baseline.throughput,
                'cost_change': new.cost_per_query - baseline.cost_per_query
            }
        
        return ComparisonResult(
            baseline_version=baseline_version,
            new_version=new_version,
            overall_improvement=overall_improvement,
            latency_improvement=latency_improvement,
            throughput_improvement=0.0,  # 简化实现
            quality_improvement=overall_improvement,
            accuracy_improvement=0.0,  # 简化实现
            cost_reduction=cost_reduction,
            efficiency_improvement=(overall_improvement - cost_reduction) / 2,
            statistical_significance={},  # 需要更多数据点进行显著性检验
            confidence_intervals={},
            metric_comparisons=metric_comparisons
        )


class BenchmarkingFramework:
    """基准测试框架"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_storage = Path(config.get('benchmark_results_path', 'data/benchmarks'))
        self.results_storage.mkdir(parents=True, exist_ok=True)
        
        # 历史结果
        self.historical_results = self._load_historical_results()
        
        logger.info("Benchmarking framework initialized")
    
    def _load_historical_results(self) -> Dict[str, Dict[str, BenchmarkResult]]:
        """加载历史基准测试结果"""
        
        historical = {}
        
        try:
            for result_file in self.results_storage.glob('*.json'):
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    version = data.get('system_version', 'unknown')
                    if version not in historical:
                        historical[version] = {}
                    
                    # 重构BenchmarkResult对象
                    benchmark_name = data.get('benchmark_name', result_file.stem)
                    historical[version][benchmark_name] = BenchmarkResult(**data)
                    
        except Exception as e:
            logger.warning(f"Failed to load historical results: {e}")
        
        return historical
    
    async def run_continuous_benchmarking(
        self,
        rag_system,
        evaluator: ComprehensiveEvaluator,
        system_version: str
    ):
        """运行持续基准测试"""
        
        runner = BenchmarkRunner(rag_system, evaluator)
        
        # 运行完整基准测试套件
        results = await runner.run_full_benchmark_suite(system_version)
        
        # 保存结果
        self._save_benchmark_results(results, system_version)
        
        # 与历史结果对比
        if self.historical_results:
            latest_baseline = max(self.historical_results.keys())
            if latest_baseline != system_version:
                comparison = runner.compare_benchmarks(
                    self.historical_results[latest_baseline],
                    results
                )
                self._save_comparison_result(comparison)
        
        return results
    
    def _save_benchmark_results(
        self,
        results: Dict[str, BenchmarkResult],
        system_version: str
    ):
        """保存基准测试结果"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for benchmark_name, result in results.items():
            file_path = self.results_storage / f"{system_version}_{benchmark_name}_{timestamp}.json"
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(asdict(result), f, indent=2)
                    
            except Exception as e:
                logger.error(f"Failed to save benchmark result: {e}")
        
        # 更新历史记录
        if system_version not in self.historical_results:
            self.historical_results[system_version] = {}
        self.historical_results[system_version].update(results)
        
        logger.info(f"Benchmark results saved for version {system_version}")
    
    def _save_comparison_result(self, comparison: ComparisonResult):
        """保存对比结果"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = self.results_storage / f"comparison_{comparison.baseline_version}_vs_{comparison.new_version}_{timestamp}.json"
        
        try:
            with open(file_path, 'w') as f:
                json.dump(asdict(comparison), f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save comparison result: {e}")
    
    def get_performance_trends(
        self,
        metric: str = 'overall_score',
        days: int = 30
    ) -> Dict[str, Any]:
        """获取性能趋势"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        trends = {}
        for version, benchmarks in self.historical_results.items():
            for benchmark_name, result in benchmarks.items():
                result_date = datetime.fromisoformat(result.timestamp)
                
                if result_date >= cutoff_date:
                    if benchmark_name not in trends:
                        trends[benchmark_name] = {'timestamps': [], 'values': []}
                    
                    trends[benchmark_name]['timestamps'].append(result.timestamp)
                    trends[benchmark_name]['values'].append(getattr(result, metric, 0))
        
        return trends
    
    def generate_performance_report(self, output_path: str):
        """生成性能报告"""
        
        report = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_versions': len(self.historical_results),
            'summary': {},
            'trends': {},
            'latest_results': {}
        }
        
        # 最新结果
        if self.historical_results:
            latest_version = max(self.historical_results.keys())
            report['latest_results'] = {
                'version': latest_version,
                'benchmarks': {
                    name: {
                        'overall_score': result.overall_score,
                        'latency_p50': result.latency_p50,
                        'throughput': result.throughput,
                        'cost_per_query': result.cost_per_query
                    }
                    for name, result in self.historical_results[latest_version].items()
                }
            }
        
        # 性能趋势
        for metric in ['overall_score', 'latency_p50', 'cost_per_query']:
            report['trends'][metric] = self.get_performance_trends(metric, days=90)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.success(f"Performance report generated: {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
    
    def export_benchmark_dashboard_data(self) -> Dict[str, Any]:
        """导出基准测试仪表板数据"""
        
        if not self.historical_results:
            return {"message": "No benchmark data available"}
        
        latest_version = max(self.historical_results.keys())
        latest_results = self.historical_results[latest_version]
        
        # 计算汇总统计
        all_scores = [result.overall_score for result in latest_results.values()]
        all_latencies = [result.latency_p50 for result in latest_results.values()]
        all_costs = [result.cost_per_query for result in latest_results.values()]
        
        dashboard_data = {
            'system_version': latest_version,
            'benchmark_count': len(latest_results),
            'summary_stats': {
                'avg_overall_score': statistics.mean(all_scores),
                'avg_latency': statistics.mean(all_latencies),
                'avg_cost': statistics.mean(all_costs),
                'total_queries_tested': sum(result.total_queries for result in latest_results.values())
            },
            'benchmark_details': {
                name: {
                    'overall_score': result.overall_score,
                    'latency_p50': result.latency_p50,
                    'latency_p95': result.latency_p95,
                    'throughput': result.throughput,
                    'success_rate': result.success_rate,
                    'cost_per_query': result.cost_per_query
                }
                for name, result in latest_results.items()
            },
            'performance_trends': self.get_performance_trends('overall_score', days=30),
            'version_history': list(self.historical_results.keys())
        }
        
        return dashboard_data