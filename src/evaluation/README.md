# 评估模块 (Evaluation Module)

RAG系统的多维度量化评估与基准测试框架，集成RAGAS、TruLens和自定义指标，建立完整的性能监控与持续改进体系。支持黄金数据集评估、基准测试、性能趋势分析和自动化质量控制。

## 🏗️ 核心架构

### 1. 综合评估框架 (Comprehensive Evaluation Framework)

```python
# src/evaluation/comprehensive_evaluation.py
@dataclass
class EvaluationMetrics:
    """多维度评估指标集合"""
    # RAGAS指标 - 基于学术标准的RAG评估
    faithfulness_score: Optional[float] = None      # 答案忠实度
    answer_relevancy_score: Optional[float] = None  # 答案相关性
    context_precision_score: Optional[float] = None # 上下文精确度
    context_recall_score: Optional[float] = None    # 上下文召回率
    answer_correctness_score: Optional[float] = None # 答案正确性
    answer_similarity_score: Optional[float] = None  # 答案相似度
    
    # TruLens指标 - 实时反馈与可解释性
    groundedness_score: Optional[float] = None      # 基于事实程度
    qa_relevance_score: Optional[float] = None      # 问答相关性
    
    # 自定义业务指标
    retrieval_precision: Optional[float] = None     # 检索精确率
    retrieval_recall: Optional[float] = None        # 检索召回率
    response_latency: Optional[float] = None        # 响应延迟
    cost_efficiency: Optional[float] = None         # 成本效益
    user_satisfaction: Optional[float] = None       # 用户满意度
    
    # 综合评估
    overall_score: Optional[float] = None           # 加权综合分数
    evaluation_timestamp: str = None               # 评估时间戳
    notes: List[str] = field(default_factory=list) # 评估备注
```

### 2. 三层评估器架构

#### RAGAS评估器 (RAGASEvaluator)
**学术标准的RAG系统评估框架**

```python
class RAGASEvaluator:
    """基于RAGAS框架的学术级评估器"""
    
    def __init__(self, llm_model=None):
        self.metrics = [
            faithfulness,        # 答案与上下文的一致性
            answer_relevancy,    # 答案与问题的相关性
            context_precision,   # 检索上下文的精确度
            context_recall,      # 检索上下文的召回率
            answer_correctness,  # 答案的事实正确性
            answer_similarity    # 答案与标准答案的相似度
        ]
    
    async def evaluate_batch(self, evaluation_cases: List[EvaluationCase]) -> Dict[str, float]:
        """批量评估RAG系统性能"""
        if not RAGAS_AVAILABLE:
            return self._mock_ragas_evaluation()  # 优雅降级
        
        dataset = self._prepare_ragas_dataset(evaluation_cases)
        results = evaluate(dataset=dataset, metrics=self.metrics, llm=self.llm_model)
        
        return {
            'faithfulness': results['faithfulness'],
            'answer_relevancy': results['answer_relevancy'],
            'context_precision': results['context_precision'],
            'context_recall': results['context_recall']
        }
```

**核心评估维度:**
- **忠实度 (Faithfulness)**: 生成答案与检索上下文的一致性
- **答案相关性 (Answer Relevancy)**: 答案对用户问题的直接相关程度
- **上下文精确度 (Context Precision)**: 检索到的相关文档比例
- **上下文召回率 (Context Recall)**: 应检索文档的实际检索比例

#### TruLens评估器 (TruLensEvaluator)
**实时反馈与可解释性评估**

```python
class TruLensEvaluator:
    """基于TruLens的实时评估与监控"""
    
    def __init__(self, rag_chain=None):
        if rag_chain:
            self.groundedness = Feedback(Groundedness().groundedness_measure_with_cot_reasons)
            self.qa_relevance = Feedback(self._qa_relevance_evaluator)
            
            # 包装RAG链进行实时监控
            self.tru_rag = TruChain(
                rag_chain,
                feedbacks=[self.groundedness, self.qa_relevance]
            )
    
    async def evaluate_real_time(self, question: str, answer: str, context: str) -> Dict[str, float]:
        """实时评估RAG查询执行"""
        with self.tru_rag as recording:
            result = await self.rag_chain.ainvoke(question)
        
        # 获取实时反馈分数
        record = recording.get()
        scores = {feedback_result.name: feedback_result.score 
                 for feedback_result in record.feedback_results}
        
        return scores
```

**TruLens特色功能:**
- **基于事实程度 (Groundedness)**: 答案是否基于提供的上下文
- **实时监控**: RAG链执行过程的实时反馈
- **可解释性**: 评估决策的推理过程追踪
- **仪表板**: 自动生成性能可视化仪表板

#### 自定义指标评估器 (CustomMetricsEvaluator)
**业务导向的定制化评估**

```python
class CustomMetricsEvaluator:
    """面向业务需求的自定义评估指标"""
    
    async def evaluate_retrieval_metrics(
        self, retrieved_docs: List[EnhancedDocument], 
        ground_truth_docs: List[str], query: str
    ) -> Dict[str, float]:
        """评估检索系统的精确率和召回率"""
        retrieved_ids = {self._get_doc_id(doc) for doc in retrieved_docs}
        ground_truth_ids = {self._get_doc_id_from_text(text) for text in ground_truth_docs}
        
        intersection = retrieved_ids.intersection(ground_truth_ids)
        
        precision = len(intersection) / len(retrieved_ids) if retrieved_ids else 0.0
        recall = len(intersection) / len(ground_truth_ids) if ground_truth_ids else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    async def evaluate_cost_efficiency(
        self, total_cost: float, quality_score: float, latency: float
    ) -> float:
        """评估系统的成本效益比"""
        normalized_cost = min(total_cost / 0.1, 1.0)
        normalized_latency = min(latency / 10.0, 1.0)
        
        efficiency = quality_score / (normalized_cost + normalized_latency + 0.1)
        return min(efficiency, 1.0)
    
    async def evaluate_user_satisfaction(
        self, response_length: int, response_clarity: float, response_completeness: float
    ) -> float:
        """基于响应特征的用户满意度评估"""
        # 长度评分：200-800字符为最优
        length_score = 1.0 if 200 <= response_length <= 800 else 0.7
        
        satisfaction = (length_score + response_clarity + response_completeness) / 3
        return min(max(satisfaction, 0.0), 1.0)
```

### 3. 综合评估器 (ComprehensiveEvaluator)

**统一的评估编排和结果聚合**

```python
class ComprehensiveEvaluator:
    """多评估器的统一编排与结果聚合"""
    
    async def evaluate_single_case(
        self, question: str, generated_answer: str, 
        retrieved_contexts: List[str], ground_truth_answer: str = None
    ) -> EvaluationMetrics:
        """单案例的多维度评估"""
        
        # 并行执行多种评估
        tasks = [
            self.ragas_evaluator.evaluate_batch([eval_case]),
            self.trulens_evaluator.evaluate_real_time(question, generated_answer, context),
            self._evaluate_custom_metrics(eval_case)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 整合评估结果
        metrics = EvaluationMetrics()
        self._merge_evaluation_results(metrics, results)
        
        # 计算加权综合分数
        metrics.overall_score = self._calculate_overall_score(metrics)
        
        return metrics
    
    def _calculate_overall_score(self, metrics: EvaluationMetrics) -> float:
        """多指标加权聚合算法"""
        scores, weights = [], []
        
        # RAGAS指标权重分配
        if metrics.faithfulness_score is not None:
            scores.append(metrics.faithfulness_score)
            weights.append(0.15)  # 忠实度权重15%
        
        if metrics.answer_relevancy_score is not None:
            scores.append(metrics.answer_relevancy_score)
            weights.append(0.15)  # 相关性权重15%
        
        # TruLens指标权重
        if metrics.groundedness_score is not None:
            scores.append(metrics.groundedness_score)
            weights.append(0.15)  # 基于事实程度权重15%
        
        # 自定义指标权重
        if metrics.cost_efficiency is not None:
            scores.append(metrics.cost_efficiency)
            weights.append(0.15)  # 成本效益权重15%
        
        # 权重归一化与加权平均
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        return sum(score * weight for score, weight in zip(scores, weights))
```

## 🎯 基准测试框架 (Benchmarking Framework)

### 标准基准测试集 (StandardBenchmarks)

```python
# src/evaluation/benchmarking_framework.py
class StandardBenchmarks:
    """标准化基准测试数据集"""
    
    def __init__(self):
        self.benchmarks = {
            'qa_accuracy': self._create_qa_benchmark(),         # 问答准确性
            'retrieval_quality': self._create_retrieval_benchmark(), # 检索质量
            'generation_speed': self._create_speed_benchmark(),     # 生成速度
            'cost_efficiency': self._create_cost_benchmark(),       # 成本效益
            'stress_test': self._create_stress_benchmark()          # 压力测试
        }
    
    def _create_qa_benchmark(self) -> List[EvaluationCase]:
        """问答准确性基准测试集"""
        return [
            EvaluationCase(
                question="什么是Transformer架构？",
                ground_truth_answer="Transformer是基于自注意力机制的深度学习架构...",
                metadata={"category": "definition", "difficulty": "medium", "domain": "nlp"}
            ),
            EvaluationCase(
                question="BERT和GPT有什么区别？",
                ground_truth_answer="BERT是双向编码器，GPT是生成式预训练变换器...",
                metadata={"category": "comparison", "difficulty": "complex", "domain": "nlp"}
            )
        ]
```

### 性能分析器 (PerformanceProfiler)

```python
class PerformanceProfiler:
    """系统性能监控与分析"""
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """获取详细性能统计"""
        stats = {}
        for operation, times in self.metrics.items():
            if times:
                stats[operation] = {
                    'count': len(times),
                    'mean': statistics.mean(times),
                    'median': statistics.median(times),
                    'p95': np.percentile(times, 95),    # 95分位延迟
                    'p99': np.percentile(times, 99),    # 99分位延迟
                    'std': statistics.stdev(times) if len(times) > 1 else 0.0
                }
        return stats
```

### 基准测试运行器 (BenchmarkRunner)

```python
class BenchmarkRunner:
    """基准测试执行与结果分析"""
    
    async def run_benchmark(
        self, benchmark_name: str, system_version: str = "current"
    ) -> BenchmarkResult:
        """执行单项基准测试"""
        
        test_cases = self.benchmarks.benchmarks[benchmark_name]
        
        if benchmark_name == 'stress_test':
            # 并发压力测试
            results, errors, total_cost = await self._run_concurrent_test(test_cases)
        else:
            # 顺序功能测试
            results, errors, total_cost = await self._run_sequential_test(test_cases)
        
        # 计算综合基准指标
        return self._calculate_benchmark_metrics(
            benchmark_name, system_version, results, total_time, total_cost, errors
        )
    
    async def _run_concurrent_test(self, test_cases: List[EvaluationCase]) -> Tuple:
        """并发执行压力测试"""
        async def execute_single_case(case):
            await self.profiler.start_profiling('concurrent_query')
            rag_result = await self.rag_system.query(case.question)
            metrics = await self.evaluator.evaluate_single_case(...)
            await self.profiler.end_profiling('concurrent_query')
            return metrics, getattr(rag_result, 'cost', 0.01)
        
        tasks = [execute_single_case(case) for case in test_cases]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return self._collect_results(task_results)
```

## 📊 评估流水线 (Evaluation Pipeline)

### LLM评估器 (LLMEvaluator)

```python
# src/evaluation/evaluation_pipeline.py
class LLMEvaluator:
    """基于大语言模型的智能评估"""
    
    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """评估答案忠实度"""
        prompt = f"""请评估生成的答案是否忠实于给定的上下文材料。评分标准：
        - 5分：答案完全基于上下文，没有添加额外信息
        - 4分：答案主要基于上下文，有少量合理推理
        - 3分：答案部分基于上下文，有一定推理成分
        - 2分：答案与上下文有关，但包含较多外部信息
        - 1分：答案与上下文关联很少或包含错误信息
        
        上下文：{context}
        生成的答案：{answer}
        
        请只输出1-5的数字评分："""
        
        score = self._get_llm_score(prompt, scale=5)
        return score / 5.0  # 归一化到0-1
    
    def evaluate_answer_relevancy(self, query: str, answer: str) -> float:
        """评估答案相关性"""
        prompt = f"""请评估生成的答案对用户问题的相关性。评分标准：
        - 5分：完全回答了问题，高度相关
        - 4分：基本回答了问题，相关性好
        - 3分：部分回答了问题，相关性一般
        - 2分：勉强涉及问题，相关性较差
        - 1分：基本没有回答问题，不相关
        
        用户问题：{query}
        生成的答案：{answer}
        
        请只输出1-5的数字评分："""
        
        score = self._get_llm_score(prompt, scale=5)
        return score / 5.0
```

### 语义评估器 (SemanticEvaluator)

```python
class SemanticEvaluator:
    """基于语义向量的评估器"""
    
    def __init__(self, embedding_model: str = "BAAI/bge-m3"):
        self.embedder = SentenceTransformer(embedding_model, device="auto")
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度"""
        embeddings = self.embedder.encode([text1, text2], convert_to_numpy=True)
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
```

### 黄金测试集生成器 (GoldenTestSetGenerator)

```python
class GoldenTestSetGenerator:
    """AI领域专业测试集构建"""
    
    def create_ai_domain_test_set(self) -> List[TestCase]:
        """创建AI领域黄金测试集"""
        return [
            # 简单事实性问题
            TestCase(
                query="什么是Transformer模型？",
                expected_answer="Transformer是基于注意力机制的神经网络架构...",
                reference_documents=["transformer", "attention mechanism"],
                difficulty_level="easy", query_type="factual"
            ),
            
            # 中等分析性问题
            TestCase(
                query="Transformer模型相比于RNN有哪些优势？",
                expected_answer="Transformer的主要优势包括并行计算、长距离依赖处理...",
                reference_documents=["transformer vs rnn", "parallel processing"],
                difficulty_level="medium", query_type="analytical"
            ),
            
            # 困难对比性问题
            TestCase(
                query="请详细对比LoRA和QLoRA在大模型微调中的异同点",
                expected_answer="LoRA和QLoRA都是参数高效微调方法...",
                reference_documents=["LoRA", "QLoRA", "parameter efficient fine-tuning"],
                difficulty_level="hard", query_type="comparative"
            )
        ]
```

## 🔧 使用方法

### 1. 基础评估流程

```python
# 初始化评估器
config = {
    'embedding_model': 'BAAI/bge-m3',
    'llm_model': 'Qwen/Qwen2-7B-Instruct',
    'evaluation_llm': your_llm_instance,
    'rag_chain': your_rag_chain
}

evaluator = ComprehensiveEvaluator(config)

# 单案例评估
metrics = await evaluator.evaluate_single_case(
    question="什么是Transformer？",
    generated_answer="Transformer是一种基于注意力机制的架构...",
    retrieved_contexts=["相关文档1", "相关文档2"],
    ground_truth_answer="标准答案..."
)

print(f"综合评分: {metrics.overall_score:.3f}")
print(f"忠实度: {metrics.faithfulness_score:.3f}")
print(f"相关性: {metrics.answer_relevancy_score:.3f}")
```

### 2. 黄金数据集评估

```python
# 黄金数据集批量评估
report = await evaluator.evaluate_golden_dataset(rag_system)

# 导出评估报告
evaluator.export_evaluation_report(report, "evaluation_report.md")

# 查看性能分析
print("弱势区域:", report.performance_analysis['weak_areas'])
print("改进建议:", report.recommendations)
```

### 3. 基准测试执行

```python
# 初始化基准测试框架
framework = BenchmarkingFramework(config)
runner = BenchmarkRunner(rag_system, evaluator)

# 运行完整基准测试套件
results = await runner.run_full_benchmark_suite("v1.0")

# 版本间性能对比
comparison = runner.compare_benchmarks(baseline_results, new_results)
print(f"整体改进: {comparison.overall_improvement:.1f}%")
print(f"延迟优化: {comparison.latency_improvement:.1f}%")
```

### 4. 持续性能监控

```python
# 持续基准测试
results = await framework.run_continuous_benchmarking(
    rag_system, evaluator, system_version="v1.1"
)

# 性能趋势分析
trends = framework.get_performance_trends(metric='overall_score', days=30)

# 生成性能报告
framework.generate_performance_report("performance_report.json")

# 导出仪表板数据
dashboard_data = framework.export_benchmark_dashboard_data()
```

## ⚙️ 配置参数

### 评估配置 (Evaluation Configuration)

```python
EVALUATION_CONFIG = {
    # 模型配置
    'embedding_model': 'BAAI/bge-m3',           # 语义相似度计算模型
    'llm_model': 'Qwen/Qwen2-7B-Instruct',     # LLM评估模型
    'device': 'auto',                           # 设备选择
    'HUGGING_FACE_TOKEN': None,                 # HF访问令牌
    
    # RAGAS配置
    'ragas_enabled': True,                      # 启用RAGAS评估
    'ragas_metrics': [                          # RAGAS评估指标
        'faithfulness', 'answer_relevancy', 
        'context_precision', 'context_recall'
    ],
    
    # TruLens配置
    'trulens_enabled': False,                   # 启用TruLens评估
    'trulens_dashboard_port': 8501,             # TruLens仪表板端口
    
    # 基准测试配置
    'benchmark_results_path': 'data/benchmarks', # 基准测试结果存储路径
    'golden_dataset_path': 'data/evaluation/golden_test_set.json',
    
    # 性能配置
    'concurrent_test_limit': 10,                # 并发测试限制
    'evaluation_timeout': 300,                  # 评估超时时间(秒)
    
    # 权重配置
    'metric_weights': {                         # 综合评分权重
        'faithfulness': 0.15,
        'answer_relevancy': 0.15,
        'context_precision': 0.10,
        'context_recall': 0.10,
        'groundedness': 0.15,
        'cost_efficiency': 0.15,
        'user_satisfaction': 0.10,
        'response_latency': 0.10
    }
}
```

## 📈 性能指标体系

### 核心质量指标

| 指标类别 | 指标名称 | 计算方法 | 目标值 | 权重 |
|---------|---------|----------|--------|------|
| **内容质量** | 忠实度 (Faithfulness) | LLM评估答案与上下文一致性 | >0.85 | 15% |
| | 答案相关性 (Answer Relevancy) | 语义相似度 + LLM评分 | >0.80 | 15% |
| | 答案正确性 (Answer Correctness) | 与标准答案对比 | >0.75 | 10% |
| **检索质量** | 上下文精确度 (Context Precision) | 相关文档 / 检索文档 | >0.70 | 10% |
| | 上下文召回率 (Context Recall) | 检索到的相关文档 / 所有相关文档 | >0.65 | 10% |
| **系统性能** | 响应延迟 (Response Latency) | 端到端查询时间 | <3.0s | 10% |
| | 成本效益 (Cost Efficiency) | 质量分数 / 成本 | >80 | 15% |
| **用户体验** | 用户满意度 (User Satisfaction) | 多维度综合评估 | >0.75 | 15% |

### 基准测试指标

```python
@dataclass
class BenchmarkResult:
    """基准测试结果数据结构"""
    # 核心性能指标
    overall_score: float          # 综合评分 (0-1)
    latency_p50: float           # 中位延迟 (秒)
    latency_p95: float           # 95分位延迟 (秒) 
    latency_p99: float           # 99分位延迟 (秒)
    throughput: float            # 吞吐量 (QPS)
    
    # RAG特定指标
    retrieval_precision: float   # 检索精确率
    retrieval_recall: float      # 检索召回率
    generation_quality: float    # 生成质量
    factual_accuracy: float      # 事实准确性
    
    # 成本效益指标
    cost_per_query: float        # 单次查询成本
    cost_efficiency_score: float # 成本效益分数
    
    # 可靠性指标
    success_rate: float          # 成功率
    error_rate: float            # 错误率
```

## 🚀 扩展功能

### 1. 自定义评估指标

```python
class CustomBusinessEvaluator:
    """业务定制化评估器"""
    
    async def evaluate_domain_specificity(self, answer: str, domain: str) -> float:
        """评估答案的领域专业性"""
        domain_keywords = self._get_domain_keywords(domain)
        keyword_coverage = sum(1 for keyword in domain_keywords if keyword in answer.lower())
        return keyword_coverage / len(domain_keywords)
    
    async def evaluate_response_completeness(self, question: str, answer: str) -> float:
        """评估回答完整性"""
        question_aspects = self._extract_question_aspects(question)
        covered_aspects = self._identify_covered_aspects(answer, question_aspects)
        return len(covered_aspects) / len(question_aspects)
```

### 2. A/B测试支持

```python
class ABTestEvaluator:
    """A/B测试评估框架"""
    
    async def compare_system_versions(
        self, version_a: str, version_b: str, test_cases: List[TestCase]
    ) -> ABTestResult:
        """对比两个系统版本的性能"""
        
        results_a = await self._evaluate_version(version_a, test_cases)
        results_b = await self._evaluate_version(version_b, test_cases)
        
        # 统计显著性检验
        significance = self._statistical_significance_test(results_a, results_b)
        
        return ABTestResult(
            version_a=version_a, version_b=version_b,
            performance_difference=results_b.mean() - results_a.mean(),
            statistical_significance=significance,
            confidence_interval=self._calculate_confidence_interval(results_a, results_b)
        )
```

### 3. 实时监控集成

```python
class RealTimeMonitor:
    """实时评估监控"""
    
    def __init__(self, evaluator: ComprehensiveEvaluator):
        self.evaluator = evaluator
        self.metrics_buffer = []
        self.alert_thresholds = {
            'overall_score': 0.7,
            'response_latency': 5.0,
            'error_rate': 0.1
        }
    
    async def monitor_query(self, query_result) -> None:
        """实时监控单次查询"""
        metrics = await self.evaluator.evaluate_single_case(...)
        self.metrics_buffer.append(metrics)
        
        # 性能告警检查
        await self._check_performance_alerts(metrics)
        
        # 定期汇总报告
        if len(self.metrics_buffer) >= 100:
            await self._generate_monitoring_report()
            self.metrics_buffer.clear()
```

## 📋 最佳实践

### 1. 评估策略设计

- **分层评估**: 结合学术指标(RAGAS)、实时反馈(TruLens)和业务指标
- **基准化**: 建立标准化测试集，支持版本间横向对比
- **自动化**: 集成CI/CD流程，实现持续评估
- **可解释性**: 提供评估决策的详细推理过程

### 2. 性能优化指导

- **弱势区域识别**: 自动分析性能瓶颈，提供针对性优化建议
- **成本效益平衡**: 在质量和成本间找到最优平衡点
- **实时调优**: 基于评估结果动态调整系统参数

### 3. 质量保障流程

- **回归测试**: 每次更新后自动运行完整评估套件
- **性能基线**: 维护性能基线，防止质量回退
- **异常告警**: 实时监控关键指标，及时发现问题

评估模块为RAG系统提供了全方位的质量保障和持续改进能力，通过多维度、多层次的评估体系，确保系统始终保持最佳性能状态。