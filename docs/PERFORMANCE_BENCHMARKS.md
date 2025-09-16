# 性能基准指南

项目内置了一套可重复的性能与质量评估框架，位于 [`src/evaluation/benchmarking_framework.py`](../src/evaluation/benchmarking_framework.py)。本指南介绍标准基准、关键指标以及如何运行与追踪结果。

## 1. 标准基准

`BenchmarkingFramework` 初始化时会创建 [`StandardBenchmarks`](../src/evaluation/benchmarking_framework.py#L103-L207) 中定义的五类测试：

| 名称 | 说明 | 典型样例 |
| --- | --- | --- |
| `qa_accuracy` | 评估问答正确性与事实性，使用定义、比较、流程类问题。 | “BERT 和 GPT 的区别？” |
| `retrieval_quality` | 检测检索召回与精度，附带期望命中文档标题。 | “深度学习中的梯度消失问题” |
| `generation_speed` | 多次提交简单查询，统计 P50 / P95 延迟与吞吐。 | “AI”“机器学习” 等短问题 |
| `cost_efficiency` | 在成本敏感场景下测量生成质量与预期成本。 | “解释神经网络的工作原理” |
| `stress_test` | 长时间/高频率执行混合查询，用于观察稳定性。 | 由框架重复生成的复杂案例 |

每个基准由若干 `EvaluationCase` 构成，可在源码中按照现有结构追加或修改测试样例。

## 2. 采集的核心指标

[`BenchmarkResult`](../src/evaluation/benchmarking_framework.py#L24-L71) 将一次测试的统计结果序列化，包括：
- **整体表现**：`overall_score`（综合加权得分）、`success_rate`、`error_rate`。
- **延迟与吞吐**：`latency_p50/p95/p99`、`throughput`（查询每秒）、`test_duration_seconds`。
- **检索与生成质量**：`retrieval_precision`、`retrieval_recall`、`generation_quality`、`factual_accuracy`。
- **成本效率**：`cost_per_query`、`cost_efficiency_score`、`gpu_utilization`、`memory_usage_mb`。
- **分布信息**：`score_distribution` 记录不同维度的细分分数，便于可视化。

若需要比较两个版本，可调用 `compare_benchmarks`，返回 [`ComparisonResult`](../src/evaluation/benchmarking_framework.py#L74-L101) 中的百分比改进与置信区间。

## 3. 运行基准测试

框架以异步方式工作，示例如下：

```python
import asyncio
from configs.config import config
from src.generation.rag_generator import RAGSystem
from src.retrieval.vector_database import VectorDatabaseManager
from src.evaluation.benchmarking_framework import BenchmarkingFramework

async def main():
    rag_config = {
        'embedding_model': config.EMBEDDING_MODEL,
        'llm_model': config.LLM_MODEL,
        'device': config.DEVICE,
        'qdrant_host': config.QDRANT_HOST,
        'qdrant_port': config.QDRANT_PORT,
        'collection_name': config.COLLECTION_NAME,
        'HUGGING_FACE_TOKEN': config.HUGGING_FACE_TOKEN,
    }

    db_manager = VectorDatabaseManager(rag_config)
    rag_system = RAGSystem(config=rag_config, db_manager=db_manager)

    framework = BenchmarkingFramework(config={'benchmark_results_path': 'data/benchmarks'})
    results = await framework.run_full_benchmark_suite(system_version="dev-2024-05")

    for name, result in results.items():
        print(name, result.overall_score, result.latency_p95)

if __name__ == "__main__":
    asyncio.run(main())
```

- `run_benchmark("qa_accuracy", version)` 可单独运行某一类测试。
- `run_full_benchmark_suite` 会依次执行全部基准，并调用 `_save_benchmark_results` 将结果写入磁盘。
- 如需对比历史版本，调用 `framework.compare_benchmarks(baseline_results, new_results)`。

> ⚠️ 基准会实际调用 `RAGSystem`，在运行前请确保向量库已经构建完毕，且所需模型可以加载。

## 4. 结果存储与可视化

- 结果目录默认位于 `data/benchmarks/`，可通过配置键 `benchmark_results_path` 覆盖。
- 每次执行会生成 `${system_version}_${benchmark_name}_${timestamp}.json`，内容即序列化后的 `BenchmarkResult`。
- `BenchmarkingFramework.load_historical_results()` 会聚合目录下的所有 JSON，供 `export_benchmark_dashboard_data()` 或 `compare_benchmarks()` 使用。
- `export_benchmark_dashboard_data()` 返回可直接用于前端展示的结构，包括最近一次得分、趋势数据和指标摘要。

## 5. 扩展建议

- **新增指标**：在 `_calculate_benchmark_metrics` 中加入自定义字段，并在 `BenchmarkResult` dataclass 中补充属性。
- **定制基准**：扩展 `StandardBenchmarks.benchmarks` 字典，或在初始化 `BenchmarkingFramework` 时传入自定义 `benchmarks` 映射。
- **压测参数**：`run_benchmark(..., system_version, max_concurrency=N)` 支持传入并发度、冷启动等待等参数，可在代码中按需扩展。
- **结果对比**：利用 `compare_benchmarks` 生成的 `ComparisonResult.metric_comparisons`，可以轻松绘制“新旧版本改进幅度”图表。

通过该框架，可以持续追踪 RAG 系统的准确率、延迟、成本与稳定性，支撑回归测试与性能优化决策。
