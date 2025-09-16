# 评估框架

`comprehensive_evaluation.py` 将 RAGAS、TruLens 与自定义指标整合成一套统一的评估体系，并提供数据结构化输出与历史记录能力。

## 数据结构
- `EvaluationMetrics`（[`comprehensive_evaluation.py#L38-L74`](./comprehensive_evaluation.py#L38-L74)）收集所有子指标并在 `__post_init__` 中补充时间戳；`notes` 字段用于记录回退或异常信息。
- `EvaluationCase`、`EvaluationReport` 分别描述单个测试样本与整体报告结构（[`comprehensive_evaluation.py#L76-L115`](./comprehensive_evaluation.py#L76-L115)）。

## 子评估器
- `RAGASEvaluator`（[`comprehensive_evaluation.py#L117-L193`](./comprehensive_evaluation.py#L117-L193)）封装 `ragas.evaluate`，在依赖缺失或执行失败时返回 `_mock_ragas_evaluation` 并设置 `using_mock=True`。
- `TruLensEvaluator`（[`comprehensive_evaluation.py#L195-L264`](./comprehensive_evaluation.py#L195-L264)）可包装 LangChain RAG 链并提供实时反馈；缺少依赖时同样进入模拟模式。
- `CustomMetricsEvaluator`（[`comprehensive_evaluation.py#L266-L332`](./comprehensive_evaluation.py#L266-L332)）实现检索精度、成本效率、用户满意度等业务指标的简化估算。

## 综合评估流程
- `ComprehensiveEvaluator` 初始化上述组件并加载黄金测试集（[`comprehensive_evaluation.py#L334-L367`](./comprehensive_evaluation.py#L334-L367)）。
- `evaluate_single_case` 在 [`comprehensive_evaluation.py#L369-L454`](./comprehensive_evaluation.py#L369-L454) 中并发执行各子评估器，收集结果后写入 `EvaluationMetrics`，并根据 `using_mock` 状态追加说明。
- `_evaluate_custom_metrics` 组合检索、成本、满意度评估；`_calculate_overall_score` 按权重聚合所有可用指标，代码位于 [`comprehensive_evaluation.py#L456-L531`](./comprehensive_evaluation.py#L456-L531)。

## 输出能力
- `generate_evaluation_report`（文件尾部）会将批量测试结果整理为 `EvaluationReport`，并支持导出为 JSON/CSV 供外部可视化。日志通过 `loguru` 记录评估过程、告警与历史数据。

整体上，该评估模块能够在真实依赖缺失时降级为模拟分数，并在 `notes` 中显式提示，保证评估结论的透明性。
