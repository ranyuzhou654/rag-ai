# 评估透明化

- [`comprehensive_evaluation.py`](./comprehensive_evaluation.py) 在 `EvaluationMetrics` 中增加 `notes` 字段，当 RAGAS 或 TruLens 回退到模拟结果时，通过 `using_mock` 标记并在报告中记录提示，防止误读评分。
