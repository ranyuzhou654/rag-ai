# RAG 系统最佳实践指南

以下建议基于仓库现有实现，总结如何高质量地使用和扩展各模块。

## 1. 架构与配置
- **保持模块化**：使用 [`RAGSystem`](../src/generation/rag_generator.py) 作为问答入口，通过依赖注入传入 [`VectorDatabaseManager`](../src/retrieval/vector_database.py) 与可选的 `AdvancedReranker`。这样可以在实验阶段快速替换检索或重排序策略。
- **统一配置来源**：所有参数应从 [`configs/config.py`](../configs/config.py) 读取，或在 `.env` 中覆盖默认值，避免在业务代码中硬编码模型名称与路径。
- **最小化全局副作用**：如果需要新增模型或嵌入器，优先扩展 [`ModelRegistry`](../src/optimization/model_registry.py)，保证所有组件复用缓存实例。

示例：
```python
from configs.config import config
from src.retrieval.vector_database import VectorDatabaseManager
from src.generation.rag_generator import RAGSystem

db_manager = VectorDatabaseManager({
    'qdrant_host': config.QDRANT_HOST,
    'qdrant_port': config.QDRANT_PORT,
    'collection_name': config.COLLECTION_NAME,
    'vector_size': 1024,
})
rag_system = RAGSystem(config={
    'embedding_model': config.EMBEDDING_MODEL,
    'llm_model': config.LLM_MODEL,
    'device': config.DEVICE,
    'HUGGING_FACE_TOKEN': config.HUGGING_FACE_TOKEN,
}, db_manager=db_manager)
```

## 2. 数据与文本处理
- **语义切分优先**：沿用 [`HierarchicalTextSplitter`](../src/processing/text_processor.py) 的章节 + 递归策略，保证 chunk 边界尽可能对齐语义；如需定制规则，可在初始化时调整 `chunk_size`、`chunk_overlap`。
- **多表示索引**：在资源允许的情况下开启 `ENABLE_MULTI_REPRESENTATION`，由 [`MultiRepresentationIndexer`](../src/processing/multi_representation_indexer.py) 生成摘要与问题，可显著提升召回率。
- **冗余校验**：处理管线结束时调用 `TextProcessor.save_processed_data`，确保 embedding 转成列表后写入 JSON，便于审查和回放。

## 3. 检索与查询优化
- **善用查询智能**：`QueryIntelligenceEngine` 提供复杂度分析、子问题拆解、HyDE 文档，请确保在配置中开启 `ENABLE_QUERY_INTELLIGENCE`，并记录 `query_analysis` 以解释回答过程。
- **混合检索调参**：`VectorDatabaseManager.search` 支持自定义 `vector_weight`、`text_weight`，可结合任务对齐权重；`HybridRetriever` 也可根据领域适当降低 BM25 权重或关闭知识图谱检索。
- **去重与多样性**：让 `SmartReranker` 在压缩前处理候选，可避免上下文重复；若新增检索源，请确保为文档生成稳定的 `chunk_id` 以便 `_deduplicate_chunks` 生效。

## 4. 答案生成策略
- **上下文压缩**：根据问题长度与模型上下文上限，配置 `use_compression` 与 `compression_method`（`sentence_extraction`、`llm_compression`、`hybrid`），必要时限制 `context_chunks`。
- **Agentic 模式**：启用 `ENABLE_AGENTIC_RAG` 后，`AgenticRAGOrchestrator` 会自动根据 `RetrievalEvaluator` 决策是否改写查询或扩展检索范围，适合开放式问题。
- **分层路由**：如需精细控制成本，可配置 [`TieredGenerationSystem`](../src/generation/tiered_generation.py) 中的模型列表，利用 `TaskRouter` 与 `CostOptimizer` 在本地模型和 API 模型间切换。

## 5. 性能优化
- **模型缓存**：所有高成本模型调用应通过 `ModelRegistry` 获取，避免在协程内重复实例化；如需测试不同模型，可在新进程中运行，防止旧缓存占用资源。
- **查询缓存**：`EnhancedQueryProcessor` 默认缓存 256 条查询向量；对于热点问题可以适当增大 `query_vector_cache_size`，或结合 [`PerformanceOptimizer`](../src/optimization/performance_optimizer.py) 记录命中率。
- **异步与批处理**：PDF 下载、摘要生成、知识图谱更新等任务均通过 `asyncio.gather`/`to_thread` 执行，扩展代码时请遵守现有信号量限制（如 `multi_rep_concurrency`）以避免过载。
- **日志监控**：所有核心路径均使用 `loguru` 输出耗时和统计；建议保留 `logs/` 目录，配合 Streamlit 前端的指标面板快速定位瓶颈。

## 6. 评估与持续改进
- **统一评估入口**：使用 [`ComprehensiveEvaluator`](../src/evaluation/comprehensive_evaluation.py) 运行单次或批量评估，缺依赖时会返回 `using_mock` 标记，便于识别退化模式。
- **基准跟踪**：通过 [`BenchmarkingFramework`](../src/evaluation/benchmarking_framework.py) 定期生成 `BenchmarkResult`，比较不同版本的延迟与准确率，结果会保存在 `data/benchmarks/`。
- **反馈闭环**：`FeedbackCollector`/`FeedbackAnalyzer` + [`ContinuousLearningOrchestrator`](../src/learning/continuous_learning_system.py) 可以自动收集用户反馈、生成洞察、触发嵌入微调与知识图谱增量更新。

## 7. 安全性与可靠性
- **凭证管理**：将 `HUGGING_FACE_TOKEN`、API Key 等保存在 `.env`，确保代码仓库不包含明文密钥；在 CI/CD 环境可通过环境变量注入。
- **向量库健康检查**：`run_rag_system.py` 的 `check_environment()` 已经检测 Qdrant 状态，建议在部署脚本中重用该逻辑，或调用 `VectorDatabaseManager.db.get_collection_stats()` 监控集合规模。
- **容错处理**：保持 `_fallback_evaluation`、`_rule_based_extraction` 等后备逻辑，当外部依赖不可用时仍能输出可解释结果。

## 8. 成本控制
- **模型路由**：利用 `TieredGenerationSystem` 的 `task_stats` 观察不同模型的调用次数与成本，必要时调整 `TaskRouter` 配置。
- **缓存复用**：`PerformanceOptimizer` 的多级缓存可以减少重复检索与嵌入开销，适用于批量问答或回归测试场景。
- **评估成本**：在 `BenchmarkingFramework.run_benchmark` 时合理设置并发度与测试数量，避免在非必要场景触发大量 API 调用。

---
遵循以上实践，可以在保持系统稳定性的同时迭代检索、生成与评估策略，快速验证改进效果。
