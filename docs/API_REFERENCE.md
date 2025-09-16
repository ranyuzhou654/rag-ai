# API 参考文档

本节整理了项目中可以直接复用的 Python 接口，按照“核心系统 → 检索组件 → 生成与路由 → 知识图谱 → 评估与学习 → 优化与配置”的顺序列出关键类和方法。每个条目都指向对应的源码，便于进一步查阅实现细节。

## 核心系统

### `GenerationResult`
- **位置**：[`src/generation/rag_generator.py`](../src/generation/rag_generator.py)
- **说明**：封装一次 RAG 生成的完整返回值，包括答案、引用片段、置信度和查询分析信息。
- **关键字段**：`answer`、`source_chunks`、`confidence`、`generation_time`、`token_count`、`query_analysis`、`retrieval_strategies`、`agentic_steps`。

### `RAGSystem`
- **类名**：`EnhancedRAGSystem`（文件底部通过 `RAGSystem = EnhancedRAGSystem` 暴露）
- **构造函数**：`__init__(self, config: Dict, db_manager: VectorDatabaseManager, reranker: Optional[AdvancedReranker] = None)`
- **主要方法**：
  - `async generate_answer(user_query: str, **kwargs) -> GenerationResult`：执行多查询 / HyDE / 标准检索，去重、可选重排后调用 LLM 生成答案。
  - `async generate_answer_agentic(user_query: str, **kwargs) -> GenerationResult`：调用 `AgenticRAGOrchestrator` 进行检索—评估—改写的循环。
- **典型用法**：
  ```python
  from configs.config import config
  from src.retrieval.vector_database import VectorDatabaseManager
  from src.generation.rag_generator import RAGSystem

  rag_config = {
      'embedding_model': config.EMBEDDING_MODEL,
      'llm_model': config.LLM_MODEL,
      'device': config.DEVICE,
      'max_context_length': 3000,
      'qdrant_host': config.QDRANT_HOST,
      'qdrant_port': config.QDRANT_PORT,
      'collection_name': config.COLLECTION_NAME,
      'HUGGING_FACE_TOKEN': config.HUGGING_FACE_TOKEN,
  }

  db_manager = VectorDatabaseManager(rag_config)
  rag_system = RAGSystem(config=rag_config, db_manager=db_manager)

  result = asyncio.run(rag_system.generate_answer("什么是Transformer架构？", top_k=5))
  print(result.answer)
  ```

### `VectorDatabaseManager` & `QdrantVectorDB`
- **位置**：[`src/retrieval/vector_database.py`](../src/retrieval/vector_database.py)
- **核心接口**：
  - `build_knowledge_base(processed_chunks_path: Path, chunks: Optional[List[Dict]] = None) -> bool`：从处理后的 JSON 或内存列表写入 Qdrant。
  - `search(query_vector: np.ndarray, query_text: str, top_k: int = 5, **kwargs) -> List[Dict]`：封装 `_hybrid_search`，同时返回向量分数与文本匹配分。
  - `QdrantVectorDB.add_chunks(chunks: List[Dict], batch_size: int = 100)`：批量 upsert，自动生成全文检索 token。
  - `QdrantVectorDB.hybrid_search(...)`：自定义向量 / 文本权重、过滤条件与耗时统计。

## 查询与检索组件

### `EnhancedQueryProcessor`
- **位置**：[`src/generation/rag_generator.py`](../src/generation/rag_generator.py)
- **职责**：共享 `SentenceTransformer` 嵌入模型，维护查询向量 LRU 缓存，并按需调用查询智能。
- **关键方法**：
  - `process_query(query: str) -> Dict`：返回原始 / 重写查询、HyDE 文档、语言识别结果等。
  - `get_vector(text: str)`：对任意文本执行缓存向量化。

### `QueryIntelligenceEngine`
- **位置**：[`src/retrieval/query_intelligence.py`](../src/retrieval/query_intelligence.py)
- **能力**：
  - `analyze_query(query: str) -> QueryAnalysisResult`：识别语言、复杂度、查询类型。
  - `get_optimized_queries(query: str) -> List[str]`：合并子问题与重写版本，默认去重。
  - `get_hyde_document(query: str) -> str`：生成 HyDE 假设文档，供稠密检索使用。

### `HybridRetriever`
- **位置**：[`src/retrieval/hybrid_retriever.py`](../src/retrieval/hybrid_retriever.py)
- **构造参数**：`vector_store: QdrantVectorStore`, `documents: List[Document]`, `kg_retriever: Optional[KnowledgeGraphRetriever] = None`, `vector_weight`, `bm25_weight`, `kg_weight`, `k`。
- **异步检索**：`async _aget_relevant_documents(query: str) -> List[EnhancedDocument]` 会并发执行稠密、BM25、知识图谱检索，并在 `_fuse_results` 中按权重融合分数。
- **结果结构**：返回的 `EnhancedDocument` 带有 `RetrievalMetadata`，记录检索方式、置信度、质量评估等信息。

### `AgenticRAGOrchestrator`
- **位置**：[`src/retrieval/agentic_rag.py`](../src/retrieval/agentic_rag.py)
- **接口**：`async agentic_retrieve_and_generate(user_query: str, **kwargs) -> Tuple[str, List[Dict], List[AgenticStep], float]`
- **说明**：每轮调用 `RetrievalEvaluator` 评估结果（Proceed / Retry / Expand / Seek More），必要时经 `QueryRefiner` 改写查询，直到满足置信度或达到迭代上限。

### `ContextualCompressor` 与 `SmartReranker`
- **位置**：[`src/retrieval/contextual_compression.py`](../src/retrieval/contextual_compression.py)
- **主要方法**：
  - `ContextualCompressor.compress_context(query, chunks, max_context_length, compression_method)`：支持句子抽取、LLM 压缩与混合模式。
  - `SmartReranker.smart_rerank(query, chunks, top_k)`：基于多样性与质量指标筛选候选。

## 生成与模型路由

### `LLMGenerator`
- **位置**：[`src/generation/rag_generator.py`](../src/generation/rag_generator.py)
- **作用**：通过 `ModelRegistry` 共享的因果语言模型生成答案；`generate_answer(query, context) -> Tuple[str, int]` 返回答案与 token 数。

### `EnhancedContextOptimizer`
- **位置**：同上
- **方法**：`optimize_context(retrieved_chunks, top_k, query=None, use_compression=True, compression_method="hybrid") -> Tuple[str, List[Dict]]`。在增强模式下会先调用 `SmartReranker`，再执行压缩。

### `TieredGenerationSystem`
- **位置**：[`src/generation/tiered_generation.py`](../src/generation/tiered_generation.py)
- **核心接口**：
  - `async execute_task(task: TaskRequest) -> TaskResponse`：根据 `TaskRouter` 的模型决策执行单个任务。
  - `async execute_workflow(workflow_tasks: List[TaskRequest]) -> List[TaskResponse]`：支持按优先级排序与条件并发。
  - `task_stats` 字段记录模型使用次数、平均延迟、总成本等。

### `LangChainRAGSystem`
- **位置**：[`src/generation/langchain_rag_system.py`](../src/generation/langchain_rag_system.py)
- **构造函数**：`__init__(self, config: Dict[str, Any])`
- **公开方法**：
  - `async query(user_query: str, **kwargs) -> LangChainRAGResult`：执行 LCEL 链，完成复杂度分析 → 混合检索 → 答案生成。
  - `get_stats() -> Dict[str, Any]`：返回向量库、重排序器等组件的初始化状态。
- **说明**：该实现集成 LangChain `RunnableSequence`，需要事先通过 `_init_hybrid_retriever` 构建文档集合。

## 知识图谱组件

### `KnowledgeGraphIndexer`
- **位置**：[`src/knowledge_graph/knowledge_extractor.py`](../src/knowledge_graph/knowledge_extractor.py)
- **功能**：
  - 使用共享 LLM (`EntityExtractor`、`RelationExtractor`) 解析文本块生成实体与关系。
  - 将结构化结果写入 SQLite，并维护 `networkx.MultiDiGraph` 便于运行时查询。
  - `async index_documents(chunks: List[Dict]) -> KnowledgeGraphSummary` 支持批量更新。

### `KnowledgeGraphRetriever` 与 `KGEnhancedRetriever`
- **位置**：[`src/knowledge_graph/kg_retriever.py`](../src/knowledge_graph/kg_retriever.py)
- **接口**：
  - `KnowledgeGraphRetriever.search(entity_name: str, max_hops: int = 1)`：按实体 / 类型 / 关系检索节点。
  - `KGEnhancedRetriever.retrieve(query: str, vector_results: List[Dict])`：在向量检索结果基础上追加图谱描述，并在 `metadata` 中标注 `kg_entities`、`kg_relations`。

## 评估与持续学习

### `ComprehensiveEvaluator`
- **位置**：[`src/evaluation/comprehensive_evaluation.py`](../src/evaluation/comprehensive_evaluation.py)
- **主要方法**：
  - `async evaluate_single_case(case: EvaluationCase, rag_system=None) -> EvaluationMetrics`
  - `async evaluate_batch(evaluation_cases: List[EvaluationCase]) -> Dict[str, float]`
  - `async evaluate_golden_dataset(rag_system) -> EvaluationReport`
  - `async evaluate_retrieval_metrics(...)`、`async evaluate_cost_efficiency(...)` 等细分评估。
- **特性**：缺失 RAGAS / TruLens 依赖时会自动切换到模拟指标，并在结果中设置 `using_mock` 标记。

### `BenchmarkingFramework`
- **位置**：[`src/evaluation/benchmarking_framework.py`](../src/evaluation/benchmarking_framework.py)
- **用途**：运行标准化基准（问答准确率、检索质量、速度、成本、压力测试）。
- **关键方法**：
  - `async run_benchmark(benchmark_name: str, system_version: str) -> BenchmarkResult`
  - `async run_full_benchmark_suite(system_version: str) -> Dict[str, BenchmarkResult]`
  - `compare_benchmarks(baseline_results, new_results) -> ComparisonResult`
  - `export_benchmark_dashboard_data() -> Dict[str, Any]`

### 反馈与持续学习
- **组件**：[`src/feedback/feedback_system.py`](../src/feedback/feedback_system.py)
  - `FeedbackCollector`：写入 / 查询 SQLite 反馈数据，支持 `store_feedback`、`get_feedback_since`。
  - `FeedbackAnalyzer`：从反馈中提取常见问题、文档相关性等洞察。
- **持续学习协调器**：[`ContinuousLearningOrchestrator`](../src/learning/continuous_learning_system.py)
  - `analyze_recent_feedback()`、`perform_incremental_learning()`、`update_knowledge_graph()`、`generate_learning_report()`。
  - 通过后台异步任务 `_periodic_*` 周期性触发分析、微调与知识图谱更新。

## 优化与配置

### `ModelRegistry`
- **位置**：[`src/optimization/model_registry.py`](../src/optimization/model_registry.py)
- **接口**：`get_sentence_transformer(model_name, device="auto")`、`get_llm(model_name, device="auto", token=None)`。通过线程锁保证模型懒加载且全局共享。

### `PerformanceOptimizer`
- **位置**：[`src/optimization/performance_optimizer.py`](../src/optimization/performance_optimizer.py)
- **能力**：
  - `async optimize_retrieval(query, retrieval_func, retrieval_params)`：集成多级缓存和成本估算。
  - `async optimize_embedding_batch(texts, embedding_model)`：缓存命中后直接返回向量，其余批量送入模型并记录统计。
  - `async accelerate_generation(task_id, generator_func, **kwargs)`：在 GPU 加速与缓存间做策略选择。
  - `get_performance_summary()`：返回缓存命中率、平均延迟等指标。

### 配置入口
- **位置**：[`configs/config.py`](../configs/config.py)
- **说明**：`Config` 类集中管理路径、模型、向量库与功能开关。配置值默认读取 `.env`，同时计算 `STORAGE_ROOT` / `DATA_DIR` / `LOG_DIR` 等派生路径。
- **常用字段**：`EMBEDDING_MODEL`、`LLM_MODEL`、`ENABLE_QUERY_INTELLIGENCE`、`ENABLE_MULTI_REPRESENTATION`、`ENABLE_AGENTIC_RAG`、`ENABLE_KNOWLEDGE_GRAPH`、`ENABLE_TIERED_GENERATION`。

---
通过以上接口文档，可以根据项目现有实现快速组合新的实验或扩展功能。每个模块都在源码中附带日志与注释，建议结合具体业务场景选择合适的调用方式。
