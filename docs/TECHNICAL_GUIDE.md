# 技术详细指南

本文档概述仓库的技术实现与扩展要点，帮助阅读者从工程角度理解数据流、关键算法和配置策略。所有章节均引用具体源码，方便深入查看。

## 1. 系统架构

```
┌──────────────┐   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
│ 数据采集层   │ → │ 文本处理层     │ → │ 检索与重排序层 │ → │ 答案生成层     │
│ MultiSource  │   │ TextProcessor  │   │ Vector DB +    │   │ EnhancedRAG /  │
│ Collector    │   │ + Multi-Rep    │   │ Query Intel +  │   │ Tiered System  │
└──────────────┘   └────────────────┘   │ Agentic RAG    │   └──────┬─────────┘
                                         └────────────────┘          │
                                                │                    │
                                                ▼                    ▼
                                        ┌────────────────┐   ┌────────────────┐
                                        │ 知识图谱与反馈 │   │ 评估与持续学习 │
                                        │ KG Indexer /   │   │ Comprehensive  │
                                        │ Feedback DB    │   │ Evaluator /    │
                                        └────────────────┘   │ Benchmarking   │
                                                             └────────────────┘
```

核心技术栈：
- **Python 3.10+** 与 `asyncio`：异步采集与并发生成。
- **SentenceTransformers** 与 **Transformers**：统一向量化与 LLM 推理（通过 [`ModelRegistry`](../src/optimization/model_registry.py) 缓存）。
- **Qdrant**：混合检索、向量存储与分词增强（[`vector_database.py`](../src/retrieval/vector_database.py)）。
- **Streamlit**：前端展示（[`app.py`](../app.py)）。
- **LangChain**：可选 LCEL RAG 管线（[`langchain_rag_system.py`](../src/generation/langchain_rag_system.py)）。

## 2. 核心组件拆解

### 2.1 数据采集
- [`MultiSourceCollector`](../src/data_ingestion/multi_source_collector.py) 异步抓取 ArXiv / Hugging Face / RSS 博客，复用单个 `aiohttp.ClientSession` 并通过信号量限制 PDF 下载并发。
- 支持 `pymupdf4llm` → PyMuPDF 的多级回退，抽取 Markdown 文本后统一存入 `data/raw/raw_collected_data.json`。

### 2.2 文本处理
- [`HierarchicalTextSplitter`](../src/processing/text_processor.py) 先按章节再递归切分，生成带 `chunk_id` 的 `TextChunk`。
- `MultilingualEmbedder` 复用 BGE-M3 嵌入模型并记录向量维度，供下游集合创建。
- [`MultiRepresentationIndexer`](../src/processing/multi_representation_indexer.py)（配置 `ENABLE_MULTI_REPRESENTATION=True` 时启用）
  - 复用共享 LLM 生成摘要与假设问题。
  - 异步批量调用、统一嵌入，并输出 `semantic_type` 为 `content/summary/question` 的多条索引。

### 2.3 向量库与检索
- [`VectorDatabaseManager`](../src/retrieval/vector_database.py) 封装 Qdrant 连接、集合初始化与 `_tokenize_for_search` 中文 1-3 字滑窗分词。
- [`HybridRetriever`](../src/retrieval/hybrid_retriever.py) 并行组合稠密向量、BM25 与知识图谱检索，输出附带 `RetrievalMetadata` 的 `EnhancedDocument`。
- [`QueryIntelligenceEngine`](../src/retrieval/query_intelligence.py) 负责复杂度分析、子问题拆解、查询重写与 HyDE 文档生成。
- 去重逻辑由 [`EnhancedRAGSystem._deduplicate_chunks`](../src/generation/rag_generator.py) 组合 chunk id 与内容相似度完成。

### 2.4 答案生成
- [`EnhancedRAGSystem`](../src/generation/rag_generator.py) 将查询智能、混合检索、重排序、上下文压缩和 LLM 生成串联，返回 `GenerationResult`。
- `AgenticRAGOrchestrator` 通过 `RetrievalEvaluator` 判断是否继续检索或改写查询，实现自我反思流程。
- [`TieredGenerationSystem`](../src/generation/tiered_generation.py) 提供模型路由：`TaskRouter` 根据复杂度 / 成本在本地模型、API 模型间切换，`TaskRequest` / `TaskResponse` 记录成本与延迟。

### 2.5 知识图谱与反馈
- [`KnowledgeGraphIndexer`](../src/knowledge_graph/knowledge_extractor.py) 使用共享 LLM 抽取实体 / 关系，写入 SQLite + `networkx.MultiDiGraph`。
- [`KGEnhancedRetriever`](../src/knowledge_graph/kg_retriever.py) 将知识图谱节点补充到检索结果，供答案引用。
- [`FeedbackCollector`](../src/feedback/feedback_system.py) 保存用户反馈、引用片段与置信度；`FeedbackAnalyzer` 生成常见问题、文档相关度等洞察。
- [`ContinuousLearningOrchestrator`](../src/learning/continuous_learning_system.py) 定期执行反馈分析、嵌入微调与知识图谱增量更新。

### 2.6 评估与基准
- [`ComprehensiveEvaluator`](../src/evaluation/comprehensive_evaluation.py) 集成 RAGAS / TruLens（缺依赖时自动 mock），评估 `faithfulness`、`context_precision`、成本效率等指标。
- [`BenchmarkingFramework`](../src/evaluation/benchmarking_framework.py) 提供标准化基准：问答准确率、检索质量、生成速度、成本效率、压力测试，并能对比不同版本。

## 3. 数据流与运行脚本

[`run_rag_system.py`](../run_rag_system.py) 的 `RAGSystemRunner` 按以下顺序 orchestrate：
1. `check_environment()`：校验 Python 版本、核心依赖、Qdrant 状态。
2. `collect_data()`：调用 `MultiSourceCollector.collect_all()` 并保存原始 JSON。
3. `process_data()`：构造处理配置，驱动 `TextProcessor.process_documents()` 输出向量化条目。
4. `build_knowledge_base()`：写入 Qdrant 集合并打印统计。
5. `test_system()`（可选）：使用 `RAGSystem.generate_answer` 对中英文样例做端到端验证。
6. `launch_frontend()`：在子进程启动 Streamlit，并打印访问地址。

命令行选项（`--skip-collect/--skip-process/--skip-build/--test/--no-frontend` 等）允许灵活组合阶段。`--quick` 会自动跳过采集，`--frontend-only` 仅启动 UI。

## 4. 关键算法细节

- **多查询检索**：`EnhancedQueryProcessor` 返回 `optimized_queries`，`EnhancedRAGSystem.generate_answer` 会限制前三个改写并平均分配检索配额。
- **HyDE 召回**：若 `hyde_document` 非空，将其编码后执行额外检索，标记 `retrieval_source='hyde'`，用于解释来源。
- **去重策略**：`_deduplicate_chunks` 先依据 chunk id，再用 SHA256 + `SequenceMatcher` 比较前 500 字，阈值 0.92。
- **上下文压缩**：`ContextualCompressor` 支持句子抽取、LLM 压缩、混合模式，并记录压缩比；`SmartReranker` 结合查询向量、chunk 质量与多样性排序。
- **Agentic 决策**：`RetrievalEvaluator` 输出 `RetrievalDecision`（Proceed / Retry / Expand Query / Seek More / Insufficient），驱动 `QueryRefiner` 或扩大检索范围。
- **模型路由**：`TaskRouter` 综合任务类型、复杂度、预算选择模型，`CostOptimizer` 追踪开销，`PerformanceMonitor` 记录延迟与成功率。

## 5. 配置与环境

所有开关集中在 [`configs/config.py`](../configs/config.py)：
- `.env` 中可设置 `EMBEDDING_MODEL`、`LLM_MODEL`、`DEVICE`、`HUGGING_FACE_TOKEN`、`QDRANT_HOST/PORT` 等。
- 开关项：`ENABLE_QUERY_INTELLIGENCE`、`ENABLE_MULTI_REPRESENTATION`、`ENABLE_AGENTIC_RAG`、`ENABLE_CONTEXTUAL_COMPRESSION`、`ENABLE_KNOWLEDGE_GRAPH`、`ENABLE_TIERED_GENERATION`。
- `STORAGE_ROOT` 默认指向 `project_data/`，内部自动创建 `data/`、`logs/`、`knowledge_graph/`、`feedback/` 等目录。

示例：
```env
EMBEDDING_MODEL=BAAI/bge-m3
LLM_MODEL=Qwen/Qwen2-7B-Instruct
ENABLE_AGENTIC_RAG=true
ENABLE_MULTI_REPRESENTATION=true
```

## 6. 扩展指南

- **新增数据源**：在 `MultiSourceCollector.blog_feeds` 中加入 RSS，或实现新的 `fetch_*` 协程并加入 `collect_all` 任务列表。
- **定制切分/嵌入**：可继承 `HierarchicalTextSplitter`、`MultilingualEmbedder` 或通过配置覆盖 `chunk_size`、`chunk_overlap`、`embedding_model`。
- **调整检索权重**：`HybridRetriever` 构造函数接受 `vector_weight`、`bm25_weight`、`kg_weight`，可在实例化时灵活传入。
- **替换生成模型**：更新 `config.LLM_MODEL` 并确保 `ModelRegistry.get_llm` 能拉取相应模型或 API；必要时扩展 `TieredGenerationSystem` 的模型配置。
- **加入新的评估指标**：`ComprehensiveEvaluator` 提供 `custom_metrics` 字段，可在初始化时传入自定义评估函数；`BenchmarkingFramework` 也支持扩展基准集合。

## 7. 性能与排错

- **模型缓存**：`ModelRegistry` 确保嵌入模型与 LLM 只加载一次，避免多组件重复占用显存。
- **查询缓存**：`EnhancedQueryProcessor` 的 `_vector_cache` 默认保留 256 个查询向量，可在配置中通过 `query_vector_cache_size` 调整。
- **异步吞吐**：PDF 下载、摘要/问题生成、知识图谱更新都通过 `asyncio.gather` 与线程池提升吞吐，注意在自定义扩展时遵守并发限制（如 `multi_rep_concurrency`）。
- **日志与监控**：所有关键步骤使用 `loguru` 打印耗时、统计信息。`PerformanceOptimizer.get_performance_summary()` 可查看缓存命中率、平均延迟。
- **常见问题排查**：
  - Qdrant 连接失败 → 检查 Docker 服务与 `QDRANT_HOST/PORT`。
  - 模型拉取超时 → 提前在 `.env` 中设置 `HF_HOME` 并下载模型。
  - 评估返回 mock 分数 → 安装 `ragas` / `trulens-eval` 或在日志中确认 `using_mock` 标记。

---
掌握以上结构即可自信地阅读、修改或扩展本项目：从采集到评估的每一层都采用模块化设计，并通过统一配置与模型缓存降低工程复杂度。
