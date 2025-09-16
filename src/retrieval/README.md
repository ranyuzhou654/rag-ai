# 检索与上下文优化

该目录整合了向量数据库操作、查询智能、上下文压缩与 Agentic RAG 流程，支撑最终的答案生成。

## 向量数据库管理
- [`QdrantVectorDB`](./vector_database.py#L1-L201) 封装集合创建、批量写入与混合检索：
  - `_ensure_collection_exists` 在初始化时检查集合是否存在并按指定维度与 HNSW 参数创建。
  - `_tokenize_for_search` 与 `_generate_chinese_tokens`（[`vector_database.py#L125-L170`](./vector_database.py#L125-L170)）对英文与中文进行分词，为混合检索提供文本信号。
- `VectorDatabaseManager.build_knowledge_base` 读取处理后的 chunk 列表并调用 `add_chunks` 批量写入数据库（[`vector_database.py#L316-L358`](./vector_database.py#L316-L358)）。
- `hybrid_search` 将向量相似度与全文 token 命中得分按权重合并（[`vector_database.py#L172-L251`](./vector_database.py#L172-L251)），并返回带 `hybrid_score`、payload 的检索结果。

## 查询智能
[`query_intelligence.py`](./query_intelligence.py) 提供多路查询增强：
- `QueryComplexityAnalyzer.analyze_complexity` 根据正则特征与长度估计问题复杂度。
- `SubQuestionGenerator`、`QueryRewriter`、`HydeDocumentGenerator` 继承 `_SharedLLMComponent`，共用 `ModelRegistry.get_llm` 创建的 tokenizer/模型。
- `QueryIntelligenceEngine` 在 [`query_intelligence.py#L310-L450`](./query_intelligence.py#L310-L450) 中聚合上述组件，暴露 `analyze_query`、`get_optimized_queries`、`get_hyde_document`、`generate_follow_up_questions` 等方法。

## 上下文压缩与重排
- `SentenceExtractor.extract_relevant_sentences` 使用共享嵌入模型与余弦相似度筛选关键句子，定义在 [`contextual_compression.py#L17-L78`](./contextual_compression.py#L17-L78)。
- `LLMCompressor.compress_context` 通过共享 LLM 将上下文压缩到目标长度（[`contextual_compression.py#L97-L150`](./contextual_compression.py#L97-L150)）。
- `SmartReranker.smart_rerank`（见 [`contextual_compression.py#L152-L239`](./contextual_compression.py#L152-L239)）对候选 chunk 执行多信号重排，为生成阶段提供更相关的上下文。
- `ContextualCompressor` 将上述能力组合成 `CompressedContext`，供 `EnhancedContextOptimizer` 直接调用。

## Agentic RAG 协调
- `RetrievalEvaluator.evaluate_retrieval` 先进行快速检查，再调用 `_llm_evaluate_retrieval` 依据模型反馈决定 `PROCEED`、`RETRY` 等策略（[`agentic_rag.py#L24-L150`](./agentic_rag.py#L24-L150)）。
- `AgenticRAGOrchestrator.agentic_retrieve_and_generate` 在 [`agentic_rag.py#L229-L360`](./agentic_rag.py#L229-L360) 中结合查询智能、向量检索与评估器反馈进行多轮检索，并记录每次 `AgenticStep`，供最终回答回溯。

## 与生成模块协同
`EnhancedQueryProcessor` 与 `EnhancedContextOptimizer`（位于 `src/generation/rag_generator.py`）分别调用 `QueryIntelligenceEngine` 与 `ContextualCompressor`/`SmartReranker`，复用共享模型实例并输出优化后的上下文，为回答生成提供高质量输入。