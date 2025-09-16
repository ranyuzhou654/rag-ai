# 答案生成与任务编排

该目录负责查询解析、上下文优化、最终回答生成以及多模型分层调度。

## 查询与上下文处理
- `EnhancedQueryProcessor` 在 [`rag_generator.py#L33-L102`](./rag_generator.py#L33-L102) 中定义，通过 `ModelRegistry.get_sentence_transformer` 初始化嵌入器并维护一个 LRU 风格的查询向量缓存；当配置提供 `QueryIntelligenceEngine` 时，`process_query` 会返回改写查询、HyDE 文档及复杂度分析。
- `EnhancedContextOptimizer`（[`rag_generator.py#L104-L175`](./rag_generator.py#L104-L175)）注入 `ContextualCompressor` 与 `SmartReranker`，根据是否启用增强模式决定仅截断还是执行重排+压缩。

## LLM 答案生成
- `LLMGenerator` 通过 `ModelRegistry.get_llm` 共享加载生成模型与 tokenizer，核心调用位于 [`rag_generator.py#L177-L208`](./rag_generator.py#L177-L208)。

```python
def generate_answer(self, query: str, context: str) -> Tuple[str, int]:
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    outputs = self.model.generate(**inputs, generation_config=self.generation_config)
    answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
```

## 增强 RAG 系统
- `EnhancedRAGSystem`（[`rag_generator.py#L210-L362`](./rag_generator.py#L210-L362)）串联多策略检索：
  1. 调用 `EnhancedQueryProcessor` 生成改写查询与 HyDE 文档。
  2. 借助 `VectorDatabaseManager.search` 执行多查询检索并记录 `retrieval_source`。
  3. `_deduplicate_chunks` 结合 `chunk_id`、SHA256 与内容相似度去重，代码位于 [`rag_generator.py#L364-L418`](./rag_generator.py#L364-L418)。
  4. 可选使用外部 `AdvancedReranker`，否则直接调用 `EnhancedContextOptimizer` 生成压缩上下文。
  5. 最终通过 `LLMGenerator.generate_answer` 生成答案，并计算置信度（[`rag_generator.py#L420-L456`](./rag_generator.py#L420-L456)）。
- `generate_answer_agentic` 利用 `AgenticRAGOrchestrator` 进行多轮检索，返回 `AgenticStep` 记录以便前端展示。

## 分层任务编排
[`tiered_generation.py`](./tiered_generation.py) 为不同复杂度的任务选择最合适的模型：
- `TaskRouter`（[`tiered_generation.py#L38-L158`](./tiered_generation.py#L38-L158)）定义本地、API 模型的能力标签与路由规则。
- `LocalModelExecutor` 通过 `ModelRegistry.get_llm` 复用本地模型实例；`APIModelExecutor` 调用 OpenAI/Claude 等云端接口（示例实现位于文件后半部分）。
- `TieredGenerationSystem.execute_task`（[`tiered_generation.py#L412-L472`](./tiered_generation.py#L412-L472)）根据任务类型与复杂度选择执行器，记录耗时、token 与成本。`execute_workflow` 支持按优先级串行或并行执行任务流，并在 `task_stats` 中累计统计信息。

这些组件共同构成从查询理解、上下文优化到最终答复与多模型调度的完整生成链路。