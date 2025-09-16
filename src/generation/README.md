# 生成链路改进

- [`rag_generator.py`](./rag_generator.py) 的 `EnhancedQueryProcessor` 使用 LRU 风格缓存复用查询向量，`LLMGenerator` 与上下文优化流程均通过 `ModelRegistry` 共享模型，并引入哈希+文本相似度的去重策略。
- [`tiered_generation.py`](./tiered_generation.py) 的 `LocalModelExecutor` 复用注册中心提供的 LLM 实例，避免多次加载大型模型。
