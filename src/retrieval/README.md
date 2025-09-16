# 检索模块更新

- [`query_intelligence.py`](./query_intelligence.py) 各类生成器继承共享 LLM 组件，通过 `ModelRegistry` 复用 tokenizer 与模型实例。
- [`contextual_compression.py`](./contextual_compression.py) 的句子提取器、压缩器与智能重排器统一使用共享嵌入模型，LLM 压缩器复用注册中心实例以减少显存占用。
- [`agentic_rag.py`](./agentic_rag.py) 中的检索评估器与查询优化器共享大模型资源，避免多次加载。
- [`vector_database.py`](./vector_database.py) 在 `_tokenize_for_search` 中新增对中文的 n-gram 切分逻辑，并允许 `build_knowledge_base` 接受预加载的 chunk 数据，以减少重复 IO。
