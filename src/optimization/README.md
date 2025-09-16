# 模型注册中心

- [`model_registry.py`](./model_registry.py) 定义了 `ModelRegistry` 与 `LLMResource`，集中管理向量模型与生成模型的生命周期，保证各子系统通过 `get_sentence_transformer` 与 `get_llm` 复用同一个实例，避免重复加载。 
