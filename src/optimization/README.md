# 模型注册中心

`model_registry.py` 用于集中管理向量模型与生成模型的生命周期，避免重复加载占用资源。

## 数据结构
- `LLMResource` 是一个 dataclass（[`model_registry.py#L24-L31`](./model_registry.py#L24-L31)），保存共享 tokenizer、模型实例与设备信息。

## 核心接口
- `ModelRegistry.get_sentence_transformer(model_name, device)` 会：
  1. 调用 `_resolve_device` 自动选择 CUDA 或 CPU（[`model_registry.py#L41-L50`](./model_registry.py#L41-L50)）。
  2. 使用线程锁 `_embedding_lock` 确保并发环境下只加载一次模型，缓存键为 `(model_name, device)`。
- `ModelRegistry.get_llm(model_name, device, token)` 共享加载因果语言模型：
  - 通过 `_llm_lock` 保护初始化，创建 `AutoTokenizer`、`AutoModelForCausalLM` 并封装为 `LLMResource`（[`model_registry.py#L52-L96`](./model_registry.py#L52-L96)）。
  - 支持传入 Hugging Face token 或自定义设备，默认按 `torch.cuda.is_available()` 选择最优硬件。

## 使用场景
- 数据处理阶段的 `MultilingualEmbedder`、`MultiRepresentationIndexer`、检索阶段的 `SentenceExtractor` 及生成阶段的 `LLMGenerator` 均通过本注册中心共享嵌入模型。
- `QueryIntelligenceEngine`、`AgenticRAGOrchestrator`、`TieredGenerationSystem` 等多处组件在构造函数中调用 `get_llm` 复用同一套 tokenizer/模型，显著减少显存占用与加载时间。

该注册中心为跨模块的模型复用提供统一入口，保证整个 RAG 流程在多线程/异步场景下都能安全共享资源。
