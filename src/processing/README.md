# 文本处理增强

- [`text_processor.py`](./text_processor.py) 通过异步 `process_documents` 调用 `MultiRepresentationIndexer`，同时将管线参数与模型名称对齐到 `configs/config.py`，并复用共享嵌入模型。
- [`multi_representation_indexer.py`](./multi_representation_indexer.py) 引入异步的 `create_multi_representations`，利用 `asyncio.to_thread` 并发生成摘要与问题，同时依赖模型注册中心复用 LLM 资源。
