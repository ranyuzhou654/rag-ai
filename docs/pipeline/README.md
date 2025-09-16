# 运行管线整理

- [`run_rag_system.py`](../../run_rag_system.py) 统一从配置文件读取模型与数据库参数，`process_data` 异步驱动文本处理，`build_knowledge_base` 复用已加载的向量数据并动态推断维度，确保脚本与前端共享一致配置。
