# 知识图谱改造

- [`knowledge_extractor.py`](./knowledge_extractor.py) 的实体/关系抽取器共享 LLM 资源，并新增异步 `build_knowledge_graph`，通过并发抽取和数据库写入的节流来加速构图。
- [`kg_retriever.py`](./kg_retriever.py) 改为使用注册中心提供的嵌入模型，保证与主检索流程一致。
