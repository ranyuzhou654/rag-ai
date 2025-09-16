# 知识图谱抽取与检索

本目录负责从文本块中抽取实体、关系并构建图数据库，同时提供基于知识图谱的检索与上下文增强能力。

## 抽取组件
- `EntityExtractor` 与 `RelationExtractor` 均继承 `_SharedLLMComponent`，通过 `ModelRegistry.get_llm` 共享 tokenizer/模型资源，位于 [`knowledge_extractor.py#L33-L247`](./knowledge_extractor.py#L33-L247)。
  - `EntityExtractor.extract_entities` 根据语言动态构造提示，并解析 LLM 输出的 JSON，失败时回退到 `_rule_based_extraction`。
  - `RelationExtractor.extract_relations` 接受实体列表并生成关系 JSON，同样包含规则回退逻辑。
- `KnowledgeGraphIndexer` 将上述抽取器与 `KnowledgeGraphDatabase` 结合，`build_knowledge_graph` 在 [`knowledge_extractor.py#L624-L733`](./knowledge_extractor.py#L624-L733) 中根据事件循环状态选择异步或顺序处理。
  - `_build_knowledge_graph_async` 使用 `asyncio.Semaphore` 控制并发，`_extract_chunk_async` 借助 `asyncio.to_thread` 调用抽取器。

## 图数据库管理
- `KnowledgeGraphDatabase`（[`knowledge_extractor.py#L409-L622`](./knowledge_extractor.py#L409-L622)）负责将实体/关系写入 SQLite，并利用 NetworkX 维护内存图，提供 `store_entity`、`store_relation`、`load_graph_from_db` 等接口。
- `get_graph_statistics` 与 `query_knowledge_graph` 支持统计节点/边分布、根据查询关键词回收相关实体与关系路径，定义在 [`knowledge_extractor.py#L735-L828`](./knowledge_extractor.py#L735-L828)。

## 检索与增强
- `KnowledgeGraphRetriever`（[`kg_retriever.py#L1-L154`](./kg_retriever.py#L1-L154)）在初始化时实例化 `KnowledgeGraphIndexer` 和共享嵌入模型。
  - `retrieve_kg_context` 调用 `query_knowledge_graph` 得到实体/关系，组装为 `KGRetrievalResult`。
  - `enhance_chunks_with_kg` 会遍历文本块内容，若包含实体或别名则拼接 `[KG]` 前缀的补充信息，并按增强数量提高混合得分。
  - `generate_kg_summary` 汇总实体与关系描述，为最终回答提供结构化摘要。

## 与主流程的集成
- `src/generation/rag_generator.py` 在 `EnhancedRAGSystem` 中可选引入 `KnowledgeGraphRetriever`，将 `kg_entities`、`kg_relations` 附加到 `GenerationResult` 中，为用户展示图谱证据。

- `build_knowledge_graph` 可由 `TextProcessor` 输出的 chunk 列表直接调用，使知识图谱与向量库同步更新。