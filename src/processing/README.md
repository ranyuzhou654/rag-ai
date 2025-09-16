# 文本处理与多表示索引

该目录提供从原始 `Document` 到可写入向量库条目的完整转换流程，核心代码在 [`text_processor.py`](./text_processor.py) 与 [`multi_representation_indexer.py`](./multi_representation_indexer.py) 中。

## 分层切分与向量化
- `HierarchicalTextSplitter` 先按章节正则再递归切分，详见 [`split_document`](./text_processor.py#L28-L78)，生成带 `chunk_id`、`metadata` 的 `TextChunk`。
- `MultilingualEmbedder` 通过 `ModelRegistry.get_sentence_transformer` 共享加载向量模型，其 `embed_chunks` 方法会批量向量化 chunk 并把结果写回 `TextChunk.embedding`。

```python
class EnhancedTextProcessor:
    def __init__(self, config: Dict):
        self.splitter = HierarchicalTextSplitter(...)
        self.embedder = MultilingualEmbedder(...)
        if config.get('enable_multi_representation', True):
            self.multi_rep_indexer = MultiRepresentationIndexer(config)
```
- 初始化逻辑位于 [`EnhancedTextProcessor.__init__`](./text_processor.py#L80-L110)，确保切分器、向量化器与多表示索引器共享同一配置与模型实例。

## 处理主流程
`process_documents` 将分层切分、向量化和多表示生成串联：

```python
for doc in documents:
    chunks = self.splitter.split_document(...)
    all_chunks.extend(chunks)
vectorized_chunks = self.embedder.embed_chunks(all_chunks)
if self.enable_multi_representation:
    multi_rep_chunks = await self.multi_rep_indexer.create_multi_representations(...)
    return self.multi_rep_indexer.generate_index_entries(multi_rep_chunks)
```
- 见 [`process_documents`](./text_processor.py#L112-L167)。启用多表示时会将 `TextChunk` 转换为 dict 传给索引器；否则返回标准向量化结果。

## 多表示索引器
`MultiRepresentationIndexer` 为每个 chunk 生成摘要、假设问题等额外表示，代码组织如下：

- 数据结构：[`MultiRepresentationChunk`](./multi_representation_indexer.py#L15-L33) 保存原文、摘要、问题及对应嵌入。
- 生成组件：`SummaryGenerator`、`QuestionGenerator` 均继承 `_SharedLLMComponent`，共享 `ModelRegistry.get_llm` 提供的 tokenizer 与模型。
- 生成流程：[`create_multi_representations`](./multi_representation_indexer.py#L170-L264) 会使用 `asyncio.to_thread` 异步调用生成器，并把摘要/问题的嵌入补齐到 `MultiRepresentationChunk`。
- 索引条目：[`generate_index_entries`](./multi_representation_indexer.py#L266-L341) 将原文、摘要、问题三类表示展开为多条索引记录，每条记录都携带 `semantic_type`、`representation_type` 等元信息，供向量库或重排器使用。

整体而言，该子模块将异步多表示生成与统一的模型注册中心结合，实现了对多语种 RAG 数据的高效预处理。