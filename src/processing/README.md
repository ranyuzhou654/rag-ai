# 文本处理与多表示索引模块 (Processing Module)

## 概述 (Overview)

文本处理模块是 RAG 系统的第二层，负责将原始文档转换为可检索的结构化文本块。该模块采用层次化分割策略和多表示索引技术，通过智能的语义增强处理，显著提升检索召回率和精度。

**核心特性：**
- 📖 **层次化文本分割**: 基于文档结构的智能分段策略
- 🌍 **多语言向量化**: 支持中英文的高质量嵌入生成
- 🔄 **多表示索引**: 原文、摘要、假设问题三重表示增强检索
- ⚡ **异步并行处理**: 高效的批量文本处理流水线
- 💾 **模型共享机制**: 统一的模型注册中心避免重复加载

## 核心架构

### 1. 数据模型层

#### TextChunk - 基础文本块
```python
@dataclass
class TextChunk:
    """优化的文本块结构 - 处理流水线的基本单位"""
    content: str                             # 文本内容
    chunk_id: str                           # 全局唯一标识符
    source_id: str                          # 源文档ID
    metadata: Dict = field(default_factory=dict)  # 章节、位置等元信息
    embedding: Optional[np.ndarray] = None   # 向量表示
```

**设计原理**: 位于 [`text_processor.py:13-20`](./text_processor.py#L13-L20)
- **轻量级设计**: 最小化内存占用，支持大规模文档处理
- **元数据丰富**: 保留章节信息、位置索引等上下文
- **向量就绪**: 预留嵌入字段支持后续向量化

#### MultiRepresentationChunk - 多表示文本块
```python
@dataclass
class MultiRepresentationChunk:
    """多表示文本块 - 增强检索的核心数据结构"""
    content: str                    # 原文内容
    chunk_id: str
    source_id: str
    metadata: Dict = field(default_factory=dict)
    
    # 原文向量
    content_embedding: Optional[np.ndarray] = None
    
    # 摘要表示
    summary: Optional[str] = None
    summary_embedding: Optional[np.ndarray] = None
    
    # 假设问题表示
    hypothetical_questions: List[str] = field(default_factory=list)
    questions_embeddings: List[np.ndarray] = field(default_factory=list)
    
    # 语义类型标识
    semantic_type: str = 'content'  # 'content', 'summary', 'question'
```

**技术创新**: 位于 [`multi_representation_indexer.py:14-33`](./multi_representation_indexer.py#L14-L33)
- **多视角表示**: 同一内容的三种不同角度表示，提升检索覆盖面
- **向量对齐**: 每种表示都有对应的嵌入向量，支持语义检索
- **类型标识**: 便于检索时进行表示类型过滤和重排序

### 2. 文本分割层

#### 层次化文本分割器
```python
class HierarchicalTextSplitter:
    """
    层次化文本分割器 - 智能保持文档结构
    - 首先按逻辑章节分割 (Abstract, Introduction, etc.)
    - 然后在章节内进行递归字符分割
    """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.section_splitter = re.compile(
            r"\n(##?|Abstract|Introduction|Conclusion|Methodology|Discussion|Related Work)\n", 
            re.IGNORECASE
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
```

**分割策略**: 位于 [`text_processor.py:22-34`](./text_processor.py#L22-L34)

1. **章节识别**: 正则表达式识别学术论文标准章节结构
2. **递归分割**: LangChain 的递归字符分割器保持语义连贯性
3. **重叠机制**: chunk_overlap 确保语义边界信息不丢失

#### 智能分割流程
```python
def split_document(self, doc_content: str, source_id: str, metadata: Dict) -> List[TextChunk]:
    """执行分层分割 - 保持结构化信息"""
    # 1. 按章节分割
    sections = self.section_splitter.split(doc_content)
    
    # 2. 处理分割结果，合并标题与内容
    processed_sections = []
    i = 0
    while i < len(sections):
        if i + 1 < len(sections) and self.section_splitter.match("\n" + sections[i+1] + "\n"):
            header = sections[i+1].strip()
            content = sections[i+2]
            processed_sections.append((header, content))
            i += 3
        else:
            processed_sections.append(("content", sections[i]))
            i += 1
    
    # 3. 章节内递归分割
    chunk_counter = 0
    for header, content in processed_sections:
        sub_chunks = self.recursive_splitter.split_text(content)
        for sub_chunk in sub_chunks:
            chunk_metadata = {
                **metadata,
                'section': header.lower(),
                'chunk_index': chunk_counter,
            }
            chunks.append(TextChunk(...))
            chunk_counter += 1
```

**核心优势**: 位于 [`text_processor.py:36-80`](./text_processor.py#L36-L80)
- **结构保持**: 保留章节信息，便于后续上下文理解
- **索引完整**: 每个chunk都有全局唯一ID和位置信息  
- **元数据传承**: 源文档的所有元数据都会传递给子块

### 3. 向量化层

#### 多语言嵌入器
```python
class MultilingualEmbedder:
    """多语言文本嵌入器 - 基于共享模型实例"""
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "auto"):
        # 通过模型注册中心获取共享实例
        self.model_name = model_name
        self.device = device
        self.embedding_model = ModelRegistry.get_sentence_transformer(
            model_name, device=device
        )
```

**批量向量化核心流程**:
```python
def embed_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
    """批量生成文本嵌入"""
    if not chunks:
        return chunks
    
    texts = [chunk.content for chunk in chunks]
    
    # 批量生成嵌入向量
    embeddings = self.embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # 将向量赋值回TextChunk对象
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
    
    return chunks
```

**技术特点**: 位于 [`text_processor.py:80+`](./text_processor.py#L80)
- **模型共享**: 通过 `ModelRegistry` 避免重复加载相同模型
- **批量优化**: 批量处理提升GPU利用率
- **进度监控**: 内置进度条便于监控处理状态
- **内存友好**: 使用 numpy 数组减少内存占用

### 4. 多表示索引层

#### 共享LLM组件基类
```python
class _SharedLLMComponent:
    """Helper mixin to reuse cached LLM instances."""
    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None, component_name: str = "LLM component"):
        resource = ModelRegistry.get_llm(model_name, device=device, token=token)
        self.device = resource.device
        self.model_name = model_name
        self.tokenizer = resource.tokenizer
        self.model = resource.model
        logger.info(f"{component_name} using shared model: {model_name}")
```

**设计原理**: 位于 [`multi_representation_indexer.py:36-45`](./multi_representation_indexer.py#L36-L45)
- **资源复用**: 所有生成组件共享同一LLM实例
- **统一接口**: 标准化的模型访问方式
- **内存优化**: 避免多次加载大型语言模型

#### 摘要生成器
```python
class SummaryGenerator(_SharedLLMComponent):
    """文档摘要生成器 - 生成简洁的内容概述"""
    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None):
        super().__init__(model_name=model_name, device=device, token=token, component_name="Summary Generator")
        
        self.generation_config = GenerationConfig(
            max_new_tokens=200,    # 摘要长度控制
            temperature=0.3,       # 低温度保证摘要质量
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
```

**生成策略**:
```python
def generate_summary(self, text: str) -> str:
    """生成文本摘要"""
    prompt = f"""请为以下文本生成一个简洁的摘要，重点突出关键信息和核心观点：

文本内容：
{text}

摘要："""
    
    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = self.model.generate(**inputs, generation_config=self.generation_config)
    
    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = response.split("摘要：")[-1].strip()
    return summary
```

**核心特性**: 位于 [`multi_representation_indexer.py:47-90`](./multi_representation_indexer.py#L47-90)
- **中文优化**: 专门针对中文文本的摘要prompt设计
- **长度控制**: 通过 `max_new_tokens` 确保摘要长度适中
- **质量保证**: 低温度和重复惩罚机制提升摘要质量

#### 假设问题生成器
```python
class QuestionGenerator(_SharedLLMComponent):
    """假设问题生成器 - 基于内容生成可能的查询问题"""
    def generate_questions(self, text: str, num_questions: int = 3) -> List[str]:
        """为文本内容生成假设问题"""
        prompt = f"""基于以下文本内容，生成 {num_questions} 个用户可能会问的相关问题。
问题应该涵盖文本的主要内容和关键信息点。

文本内容：
{text}

请生成问题（每个问题占一行）："""
        
        # 生成和解析逻辑...
        questions = self._extract_questions(response, num_questions)
        return questions
```

**问题质量优化**:
```python
def _extract_questions(self, response: str, expected_count: int) -> List[str]:
    """从生成的回复中提取问题"""
    lines = response.split('\n')
    questions = []
    
    for line in lines:
        line = line.strip()
        if line and ('?' in line or '？' in line):
            # 清理问题格式
            question = re.sub(r'^\d+\.?\s*', '', line)  # 移除序号
            question = question.strip('- ').strip()
            if len(question) > 10:  # 过滤过短问题
                questions.append(question)
    
    return questions[:expected_count]  # 返回指定数量
```

**技术亮点**: 位于 [`multi_representation_indexer.py:90+`](./multi_representation_indexer.py#L90)
- **多样性控制**: 生成多个不同角度的问题
- **格式规范**: 自动清理生成结果的格式
- **质量过滤**: 过滤过短或无效的问题

#### 多表示索引器核心类
```python
class MultiRepresentationIndexer:
    """多表示索引器 - 协调摘要和问题生成"""
    def __init__(self, config: Dict):
        model_name = config.get('llm_model', 'Qwen/Qwen2-7B-Instruct')
        device = config.get('device', 'auto')
        token = config.get('hugging_face_token')
        
        self.summary_generator = SummaryGenerator(model_name, device, token)
        self.question_generator = QuestionGenerator(model_name, device, token)
        self.embedder = MultilingualEmbedder(
            config.get('embedding_model', 'BAAI/bge-m3'), device
        )
```

**异步多表示生成流程**:
```python
async def create_multi_representations(self, chunks_data: List[Dict]) -> List[MultiRepresentationChunk]:
    """异步生成多表示数据"""
    multi_rep_chunks = []
    
    for chunk_data in chunks_data:
        chunk = MultiRepresentationChunk(
            content=chunk_data['content'],
            chunk_id=chunk_data['chunk_id'],
            source_id=chunk_data['source_id'],
            metadata=chunk_data['metadata'],
            content_embedding=chunk_data.get('embedding')
        )
        
        # 异步生成摘要和问题
        summary_task = asyncio.to_thread(self.summary_generator.generate_summary, chunk.content)
        questions_task = asyncio.to_thread(self.question_generator.generate_questions, chunk.content)
        
        summary, questions = await asyncio.gather(summary_task, questions_task)
        
        # 设置生成结果
        chunk.summary = summary
        chunk.hypothetical_questions = questions
        
        # 生成摘要和问题的嵌入
        if summary:
            summary_embedding = self.embedder.embedding_model.encode([summary])[0]
            chunk.summary_embedding = summary_embedding
        
        if questions:
            questions_embeddings = self.embedder.embedding_model.encode(questions)
            chunk.questions_embeddings = list(questions_embeddings)
        
        multi_rep_chunks.append(chunk)
    
    return multi_rep_chunks
```

**核心特性**: 位于 [`multi_representation_indexer.py:170-264`](./multi_representation_indexer.py#L170-L264)
- **并发生成**: `asyncio.gather` 同时生成摘要和问题
- **线程池**: `asyncio.to_thread` 将同步LLM调用转为异步
- **向量对齐**: 为每种表示生成对应的嵌入向量
- **错误处理**: 单个chunk失败不影响整体流程

#### 索引条目生成
```python
def generate_index_entries(self, multi_rep_chunks: List[MultiRepresentationChunk]) -> List[Dict]:
    """将多表示数据展开为索引条目"""
    index_entries = []
    
    for chunk in multi_rep_chunks:
        # 1. 原文条目
        index_entries.append({
            'chunk_id': chunk.chunk_id,
            'content': chunk.content,
            'embedding': chunk.content_embedding.tolist(),
            'semantic_type': 'original',
            'representation_type': 'content',
            'source_id': chunk.source_id,
            'metadata': chunk.metadata
        })
        
        # 2. 摘要条目
        if chunk.summary and chunk.summary_embedding is not None:
            index_entries.append({
                'chunk_id': f"{chunk.chunk_id}_summary",
                'content': chunk.summary,
                'embedding': chunk.summary_embedding.tolist(),
                'semantic_type': 'summary',
                'representation_type': 'summary',
                'source_id': chunk.source_id,
                'metadata': {**chunk.metadata, 'original_chunk_id': chunk.chunk_id}
            })
        
        # 3. 问题条目
        for i, (question, q_embedding) in enumerate(zip(chunk.hypothetical_questions, chunk.questions_embeddings)):
            index_entries.append({
                'chunk_id': f"{chunk.chunk_id}_question_{i}",
                'content': question,
                'embedding': q_embedding.tolist(),
                'semantic_type': 'question',
                'representation_type': 'hypothetical_question',
                'source_id': chunk.source_id,
                'metadata': {**chunk.metadata, 'original_chunk_id': chunk.chunk_id, 'question_index': i}
            })
    
    return index_entries
```

**设计优势**: 位于 [`multi_representation_indexer.py:266-341`](./multi_representation_indexer.py#L266-L341)
- **扁平化存储**: 将复杂的多表示结构转为向量库友好的格式
- **类型标识**: 每个条目都有明确的语义类型和表示类型
- **追溯机制**: 通过 `original_chunk_id` 可以追溯到原始文本块
- **元数据传承**: 保持所有必要的上下文信息

### 5. 处理流水线

#### 增强文本处理器
```python
class EnhancedTextProcessor:
    """增强文本处理器 - 统一的处理流水线入口"""
    def __init__(self, config: Dict):
        # 初始化各组件
        self.splitter = HierarchicalTextSplitter(
            chunk_size=config.get('chunk_size', 512),
            chunk_overlap=config.get('chunk_overlap', 50)
        )
        self.embedder = MultilingualEmbedder(
            config.get('embedding_model', 'BAAI/bge-m3'),
            config.get('device', 'auto')
        )
        self.enable_multi_representation = config.get('enable_multi_representation', True)
        if self.enable_multi_representation:
            self.multi_rep_indexer = MultiRepresentationIndexer(config)
```

#### 文档处理主流程
```python
async def process_documents(self, documents: List[Dict]) -> List[Dict]:
    """处理文档集合的主入口"""
    logger.info(f"🔄 开始处理 {len(documents)} 个文档...")
    
    all_chunks = []
    for doc in documents:
        # 1. 分层文本分割
        chunks = self.splitter.split_document(
            doc_content=doc['content'],
            source_id=doc['id'],
            metadata={
                'title': doc.get('title', ''),
                'source': doc.get('source', ''),
                'url': doc.get('url', ''),
                'published_date': doc.get('published_date'),
                **doc.get('metadata', {})
            }
        )
        all_chunks.extend(chunks)
    
    logger.info(f"📄 分割产生 {len(all_chunks)} 个文本块")
    
    # 2. 批量向量化
    vectorized_chunks = self.embedder.embed_chunks(all_chunks)
    logger.info(f"🧮 完成 {len(vectorized_chunks)} 个块的向量化")
    
    # 3. 多表示索引（可选）
    if self.enable_multi_representation:
        logger.info("🔄 开始生成多表示索引...")
        
        # 转换为dict格式传递给索引器
        chunks_data = [
            {
                'content': chunk.content,
                'chunk_id': chunk.chunk_id,
                'source_id': chunk.source_id,
                'metadata': chunk.metadata,
                'embedding': chunk.embedding
            }
            for chunk in vectorized_chunks
        ]
        
        multi_rep_chunks = await self.multi_rep_indexer.create_multi_representations(chunks_data)
        result = self.multi_rep_indexer.generate_index_entries(multi_rep_chunks)
        
        logger.success(f"✅ 生成 {len(result)} 个多表示索引条目")
        return result
    else:
        # 标准索引格式
        result = [
            {
                'chunk_id': chunk.chunk_id,
                'content': chunk.content,
                'embedding': chunk.embedding.tolist(),
                'semantic_type': 'original',
                'representation_type': 'content',
                'source_id': chunk.source_id,
                'metadata': chunk.metadata
            }
            for chunk in vectorized_chunks
        ]
        logger.success(f"✅ 生成 {len(result)} 个标准索引条目")
        return result
```

**流程特点**: 位于 [`text_processor.py:112-167`](./text_processor.py#L112-L167)
- **模块化设计**: 分割、向量化、多表示生成各司其职
- **配置驱动**: 通过配置控制是否启用多表示索引
- **批量优化**: 先完成所有分割再统一向量化，提升效率
- **格式统一**: 无论是否启用多表示，输出格式都保持一致

## 性能与优化

### 1. 内存管理
- **批量处理**: 32个文本一批进行向量化，平衡内存与速度
- **模型共享**: 通过 `ModelRegistry` 确保全局单例，避免重复加载
- **异步处理**: `asyncio` 并发生成多种表示，提升整体效率

### 2. 质量控制
- **长度过滤**: 摘要最大200 tokens，问题最少10字符
- **重复去除**: 自动去除编号、格式符号等干扰信息
- **温度控制**: 较低的生成温度确保输出质量稳定

### 3. 可扩展性
- **插件化架构**: 新的表示类型可以通过继承 `_SharedLLMComponent` 轻松添加
- **配置灵活**: 所有关键参数都可通过配置文件调整
- **模型无关**: 支持任意兼容的 HuggingFace 模型

## 使用示例

```python
from src.processing.text_processor import EnhancedTextProcessor

# 配置处理器
config = {
    'chunk_size': 512,
    'chunk_overlap': 50,
    'embedding_model': 'BAAI/bge-m3',
    'llm_model': 'Qwen/Qwen2-7B-Instruct',
    'device': 'auto',
    'enable_multi_representation': True
}

processor = EnhancedTextProcessor(config)

# 处理文档
documents = [
    {
        'id': 'doc_1',
        'title': 'AI研究论文',
        'content': '这是一篇关于人工智能的研究论文...',
        'source': 'arxiv',
        'metadata': {'authors': ['张三', '李四']}
    }
]

# 异步处理
index_entries = await processor.process_documents(documents)

# 结果分析
for entry in index_entries:
    print(f"类型: {entry['semantic_type']}, 内容: {entry['content'][:50]}...")
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 512 | 文本块最大长度 |
| `chunk_overlap` | 50 | 文本块重叠长度 |
| `embedding_model` | "BAAI/bge-m3" | 嵌入模型名称 |
| `llm_model` | "Qwen/Qwen2-7B-Instruct" | 生成模型名称 |
| `enable_multi_representation` | true | 是否启用多表示索引 |
| `device` | "auto" | 计算设备选择 |

## 扩展指南

### 添加新的表示类型
1. 继承 `_SharedLLMComponent` 创建新的生成器
2. 在 `MultiRepresentationChunk` 中添加对应字段
3. 在 `create_multi_representations` 中添加生成逻辑
4. 在 `generate_index_entries` 中添加条目生成代码

### 优化建议
- **GPU加速**: 配置合适的 `device` 参数利用GPU加速
- **批次调优**: 根据显存大小调整向量化批次大小
- **模型选择**: 根据精度需求选择合适的嵌入和生成模型
- **缓存策略**: 对于重复文档可以考虑添加处理结果缓存

文本处理模块通过先进的层次化分割和多表示索引技术，为 RAG 系统提供了高质量的文本预处理能力，是整个检索系统性能的重要基础。