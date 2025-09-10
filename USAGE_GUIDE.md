# Ultimate RAG 系统使用指南

## 目录
1. [快速开始](#快速开始)
2. [环境配置](#环境配置)
3. [系统初始化](#系统初始化)
4. [核心功能使用](#核心功能使用)
5. [高级功能配置](#高级功能配置)
6. [API 接口说明](#api-接口说明)
7. [性能优化建议](#性能优化建议)
8. [故障排除](#故障排除)

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装 Qdrant (向量数据库)
docker run -p 6333:6333 qdrant/qdrant
```

### 2. 环境配置

创建 `.env` 文件并配置以下参数：

```env
# 必需配置
STORAGE_ROOT=/path/to/your/storage
HF_HOME=/path/to/huggingface/cache
HUGGING_FACE_TOKEN=your_hf_token

# 模型配置
EMBEDDING_MODEL=BAAI/bge-m3
LLM_MODEL=Qwen/Qwen2-7B-Instruct
DEVICE=auto

# 向量数据库
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=ai_papers

# RAG 参数
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_TOKENS=4096
TEMPERATURE=0.1

# 功能开关
ENABLE_QUERY_INTELLIGENCE=true
ENABLE_MULTI_REPRESENTATION=true
ENABLE_AGENTIC_RAG=true
ENABLE_CONTEXTUAL_COMPRESSION=true
ENABLE_KNOWLEDGE_GRAPH=true
ENABLE_TIERED_GENERATION=true

# API 模型 (可选)
GPT4_API_KEY=your_gpt4_key
GPT4_API_BASE=https://api.openai.com/v1
CLAUDE_API_KEY=your_claude_key
```

### 3. 快速启动示例

```python
from src.generation.ultimate_rag_system import UltimateRAGSystem
from configs.config import config

# 初始化系统
rag_system = UltimateRAGSystem(config)

# 添加文档
documents = [
    {"title": "论文标题", "content": "论文内容", "url": "https://..."},
    # 更多文档...
]
await rag_system.add_documents(documents)

# 查询
result = await rag_system.generate_answer(
    user_query="什么是 Transformer 架构？",
    mode="ultimate"
)

print(f"答案: {result.answer}")
print(f"置信度: {result.confidence}")
print(f"引用来源: {[ref['title'] for ref in result.references]}")
```

## 环境配置

### 配置文件说明

系统使用 `configs/config.py` 作为主配置文件，支持以下配置类别：

#### 路径配置
```python
PROJECT_ROOT = Path(__file__).parent.parent
STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", PROJECT_ROOT / "project_data"))
DATA_DIR = STORAGE_ROOT / "data"
MODEL_DIR = STORAGE_ROOT / "models"
LOG_DIR = STORAGE_ROOT / "logs"
```

#### 模型配置
```python
# 基础模型
EMBEDDING_MODEL = "BAAI/bge-m3"  # 嵌入模型
LLM_MODEL = "Qwen/Qwen2-7B-Instruct"  # 生成模型
FAST_MODEL = "Qwen/Qwen2-1.5B-Instruct"  # 快速模型

# 设备配置
DEVICE = "auto"  # auto/cuda/cpu
```

#### 功能开关
```python
ENABLE_QUERY_INTELLIGENCE = True    # 查询智能化
ENABLE_MULTI_REPRESENTATION = True  # 多重表示
ENABLE_AGENTIC_RAG = True          # 智能体RAG
ENABLE_CONTEXTUAL_COMPRESSION = True # 上下文压缩
ENABLE_KNOWLEDGE_GRAPH = True       # 知识图谱
ENABLE_TIERED_GENERATION = True     # 分层生成
```

## 系统初始化

### 1. 基础初始化

```python
from src.generation.ultimate_rag_system import UltimateRAGSystem
from configs.config import config
import asyncio

async def initialize_system():
    # 创建系统实例
    rag_system = UltimateRAGSystem(config)
    
    # 系统会自动初始化以下组件：
    # - 嵌入模型和生成模型
    # - 向量数据库连接
    # - 知识图谱数据库
    # - 反馈系统数据库
    
    return rag_system

# 运行初始化
rag_system = asyncio.run(initialize_system())
```

### 2. 数据导入

#### 从文件导入
```python
from src.data_ingestion.document_processor import DocumentProcessor

processor = DocumentProcessor()

# 处理 PDF 文档
pdf_chunks = await processor.process_pdf("/path/to/document.pdf")

# 处理文本文档
text_chunks = await processor.process_text("/path/to/document.txt")

# 添加到系统
await rag_system.add_documents(pdf_chunks + text_chunks)
```

#### 从 ArXiv 导入
```python
from src.data_ingestion.arxiv_fetcher import ArxivFetcher

fetcher = ArxivFetcher()

# 搜索并导入论文
papers = await fetcher.search_and_fetch(
    query="transformer attention mechanism",
    max_results=50
)

await rag_system.add_documents(papers)
```

### 3. 知识图谱构建

```python
# 系统会自动从添加的文档中构建知识图谱
# 也可以手动触发知识图谱更新
await rag_system.kg_indexer.rebuild_knowledge_graph()

# 查看知识图谱统计
stats = await rag_system.kg_retriever.get_graph_statistics()
print(f"实体数量: {stats['entities']}")
print(f"关系数量: {stats['relations']}")
print(f"连接数量: {stats['edges']}")
```

## 核心功能使用

### 1. 基础查询

```python
# 基础模式 - 简单向量检索
result = await rag_system.generate_answer(
    user_query="什么是注意力机制？",
    mode="basic"
)
```

### 2. 增强查询

```python
# 增强模式 - 包含查询智能化和多重表示
result = await rag_system.generate_answer(
    user_query="Transformer 和 BERT 有什么区别？",
    mode="enhanced",
    max_chunks=10,
    enable_reranking=True
)
```

### 3. 智能体查询

```python
# 智能体模式 - 迭代检索和验证
result = await rag_system.generate_answer(
    user_query="请详细解释多头注意力机制的工作原理",
    mode="agentic",
    max_iterations=3,
    confidence_threshold=0.8
)
```

### 4. 终极模式

```python
# 终极模式 - 所有高级功能
result = await rag_system.generate_answer(
    user_query="如何设计一个高效的预训练语言模型？",
    mode="ultimate",
    enable_kg_enhancement=True,
    use_tiered_generation=True,
    compression_method="hybrid"
)

# 查看详细结果
print(f"答案: {result.answer}")
print(f"置信度: {result.confidence}")
print(f"使用的模型: {result.model_used}")
print(f"处理时间: {result.generation_time:.2f}s")
print(f"引用来源: {len(result.references)} 篇")

# 查看检索到的知识图谱信息
if result.kg_context:
    print(f"相关实体: {len(result.kg_context['entities'])}")
    print(f"相关关系: {len(result.kg_context['relations'])}")
```

### 5. 批量处理

```python
# 批量查询处理
queries = [
    "什么是 GPT？",
    "BERT 的预训练任务有哪些？",
    "如何评估语言模型的性能？"
]

results = []
for query in queries:
    result = await rag_system.generate_answer(query, mode="enhanced")
    results.append(result)
    
    # 添加用户反馈
    await rag_system.feedback_system.add_feedback(
        query=query,
        answer=result.answer,
        rating=5,  # 1-5 评分
        references=result.references
    )
```

## 高级功能配置

### 1. 查询智能化配置

```python
# 自定义查询智能化参数
query_config = {
    "enable_query_rewriting": True,
    "enable_sub_questions": True,
    "enable_hyde": True,
    "max_sub_questions": 3,
    "rewriting_methods": ["simplification", "expansion", "clarification"]
}

result = await rag_system.generate_answer(
    user_query="复杂的技术问题",
    mode="enhanced",
    query_intelligence_config=query_config
)
```

### 2. 上下文压缩配置

```python
# 配置上下文压缩策略
compression_config = {
    "method": "hybrid",  # sentence_extraction, llm_compression, hybrid
    "target_ratio": 0.6,  # 压缩到 60%
    "preserve_key_sentences": True,
    "enable_reranking": True
}

result = await rag_system.generate_answer(
    user_query="需要大量上下文的问题",
    mode="ultimate",
    compression_config=compression_config
)
```

### 3. 分层生成配置

```python
# 配置模型选择策略
tiered_config = {
    "complexity_threshold": 0.7,
    "cost_optimization": True,
    "fallback_enabled": True,
    "model_preferences": {
        "simple": "local_fast",
        "medium": "local_main", 
        "complex": "gpt4_api"
    }
}

result = await rag_system.generate_answer(
    user_query="复杂的研究问题",
    mode="ultimate",
    tiered_config=tiered_config
)
```

### 4. 知识图谱增强

```python
# 启用知识图谱增强检索
kg_config = {
    "entity_expansion": True,
    "relation_traversal": True,
    "max_hop_distance": 2,
    "entity_relevance_threshold": 0.6
}

result = await rag_system.generate_answer(
    user_query="关于实体间关系的问题",
    mode="ultimate",
    kg_config=kg_config
)
```

## API 接口说明

### 1. 核心 API

#### generate_answer()
```python
async def generate_answer(
    self, 
    user_query: str,
    mode: str = "ultimate",  # basic, enhanced, agentic, ultimate
    **kwargs
) -> UltimateGenerationResult
```

**参数说明:**
- `user_query`: 用户查询
- `mode`: 生成模式
- `max_chunks`: 最大检索块数 (默认: 10)
- `enable_reranking`: 启用重排序 (默认: True)
- `compression_method`: 压缩方法 (默认: "hybrid")
- `confidence_threshold`: 置信度阈值 (默认: 0.7)

**返回值:**
```python
@dataclass
class UltimateGenerationResult:
    answer: str
    confidence: float
    references: List[Dict]
    kg_context: Optional[Dict]
    query_analysis: Optional[Dict]
    agentic_steps: List[Dict]
    generation_time: float
    model_used: str
    retrieval_stats: Dict
```

### 2. 文档管理 API

#### add_documents()
```python
async def add_documents(
    self,
    documents: List[Dict],
    enable_kg_extraction: bool = True,
    batch_size: int = 100
) -> Dict[str, Any]
```

#### remove_documents()
```python
async def remove_documents(
    self,
    document_ids: List[str]
) -> Dict[str, Any]
```

#### update_documents()
```python
async def update_documents(
    self,
    documents: List[Dict]
) -> Dict[str, Any]
```

### 3. 知识图谱 API

#### search_entities()
```python
async def search_entities(
    self,
    entity_name: str,
    entity_type: str = None,
    limit: int = 10
) -> List[Dict]
```

#### get_entity_relations()
```python
async def get_entity_relations(
    self,
    entity_id: str,
    max_depth: int = 2
) -> Dict[str, Any]
```

### 4. 反馈系统 API

#### add_feedback()
```python
async def add_feedback(
    self,
    query: str,
    answer: str,
    rating: int,
    feedback_text: str = None,
    references: List[Dict] = None
) -> Dict[str, Any]
```

#### get_feedback_stats()
```python
async def get_feedback_stats(
    self,
    start_date: datetime = None,
    end_date: datetime = None
) -> Dict[str, Any]
```

### 5. 评估系统 API

#### run_evaluation()
```python
async def run_evaluation(
    self,
    test_queries: List[Dict],
    modes: List[str] = ["basic", "enhanced", "ultimate"]
) -> Dict[str, Any]
```

## 性能优化建议

### 1. 模型选择优化

```python
# 根据查询复杂度自动选择模型
config.DEFAULT_RAG_MODE = "ultimate"  # 平衡性能和质量
config.ENABLE_TIERED_GENERATION = True  # 启用分层生成

# 为简单查询使用快速模型
config.FAST_MODEL = "Qwen/Qwen2-1.5B-Instruct"
```

### 2. 缓存配置

```python
# 设置模型缓存目录
config.HF_HOME = "/path/to/fast/ssd/cache"

# 启用查询结果缓存
config.ENABLE_QUERY_CACHE = True
config.CACHE_TTL = 3600  # 1小时
```

### 3. 批处理优化

```python
# 批量处理文档
batch_size = 50 if torch.cuda.is_available() else 10

documents_batches = [documents[i:i+batch_size] 
                    for i in range(0, len(documents), batch_size)]

for batch in documents_batches:
    await rag_system.add_documents(batch, batch_size=batch_size)
```

### 4. 内存管理

```python
# 定期清理缓存
import gc
import torch

async def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 每处理1000个查询后清理
query_count = 0
for query in queries:
    result = await rag_system.generate_answer(query)
    query_count += 1
    
    if query_count % 1000 == 0:
        await cleanup_memory()
```

## 故障排除

### 1. 常见错误及解决方案

#### 内存不足错误
```python
# 错误: CUDA out of memory
# 解决方案: 减少批处理大小或使用CPU
config.DEVICE = "cpu"
config.CHUNK_SIZE = 256  # 减少块大小
```

#### 模型加载失败
```python
# 错误: Model not found
# 解决方案: 检查HF_TOKEN和网络连接
export HF_TOKEN=your_token
pip install --upgrade transformers
```

#### 向量数据库连接失败
```python
# 错误: Connection refused to Qdrant
# 解决方案: 确保Qdrant服务运行
docker run -d -p 6333:6333 qdrant/qdrant
```

### 2. 性能调优

#### 查询速度慢
```python
# 优化策略:
1. 启用分层生成: config.ENABLE_TIERED_GENERATION = True
2. 减少检索块数: max_chunks = 5
3. 使用句子提取压缩: compression_method = "sentence_extraction"
4. 禁用知识图谱增强: enable_kg_enhancement = False
```

#### 答案质量不佳
```python
# 优化策略:
1. 启用智能体模式: mode = "agentic"
2. 增加检索块数: max_chunks = 15
3. 启用查询重写: enable_query_rewriting = True
4. 使用混合压缩: compression_method = "hybrid"
```

### 3. 日志和监控

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.INFO)

# 监控系统状态
stats = await rag_system.get_system_stats()
print(f"文档数量: {stats['document_count']}")
print(f"向量数量: {stats['vector_count']}")
print(f"知识图谱实体数: {stats['entity_count']}")
print(f"平均查询时间: {stats['avg_query_time']:.2f}s")
```

### 4. 数据备份和恢复

```python
# 备份数据
await rag_system.backup_data("/path/to/backup")

# 恢复数据  
await rag_system.restore_data("/path/to/backup")

# 增量备份
await rag_system.incremental_backup("/path/to/backup")
```

## 使用最佳实践

### 1. 查询优化
- 使用具体、明确的查询词
- 避免过于宽泛或模糊的问题
- 利用查询智能化功能自动优化查询

### 2. 文档质量
- 确保文档内容结构清晰
- 添加丰富的元数据信息
- 定期更新和维护文档库

### 3. 反馈循环
- 积极收集用户反馈
- 定期运行评估流程
- 基于反馈数据微调嵌入模型

### 4. 系统监控
- 监控查询响应时间
- 跟踪系统资源使用情况
- 定期检查答案质量指标

通过遵循本使用指南，您可以充分利用 Ultimate RAG 系统的强大功能，构建高质量的智能问答应用。