# 数据采集子系统 (Data Ingestion Module)

## 概述 (Overview)

数据采集子系统是 RAG 系统的第一层，负责从多个异构数据源中自动采集、解析和预处理原始文档。该模块采用企业级异步架构，支持高并发数据采集，具备智能缓存和错误恢复机制。

**核心特性：**
- 🌐 **多数据源支持**: ArXiv 学术论文、Hugging Face Papers、主流 AI 博客
- ⚡ **异步高性能**: 基于 aiohttp 的并发采集，信号量控制资源使用
- 💾 **智能缓存**: 自动去重和增量更新机制
- 🔄 **容错设计**: 多层次错误处理和降级策略
- 📄 **PDF 智能解析**: 双引擎文本提取（pymupdf4llm + PyMuPDF）

## 核心架构

### 1. 统一数据模型

```python
@dataclass
class Document:
    """通用文档数据结构 - 所有数据源的标准化接口"""
    id: str                          # 全局唯一标识符
    source: str                      # 数据源标识 (arxiv/huggingface/blog)
    title: str                       # 文档标题
    content: str                     # 提取的正文内容
    url: Optional[str] = None        # 原始链接
    published_date: Optional[datetime] = None  # 发布时间
    metadata: Dict = field(default_factory=dict)  # 扩展元数据
```

**设计原理：**
- **标准化接口**: 位于 [`multi_source_collector.py:16-25`](./multi_source_collector.py#L16-L25)，确保所有数据源输出统一的数据结构
- **元数据扩展性**: 通过 `metadata` 字段支持不同数据源的特有信息（如作者、摘要、标签等）
- **类型安全**: 使用 `@dataclass` 提供编译时类型检查和自动序列化

### 2. 多源采集器核心类

```python
class MultiSourceCollector:
    """
    企业级多数据源收集器
    - 支持ArXiv, Hugging Face Papers, 主流AI博客
    - 支持异步IO，提升采集效率  
    - 内置缓存，避免重复处理
    """
```

**核心组件：**

#### 缓存机制
```python
def _load_processed_ids(self) -> Set[str]:
    """从 raw_collected_data.json 加载已处理文档ID，实现增量更新"""
    if self.raw_data_path.exists():
        try:
            with open(self.raw_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {item.get('id', '') for item in data}
        except json.JSONDecodeError:
            return set()
    return set()
```
**原理**: 位于 [`multi_source_collector.py:51-60`](./multi_source_collector.py#L51-L60)，通过内存集合快速检查是否已处理，避免重复采集相同文档。

#### 异步协调器
```python
async def collect_all(self, days_back: int = 7) -> List[Document]:
    """主入口 - 并发执行所有数据源采集任务"""
    async with aiohttp.ClientSession() as session:
        tasks = [
            self.fetch_arxiv_papers(session, days_back=days_back),
            self.fetch_huggingface_papers(),
            self.fetch_blog_posts()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
```
**关键特性**: 位于 [`multi_source_collector.py:62-73`](./multi_source_collector.py#L62-L73)
- **连接池复用**: 单一 `ClientSession` 避免连接开销
- **并发执行**: `asyncio.gather` 同时执行所有采集任务
- **异常隔离**: `return_exceptions=True` 确保单个数据源失败不影响其他源

## 数据源详解

### 1. ArXiv 学术论文采集

#### API 查询与解析
```python
async def fetch_arxiv_papers(self, session: aiohttp.ClientSession, 
                           query: str = "cat:cs.AI OR cat:cs.CL OR cat:cs.LG", 
                           days_back: int = 7) -> List[Document]:
```

**查询策略**: 位于 [`multi_source_collector.py:97-100`](./multi_source_collector.py#L97-L100)
- **多类别覆盖**: 计算机科学 AI、计算语言学、机器学习三大核心领域
- **时间窗口**: 可配置天数回溯，默认7天获取最新论文
- **XML解析**: 使用 `xml.etree.ElementTree` 解析 ArXiv Atom Feed

#### PDF 并发下载与解析
```python
async def download_and_extract_pdfs(self, papers: List[Dict], 
                                  session: aiohttp.ClientSession) -> List[Document]:
    """并发下载PDF并提取文本"""
    semaphore = asyncio.Semaphore(5)  # 限制并发数避免服务器压力
    tasks = [self._process_single_pdf(paper, semaphore, session) for paper in papers]
    results = await asyncio.gather(*tasks)
    return [doc for doc in results if doc]
```

**性能优化**: 位于 [`multi_source_collector.py:151-156`](./multi_source_collector.py#L151-L156)
- **信号量控制**: 最多5个并发下载，避免触发 ArXiv 限流
- **批量处理**: 异步并发处理所有PDF，提升整体效率

#### 双引擎文本提取
```python
def _extract_text_from_pdf(self, pdf_path: Path) -> str:
    """使用pymupdf4llm智能提取PDF文本，失败时回退到PyMuPDF"""
    try:
        # 主引擎: pymupdf4llm - 智能Markdown转换
        with fitz.open(pdf_path) as doc:
            page_count = doc.page_count
            pages_to_process = list(range(min(page_count, 5)))
        
        md_text = pymupdf4llm.to_markdown(str(pdf_path), pages=pages_to_process)
        if md_text and len(md_text.strip()) > 100:
            return md_text
    except Exception as e:
        logger.warning(f"pymupdf4llm failed, trying fallback: {e}")
    
    # 备用引擎: PyMuPDF - 基础文本提取
    try:
        with fitz.open(pdf_path) as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""
```

**技术原理**: 位于 [`multi_source_collector.py:187-209`](./multi_source_collector.py#L187-L209)
- **智能解析**: `pymupdf4llm` 保持文档结构，生成高质量 Markdown
- **性能优化**: 只处理前5页，平衡内容质量与处理速度
- **降级策略**: 双引擎确保即使智能解析失败也能获取基础文本
- **质量过滤**: 文本长度阈值确保提取内容的有效性

### 2. Hugging Face Papers 采集

```python
async def fetch_huggingface_papers(self, limit: int = 20) -> List[Document]:
    """采集 Hugging Face 热门论文"""
    try:
        api = HfApi()
        papers = api.list_papers(query="artificial intelligence")
        papers_to_process = list(papers)[:limit]
        
        hf_docs = []
        for paper in papers_to_process:
            paper_id = f"hf_{paper.id}"
            hf_docs.append(Document(
                id=paper_id, source="huggingface", title=paper.title,
                content=f"Title: {paper.title}. This is a trending paper on Hugging Face.",
                url=f"https://huggingface.co/papers/{paper.id}",
                published_date=paper.published_at,
                metadata={'authors': paper.authors}
            ))
```

**特点**: 位于 [`multi_source_collector.py:212-235`](./multi_source_collector.py#L212-L235)
- **API集成**: 直接使用 `HfApi` 获取社区热门论文
- **元数据丰富**: 保留作者信息和发布时间
- **快速获取**: 无需下载PDF，直接构建文档记录

### 3. AI 博客 RSS 采集

#### 多源RSS配置
```python
self.blog_feeds = {
    "Google AI": "http://feeds.feedburner.com/blogspot/gJZg",
    "OpenAI": "https://openai.com/blog/rss.xml", 
    "BAIR": "https://bair.berkeley.edu/blog/feed.xml",
    "DeepMind": "https://deepmind.google/blog/rss.xml"
}
```

#### 并发RSS解析
```python
async def fetch_blog_posts(self) -> List[Document]:
    """并发获取所有博客源的最新文章"""
    blog_docs = []
    tasks = []
    
    for feed_name, feed_url in self.blog_feeds.items():
        task = asyncio.create_task(self._parse_feed(feed_name, feed_url))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

**设计优势**: 位于 [`multi_source_collector.py:238+`](./multi_source_collector.py#L238)
- **主流覆盖**: 涵盖 Google AI、OpenAI、BAIR、DeepMind 等顶级机构
- **并发解析**: 每个RSS源独立异步处理
- **容错机制**: 单个源失败不影响其他源的采集

## 性能与可靠性

### 1. 性能优化策略
- **连接池管理**: 复用 `aiohttp.ClientSession`，减少连接建立开销
- **并发控制**: 信号量限制同时下载数，避免服务器限流
- **内存缓存**: `Set[str]` 快速查找已处理文档ID
- **分页处理**: PDF只处理前5页，平衡质量与速度

### 2. 错误处理机制
- **分层降级**: PDF解析失败时回退到摘要
- **异常隔离**: 单个文档处理失败不影响批次
- **重试逻辑**: 网络请求失败时的自动重试
- **日志记录**: 详细的错误日志便于问题诊断

### 3. 数据持久化
```python
def _save_raw_data(self, docs: List[Document]):
    """保存原始数据到JSON文件"""
    output_data = [doc.__dict__ for doc in docs]
    with open(self.raw_data_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
```

**特性**: 位于 [`multi_source_collector.py:89-94`](./multi_source_collector.py#L89-L94)
- **UTF-8编码**: 支持中文等多语言内容
- **格式化输出**: `indent=2` 便于人工查看和调试
- **类型处理**: `default=str` 自动序列化 datetime 等特殊类型

## 使用示例

```python
from pathlib import Path
from src.data_ingestion.multi_source_collector import MultiSourceCollector

# 初始化采集器
collector = MultiSourceCollector(data_dir=Path("./data"))

# 异步采集所有数据源
documents = await collector.collect_all(days_back=7)

# 结果统计
print(f"总计采集文档: {len(documents)}")
for source in ["arxiv", "huggingface", "blog"]:
    count = len([d for d in documents if d.source == source])
    print(f"{source}: {count} 篇")
```

## 扩展指南

### 添加新数据源
1. 在 `MultiSourceCollector` 中添加新的采集方法
2. 返回 `List[Document]` 格式的标准化数据
3. 在 `collect_all` 方法中注册新任务
4. 添加相应的错误处理和日志记录

### 性能调优
- 调整信号量数量 `asyncio.Semaphore(N)` 控制并发度
- 修改 `days_back` 参数控制数据时间范围
- 配置 `limit` 参数限制单源文档数量
- 优化PDF页数处理范围平衡质量与速度

数据采集子系统为整个RAG系统提供了高质量、多样化的原始语料，其企业级设计确保了系统的可靠性和可扩展性。