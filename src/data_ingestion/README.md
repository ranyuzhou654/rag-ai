# 数据采集子系统

`multi_source_collector.py` 实现了从论文库与主流 AI 博客同步抓取语料的完整流程，核心由 `Document` 数据结构与 `MultiSourceCollector` 组成。

## 数据模型
```python
@dataclass
class Document:
    id: str
    source: str
    title: str
    content: str
    url: Optional[str] = None
    published_date: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)
```
- 文档在 [`Document`](./multi_source_collector.py#L16-L24) 中统一描述，保证不同来源的数据可以被后续处理组件直接消费。
- `_load_processed_ids` 会从 `raw_collected_data.json` 中加载已处理 ID，避免重复采集。

## 采集流程
`collect_all` 是异步入口，负责串联各数据源并保存结果：

```python
async def collect_all(self, days_back: int = 7) -> List[Document]:
    async with aiohttp.ClientSession() as session:
        tasks = [
            self.fetch_arxiv_papers(session, days_back=days_back),
            self.fetch_huggingface_papers(),
            self.fetch_blog_posts()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    self._save_raw_data(all_docs)
```
- 位于 [`collect_all`](./multi_source_collector.py#L41-L74)，复用单一 `ClientSession` 并并发执行各采集协程。
- `_save_raw_data` 会以 JSON 写回 `data/raw/raw_collected_data.json`，供 `TextProcessor` 使用。

## ArXiv 论文抓取
- `fetch_arxiv_papers` 通过公共 API 拉取最近提交的论文，解析 XML 后交给 `download_and_extract_pdfs` 下载 PDF、抽取正文。
- `_process_single_pdf` 会在共享信号量下下载 PDF，优先使用 `pymupdf4llm` 提取 Markdown，不足时回退到 `fitz`：

```python
async with session.get(paper_meta['pdf_url']) as response:
    if response.status == 200:
        with open(pdf_path, 'wb') as f:
            f.write(await response.read())
extracted_text = self._extract_text_from_pdf(pdf_path)
return Document(..., content=extracted_text or paper_meta['abstract'])
```
- 上述逻辑见 [`_process_single_pdf`](./multi_source_collector.py#L96-L142)，确保即使 PDF 解析失败也能回退到摘要。

## Hugging Face Papers
- `fetch_huggingface_papers` 利用 `HfApi.list_papers` 获取热点论文，并构造带有作者与链接的 `Document`。函数定义位于 [`fetch_huggingface_papers`](./multi_source_collector.py#L151-L181)。

## RSS 博客
- `fetch_blog_posts` 针对 `self.blog_feeds` 中配置的 RSS 地址并发解析，`_parse_feed` 会跳过已采集的文章并保留最近条目。
- 通过 `feedparser` 提取摘要、发布时间等字段，代码位于 [`_parse_feed`](./multi_source_collector.py#L190-L221)。

整个采集模块的输出就是 `Document` 列表，后续由处理阶段统一切分、向量化。