# src/data_ingestion/multi_source_collector.py
import asyncio
import aiohttp
import feedparser
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from huggingface_hub import HfApi
from loguru import logger
import pymupdf4llm
import fitz  # PyMuPDF
import json

@dataclass
class Document:
    """通用文档数据结构"""
    id: str
    source: str
    title: str
    content: str
    url: Optional[str] = None
    published_date: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)


class MultiSourceCollector:
    """
    企业级多数据源收集器
    - 支持ArXiv, Hugging Face Papers, 主流AI博客
    - 支持异步IO，提升采集效率
    - 内置缓存，避免重复处理
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.raw_data_path = self.data_dir / "raw_collected_data.json"
        self.pdf_dir = self.data_dir / "pdfs"
        self.pdf_dir.mkdir(exist_ok=True)
        self.processed_ids: Set[str] = self._load_processed_ids()

        # AI博客RSS源
        self.blog_feeds = {
            "Google AI": "http://feeds.feedburner.com/blogspot/gJZg",
            "OpenAI": "https://openai.com/blog/rss.xml",
            "BAIR": "https://bair.berkeley.edu/blog/feed.xml",
            "DeepMind": "https://deepmind.google/blog/rss.xml"
        }

    def _load_processed_ids(self) -> Set[str]:
        """加载已处理文档的ID，用于缓存"""
        if self.raw_data_path.exists():
            try:
                with open(self.raw_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {item.get('id', '') for item in data}
            except json.JSONDecodeError:
                return set()
        return set()

    async def collect_all(self, days_back: int = 7) -> List[Document]:
        """从所有配置的数据源并行收集数据"""
        logger.info("🚀 Starting data collection from all sources...")
        
        tasks = [
            self.fetch_arxiv_papers(days_back=days_back),
            self.fetch_huggingface_papers(),
            self.fetch_blog_posts()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_docs = []
        for res in results:
            if isinstance(res, list):
                all_docs.extend(res)
            elif isinstance(res, Exception):
                logger.error(f"A collector task failed: {res}")
        
        logger.success(f"✅ Total documents collected: {len(all_docs)}")

        # 保存到原始数据文件
        self._save_raw_data(all_docs)

        return all_docs

    def _save_raw_data(self, docs: List[Document]):
        """保存原始数据到JSON文件"""
        output_data = [doc.__dict__ for doc in docs]
        with open(self.raw_data_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"💾 Raw data saved to {self.raw_data_path}")

    # --- ArXiv Collector ---
    async def fetch_arxiv_papers(self, query: str = "cat:cs.AI OR cat:cs.CL OR cat:cs.LG", days_back: int = 7) -> List[Document]:
        logger.info("🔍 Collecting from ArXiv...")
        base_url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': query, 'start': 0, 'max_results': 100,
            'sortBy': 'submittedDate', 'sortOrder': 'descending'
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(base_url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"ArXiv API request failed: {response.status}")
                        return []
                    
                    xml_content = await response.text()
                    parsed_papers = self._parse_arxiv_response(xml_content, days_back)
                    
                    # 下载并解析PDF
                    processed_papers = await self.download_and_extract_pdfs(parsed_papers)
                    logger.success(f"ArXiv collection successful: {len(processed_papers)} papers.")
                    return processed_papers
            except Exception as e:
                logger.error(f"Error fetching ArXiv papers: {e}")
                return []
    
    def _parse_arxiv_response(self, xml_content: str, days_back: int) -> List[Dict]:
        """解析ArXiv API的XML响应"""
        papers = []
        root = ET.fromstring(xml_content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for entry in root.findall('atom:entry', ns):
            arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
            if f"arxiv_{arxiv_id}" in self.processed_ids:
                continue

            published_str = entry.find('atom:published', ns).text
            published = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
            if published.replace(tzinfo=None) < cutoff_date:
                continue
            
            pdf_url = next((link.get('href') for link in entry.findall('atom:link', ns) if link.get('type') == 'application/pdf'), f"https://arxiv.org/pdf/{arxiv_id}.pdf")
            
            papers.append({
                'id': f"arxiv_{arxiv_id}",
                'title': entry.find('atom:title', ns).text.strip(),
                'abstract': entry.find('atom:summary', ns).text.strip(),
                'url': entry.find('atom:id', ns).text,
                'pdf_url': pdf_url,
                'published': published
            })
        return papers

    async def download_and_extract_pdfs(self, papers: List[Dict]) -> List[Document]:
        """并发下载PDF并提取文本"""
        semaphore = asyncio.Semaphore(5)
        tasks = [self._process_single_pdf(paper, semaphore) for paper in papers]
        results = await asyncio.gather(*tasks)
        return [doc for doc in results if doc]

    async def _process_single_pdf(self, paper_meta: Dict, semaphore: asyncio.Semaphore) -> Optional[Document]:
        """处理单个PDF的下载和文本提取"""
        async with semaphore:
            pdf_path = self.pdf_dir / f"{paper_meta['id']}.pdf"
            try:
                # Download PDF
                if not pdf_path.exists():
                    async with aiohttp.ClientSession() as session:
                        async with session.get(paper_meta['pdf_url']) as response:
                            if response.status == 200:
                                with open(pdf_path, 'wb') as f:
                                    f.write(await response.read())
                            else:
                                logger.warning(f"Failed to download {paper_meta['pdf_url']}")
                                return None
                
                # Extract text
                extracted_text = self._extract_text_from_pdf(pdf_path)
                if not extracted_text:
                    extracted_text = paper_meta['abstract'] # Fallback to abstract

                return Document(
                    id=paper_meta['id'], source="arxiv", title=paper_meta['title'],
                    content=extracted_text, url=paper_meta['url'],
                    published_date=paper_meta['published'], metadata={'abstract': paper_meta['abstract']}
                )
            except Exception as e:
                logger.error(f"Error processing PDF {paper_meta['id']}: {e}")
                return None
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """使用pymupdf4llm智能提取PDF文本，并动态处理页数"""
        try:
            with fitz.open(pdf_path) as doc:
                page_count = doc.page_count
                pages_to_process = list(range(min(page_count, 5))) # Process up to 5 pages
            
            if not pages_to_process:
                return ""

            md_text = pymupdf4llm.to_markdown(str(pdf_path), pages=pages_to_process)
            if md_text and len(md_text.strip()) > 100:
                return md_text
        except Exception as e:
            logger.warning(f"pymupdf4llm failed for {pdf_path}, trying fallback: {e}")
        
        # Fallback using basic PyMuPDF
        try:
            with fitz.open(pdf_path) as doc:
                return "".join(page.get_text() for page in doc)
        except Exception as e:
            logger.error(f"PDF text extraction failed for {pdf_path}: {e}")
            return ""

    # --- Hugging Face Papers Collector ---
    async def fetch_huggingface_papers(self, limit: int = 20) -> List[Document]:
        logger.info("🤗 Collecting from Hugging Face Papers...")
        try:
            api = HfApi()
            # API现在需要一个查询参数
            papers = api.list_papers(query="artificial intelligence")
            # 手动截取所需数量
            papers_to_process = list(papers)[:limit]

            hf_docs = []
            for paper in papers_to_process:
                paper_id = f"hf_{paper.id}"
                hf_docs.append(Document(
                    id=paper_id, source="huggingface", title=paper.title,
                    content=f"Title: {paper.title}. This is a trending paper on Hugging Face.", # Placeholder content
                    url=f"https://huggingface.co/papers/{paper.id}",
                    published_date=paper.published_at,
                    metadata={'authors': paper.authors}
                ))
            logger.success(f"Hugging Face collection successful: {len(hf_docs)} papers.")
            return hf_docs
        except Exception as e:
            logger.error(f"Error fetching Hugging Face papers: {e}")
            return []

    # --- Blog Collector ---
    async def fetch_blog_posts(self) -> List[Document]:
        logger.info("📝 Collecting from AI Blogs...")
        tasks = [self._parse_feed(name, url) for name, url in self.blog_feeds.items()]
        results = await asyncio.gather(*tasks)
        all_posts = [post for feed_posts in results for post in feed_posts]
        logger.success(f"Blog collection successful: {len(all_posts)} posts.")
        return all_posts

    async def _parse_feed(self, name: str, url: str) -> List[Document]:
        """解析单个RSS源"""
        posts = []
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]: # Get latest 5 posts
                post_id = f"blog_{name.lower().replace(' ', '')}_{entry.id}"
                if post_id in self.processed_ids:
                    continue
                
                posts.append(Document(
                    id=post_id, source=f"blog_{name}", title=entry.title,
                    content=entry.summary, url=entry.link,
                    published_date=datetime(*entry.published_parsed[:6]) if 'published_parsed' in entry else datetime.now(),
                ))
            return posts
        except Exception as e:
            logger.error(f"Failed to parse blog feed {name}: {e}")
            return []

