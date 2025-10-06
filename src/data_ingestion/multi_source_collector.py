# src/data_ingestion/multi_source_collector.py
import asyncio
import os
import aiohttp
from aiohttp import ClientTimeout
import feedparser
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Union, Any, Tuple
from dataclasses import dataclass, field
from huggingface_hub import HfApi
from loguru import logger
import pymupdf4llm
import fitz  # PyMuPDF
import json
import hashlib

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
    
    # 新增字段用于元数据优先策略
    abstract: Optional[str] = None  # 论文摘要或简短描述
    authors: Optional[List[str]] = None  # 作者列表
    keywords: Optional[List[str]] = None  # 关键词
    doi: Optional[str] = None  # DOI标识符
    arxiv_id: Optional[str] = None  # ArXiv ID
    pdf_url: Optional[str] = None  # PDF下载链接
    is_full_text: bool = False  # 标识是否已获取全文
    tldr: Optional[str] = None  # 简短摘要
    concepts: Optional[List[str]] = None  # 主题概念或标签

    # 引用信息
    citation_info: Dict = field(default_factory=dict)  # 引用格式信息


class MultiSourceCollector:
    """
    企业级多数据源收集器 - 元数据优先策略
    - 支持ArXiv, Hugging Face Papers, 主流AI博客
    - 采用"元数据+按需全文"策略，优化存储和性能
    - 支持异步IO，提升采集效率
    - 内置缓存，避免重复处理
    - 支持引用信息生成和来源追溯
    """

    def __init__(self, data_dir: Path, metadata_only: bool = True):
        self.data_dir = data_dir
        self.raw_data_path = self.data_dir / "raw_collected_data.json"
        self.metadata_path = self.data_dir / "metadata_index.json"  # 元数据索引
        self.pdf_dir = self.data_dir / "pdfs"
        self.pdf_cache_dir = self.data_dir / "pdf_cache"  # PDF缓存目录
        
        # 创建必要目录
        for dir_path in [self.pdf_dir, self.pdf_cache_dir]:
            dir_path.mkdir(exist_ok=True)
            
        self.processed_ids: Set[str] = self._load_processed_ids()
        self.metadata_only = metadata_only  # 控制是否只收集元数据
        
        # 缓存管理
        self.pdf_download_queue = asyncio.Queue()  # PDF下载队列
        self.metadata_cache: Dict[str, Dict] = {}  # 元数据缓存
        self._metadata_cache_loaded = False
        self.http_timeout = ClientTimeout(total=60)
        self.session_headers = {
            "User-Agent": "rag-ai-collector/1.0"
        }
        self.semantic_scholar_enabled = os.getenv("ENABLE_SEMANTIC_SCHOLAR", "true").lower() == "true"
        self.semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

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
        docs = await self._collect_from_sources(days_back=days_back, metadata_only=self.metadata_only)
        if docs:
            self._save_raw_data(docs)
        logger.success(f"✅ Total documents collected: {len(docs)}")
        return docs

    async def collect_recent(self, days_back: int = 30, max_papers: Optional[int] = None) -> List[Document]:
        """收集最近一段时间的数据，可设置最大数量"""
        docs = await self._collect_from_sources(
            days_back=days_back,
            metadata_only=self.metadata_only,
            max_papers=max_papers
        )
        if docs:
            self._save_raw_data(docs)
        logger.success(f"✅ Recent collection complete: {len(docs)} documents")
        return docs

    async def collect_metadata_only(
        self,
        days_back: int = 7,
        max_papers: Optional[int] = None
    ) -> List[Document]:
        """仅收集元数据，不下载全文"""
        docs = await self._collect_from_sources(
            days_back=days_back,
            metadata_only=True,
            max_papers=max_papers
        )
        logger.success(f"✅ Metadata-only collection complete: {len(docs)} documents")
        return docs

    async def _collect_from_sources(
        self,
        days_back: int,
        metadata_only: bool,
        max_papers: Optional[int] = None,
        update_index: bool = True
    ) -> List[Document]:
        """内部方法：统一从各源收集数据"""
        mode_label = "metadata" if metadata_only else "full"
        logger.info(f"🚀 Starting data collection ({mode_label} mode)...")

        previous_mode = self.metadata_only
        self.metadata_only = metadata_only

        try:
            all_docs: List[Document] = []
            async with aiohttp.ClientSession(timeout=self.http_timeout, headers=self.session_headers) as session:
                tasks = [
                    self.fetch_arxiv_papers(session, days_back=days_back),
                    self.fetch_openalex_papers(session, days_back=days_back),
                    self.fetch_huggingface_papers(),
                    self.fetch_blog_posts()
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for res in results:
                    if isinstance(res, list):
                        all_docs.extend(res)
                    elif isinstance(res, Exception):
                        logger.error(f"❌ A collector task failed: {res}")

                if all_docs:
                    await self._enrich_with_semantic_scholar(session, all_docs)

            if max_papers and len(all_docs) > max_papers:
                all_docs = all_docs[:max_papers]

            if all_docs and update_index:
                self._append_to_metadata_index(all_docs)

            return all_docs

        finally:
            self.metadata_only = previous_mode

    async def collect_arxiv_history(
        self,
        years: int = 10,
        categories: str = "cat:cs.AI OR cat:cs.CL OR cat:cs.LG",
        batch_days: int = 30,
        max_results_per_query: int = 200,
        max_total: Optional[int] = None,
        metadata_only: Optional[bool] = None
    ) -> List[Document]:
        """按时间窗口收集近几年 ArXiv 论文"""
        end_date = datetime.utcnow()
        start_boundary = end_date - timedelta(days=365 * years)
        logger.info(
            f"🚀 Collecting ArXiv papers from {start_boundary.date()} to {end_date.date()}"
        )

        collected: List[Document] = []

        target_mode = self.metadata_only if metadata_only is None else metadata_only
        previous_mode = self.metadata_only
        self.metadata_only = target_mode

        try:
            async with aiohttp.ClientSession(timeout=self.http_timeout, headers=self.session_headers) as session:
                window_end = end_date
                while window_end > start_boundary:
                    window_start = max(start_boundary, window_end - timedelta(days=batch_days))
                    docs = await self._fetch_arxiv_window(
                        session=session,
                        base_query=categories,
                        window_start=window_start,
                        window_end=window_end,
                        max_results=max_results_per_query
                    )
                    if docs:
                        collected.extend(docs)
                        logger.info(
                            f"📚 Collected {len(docs)} papers between {window_start.date()} and {window_end.date()}"
                        )
                    window_end = window_start
                    if max_total and len(collected) >= max_total:
                        collected = collected[:max_total]
                        break
                    await asyncio.sleep(1)

            self._save_raw_data(collected)
            logger.success(f"✅ Historical collection complete: {len(collected)} documents")
            return collected
        finally:
            self.metadata_only = previous_mode

    def _save_raw_data(self, docs: List[Document]):
        """保存原始数据到JSON文件"""
        output_data = [doc.__dict__ for doc in docs]
        with open(self.raw_data_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"💾 Raw data saved to {self.raw_data_path}")
        self.processed_ids.update(doc.id for doc in docs)

    def _load_metadata_index(self) -> Dict[str, Dict]:
        """加载元数据索引到内存缓存"""
        if not self._metadata_cache_loaded:
            if self.metadata_path.exists():
                try:
                    with open(self.metadata_path, 'r', encoding='utf-8') as f:
                        self.metadata_cache = json.load(f)
                except Exception as exc:
                    logger.error(f"❌ Failed to load metadata index: {exc}")
                    self.metadata_cache = {}
            self._metadata_cache_loaded = True
        return self.metadata_cache

    def _persist_metadata_index(self):
        """将内存中的元数据索引写回磁盘"""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata_cache, f, ensure_ascii=False, indent=2)

    def _get_metadata_entry(self, doc_id: str) -> Optional[Dict]:
        index = self._load_metadata_index()
        return index.get(doc_id)

    def _update_metadata_entry(self, doc_id: str, updates: Dict):
        index = self._load_metadata_index()
        entry = index.setdefault(doc_id, {})
        entry.update(updates)
        self.metadata_cache[doc_id] = entry
        self._persist_metadata_index()

    # --- ArXiv Collector ---
    async def fetch_arxiv_papers(self, session: aiohttp.ClientSession, query: str = "cat:cs.AI OR cat:cs.CL OR cat:cs.LG", days_back: int = 7) -> List[Document]:
        logger.info("🔍 Collecting from ArXiv...")
        base_url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': query, 'start': 0, 'max_results': 100,
            'sortBy': 'submittedDate', 'sortOrder': 'descending'
        }

        try:
            async with session.get(base_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"ArXiv API request failed: {response.status}")
                    return []

                xml_content = await response.text()
                cutoff = datetime.utcnow() - timedelta(days=days_back)
                parsed_papers = self._parse_arxiv_response(xml_content, start_date=cutoff)

                # 下载并解析PDF
                processed_papers = await self.download_and_extract_pdfs(parsed_papers, session)
                logger.success(f"ArXiv collection successful: {len(processed_papers)} papers.")
                return processed_papers
        except Exception as e:
            logger.error(f"Error fetching ArXiv papers: {e}")
            return []

    async def fetch_openalex_papers(
        self,
        session: aiohttp.ClientSession,
        days_back: int = 7,
        per_page: int = 100,
        max_pages: int = 2
    ) -> List[Document]:
        """从OpenAlex获取论文元数据"""
        logger.info("🔍 Collecting from OpenAlex...")

        cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).date()
        params = {
            'filter': f'from_publication_date:{cutoff_date.isoformat()}',
            'per-page': per_page,
            'cursor': '*',
            'select': ','.join([
                'id', 'display_name', 'publication_year', 'publication_date',
                'doi', 'authorships', 'concepts', 'abstract_inverted_index',
                'primary_location', 'host_venue'
            ]),
            'sort': 'publication_date:desc'
        }

        base_url = "https://api.openalex.org/works"
        collected: List[Document] = []

        for page in range(max_pages):
            try:
                async with session.get(base_url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"❌ OpenAlex request failed: HTTP {response.status}")
                        break
                    payload = await response.json()
            except Exception as exc:
                logger.error(f"❌ OpenAlex fetch error: {exc}")
                break

            works = payload.get('results', [])
            if not works:
                break

            for work in works:
                document = self._convert_openalex_work(work)
                if document:
                    collected.append(document)

            next_cursor = payload.get('meta', {}).get('next_cursor')
            if not next_cursor:
                break
            params['cursor'] = next_cursor

            if len(collected) >= per_page * max_pages:
                break

            await asyncio.sleep(0.5)  # polite pacing

        logger.success(f"OpenAlex collection successful: {len(collected)} records")
        return collected

    def _convert_openalex_work(self, work: Dict[str, Any]) -> Optional[Document]:
        work_id = work.get('id')
        if not work_id:
            return None

        short_id = work_id.split('/')[-1]
        doc_id = f"openalex_{short_id}"
        if doc_id in self.processed_ids:
            return None

        title = work.get('display_name', '')
        abstract = self._decode_openalex_abstract(work.get('abstract_inverted_index'))

        authors = []
        for authorship in work.get('authorships', []) or []:
            author = authorship.get('author', {})
            name = author.get('display_name')
            if name:
                authors.append(name)

        concepts = [
            concept.get('display_name')
            for concept in (work.get('concepts') or [])[:5]
            if concept.get('display_name')
        ]

        landing_page = work.get('primary_location', {}) or {}
        host_venue = landing_page.get('source') or {}
        pdf_url = landing_page.get('pdf_url') or None
        landing_url = landing_page.get('landing_page_url') or work_id

        published_date = None
        publication_date = work.get('publication_date')
        if publication_date:
            try:
                published_date = datetime.fromisoformat(publication_date)
            except ValueError:
                published_date = None
        if published_date is None and work.get('publication_year'):
            try:
                published_date = datetime(int(work['publication_year']), 1, 1)
            except Exception:  # pragma: no cover - defensive fallback
                published_date = None

        metadata = {
            'source': 'openalex',
            'openalex_id': short_id,
            'host_venue': host_venue.get('display_name'),
            'is_oa': landing_page.get('is_oa'),
            'oa_status': landing_page.get('open_access', {}).get('status') if isinstance(landing_page.get('open_access'), dict) else None,
            'landing_page': landing_url,
            'concepts': concepts
        }

        doc = Document(
            id=doc_id,
            source='openalex',
            title=title,
            content=abstract or title,
            url=landing_url,
            published_date=published_date,
            metadata=metadata,
            abstract=abstract or None,
            authors=authors or None,
            keywords=concepts or None,
            doi=work.get('doi'),
            pdf_url=pdf_url,
            is_full_text=False,
            concepts=concepts or None,
            citation_info={
                'source': 'OpenAlex',
                'openalex_id': short_id,
                'publication_year': work.get('publication_year')
            }
        )
        return doc

    def _decode_openalex_abstract(self, inverted_index: Optional[Dict[str, Any]]) -> Optional[str]:
        if not inverted_index:
            return None

        positions: List[Tuple[int, str]] = []
        for token, indexes in inverted_index.items():
            for idx in indexes:
                positions.append((idx, token))

        if not positions:
            return None

        positions.sort(key=lambda item: item[0])
        return ' '.join(token for _, token in positions)

    async def _enrich_with_semantic_scholar(
        self,
        session: aiohttp.ClientSession,
        documents: List[Document],
        max_total: int = 40,
        max_concurrent: int = 3
    ):
        if not self.semantic_scholar_enabled or not documents:
            return

        identifiers: List[Tuple[Document, str]] = []
        seen_ids: Set[str] = set()
        for doc in documents:
            identifier = self._resolve_semantic_scholar_identifier(doc)
            if not identifier or identifier in seen_ids:
                continue
            seen_ids.add(identifier)
            identifiers.append((doc, identifier))
            if len(identifiers) >= max_total:
                break

        if not identifiers:
            return

        before_enriched = sum(1 for doc in documents if doc.tldr)
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []
        for doc, identifier in identifiers:
            tasks.append(self._fetch_and_merge_semantic_scholar(session, semaphore, doc, identifier))

        await asyncio.gather(*tasks, return_exceptions=True)

        after_enriched = sum(1 for doc in documents if doc.tldr)
        new_count = max(after_enriched - before_enriched, 0)
        if new_count:
            logger.info(f"✨ Semantic Scholar enriched {new_count} documents")

    async def _fetch_and_merge_semantic_scholar(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        document: Document,
        identifier: str
    ):
        async with semaphore:
            data = await self._fetch_semantic_scholar_metadata(session, identifier)
            if data:
                self._merge_semantic_scholar_data(document, data)

    def _resolve_semantic_scholar_identifier(self, document: Document) -> Optional[str]:
        if document.doi:
            return f"DOI:{document.doi}"
        if document.arxiv_id:
            return f"ARXIV:{document.arxiv_id}"
        external_ids = document.metadata.get('external_ids') if isinstance(document.metadata, dict) else None
        if external_ids:
            doi = external_ids.get('DOI')
            if doi:
                return f"DOI:{doi}"
        return None

    async def _fetch_semantic_scholar_metadata(
        self,
        session: aiohttp.ClientSession,
        identifier: str
    ) -> Optional[Dict[str, Any]]:
        base_url = f"https://api.semanticscholar.org/graph/v1/paper/{identifier}"
        params = {
            'fields': 'title,abstract,tldr,publicationDate,authors,isOpenAccess,url,externalIds'
        }
        headers = dict(self.session_headers)
        if self.semantic_scholar_api_key:
            headers['x-api-key'] = self.semantic_scholar_api_key

        try:
            async with session.get(base_url, params=params, headers=headers) as response:
                if response.status != 200:
                    if response.status != 404:
                        logger.debug(f"Semantic Scholar request failed for {identifier}: {response.status}")
                    return None
                return await response.json()
        except Exception as exc:
            logger.debug(f"Semantic Scholar fetch error for {identifier}: {exc}")
            return None

    def _merge_semantic_scholar_data(self, document: Document, data: Dict[str, Any]):
        tldr = None
        tldr_field = data.get('tldr')
        if isinstance(tldr_field, dict):
            tldr = tldr_field.get('text')
        elif isinstance(tldr_field, str):  # pragma: no cover - future proofing
            tldr = tldr_field

        if tldr:
            document.tldr = tldr
            document.metadata.setdefault('semantic_scholar', {})['tldr'] = tldr

        if data.get('abstract') and (not document.abstract or len(document.abstract) < len(data['abstract'])):
            document.abstract = data['abstract']
            document.content = document.abstract

        if data.get('authors'):
            authors = [author.get('name') for author in data['authors'] if author.get('name')]
            if authors:
                document.authors = authors

        if data.get('publicationDate') and not document.published_date:
            try:
                document.published_date = datetime.fromisoformat(data['publicationDate'])
            except Exception:  # pragma: no cover
                pass

        if isinstance(document.metadata, dict):
            sem_meta = document.metadata.setdefault('semantic_scholar', {})
            sem_meta['paper_id'] = data.get('paperId')
            sem_meta['url'] = data.get('url')
            sem_meta['is_open_access'] = data.get('isOpenAccess')

    async def _fetch_arxiv_window(
        self,
        session: aiohttp.ClientSession,
        base_query: str,
        window_start: datetime,
        window_end: datetime,
        max_results: int
    ) -> List[Document]:
        query = (
            f"({base_query}) AND submittedDate:"
            f"[{window_start.strftime('%Y%m%d%H%M%S')} TO {window_end.strftime('%Y%m%d%H%M%S')}]"
        )
        base_url = "http://export.arxiv.org/api/query"
        start = 0
        collected: List[Document] = []

        while True:
            params = {
                'search_query': query,
                'start': start,
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            async with session.get(base_url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"ArXiv request failed ({response.status}) for {window_start.date()}-{window_end.date()}")
                    break
                xml_content = await response.text()
                parsed = self._parse_arxiv_response(
                    xml_content,
                    start_date=window_start,
                    end_date=window_end
                )
                if not parsed:
                    break
                docs = await self.download_and_extract_pdfs(parsed, session)
                collected.extend(docs)
                if len(parsed) < max_results:
                    break
            start += max_results
            await asyncio.sleep(1)
        return collected
    
    def _parse_arxiv_response(
        self,
        xml_content: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """解析ArXiv API的XML响应，可按时间范围过滤"""
        papers = []
        root = ET.fromstring(xml_content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('atom:entry', ns):
            arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
            if f"arxiv_{arxiv_id}" in self.processed_ids:
                continue

            published_str = entry.find('atom:published', ns).text
            published = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
            naive_published = published.replace(tzinfo=None)
            if start_date and naive_published < start_date:
                continue
            if end_date and naive_published > end_date:
                continue

            pdf_url = next((link.get('href') for link in entry.findall('atom:link', ns) if link.get('type') == 'application/pdf'), f"https://arxiv.org/pdf/{arxiv_id}.pdf")
            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
            categories = [cat.get('term') for cat in entry.findall('atom:category', ns)]

            papers.append({
                'id': f"arxiv_{arxiv_id}",
                'title': entry.find('atom:title', ns).text.strip(),
                'abstract': entry.find('atom:summary', ns).text.strip(),
                'url': entry.find('atom:id', ns).text,
                'pdf_url': pdf_url,
                'published': published,
                'authors': authors,
                'categories': categories
            })
        return papers

    async def download_and_extract_pdfs(self, papers: List[Dict], session: aiohttp.ClientSession) -> List[Document]:
        """并发下载PDF并提取文本"""
        if self.metadata_only:
            docs: List[Document] = []
            for paper in papers:
                metadata = {
                    'source': 'arxiv',
                    'authors': paper.get('authors', []),
                    'categories': paper.get('categories', []),
                    'published': paper.get('published').isoformat() if paper.get('published') else None,
                    'pdf_url': paper.get('pdf_url'),
                    'url': paper.get('url')
                }
                abstract = paper.get('abstract', '')
                doc = Document(
                    id=paper['id'],
                    source='arxiv',
                    title=paper.get('title', ''),
                    content=abstract,
                    url=paper.get('url'),
                    published_date=paper.get('published'),
                    metadata=metadata,
                    abstract=abstract,
                    authors=paper.get('authors', []),
                    doi=None,
                    arxiv_id=paper['id'].replace('arxiv_', ''),
                    pdf_url=paper.get('pdf_url'),
                    is_full_text=False
                )
                docs.append(doc)
            return docs

        semaphore = asyncio.Semaphore(5)
        tasks = [self._process_single_pdf(paper, semaphore, session) for paper in papers]
        results = await asyncio.gather(*tasks)
        return [doc for doc in results if doc]

    async def _process_single_pdf(self, paper_meta: Dict, semaphore: asyncio.Semaphore, session: aiohttp.ClientSession) -> Optional[Document]:
        """处理单个PDF的下载和文本提取"""
        async with semaphore:
            pdf_path = self.pdf_dir / f"{paper_meta['id']}.pdf"
            try:
                # Download PDF
                if not pdf_path.exists():
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

                metadata = {
                    'source': 'arxiv',
                    'abstract': paper_meta.get('abstract'),
                    'authors': paper_meta.get('authors', []),
                    'categories': paper_meta.get('categories', []),
                    'pdf_url': paper_meta.get('pdf_url')
                }

                return Document(
                    id=paper_meta['id'],
                    source="arxiv",
                    title=paper_meta['title'],
                    content=extracted_text,
                    url=paper_meta['url'],
                    published_date=paper_meta['published'],
                    metadata=metadata,
                    abstract=paper_meta.get('abstract'),
                    authors=paper_meta.get('authors', []),
                    arxiv_id=paper_meta['id'].replace('arxiv_', ''),
                    pdf_url=paper_meta.get('pdf_url'),
                    is_full_text=True
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
    
    # --- 新增方法：批量预热缓存 ---
    async def preload_popular_papers(self, paper_ids: List[str]):
        """预热加载热门论文的全文"""
        logger.info(f"🔥 Preloading {len(paper_ids)} popular papers...")
        
        semaphore = asyncio.Semaphore(3)  # 限制并发数
        tasks = [self._preload_single_paper(paper_id, semaphore) for paper_id in paper_ids]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.success("✅ Popular papers preloading completed")
    
    async def _preload_single_paper(self, paper_id: str, semaphore: asyncio.Semaphore):
        """预加载单篇论文"""
        async with semaphore:
            full_text = await self.fetch_full_text_on_demand(paper_id)
            if full_text:
                logger.info(f"✅ Preloaded: {paper_id}")
            else:
                logger.warning(f"⚠️ Failed to preload: {paper_id}")
    
    # --- 新增方法：增量更新 ---
    async def daily_incremental_update(self) -> List[Document]:
        """每日增量更新元数据"""
        logger.info("🔄 Starting daily incremental metadata update...")
        
        # 只获取最近1天的数据
        new_docs = await self._collect_from_sources(
            days_back=1,
            metadata_only=True,
            update_index=False
        )
        
        # 过滤出真正的新文档
        truly_new_docs = [doc for doc in new_docs if doc.id not in self.processed_ids]
        
        if truly_new_docs:
            # 保存新数据并刷新索引
            self._append_to_metadata_index(truly_new_docs)
            logger.success(f"✅ Daily update completed: {len(truly_new_docs)} new documents")
        else:
            logger.info("💯 No new documents found in daily update")
        
        return truly_new_docs
    
    def _append_to_metadata_index(self, new_docs: List[Document]):
        """将新文档附加到元数据索引"""
        if not new_docs:
            return

        index = self._load_metadata_index()

        for doc in new_docs:
            index[doc.id] = {
                'title': doc.title,
                'abstract': doc.abstract,
                'authors': doc.authors,
                'keywords': doc.keywords,
                'url': doc.url,
                'pdf_url': doc.pdf_url,
                'doi': doc.doi,
                'arxiv_id': doc.arxiv_id,
                'published_date': doc.published_date.isoformat() if doc.published_date else None,
                'source': doc.source,
                'citation_info': doc.citation_info,
                'is_full_text': doc.is_full_text,
                'tldr': doc.tldr,
                'concepts': doc.concepts
            }

        self.metadata_cache = index
        self._persist_metadata_index()
        self.processed_ids.update(doc.id for doc in new_docs if doc.id)
        logger.info(f"💾 Metadata index updated with {len(new_docs)} new entries")

    async def fetch_full_text_on_demand(
        self,
        doc_id: str,
        session: Optional[aiohttp.ClientSession] = None
    ) -> Optional[str]:
        """按需获取论文全文并缓存结果"""
        metadata = self._get_metadata_entry(doc_id)
        if not metadata:
            logger.warning(f"⚠️ Metadata not found for document {doc_id}")
            return None

        pdf_url = metadata.get('pdf_url')
        if not pdf_url:
            logger.warning(f"⚠️ No PDF URL available for {doc_id}")
            return None

        pdf_path = self.pdf_cache_dir / f"{doc_id}.pdf"

        # 如果缓存存在，直接读取
        if pdf_path.exists():
            text = self._extract_text_from_pdf(pdf_path)
            if text:
                self._update_metadata_entry(doc_id, {
                    'is_full_text': True,
                    'last_full_text_fetch': datetime.utcnow().isoformat()
                })
                return text

        own_session = False
        client = session
        if client is None:
            client = aiohttp.ClientSession(timeout=self.http_timeout, headers=self.session_headers)
            own_session = True

        try:
            async with client.get(pdf_url) as response:
                if response.status != 200:
                    logger.error(f"❌ Failed to download PDF for {doc_id}: HTTP {response.status}")
                    return None
                pdf_path.parent.mkdir(exist_ok=True, parents=True)
                with open(pdf_path, 'wb') as f:
                    f.write(await response.read())

            text = self._extract_text_from_pdf(pdf_path)
            if not text:
                logger.warning(f"⚠️ Failed to extract text for {doc_id}")
                return None

            self._update_metadata_entry(doc_id, {
                'is_full_text': True,
                'last_full_text_fetch': datetime.utcnow().isoformat()
            })
            return text

        except Exception as exc:
            logger.error(f"❌ Error fetching full text for {doc_id}: {exc}")
            return None

        finally:
            if own_session and client:
                await client.close()
