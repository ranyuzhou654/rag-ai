# src/data_ingestion/langchain_document_pipeline.py
"""
基于LangChain的现代化文档处理管道
替换原有的multi_source_collector，提供更强大和标准化的文档处理能力
"""

from typing import List, Dict, Optional, Any, AsyncIterator, Union
from dataclasses import dataclass
from pathlib import Path
import asyncio
import hashlib
from datetime import datetime
from loguru import logger

# LangChain核心组件
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
    RSSFeedLoader,
    DirectoryLoader,
    TextLoader,
    CSVLoader,
    JSONLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter
)

# 本地组件
from src.processing.weighted_embedding_fusion import MultiRepresentation, WeightedEmbeddingFusion
from src.processing.multi_representation_indexer import MultiRepresentationIndexer


@dataclass
class DocumentSource:
    """文档源配置"""
    source_type: str  # 'web', 'rss', 'pdf', 'directory', 'file'
    source_path: str  # URL或文件路径
    metadata: Dict[str, Any]
    enabled: bool = True


@dataclass
class ProcessingResult:
    """处理结果"""
    documents: List[Document]
    multi_representations: List[MultiRepresentation]
    processing_stats: Dict[str, Any]
    errors: List[str]


class DocumentLoader:
    """统一文档加载器"""
    
    def __init__(self):
        self.supported_loaders = {
            'web': self._load_web,
            'rss': self._load_rss,
            'pdf': self._load_pdf,
            'directory': self._load_directory,
            'text': self._load_text,
            'csv': self._load_csv,
            'json': self._load_json
        }
        logger.info("Document loader initialized with support for: " + 
                   ", ".join(self.supported_loaders.keys()))
    
    async def load_from_source(self, source: DocumentSource) -> List[Document]:
        """从源加载文档"""
        
        if not source.enabled:
            logger.info(f"Skipping disabled source: {source.source_path}")
            return []
        
        if source.source_type not in self.supported_loaders:
            logger.error(f"Unsupported source type: {source.source_type}")
            return []
        
        try:
            logger.info(f"Loading documents from {source.source_type}: {source.source_path}")
            
            # 调用对应的加载器
            loader_func = self.supported_loaders[source.source_type]
            documents = await loader_func(source)
            
            # 添加源元数据
            for doc in documents:
                doc.metadata.update({
                    'source_type': source.source_type,
                    'source_path': source.source_path,
                    'loaded_at': datetime.now().isoformat(),
                    **source.metadata
                })
            
            logger.success(f"Loaded {len(documents)} documents from {source.source_type}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load from {source.source_path}: {e}")
            return []
    
    async def _load_web(self, source: DocumentSource) -> List[Document]:
        """加载网页文档"""
        urls = [source.source_path] if isinstance(source.source_path, str) else source.source_path
        
        loader = WebBaseLoader(
            web_paths=urls,
            header_template={'User-Agent': 'RAG-System/1.0'}
        )
        
        # WebBaseLoader通常是同步的，包装为异步
        def _sync_load():
            return loader.load()
        
        # 在线程池中执行同步加载
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(None, _sync_load)
        
        return documents
    
    async def _load_rss(self, source: DocumentSource) -> List[Document]:
        """加载RSS订阅"""
        loader = RSSFeedLoader(urls=[source.source_path])
        
        def _sync_load():
            return loader.load()
        
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(None, _sync_load)
        
        return documents
    
    async def _load_pdf(self, source: DocumentSource) -> List[Document]:
        """加载PDF文档"""
        file_path = Path(source.source_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # 尝试使用UnstructuredPDFLoader，回退到PyPDFLoader
        try:
            loader = UnstructuredPDFLoader(str(file_path))
        except ImportError:
            logger.warning("Unstructured not available, using PyPDFLoader")
            loader = PyPDFLoader(str(file_path))
        
        def _sync_load():
            return loader.load()
        
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(None, _sync_load)
        
        return documents
    
    async def _load_directory(self, source: DocumentSource) -> List[Document]:
        """加载目录中的文档"""
        dir_path = Path(source.source_path)
        
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Directory not found: {dir_path}")
        
        # 支持的文件类型
        glob_patterns = source.metadata.get('glob_patterns', ['**/*.txt', '**/*.md', '**/*.pdf'])
        
        documents = []
        for pattern in glob_patterns:
            loader = DirectoryLoader(
                str(dir_path),
                glob=pattern,
                loader_cls=self._get_loader_for_extension
            )
            
            def _sync_load():
                return loader.load()
            
            loop = asyncio.get_event_loop()
            batch_docs = await loop.run_in_executor(None, _sync_load)
            documents.extend(batch_docs)
        
        return documents
    
    async def _load_text(self, source: DocumentSource) -> List[Document]:
        """加载文本文件"""
        loader = TextLoader(source.source_path, encoding='utf-8')
        
        def _sync_load():
            return loader.load()
        
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(None, _sync_load)
        
        return documents
    
    async def _load_csv(self, source: DocumentSource) -> List[Document]:
        """加载CSV文件"""
        loader = CSVLoader(
            file_path=source.source_path,
            csv_args=source.metadata.get('csv_args', {})
        )
        
        def _sync_load():
            return loader.load()
        
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(None, _sync_load)
        
        return documents
    
    async def _load_json(self, source: DocumentSource) -> List[Document]:
        """加载JSON文件"""
        jq_schema = source.metadata.get('jq_schema', '.')
        
        loader = JSONLoader(
            file_path=source.source_path,
            jq_schema=jq_schema
        )
        
        def _sync_load():
            return loader.load()
        
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(None, _sync_load)
        
        return documents
    
    def _get_loader_for_extension(self, file_path: str):
        """根据文件扩展名选择加载器"""
        extension = Path(file_path).suffix.lower()
        
        if extension == '.pdf':
            return PyPDFLoader
        elif extension in ['.txt', '.md']:
            return TextLoader
        elif extension == '.csv':
            return CSVLoader
        elif extension == '.json':
            return JSONLoader
        else:
            return TextLoader  # 默认使用文本加载器


class DocumentSplitter:
    """智能文档切分器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 默认切分器配置
        self.chunk_size = config.get('chunk_size', 512)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        self.separators = config.get('separators', ['\n\n', '\n', '。', '！', '？', ';', ',', ' '])
        
        # 初始化切分器
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len
        )
        
        # Token-based切分器（用于长文档）
        self.token_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        logger.info(f"Document splitter initialized: chunk_size={self.chunk_size}, "
                   f"overlap={self.chunk_overlap}")
    
    async def split_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档"""
        
        all_chunks = []
        
        for doc in documents:
            try:
                # 根据文档长度选择切分策略
                content_length = len(doc.page_content)
                
                if content_length <= self.chunk_size:
                    # 短文档直接使用
                    chunks = [doc]
                elif content_length <= 10000:
                    # 中等长度文档使用递归切分
                    chunks = await self._recursive_split(doc)
                else:
                    # 长文档使用token切分
                    chunks = await self._token_split(doc)
                
                # 为每个块添加块ID和位置信息
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        'chunk_id': self._generate_chunk_id(chunk),
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk.page_content),
                        'split_method': self._get_split_method(content_length)
                    })
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Failed to split document: {e}")
                # 如果切分失败，保留原文档
                all_chunks.append(doc)
        
        logger.info(f"Document splitting completed: {len(documents)} docs -> {len(all_chunks)} chunks")
        return all_chunks
    
    async def _recursive_split(self, doc: Document) -> List[Document]:
        """递归切分"""
        def _sync_split():
            return self.recursive_splitter.split_documents([doc])
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_split)
    
    async def _token_split(self, doc: Document) -> List[Document]:
        """Token切分"""
        def _sync_split():
            return self.token_splitter.split_documents([doc])
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_split)
    
    def _generate_chunk_id(self, chunk: Document) -> str:
        """生成块ID"""
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
        return f"chunk_{content_hash[:12]}"
    
    def _get_split_method(self, content_length: int) -> str:
        """获取切分方法名称"""
        if content_length <= self.chunk_size:
            return 'none'
        elif content_length <= 10000:
            return 'recursive'
        else:
            return 'token'


class LangChainDocumentPipeline:
    """基于LangChain的文档处理管道"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化组件
        self.document_loader = DocumentLoader()
        self.document_splitter = DocumentSplitter(config)
        
        # 多表示索引器（如果启用）
        self.enable_multi_representation = config.get('enable_multi_representation', True)
        if self.enable_multi_representation:
            self.multi_rep_indexer = MultiRepresentationIndexer(config)
        
        # 统计信息
        self.stats = {
            'total_sources': 0,
            'successful_sources': 0,
            'total_documents': 0,
            'total_chunks': 0,
            'processing_errors': []
        }
        
        logger.info("LangChain Document Pipeline initialized")
    
    async def process_sources(self, sources: List[DocumentSource]) -> ProcessingResult:
        """处理多个文档源"""
        
        logger.info(f"Starting processing of {len(sources)} document sources")
        
        all_documents = []
        all_multi_representations = []
        processing_errors = []
        
        # 更新统计
        self.stats['total_sources'] = len(sources)
        
        # 并行处理多个源（限制并发数）
        semaphore = asyncio.Semaphore(5)  # 最多同时处理5个源
        
        async def process_single_source(source):
            async with semaphore:
                return await self._process_single_source(source)
        
        # 执行并行处理
        tasks = [process_single_source(source) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集结果
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Source {sources[i].source_path} failed: {result}"
                processing_errors.append(error_msg)
                logger.error(error_msg)
            else:
                documents, multi_reps = result
                all_documents.extend(documents)
                all_multi_representations.extend(multi_reps)
                self.stats['successful_sources'] += 1
        
        # 更新统计
        self.stats['total_documents'] = len(all_documents)
        self.stats['total_chunks'] = len([doc for doc in all_documents if 'chunk_id' in doc.metadata])
        self.stats['processing_errors'] = processing_errors
        
        logger.info(f"Processing completed: {self.stats['successful_sources']}/{self.stats['total_sources']} sources, "
                   f"{self.stats['total_documents']} total documents")
        
        return ProcessingResult(
            documents=all_documents,
            multi_representations=all_multi_representations,
            processing_stats=self.stats.copy(),
            errors=processing_errors
        )
    
    async def _process_single_source(self, source: DocumentSource) -> tuple[List[Document], List[MultiRepresentation]]:
        """处理单个文档源"""
        
        try:
            # 1. 加载文档
            documents = await self.document_loader.load_from_source(source)
            
            if not documents:
                return [], []
            
            # 2. 切分文档
            chunked_documents = await self.document_splitter.split_documents(documents)
            
            # 3. 生成多表示（如果启用）
            multi_representations = []
            if self.enable_multi_representation:
                multi_representations = await self._generate_multi_representations(chunked_documents)
            
            return chunked_documents, multi_representations
            
        except Exception as e:
            logger.error(f"Error processing source {source.source_path}: {e}")
            raise
    
    async def _generate_multi_representations(self, documents: List[Document]) -> List[MultiRepresentation]:
        """生成多表示"""
        
        try:
            multi_representations = []
            
            for doc in documents:
                # 生成摘要和假设问题
                summary, questions = await self.multi_rep_indexer.generate_representations(
                    doc.page_content
                )
                
                multi_rep = MultiRepresentation(
                    original_content=doc.page_content,
                    summary=summary,
                    hypothetical_questions=questions,
                    metadata=doc.metadata
                )
                
                multi_representations.append(multi_rep)
            
            logger.debug(f"Generated multi-representations for {len(documents)} documents")
            return multi_representations
            
        except Exception as e:
            logger.warning(f"Multi-representation generation failed: {e}")
            return []
    
    async def process_single_file(self, file_path: str, source_type: str = None) -> ProcessingResult:
        """处理单个文件（便捷方法）"""
        
        if source_type is None:
            # 根据文件扩展名自动判断类型
            extension = Path(file_path).suffix.lower()
            type_mapping = {
                '.pdf': 'pdf',
                '.txt': 'text',
                '.md': 'text',
                '.csv': 'csv',
                '.json': 'json'
            }
            source_type = type_mapping.get(extension, 'text')
        
        source = DocumentSource(
            source_type=source_type,
            source_path=file_path,
            metadata={'single_file': True}
        )
        
        return await self.process_sources([source])
    
    async def process_directory(self, directory_path: str, glob_patterns: List[str] = None) -> ProcessingResult:
        """处理目录（便捷方法）"""
        
        if glob_patterns is None:
            glob_patterns = ['**/*.txt', '**/*.md', '**/*.pdf']
        
        source = DocumentSource(
            source_type='directory',
            source_path=directory_path,
            metadata={'glob_patterns': glob_patterns}
        )
        
        return await self.process_sources([source])
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_sources': 0,
            'successful_sources': 0,
            'total_documents': 0,
            'total_chunks': 0,
            'processing_errors': []
        }