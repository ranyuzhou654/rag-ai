# src/retrieval/vector_database.py
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import asdict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, Match, Range, GeoBoundingBox,
    SearchRequest, RecommendRequest, ScrollRequest
)
from loguru import logger
import uuid
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
import re
from rank_bm25 import BM25Okapi
from collections import defaultdict
import hashlib

class QdrantVectorDB:
    """
    Qdrant向量数据库管理器 - 企业级版本
    核心功能：
    - 高效的向量检索 + 混合搜索能力
    - 支持语义检索 + 关键词匹配 + 元数据过滤
    - 支持学术论文特定的过滤（作者、年份、期刊、分类等）
    - 支持分布式部署和水平扩展
    - 内置性能监控和统计分析
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "ai_papers",
        vector_size: int = 1024,  # BGE-M3的向量维度
        timeout: int = 60,
        enable_hybrid_search: bool = True,
        bm25_weight: float = 0.3,  # BM25权重
        vector_weight: float = 0.7  # 向量相似度权重
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.enable_hybrid_search = enable_hybrid_search
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        
        # 混合搜索组件
        self.bm25_index = None
        self.document_texts = []  # 用于BM25索引
        self.doc_id_mapping = {}  # ID映射
        
        # 性能统计
        self.search_stats = {
            'total_searches': 0,
            'semantic_searches': 0,
            'keyword_searches': 0,
            'hybrid_searches': 0,
            'filtered_searches': 0,
            'avg_search_time': 0.0,
            'cache_hits': 0
        }
        
        # 初始化客户端
        try:
            self.client = QdrantClient(host=host, port=port, timeout=timeout) # 应用超时参数
            logger.info(f"成功连接到Qdrant: {host}:{port} (超时设置为 {timeout}s)")
            
            # 检查集合是否存在，不存在则创建
            self._ensure_collection_exists()
            
        except Exception as e:
            logger.error(f"连接Qdrant失败: {e}")
            raise
    
    def _ensure_collection_exists(self):
        """确保集合存在，不存在则创建"""
        try:
            # 检查集合是否存在
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"创建新集合: {self.collection_name}")
                
                # 创建集合配置
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE  # 余弦相似度
                    ),
                    # 优化配置
                    optimizers_config=models.OptimizersConfig(
                        default_segment_number=4,  # 增加分段数以支持并发
                        max_segment_size=50000,    # 增大分段以提升性能
                        memmap_threshold=50000,    # 更高的内存映射阈值
                        indexing_threshold=20000,  # 更高的索引阈值
                        flush_interval_sec=30,     # 刷新间隔
                        max_optimization_threads=2 # 优化线程数
                    ),
                    # HNSW索引参数优化（为学术论文检索调优）
                    hnsw_config=models.HnswConfig(
                        m=32,                      # 增加连接数提升召回率
                        ef_construct=400,          # 更高的构建搜索宽度
                        full_scan_threshold=50000, # 更高的全扫描阈值
                        max_indexing_threads=4,    # 索引线程数
                        on_disk=False              # 内存索引（更快）
                    )
                )
                
                # 创建索引以支持高效过滤
                self._create_payload_indexes()
                
                logger.info("✅ 集合创建成功")
            else:
                # 获取集合信息
                info = self.client.get_collection(self.collection_name)
                logger.info(f"使用现有集合: {self.collection_name}")
                logger.info(f"当前向量数量: {info.points_count}")
                
                # 确保索引存在
                self._create_payload_indexes()
                
                # 如果启用混合搜索，加载现有文档用于BM25
                if self.enable_hybrid_search:
                    self._load_existing_documents_for_bm25()

        except Exception as e:
            logger.error(f"确保集合存在失败: {e}")

    def add_chunks(self, chunks: List[Dict], batch_size: int = 100) -> bool:
        """
        批量添加文本块到向量数据库
        
        Args:
            chunks: 文本块列表 (包含embedding)
            batch_size: 批次大小
            
        Returns:
            bool: 是否成功
        """
        logger.info(f"开始添加 {len(chunks)} 个向量到数据库...")
        
        total_added = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            points = []
            
            for chunk in batch:
                # 检查是否有embedding
                if 'embedding' not in chunk or chunk['embedding'] is None:
                    logger.warning(f"跳过无embedding的chunk: {chunk.get('chunk_id', 'unknown')}")
                    continue
                
                # 准备向量点
                try:
                    point = PointStruct(
                        id=str(uuid.uuid4()),  # 生成唯一ID
                        vector=chunk['embedding'],
                        payload={
                            'chunk_id': chunk['chunk_id'],
                            'source_id': chunk['source_id'],
                            'content': chunk['content'],
                            'semantic_type': chunk.get('semantic_type', 'content'),
                            'metadata': chunk.get('metadata', {}),
                            # 添加全文检索字段
                            'text_tokens': self._tokenize_for_search(chunk['content'])
                        }
                    )
                    points.append(point)
                    
                except Exception as e:
                    logger.error(f"准备向量点失败: {e}")
                    continue
            
            if points:
                try:
                    # 批量上传
                    operation_info = self.client.upsert(
                        collection_name=self.collection_name,
                        wait=True,  # 等待操作完成
                        points=points
                    )
                    
                    batch_count = len(points)
                    total_added += batch_count
                    logger.info(f"批次 {i//batch_size + 1}: 成功添加 {batch_count} 个向量")
                    
                except Exception as e:
                    logger.error(f"批次上传失败: {e}")
                    continue
        
        # 统计信息
        collection_info = self.client.get_collection(self.collection_name)
        success_rate = total_added / len(chunks) * 100
        
        logger.info(f"✅ 向量添加完成!")
        logger.info(f"   成功添加: {total_added}/{len(chunks)} ({success_rate:.1f}%)")
        logger.info(f"   数据库总向量数: {collection_info.points_count}")
        
        # 更新BM25索引
        if success_rate > 90 and self.enable_hybrid_search:
            self.update_bm25_index(chunks)
        
        return success_rate > 90  # 成功率大于90%认为成功
    
    def _tokenize_for_search(self, text: str) -> List[str]:
        """为全文检索准备token"""
        import re

        # 英文和数字分词
        tokens = re.findall(r'\b[a-z0-9]{2,}\b', text.lower())

        # 简单的中文分词，通过生成1-3字的滑动窗口
        chinese_segments = re.findall(r'[\u4e00-\u9fa5]+', text)
        for segment in chinese_segments:
            tokens.extend(self._generate_chinese_tokens(segment))

        return list({token for token in tokens if token})  # 去重

    def _generate_chinese_tokens(self, segment: str) -> List[str]:
        if not segment:
            return []

        tokens = set()
        length = len(segment)

        for size in (1, 2, 3):
            for i in range(length - size + 1):
                tokens.add(segment[i:i + size])

        return list(tokens)

    def update_bm25_index(self, chunks: List[Dict]) -> None:
        """根据新增或更新的chunk刷新BM25索引"""
        if not self.enable_hybrid_search:
            return

        updated = False
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id')
            content = chunk.get('content')
            if not chunk_id or not content:
                continue

            if chunk_id in self.doc_id_mapping:
                index = self.doc_id_mapping[chunk_id]
                self.document_texts[index] = content
            else:
                self.doc_id_mapping[chunk_id] = len(self.document_texts)
                self.document_texts.append(content)
            updated = True

        if not updated:
            logger.debug("BM25 index update skipped: no valid chunks provided")
            return

        tokenized_corpus = []
        for text in self.document_texts:
            tokens = self._tokenize_for_search(text)
            if not tokens:
                # Fallback: basic whitespace split to avoid empty document
                tokens = text.split()
            tokenized_corpus.append(tokens)

        if not any(tokenized_corpus):
            logger.warning("BM25 index update skipped: corpus produced no tokens")
            return

        self.bm25_index = BM25Okapi(tokenized_corpus)
        logger.success(f"BM25 index updated with {len(self.document_texts)} documents")
    
    def hybrid_search(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        filter_condition: Optional[Dict] = None
    ) -> List[Dict]:
        """
        混合检索：向量相似度 + 文本匹配
        这是工业级RAG系统的标准做法
        
        Args:
            query_vector: 查询向量
            query_text: 查询文本
            top_k: 返回结果数
            vector_weight: 向量检索权重
            text_weight: 文本检索权重
            filter_condition: 过滤条件
            
        Returns:
            检索结果列表
        """
        try:
            start_time = time.time()
            
            # 1. 向量相似度检索
            vector_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k * 2,  # 多检索一些候选
                query_filter=self._build_filter(filter_condition) if filter_condition else None,
                with_payload=True,
                with_vectors=False  # 不返回向量以节省带宽
            )
            
            # 2. 文本关键词匹配得分
            query_tokens = set(self._tokenize_for_search(query_text))
            
            results = []
            for hit in vector_results:
                payload = hit.payload
                
                # 计算文本匹配得分
                content_tokens = set(payload.get('text_tokens', []))
                text_score = len(query_tokens.intersection(content_tokens)) / max(len(query_tokens), 1)
                
                # 混合得分
                vector_score = hit.score
                hybrid_score = vector_weight * vector_score + text_weight * text_score
                
                result = {
                    'chunk_id': payload['chunk_id'],
                    'source_id': payload['source_id'],
                    'content': payload['content'],
                    'semantic_type': payload.get('semantic_type', 'content'),
                    'metadata': payload.get('metadata', {}),
                    'scores': {
                        'vector_score': float(vector_score),
                        'text_score': text_score,
                        'hybrid_score': hybrid_score
                    }
                }
                results.append(result)
            
            # 3. 按混合得分重新排序
            results.sort(key=lambda x: x['scores']['hybrid_score'], reverse=True)
            results = results[:top_k]
            
            search_time = time.time() - start_time
            logger.info(f"混合检索完成: {len(results)} 个结果, 耗时 {search_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return []
    
    def _build_filter(self, filter_condition: Dict) -> models.Filter:
        """构建查询过滤器"""
        conditions = []
        
        if 'semantic_type' in filter_condition:
            conditions.append(
                models.FieldCondition(
                    key="semantic_type",
                    match=models.MatchValue(value=filter_condition['semantic_type'])
                )
            )
        
        if 'source_type' in filter_condition:
            conditions.append(
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=filter_condition['source_type'])
                )
            )
        
        return models.Filter(must=conditions) if conditions else None
    
    def advanced_academic_search(
        self, 
        query_vector: np.ndarray, 
        query_text: str,
        authors: Optional[List[str]] = None,
        year_range: Optional[Tuple[int, int]] = None,
        sources: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        has_full_text: Optional[bool] = None,
        language: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        高级学术论文搜索，支持多维度过滤
        
        Args:
            query_vector: 查询向量
            query_text: 查询文本
            authors: 作者名单
            year_range: 年份范围 (start_year, end_year)
            sources: 数据源列表 ['arxiv', 'huggingface', 'blogs']
            categories: ArXiv分类列表
            has_full_text: 是否必须有全文
            language: 语言过滤
            top_k: 返回结果数量
        """
        start_time = time.time()
        self.search_stats['total_searches'] += 1
        self.search_stats['filtered_searches'] += 1
        
        try:
            # 构建过滤条件
            filter_conditions = []
            
            # 作者过滤
            if authors:
                author_conditions = []
                for author in authors:
                    author_conditions.append(
                        models.FieldCondition(
                            key="authors",
                            match=models.MatchValue(value=author)
                        )
                    )
                filter_conditions.append(models.Filter(should=author_conditions))
            
            # 年份过滤
            if year_range:
                start_year, end_year = year_range
                filter_conditions.append(
                    models.FieldCondition(
                        key="year",
                        range=models.Range(
                            gte=start_year,
                            lte=end_year
                        )
                    )
                )
            
            # 数据源过滤
            if sources:
                source_conditions = []
                for source in sources:
                    source_conditions.append(
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=source)
                        )
                    )
                filter_conditions.append(models.Filter(should=source_conditions))
            
            # 分类过滤
            if categories:
                category_conditions = []
                for category in categories:
                    category_conditions.append(
                        models.FieldCondition(
                            key="categories",
                            match=models.MatchValue(value=category)
                        )
                    )
                filter_conditions.append(models.Filter(should=category_conditions))
            
            # 全文过滤
            if has_full_text is not None:
                filter_conditions.append(
                    models.FieldCondition(
                        key="has_full_text",
                        match=models.MatchValue(value=has_full_text)
                    )
                )
            
            # 语言过滤
            if language:
                filter_conditions.append(
                    models.FieldCondition(
                        key="language",
                        match=models.MatchValue(value=language)
                    )
                )
            
            # 组合过滤条件
            search_filter = models.Filter(must=filter_conditions) if filter_conditions else None
            
            # 执行语义搜索
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=top_k * 2,  # 获取更多结果用于混合重排
                with_payload=True,
                with_vectors=False
            )
            
            # 如果启用混合搜索，结合BM25结果
            if self.enable_hybrid_search and self.bm25_index and query_text:
                # BM25搜索
                query_tokens = query_text.lower().split()
                bm25_scores = self.bm25_index.get_scores(query_tokens)
                
                # 重新计算混合得分
                results = []
                for point in search_result:
                    vector_score = point.score
                    
                    # 查找BM25得分
                    bm25_score = 0.0
                    doc_id = str(point.id)
                    for bm25_idx, mapped_id in self.doc_id_mapping.items():
                        if mapped_id == doc_id and bm25_idx < len(bm25_scores):
                            bm25_score = bm25_scores[bm25_idx]
                            break
                    
                    # 归一化BM25得分
                    normalized_bm25 = min(bm25_score / (max(bm25_scores) + 1e-8), 1.0)
                    
                    # 计算混合得分
                    hybrid_score = (
                        self.vector_weight * vector_score + 
                        self.bm25_weight * normalized_bm25
                    )
                    
                    result = {
                        'id': point.id,
                        'content': point.payload.get('content', ''),
                        'metadata': point.payload.get('metadata', {}),
                        'scores': {
                            'vector_score': float(vector_score),
                            'bm25_score': float(normalized_bm25),
                            'hybrid_score': float(hybrid_score)
                        }
                    }
                    results.append(result)
                
                # 按混合得分排序
                results.sort(key=lambda x: x['scores']['hybrid_score'], reverse=True)
                results = results[:top_k]
            else:
                # 纯语义搜索结果
                results = []
                for point in search_result[:top_k]:
                    result = {
                        'id': point.id,
                        'content': point.payload.get('content', ''),
                        'metadata': point.payload.get('metadata', {}),
                        'scores': {
                            'vector_score': float(point.score),
                            'hybrid_score': float(point.score)
                        }
                    }
                    results.append(result)
            
            search_time = time.time() - start_time
            self.search_stats['avg_search_time'] = (
                (self.search_stats['avg_search_time'] * (self.search_stats['total_searches'] - 1) + search_time) /
                self.search_stats['total_searches']
            )
            
            logger.info(
                f"✅ 高级学术搜索完成: {len(results)}个结果, "
                f"过滤条件: {len(filter_conditions)}, 耗时: {search_time:.3f}s"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 高级学术搜索失败: {e}")
            return []
    
    def get_trending_papers(self, days: int = 7, limit: int = 20) -> List[Dict]:
        """获取近期热门论文"""
        try:
            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # 构建日期过滤器
            date_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="published_date",
                        range=models.Range(
                            gte=start_date.timestamp(),
                            lte=end_date.timestamp()
                        )
                    )
                ]
            )
            
            # 滚动查询获取近期论文
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=date_filter,
                limit=limit * 3,  # 获取更多结果用于排序
                with_payload=True,
                with_vectors=False
            )
            
            papers = scroll_result[0]
            
            # 按某种热度指标排序（这里简化为按发表时间）
            papers.sort(
                key=lambda x: x.payload.get('published_date', 0), 
                reverse=True
            )
            
            results = []
            for paper in papers[:limit]:
                result = {
                    'id': paper.id,
                    'title': paper.payload.get('title', ''),
                    'authors': paper.payload.get('authors', []),
                    'abstract': paper.payload.get('abstract', ''),
                    'published_date': paper.payload.get('published_date'),
                    'source': paper.payload.get('source', ''),
                    'url': paper.payload.get('url', ''),
                    'categories': paper.payload.get('categories', [])
                }
                results.append(result)
            
            logger.info(f"✅ 获取近{days}天热门论文: {len(results)}篇")
            return results
            
        except Exception as e:
            logger.error(f"❌ 获取热门论文失败: {e}")
            return []
    
    def get_papers_by_author(self, author_name: str, limit: int = 50) -> List[Dict]:
        """按作者获取论文"""
        try:
            author_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="authors",
                        match=models.MatchValue(value=author_name)
                    )
                ]
            )
            
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=author_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            papers = scroll_result[0]
            
            results = []
            for paper in papers:
                result = {
                    'id': paper.id,
                    'title': paper.payload.get('title', ''),
                    'authors': paper.payload.get('authors', []),
                    'published_date': paper.payload.get('published_date'),
                    'source': paper.payload.get('source', ''),
                    'categories': paper.payload.get('categories', [])
                }
                results.append(result)
            
            logger.info(f"✅ 获取作者 '{author_name}' 的论文: {len(results)}篇")
            return results
            
        except Exception as e:
            logger.error(f"❌ 获取作者论文失败: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        try:
            info = self.client.get_collection(self.collection_name)
            
            # 获取数据源统计
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            papers = scroll_result[0]
            source_counts = defaultdict(int)
            year_counts = defaultdict(int)
            category_counts = defaultdict(int)
            
            for paper in papers:
                payload = paper.payload or {}
                
                # 数据源统计
                source = payload.get('source', 'unknown')
                source_counts[source] += 1
                
                # 年份统计
                year = payload.get('year', 'unknown')
                year_counts[year] += 1
                
                # 分类统计
                categories = payload.get('categories', [])
                for category in categories:
                    category_counts[category] += 1
            
            return {
                'total_points': info.points_count,
                'vector_size': info.config.params.vectors.size,
                'distance_metric': info.config.params.vectors.distance.name,
                'status': info.status.name,
                'optimizer_status': getattr(info, 'optimizer_status', 'unknown'),
                'search_stats': self.search_stats,
                'data_distribution': {
                    'sources': dict(source_counts),
                    'years': dict(sorted(year_counts.items())),
                    'top_categories': dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10])
                },
                'hybrid_search_enabled': self.enable_hybrid_search,
                'bm25_documents': len(self.document_texts) if self.bm25_index else 0
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

class VectorDatabaseManager:
    """向量数据库管理器 - 企业级版本"""
    
    def __init__(self, config: Dict):
        self.db = QdrantVectorDB(
            host=config.get('qdrant_host', 'localhost'),
            port=config.get('qdrant_port', 6333),
            collection_name=config.get('collection_name', 'ai_papers'),
            vector_size=config.get('vector_size', 1024),
            timeout=config.get('qdrant_timeout', 120),
            enable_hybrid_search=config.get('enable_hybrid_search', True),
            bm25_weight=config.get('bm25_weight', 0.3),
            vector_weight=config.get('vector_weight', 0.7)
        )
    
    def build_knowledge_base(self, processed_chunks_path: Path, chunks: Optional[List[Dict]] = None) -> bool:
        """
        从处理好的文本块构建知识库

        Args:
            processed_chunks_path: 处理后的文本块JSON文件路径
            
        Returns:
            bool: 是否构建成功
        """
        logger.info(f"开始构建知识库: {processed_chunks_path}")
        
        try:
            # 加载处理后的数据
            if chunks is None:
                with open(processed_chunks_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
            
            logger.info(f"加载到 {len(chunks)} 个文本块")
            
            # 过滤有效的chunks（必须有embedding）
            valid_chunks = [c for c in chunks if c.get('embedding') is not None]
            logger.info(f"有效文本块: {len(valid_chunks)}/{len(chunks)}")
            
            if not valid_chunks:
                logger.error("没有有效的文本块（缺少embedding）")
                return False
            
            # 添加到向量数据库
            success = self.db.add_chunks(valid_chunks)
            
            if success:
                # 打印统计信息
                stats = self.db.get_collection_stats()
                logger.info("知识库构建成功！")
                logger.info(f"数据库统计: {stats}")
                
                return True
            else:
                logger.error("知识库构建失败")
                return False
                
        except Exception as e:
            logger.error(f"构建知识库时出错: {e}")
            return False
    
    def search(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 5,
        search_type: str = "hybrid",
        **kwargs
    ) -> List[Dict]:
        """执行检索"""
        if search_type == "academic":
            return self.db.advanced_academic_search(
                query_vector=query_vector,
                query_text=query_text,
                top_k=top_k,
                **kwargs
            )
        else:
            return self.db.hybrid_search(
                query_vector=query_vector,
                query_text=query_text,
                top_k=top_k,
                **kwargs
            )
    
    def get_trending_papers(self, days: int = 7, limit: int = 20) -> List[Dict]:
        """获取近期热门论文"""
        return self.db.get_trending_papers(days=days, limit=limit)
    
    def get_papers_by_author(self, author_name: str, limit: int = 50) -> List[Dict]:
        """按作者获取论文"""
        return self.db.get_papers_by_author(author_name=author_name, limit=limit)
    
    def get_comprehensive_stats(self) -> Dict:
        """获取综合统计信息"""
        return self.db.get_collection_stats()

# 使用示例和测试
async def main():
    from pathlib import Path
    import json
    from sentence_transformers import SentenceTransformer
    
    # 配置
    config = {
        'qdrant_host': 'localhost',
        'qdrant_port': 6333,
        'collection_name': 'ai_papers',
        'vector_size': 1024,  # BGE-M3维度
        'qdrant_timeout': 180 # 测试时使用更长的超时
    }
    
    # 初始化管理器
    db_manager = VectorDatabaseManager(config)
    
    # 构建知识库
    processed_data_path = Path("data/processed/processed_chunks.json")
    
    if processed_data_path.exists():
        logger.info("开始构建向量知识库...")
        success = db_manager.build_knowledge_base(processed_data_path)
        
        if success:
            print("✅ 知识库构建成功！")
            
            # 测试检索
            print("\n🔍 测试检索功能...")
            
            # 加载embedding模型进行测试
            embedder = SentenceTransformer('BAAI/bge-m3')
            
            test_queries = [
                "What are the latest developments in large language models?",
                "如何改进Transformer模型的效率？",
                "computer vision deep learning advances"
            ]
            
            for query in test_queries:
                print(f"\n查询: {query}")
                query_vector = embedder.encode([query], convert_to_numpy=True)[0]
                
                results = db_manager.search(
                    query_vector=query_vector,
                    query_text=query,
                    top_k=3
                )
                
                for i, result in enumerate(results, 1):
                    print(f"  {i}. [{result['semantic_type']}] {result['content'][:100]}...")
                    print(f"     得分: {result['scores']['hybrid_score']:.3f}")
        
        else:
            print("❌ 知识库构建失败")
    else:
        print(f"❌ 找不到处理后的数据文件: {processed_data_path}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
