# src/retrieval/vector_database.py
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import asdict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from loguru import logger
import uuid
from pathlib import Path
import json
import time

class QdrantVectorDB:
    """
    Qdrant向量数据库管理器
    核心功能：高效的向量检索 + 混合搜索能力
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "ai_papers",
        vector_size: int = 1024,  # BGE-M3的向量维度
        timeout: int = 60 # 新增超时参数
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        
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
                        default_segment_number=2,  # 分段数
                        max_segment_size=20000,    # 最大分段大小
                        memmap_threshold=20000,    # 内存映射阈值
                        indexing_threshold=10000   # 索引阈值
                    ),
                    # HNSW索引参数优化
                    hnsw_config=models.HnswConfig(
                        m=16,                      # 每层连接数
                        ef_construct=200,          # 构建时搜索宽度
                        full_scan_threshold=10000  # 全扫描阈值
                    )
                )
                
                logger.info("✅ 集合创建成功")
            else:
                # 获取集合信息
                info = self.client.get_collection(self.collection_name)
                logger.info(f"使用现有集合: {self.collection_name}")
                logger.info(f"当前向量数量: {info.points_count}")

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
    
    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        try:
            info = self.client.get_collection(self.collection_name)
            
            return {
                'total_points': info.points_count,
                'vector_size': info.config.params.vectors.size,
                'distance_metric': info.config.params.vectors.distance.name,
                'status': info.status.name,
                'optimizer_status': info.optimizer_status
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

class VectorDatabaseManager:
    """向量数据库管理器"""
    
    def __init__(self, config: Dict):
        self.db = QdrantVectorDB(
            host=config.get('qdrant_host', 'localhost'),
            port=config.get('qdrant_port', 6333),
            collection_name=config.get('collection_name', 'ai_papers'),
            vector_size=config.get('vector_size', 1024),
            timeout=config.get('qdrant_timeout', 120) # 从配置读取或默认120秒
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
        **kwargs
    ) -> List[Dict]:
        """执行检索"""
        return self.db.hybrid_search(
            query_vector=query_vector,
            query_text=query_text,
            top_k=top_k,
            **kwargs
        )

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

