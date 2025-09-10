# src/knowledge_graph/kg_retriever.py
from typing import List, Dict, Optional, Tuple, Any
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .knowledge_extractor import KnowledgeGraphIndexer

@dataclass
class KGRetrievalResult:
    """知识图谱检索结果"""
    entity: str
    entity_type: str
    related_info: str
    confidence: float
    source_type: str  # 'entity', 'relation', 'path'
    metadata: Dict[str, Any]

class KnowledgeGraphRetriever:
    """知识图谱检索器"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 初始化知识图谱索引器
        self.kg_indexer = KnowledgeGraphIndexer(config)
        
        # 初始化embedding模型用于语义相似度计算
        self.embedder = SentenceTransformer(
            config.get('embedding_model', 'BAAI/bge-m3'),
            device=config.get('device', 'auto')
        )
        
        logger.info("Knowledge Graph Retriever initialized")
    
    def retrieve_kg_context(
        self,
        query: str,
        top_k: int = 10,
        include_relations: bool = True,
        include_paths: bool = True,
        max_hops: int = 2
    ) -> List[KGRetrievalResult]:
        """从知识图谱检索相关上下文"""
        
        try:
            # 确保图已加载到内存
            if self.kg_indexer.kg_db.graph.number_of_nodes() == 0:
                self.kg_indexer.kg_db.load_graph_from_db()
            
            # 查询知识图谱
            kg_results = self.kg_indexer.query_knowledge_graph(
                query=query,
                max_entities=top_k,
                max_hops=max_hops
            )
            
            if not kg_results:
                return []
            
            # 转换为检索结果格式
            retrieval_results = []
            
            for kg_result in kg_results:
                if kg_result['type'] == 'entity':
                    # 实体信息
                    entity_info = kg_result['info']
                    info_text = f"实体: {entity_info['name']}\n"
                    info_text += f"类型: {entity_info['entity_type']}\n"
                    if entity_info['properties']:
                        info_text += f"属性: {json.dumps(entity_info['properties'], ensure_ascii=False)}\n"
                    
                    result = KGRetrievalResult(
                        entity=entity_info['name'],
                        entity_type=entity_info['entity_type'],
                        related_info=info_text,
                        confidence=entity_info['confidence'],
                        source_type='entity',
                        metadata={
                            'aliases': entity_info['aliases'],
                            'source_chunks': entity_info['source_chunks']
                        }
                    )
                    retrieval_results.append(result)
                
                elif kg_result['type'] == 'relation' and include_relations:
                    # 关系信息
                    info_text = f"关系: {kg_result['source_entity']} --{kg_result['relation']}--> {kg_result['target_entity']}\n"
                    info_text += f"跳数: {kg_result['hops']}\n"
                    if include_paths:
                        info_text += f"路径: {' -> '.join(kg_result['path'])}\n"
                    
                    result = KGRetrievalResult(
                        entity=kg_result['target_entity'],
                        entity_type='RELATION',
                        related_info=info_text,
                        confidence=kg_result['confidence'],
                        source_type='relation',
                        metadata={
                            'relation_type': kg_result['relation'],
                            'source_entity': kg_result['source_entity'],
                            'path': kg_result['path'],
                            'hops': kg_result['hops']
                        }
                    )
                    retrieval_results.append(result)
            
            # 按置信度排序
            retrieval_results.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.info(f"KG retrieval found {len(retrieval_results)} relevant items")
            return retrieval_results[:top_k]
            
        except Exception as e:
            logger.error(f"KG retrieval failed: {e}")
            return []
    
    def enhance_chunks_with_kg(
        self,
        query: str,
        chunks: List[Dict],
        kg_weight: float = 0.3
    ) -> List[Dict]:
        """用知识图谱信息增强文档块"""
        
        # 获取KG上下文
        kg_results = self.retrieve_kg_context(query, top_k=20)
        
        if not kg_results:
            return chunks
        
        # 为每个chunk添加相关的KG信息
        enhanced_chunks = []
        
        for chunk in chunks:
            enhanced_chunk = chunk.copy()
            
            # 查找与chunk内容相关的KG信息
            relevant_kg_info = []
            chunk_content = chunk['content'].lower()
            
            for kg_result in kg_results:
                # 检查实体是否在chunk中提到
                if (kg_result.entity.lower() in chunk_content or
                    any(alias.lower() in chunk_content 
                        for alias in kg_result.metadata.get('aliases', []))):
                    relevant_kg_info.append(kg_result)
            
            if relevant_kg_info:
                # 添加KG增强信息
                kg_context = "\n".join([
                    f"[KG] {kg.related_info}" 
                    for kg in relevant_kg_info[:3]  # 限制数量
                ])
                
                enhanced_chunk['kg_enhanced_content'] = f"{chunk['content']}\n\n{kg_context}"
                enhanced_chunk['has_kg_enhancement'] = True
                enhanced_chunk['kg_entities'] = [kg.entity for kg in relevant_kg_info]
                
                # 调整分数（如果存在）
                if 'scores' in enhanced_chunk:
                    original_score = enhanced_chunk['scores'].get('hybrid_score', 0)
                    kg_boost = len(relevant_kg_info) * 0.1
                    enhanced_chunk['scores']['kg_enhanced_score'] = original_score + kg_boost
            else:
                enhanced_chunk['has_kg_enhancement'] = False
            
            enhanced_chunks.append(enhanced_chunk)
        
        logger.info(f"Enhanced {sum(1 for c in enhanced_chunks if c['has_kg_enhancement'])} chunks with KG info")
        return enhanced_chunks
    
    def generate_kg_summary(self, query: str, max_length: int = 500) -> str:
        """生成查询相关的知识图谱摘要"""
        
        kg_results = self.retrieve_kg_context(query, top_k=15)
        
        if not kg_results:
            return ""
        
        # 按类型分组
        entities = [r for r in kg_results if r.source_type == 'entity']
        relations = [r for r in kg_results if r.source_type == 'relation']
        
        summary_parts = []
        
        # 实体摘要
        if entities:
            entity_names = [e.entity for e in entities[:5]]
            summary_parts.append(f"相关实体: {', '.join(entity_names)}")
        
        # 关系摘要
        if relations:
            key_relations = []
            for rel in relations[:3]:
                rel_desc = f"{rel.metadata['source_entity']}与{rel.entity}的关系是{rel.metadata['relation_type']}"
                key_relations.append(rel_desc)
            
            if key_relations:
                summary_parts.append(f"关键关系: {'; '.join(key_relations)}")
        
        summary = "\n".join(summary_parts)
        
        # 截断到指定长度
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary
    
    def find_entity_connections(
        self,
        entity1: str,
        entity2: str,
        max_path_length: int = 3
    ) -> List[Dict]:
        """查找两个实体之间的连接路径"""
        
        try:
            import networkx as nx
            
            # 确保图已加载
            if self.kg_indexer.kg_db.graph.number_of_nodes() == 0:
                self.kg_indexer.kg_db.load_graph_from_db()
            
            graph = self.kg_indexer.kg_db.graph.to_undirected()  # 转换为无向图进行路径搜索
            
            if entity1 not in graph or entity2 not in graph:
                return []
            
            # 找到所有简单路径
            try:
                paths = list(nx.all_simple_paths(
                    graph, entity1, entity2, cutoff=max_path_length
                ))
            except nx.NetworkXNoPath:
                return []
            
            connection_results = []
            for path in paths[:10]:  # 限制路径数量
                path_info = {
                    'path': path,
                    'length': len(path) - 1,
                    'relations': []
                }
                
                # 获取路径上的关系
                for i in range(len(path) - 1):
                    source, target = path[i], path[i + 1]
                    # 从有向图中获取边信息
                    if self.kg_indexer.kg_db.graph.has_edge(source, target):
                        edge_data = self.kg_indexer.kg_db.graph[source][target]
                        for edge_id, edge_attrs in edge_data.items():
                            path_info['relations'].append({
                                'source': source,
                                'target': target,
                                'relation': edge_attrs.get('relation_type', 'RELATED'),
                                'confidence': edge_attrs.get('confidence', 0.5)
                            })
                            break
                    elif self.kg_indexer.kg_db.graph.has_edge(target, source):
                        edge_data = self.kg_indexer.kg_db.graph[target][source]
                        for edge_id, edge_attrs in edge_data.items():
                            path_info['relations'].append({
                                'source': target,
                                'target': source,
                                'relation': edge_attrs.get('relation_type', 'RELATED'),
                                'confidence': edge_attrs.get('confidence', 0.5)
                            })
                            break
                
                connection_results.append(path_info)
            
            # 按路径长度排序
            connection_results.sort(key=lambda x: x['length'])
            
            logger.info(f"Found {len(connection_results)} connections between {entity1} and {entity2}")
            return connection_results
            
        except Exception as e:
            logger.error(f"Error finding entity connections: {e}")
            return []
    
    def get_entity_neighborhood(
        self,
        entity: str,
        radius: int = 1,
        max_neighbors: int = 20
    ) -> Dict[str, Any]:
        """获取实体的邻居子图"""
        
        try:
            if self.kg_indexer.kg_db.graph.number_of_nodes() == 0:
                self.kg_indexer.kg_db.load_graph_from_db()
            
            graph = self.kg_indexer.kg_db.graph
            
            if entity not in graph:
                return {'nodes': [], 'edges': [], 'center': entity}
            
            # 获取指定半径内的邻居
            neighbors = set([entity])
            current_layer = {entity}
            
            for r in range(radius):
                next_layer = set()
                for node in current_layer:
                    next_layer.update(graph.neighbors(node))
                    next_layer.update(graph.predecessors(node))
                
                neighbors.update(next_layer)
                current_layer = next_layer
                
                if len(neighbors) >= max_neighbors:
                    break
            
            # 限制邻居数量
            if len(neighbors) > max_neighbors:
                # 保留中心节点和度数最高的邻居
                degree_scores = [(n, graph.degree(n)) for n in neighbors if n != entity]
                degree_scores.sort(key=lambda x: x[1], reverse=True)
                neighbors = {entity} | {n for n, _ in degree_scores[:max_neighbors-1]}
            
            # 构建子图数据
            subgraph_nodes = []
            subgraph_edges = []
            
            for node in neighbors:
                node_data = graph.nodes[node]
                subgraph_nodes.append({
                    'id': node,
                    'type': node_data.get('entity_type', 'UNKNOWN'),
                    'confidence': node_data.get('confidence', 0.5)
                })
            
            for source in neighbors:
                for target in neighbors:
                    if graph.has_edge(source, target):
                        for edge_data in graph[source][target].values():
                            subgraph_edges.append({
                                'source': source,
                                'target': target,
                                'relation': edge_data.get('relation_type', 'RELATED'),
                                'confidence': edge_data.get('confidence', 0.5)
                            })
            
            return {
                'center': entity,
                'nodes': subgraph_nodes,
                'edges': subgraph_edges,
                'radius': radius,
                'total_nodes': len(subgraph_nodes)
            }
            
        except Exception as e:
            logger.error(f"Error getting entity neighborhood: {e}")
            return {'nodes': [], 'edges': [], 'center': entity}

class KGEnhancedRetriever:
    """知识图谱增强的检索器"""
    
    def __init__(self, vector_db_manager, kg_retriever: KnowledgeGraphRetriever):
        self.vector_db = vector_db_manager
        self.kg_retriever = kg_retriever
        self.embedder = kg_retriever.embedder
        
        logger.info("KG Enhanced Retriever initialized")
    
    def hybrid_kg_vector_search(
        self,
        query: str,
        top_k: int = 10,
        kg_weight: float = 0.3,
        vector_weight: float = 0.7
    ) -> List[Dict]:
        """混合知识图谱和向量检索"""
        
        # 1. 向量检索
        query_vector = self.embedder.encode([query], convert_to_numpy=True)[0]
        vector_results = self.vector_db.search(
            query_vector=query_vector,
            query_text=query,
            top_k=top_k * 2  # 获取更多候选
        )
        
        # 2. KG增强
        kg_enhanced_results = self.kg_retriever.enhance_chunks_with_kg(
            query=query,
            chunks=vector_results,
            kg_weight=kg_weight
        )
        
        # 3. 重新计算综合分数
        for result in kg_enhanced_results:
            original_score = result.get('scores', {}).get('hybrid_score', 0.5)
            
            # KG增强加分
            kg_boost = 0
            if result.get('has_kg_enhancement', False):
                kg_entities_count = len(result.get('kg_entities', []))
                kg_boost = kg_entities_count * kg_weight * 0.2
            
            # 计算最终分数
            final_score = vector_weight * original_score + kg_boost
            
            if 'scores' not in result:
                result['scores'] = {}
            result['scores']['kg_hybrid_score'] = final_score
        
        # 4. 按综合分数排序
        kg_enhanced_results.sort(
            key=lambda x: x.get('scores', {}).get('kg_hybrid_score', 0),
            reverse=True
        )
        
        logger.info(f"KG-enhanced hybrid search returned {len(kg_enhanced_results[:top_k])} results")
        return kg_enhanced_results[:top_k]
    
    def generate_kg_augmented_context(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        max_context_length: int = 3000
    ) -> str:
        """生成知识图谱增强的上下文"""
        
        # 获取KG摘要
        kg_summary = self.kg_retriever.generate_kg_summary(query, max_length=500)
        
        # 构建增强上下文
        context_parts = []
        
        if kg_summary:
            context_parts.append(f"[知识图谱背景]\n{kg_summary}")
        
        # 添加文档块内容
        for i, chunk in enumerate(retrieved_chunks):
            if chunk.get('has_kg_enhancement', False):
                # 使用KG增强的内容
                content = chunk['kg_enhanced_content']
            else:
                content = chunk['content']
            
            context_parts.append(f"[文档 {i+1}]\n{content}")
        
        # 组合并截断
        full_context = "\n\n".join(context_parts)
        
        if len(full_context) > max_context_length:
            full_context = full_context[:max_context_length] + "..."
        
        return full_context

# 使用示例
async def main():
    """测试知识图谱检索器"""
    config = {
        'llm_model': 'Qwen/Qwen2-7B-Instruct',
        'embedding_model': 'BAAI/bge-m3',
        'device': 'auto',
        'HUGGING_FACE_TOKEN': None,
        'knowledge_graph_db_path': 'data/knowledge_graph/test_kg.db'
    }
    
    kg_retriever = KnowledgeGraphRetriever(config)
    
    # 测试KG检索
    query = "Transformer模型的注意力机制"
    kg_results = kg_retriever.retrieve_kg_context(query, top_k=5)
    
    print("知识图谱检索结果:")
    for result in kg_results:
        print(f"实体: {result.entity}")
        print(f"类型: {result.entity_type}")
        print(f"信息: {result.related_info}")
        print(f"置信度: {result.confidence}")
        print("---")
    
    # 测试实体连接
    connections = kg_retriever.find_entity_connections("Transformer", "BERT")
    print(f"\nTransformer和BERT的连接路径:")
    for conn in connections:
        print(f"路径: {' -> '.join(conn['path'])}")
        print(f"长度: {conn['length']}")
        print("---")

if __name__ == "__main__":
    asyncio.run(main())