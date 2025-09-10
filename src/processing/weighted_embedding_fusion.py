# src/processing/weighted_embedding_fusion.py
"""
加权嵌入融合策略 - 实现多表示向量的智能融合
基于improvements.md中的建议：原文0.6 + 摘要0.3 + 问题0.1
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import asyncio
from loguru import logger

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


@dataclass
class MultiRepresentation:
    """多表示文档结构"""
    original_content: str
    summary: str
    hypothetical_questions: List[str]
    metadata: Dict[str, Any]


@dataclass
class FusedEmbedding:
    """融合嵌入结果"""
    combined_vector: np.ndarray
    component_vectors: Dict[str, np.ndarray]  # 'original', 'summary', 'questions'
    fusion_weights: Dict[str, float]
    metadata: Dict[str, Any]


class WeightedEmbeddingFusion:
    """加权嵌入融合器"""
    
    def __init__(
        self,
        embeddings_model: Embeddings,
        original_weight: float = 0.6,
        summary_weight: float = 0.3,
        questions_weight: float = 0.1,
        normalize_vectors: bool = True
    ):
        self.embeddings_model = embeddings_model
        self.original_weight = original_weight
        self.summary_weight = summary_weight
        self.questions_weight = questions_weight
        self.normalize_vectors = normalize_vectors
        
        # 权重归一化
        total_weight = original_weight + summary_weight + questions_weight
        if total_weight > 0:
            self.original_weight /= total_weight
            self.summary_weight /= total_weight
            self.questions_weight /= total_weight
        
        logger.info(f"Weighted Embedding Fusion initialized with weights: "
                   f"Original={self.original_weight:.3f}, "
                   f"Summary={self.summary_weight:.3f}, "
                   f"Questions={self.questions_weight:.3f}")
    
    async def fuse_representations(
        self, 
        multi_rep: MultiRepresentation
    ) -> FusedEmbedding:
        """融合多表示向量"""
        
        try:
            # 并行计算各部分的嵌入
            tasks = []
            
            # 原文嵌入
            if multi_rep.original_content:
                tasks.append(self._embed_text(multi_rep.original_content, "original"))
            
            # 摘要嵌入
            if multi_rep.summary:
                tasks.append(self._embed_text(multi_rep.summary, "summary"))
            
            # 问题嵌入（平均多个问题）
            if multi_rep.hypothetical_questions:
                questions_text = " ".join(multi_rep.hypothetical_questions)
                tasks.append(self._embed_text(questions_text, "questions"))
            
            # 并行执行嵌入计算
            embeddings_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 收集有效的嵌入向量
            component_vectors = {}
            valid_weights = {}
            
            for result in embeddings_results:
                if isinstance(result, tuple) and len(result) == 2:
                    vector, component_type = result
                    if vector is not None:
                        component_vectors[component_type] = vector
                        
                        # 设置对应权重
                        if component_type == "original":
                            valid_weights[component_type] = self.original_weight
                        elif component_type == "summary":
                            valid_weights[component_type] = self.summary_weight
                        elif component_type == "questions":
                            valid_weights[component_type] = self.questions_weight
            
            # 重新归一化有效权重
            if valid_weights:
                total_valid_weight = sum(valid_weights.values())
                if total_valid_weight > 0:
                    valid_weights = {k: v / total_valid_weight for k, v in valid_weights.items()}
            
            # 执行加权融合
            combined_vector = self._weighted_fusion(component_vectors, valid_weights)
            
            # 创建融合结果
            fused_embedding = FusedEmbedding(
                combined_vector=combined_vector,
                component_vectors=component_vectors,
                fusion_weights=valid_weights,
                metadata={
                    "original_length": len(multi_rep.original_content),
                    "summary_length": len(multi_rep.summary) if multi_rep.summary else 0,
                    "num_questions": len(multi_rep.hypothetical_questions),
                    "fusion_method": "weighted_average",
                    **multi_rep.metadata
                }
            )
            
            logger.debug(f"Fusion completed: {len(component_vectors)} components -> "
                        f"{combined_vector.shape[0]}D vector")
            
            return fused_embedding
            
        except Exception as e:
            logger.error(f"Embedding fusion failed: {e}")
            raise
    
    async def _embed_text(self, text: str, component_type: str) -> Tuple[Optional[np.ndarray], str]:
        """嵌入单个文本"""
        try:
            if not text or not text.strip():
                return None, component_type
            
            # 使用LangChain嵌入模型
            if hasattr(self.embeddings_model, 'aembed_query'):
                vector = await self.embeddings_model.aembed_query(text)
            else:
                # 回退到同步方法
                vector = self.embeddings_model.embed_query(text)
            
            vector = np.array(vector)
            
            # 向量归一化（如果需要）
            if self.normalize_vectors:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            
            return vector, component_type
            
        except Exception as e:
            logger.warning(f"Failed to embed {component_type} text: {e}")
            return None, component_type
    
    def _weighted_fusion(
        self, 
        component_vectors: Dict[str, np.ndarray], 
        weights: Dict[str, float]
    ) -> np.ndarray:
        """执行加权融合"""
        
        if not component_vectors:
            raise ValueError("No valid component vectors for fusion")
        
        # 确保所有向量维度一致
        vector_dims = [v.shape[0] for v in component_vectors.values()]
        if len(set(vector_dims)) > 1:
            raise ValueError(f"Inconsistent vector dimensions: {vector_dims}")
        
        # 初始化融合向量
        dim = vector_dims[0]
        combined_vector = np.zeros(dim)
        
        # 加权求和
        for component_type, vector in component_vectors.items():
            weight = weights.get(component_type, 0.0)
            combined_vector += weight * vector
        
        # 最终归一化
        if self.normalize_vectors:
            norm = np.linalg.norm(combined_vector)
            if norm > 0:
                combined_vector = combined_vector / norm
        
        return combined_vector
    
    async def batch_fuse_representations(
        self,
        multi_reps: List[MultiRepresentation]
    ) -> List[FusedEmbedding]:
        """批量融合多表示向量"""
        
        logger.info(f"Starting batch fusion for {len(multi_reps)} documents")
        
        # 并行处理多个文档
        tasks = [self.fuse_representations(multi_rep) for multi_rep in multi_reps]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集成功的结果
        fused_embeddings = []
        failed_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, FusedEmbedding):
                fused_embeddings.append(result)
            else:
                logger.error(f"Document {i} fusion failed: {result}")
                failed_count += 1
        
        logger.info(f"Batch fusion completed: {len(fused_embeddings)} successful, {failed_count} failed")
        
        return fused_embeddings
    
    def update_weights(
        self,
        original_weight: float,
        summary_weight: float,
        questions_weight: float
    ):
        """动态更新融合权重"""
        
        self.original_weight = original_weight
        self.summary_weight = summary_weight
        self.questions_weight = questions_weight
        
        # 权重归一化
        total_weight = original_weight + summary_weight + questions_weight
        if total_weight > 0:
            self.original_weight /= total_weight
            self.summary_weight /= total_weight
            self.questions_weight /= total_weight
        
        logger.info(f"Weights updated: "
                   f"Original={self.original_weight:.3f}, "
                   f"Summary={self.summary_weight:.3f}, "
                   f"Questions={self.questions_weight:.3f}")
    
    def analyze_fusion_quality(self, fused_embedding: FusedEmbedding) -> Dict[str, Any]:
        """分析融合质量"""
        
        analysis = {
            "vector_dimension": fused_embedding.combined_vector.shape[0],
            "vector_norm": float(np.linalg.norm(fused_embedding.combined_vector)),
            "num_components": len(fused_embedding.component_vectors),
            "active_weights": fused_embedding.fusion_weights,
            "component_similarities": {}
        }
        
        # 计算各组件向量之间的相似度
        component_items = list(fused_embedding.component_vectors.items())
        for i, (name1, vec1) in enumerate(component_items):
            for name2, vec2 in component_items[i+1:]:
                similarity = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
                analysis["component_similarities"][f"{name1}_vs_{name2}"] = similarity
        
        return analysis


class AdaptiveWeightingStrategy:
    """自适应权重策略 - 基于内容特性动态调整权重"""
    
    def __init__(self, base_fusion: WeightedEmbeddingFusion):
        self.base_fusion = base_fusion
    
    def calculate_adaptive_weights(self, multi_rep: MultiRepresentation) -> Tuple[float, float, float]:
        """基于内容特性计算自适应权重"""
        
        original_len = len(multi_rep.original_content)
        summary_len = len(multi_rep.summary) if multi_rep.summary else 0
        num_questions = len(multi_rep.hypothetical_questions)
        
        # 基础权重
        original_weight = 0.6
        summary_weight = 0.3
        questions_weight = 0.1
        
        # 根据原文长度调整权重
        if original_len > 2000:  # 长文档，增加摘要权重
            original_weight *= 0.8
            summary_weight *= 1.3
        elif original_len < 200:  # 短文档，增加原文权重
            original_weight *= 1.2
            summary_weight *= 0.7
        
        # 根据摘要质量调整权重
        if summary_len > 0:
            summary_ratio = summary_len / original_len if original_len > 0 else 0
            if summary_ratio > 0.8:  # 摘要太长，降低权重
                summary_weight *= 0.7
            elif summary_ratio < 0.1:  # 摘要太短，降低权重
                summary_weight *= 0.8
        else:
            # 没有摘要，将权重分配给原文和问题
            original_weight += summary_weight * 0.7
            questions_weight += summary_weight * 0.3
            summary_weight = 0.0
        
        # 根据问题数量调整权重
        if num_questions > 3:  # 问题较多，增加权重
            questions_weight *= 1.2
        elif num_questions == 0:  # 没有问题，将权重分配给其他组件
            original_weight += questions_weight * 0.6
            summary_weight += questions_weight * 0.4
            questions_weight = 0.0
        
        # 归一化权重
        total = original_weight + summary_weight + questions_weight
        if total > 0:
            original_weight /= total
            summary_weight /= total
            questions_weight /= total
        
        return original_weight, summary_weight, questions_weight
    
    async def adaptive_fuse_representations(
        self, 
        multi_rep: MultiRepresentation
    ) -> FusedEmbedding:
        """使用自适应权重融合表示"""
        
        # 计算自适应权重
        original_w, summary_w, questions_w = self.calculate_adaptive_weights(multi_rep)
        
        # 临时更新权重
        old_weights = (
            self.base_fusion.original_weight,
            self.base_fusion.summary_weight,
            self.base_fusion.questions_weight
        )
        
        self.base_fusion.update_weights(original_w, summary_w, questions_w)
        
        try:
            # 执行融合
            result = await self.base_fusion.fuse_representations(multi_rep)
            
            # 在结果中记录使用的自适应权重
            result.metadata["adaptive_weights"] = {
                "original": original_w,
                "summary": summary_w,
                "questions": questions_w
            }
            
            return result
            
        finally:
            # 恢复原权重
            self.base_fusion.update_weights(*old_weights)