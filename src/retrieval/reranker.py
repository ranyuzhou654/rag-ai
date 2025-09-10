# src/retrieval/reranker.py
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch
from loguru import logger
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class RerankResult:
    """重排序结果"""
    chunk_id: str
    content: str
    rerank_score: float
    metadata: Dict

class AdvancedReranker:
    """
    高级重排序器
    - 结合Cross-Encoder精排和MMR多样性
    """
    def __init__(
        self,
        cross_encoder_model: str = "BAAI/bge-reranker-base",
        bi_encoder_model: str = "BAAI/bge-m3", # For MMR
        device: str = "auto",
    ):
        self.device = self._get_device(device)
        logger.info(f"Loading reranker models on device: {self.device}")
        self.cross_encoder = CrossEncoder(cross_encoder_model, device=self.device)
        self.bi_encoder = SentenceTransformer(bi_encoder_model, device=self.device)
        logger.success("Reranker models loaded.")

    def _get_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def rerank_with_mmr(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        top_k: int = 5,
        lambda_mult: float = 0.5
    ) -> List[RerankResult]:
        """
        执行重排序和MMR多样性优化
        """
        if not retrieved_chunks:
            return []

        # 1. Cross-Encoder 精排
        pairs = [[query, chunk['content']] for chunk in retrieved_chunks]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(retrieved_chunks, scores):
            chunk['rerank_score'] = score
        
        # 按精排分数排序
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x['rerank_score'], reverse=True)

        # 2. MMR 多样性选择
        final_results = self.maximal_marginal_relevance(query, sorted_chunks, top_k, lambda_mult)
        
        logger.info(f"Reranking complete. Selected {len(final_results)} diverse documents.")
        return [RerankResult(
            chunk_id=chunk['chunk_id'],
            content=chunk['content'],
            rerank_score=chunk['rerank_score'],
            metadata=chunk.get('metadata', {})
        ) for chunk in final_results]
        
    def maximal_marginal_relevance(self, query: str, docs: list, top_k: int, lambda_val: float) -> list:
        """
        MMR算法实现
        """
        if not docs:
            return []
            
        doc_contents = [doc['content'] for doc in docs]
        doc_embeddings = self.bi_encoder.encode(doc_contents, convert_to_tensor=True)
        query_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)

        # Move to CPU for sklearn compatibility
        doc_embeddings_np = doc_embeddings.cpu().numpy()
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)

        # Calculate relevance (query-doc similarity)
        relevance_scores = cosine_similarity(query_embedding_np, doc_embeddings_np)[0]

        # Calculate diversity (doc-doc similarity)
        similarity_matrix = cosine_similarity(doc_embeddings_np)
        
        # MMR loop
        selected_indices = []
        candidate_indices = list(range(len(docs)))
        
        # Start with the most relevant document
        best_initial_idx = np.argmax(relevance_scores)
        selected_indices.append(best_initial_idx)
        candidate_indices.remove(best_initial_idx)

        while len(selected_indices) < min(top_k, len(docs)):
            mmr_scores = []
            for idx in candidate_indices:
                relevance = relevance_scores[idx]
                max_similarity = np.max(similarity_matrix[idx, selected_indices]) if selected_indices else 0
                mmr = lambda_val * relevance - (1 - lambda_val) * max_similarity
                mmr_scores.append((mmr, idx))
            
            if not mmr_scores:
                break
                
            best_mmr_idx = max(mmr_scores, key=lambda x: x[0])[1]
            selected_indices.append(best_mmr_idx)
            candidate_indices.remove(best_mmr_idx)
        
        return [docs[i] for i in selected_indices]
