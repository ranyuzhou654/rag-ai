# src/retrieval/contextual_compression.py
from typing import List, Dict, Optional, Tuple
import re
import asyncio
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class CompressedContext:
    """压缩的上下文结构"""
    compressed_content: str
    original_chunks: List[Dict]
    compression_ratio: float
    relevance_scores: List[float]
    extracted_sentences: List[str]

class SentenceExtractor:
    """句子提取器 - 从文档块中提取最相关的句子"""
    
    def __init__(self, embedding_model: str = "BAAI/bge-m3", device: str = "auto"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        logger.info(f"Sentence Extractor initialized with {embedding_model}")
    
    def extract_relevant_sentences(
        self, 
        query: str, 
        chunks: List[Dict], 
        top_k_sentences: int = 10,
        min_sentence_length: int = 20
    ) -> List[Tuple[str, float, int]]:
        """提取与查询最相关的句子"""
        
        all_sentences = []
        sentence_to_chunk_map = []
        
        # Split chunks into sentences
        for chunk_idx, chunk in enumerate(chunks):
            content = chunk['content']
            # Simple sentence splitting (can be improved with more sophisticated methods)
            sentences = self._split_into_sentences(content)
            
            for sentence in sentences:
                if len(sentence.strip()) >= min_sentence_length:
                    all_sentences.append(sentence.strip())
                    sentence_to_chunk_map.append(chunk_idx)
        
        if not all_sentences:
            return []
        
        # Encode query and sentences
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0]
        sentence_embeddings = self.embedder.encode(
            all_sentences, 
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Calculate similarity scores
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), 
            sentence_embeddings
        )[0]
        
        # Rank sentences by similarity
        sentence_scores = list(zip(all_sentences, similarities, sentence_to_chunk_map))
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k sentences
        return sentence_scores[:top_k_sentences]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # Handle both Chinese and English text
        # Chinese: split by Chinese periods, exclamations, questions
        # English: split by periods, exclamations, questions
        
        # First split by common sentence ending patterns
        sentences = re.split(r'[。！？!?]+\s*', text)
        
        # Further split by periods followed by space and capital letter (for English)
        refined_sentences = []
        for sentence in sentences:
            # Split English sentences
            english_splits = re.split(r'\.(?=\s+[A-Z])', sentence)
            refined_sentences.extend(english_splits)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in refined_sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                # Ensure sentence ends with punctuation
                if not sentence.endswith(('.', '。', '!', '！', '?', '？')):
                    sentence += '。' if re.search(r'[\u4e00-\u9fff]', sentence) else '.'
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

class LLMCompressor:
    """基于LLM的内容压缩器"""
    
    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        logger.info(f"Loading LLM Compressor: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            token=token
        )
        
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
    
    def compress_context(
        self, 
        query: str, 
        context: str, 
        target_length: int = 300
    ) -> str:
        """使用LLM压缩上下文"""
        
        # Detect language
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', query))
        
        if is_chinese:
            prompt = f"""请从以下文档内容中提取与问题最相关的关键信息，压缩成大约{target_length}字的简洁内容，保留所有重要细节：

问题：{query}

文档内容：
{context}

提取的关键信息："""
        else:
            prompt = f"""Extract the most relevant key information from the following document content related to the question, compress it to approximately {target_length} words while preserving all important details:

Question: {query}

Document content:
{context}

Extracted key information:"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract compressed content
            if is_chinese:
                compressed_part = response.split("提取的关键信息：")[-1].strip()
            else:
                compressed_part = response.split("Extracted key information:")[-1].strip()
            
            # Clean up the response
            compressed_content = compressed_part.split('\n')[0].strip()
            
            return compressed_content if len(compressed_content) > 50 else context[:target_length]
            
        except Exception as e:
            logger.error(f"LLM compression failed: {e}")
            # Fallback: simple truncation
            return context[:target_length] + "..." if len(context) > target_length else context

class ContextualCompressor:
    """上下文压缩器 - 集成多种压缩策略"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize sentence extractor
        self.sentence_extractor = SentenceExtractor(
            embedding_model=config.get('embedding_model', 'BAAI/bge-m3'),
            device=config.get('device', 'auto')
        )
        
        # Initialize LLM compressor if available
        model_name = config.get('llm_model')
        token = config.get('HUGGING_FACE_TOKEN')
        device = config.get('device', 'auto')
        
        self.llm_available = False
        if model_name:
            try:
                self.llm_compressor = LLMCompressor(
                    model_name=model_name, device=device, token=token
                )
                self.llm_available = True
                logger.success("Contextual Compressor with LLM support initialized")
            except Exception as e:
                logger.error(f"Failed to initialize LLM compressor: {e}")
        else:
            logger.warning("Contextual Compressor initialized without LLM support")
    
    def compress_context(
        self, 
        query: str, 
        chunks: List[Dict], 
        max_context_length: int = 2000,
        compression_method: str = "sentence_extraction"  # "sentence_extraction", "llm_compression", "hybrid"
    ) -> CompressedContext:
        """压缩上下文"""
        
        if not chunks:
            return CompressedContext(
                compressed_content="",
                original_chunks=[],
                compression_ratio=1.0,
                relevance_scores=[],
                extracted_sentences=[]
            )
        
        original_content = "\n".join([chunk['content'] for chunk in chunks])
        original_length = len(original_content)
        
        if compression_method == "sentence_extraction":
            compressed_content, extracted_sentences, relevance_scores = self._extract_sentences(
                query, chunks, max_context_length
            )
        elif compression_method == "llm_compression" and self.llm_available:
            compressed_content = self._llm_compress(query, original_content, max_context_length)
            extracted_sentences = []
            relevance_scores = []
        elif compression_method == "hybrid":
            # First extract sentences, then use LLM to refine
            sentence_compressed, extracted_sentences, relevance_scores = self._extract_sentences(
                query, chunks, max_context_length * 2  # Allow more for LLM refinement
            )
            if self.llm_available:
                compressed_content = self._llm_compress(query, sentence_compressed, max_context_length)
            else:
                compressed_content = sentence_compressed
        else:
            # Fallback: simple truncation
            compressed_content = original_content[:max_context_length]
            extracted_sentences = []
            relevance_scores = []
        
        compression_ratio = len(compressed_content) / max(original_length, 1)
        
        logger.info(f"Context compression: {original_length} -> {len(compressed_content)} chars ({compression_ratio:.2f})")
        
        return CompressedContext(
            compressed_content=compressed_content,
            original_chunks=chunks,
            compression_ratio=compression_ratio,
            relevance_scores=relevance_scores,
            extracted_sentences=extracted_sentences
        )
    
    def _extract_sentences(
        self, 
        query: str, 
        chunks: List[Dict], 
        max_length: int
    ) -> Tuple[str, List[str], List[float]]:
        """基于句子提取的压缩"""
        
        # Extract top relevant sentences
        top_sentences = self.sentence_extractor.extract_relevant_sentences(
            query=query,
            chunks=chunks,
            top_k_sentences=20  # Extract more than needed
        )
        
        if not top_sentences:
            content = "\n".join([chunk['content'] for chunk in chunks])
            return content[:max_length], [], []
        
        # Select sentences until we reach the length limit
        selected_sentences = []
        selected_scores = []
        current_length = 0
        
        for sentence, score, chunk_idx in top_sentences:
            if current_length + len(sentence) <= max_length:
                selected_sentences.append(sentence)
                selected_scores.append(float(score))
                current_length += len(sentence) + 1  # +1 for separator
            else:
                break
        
        compressed_content = " ".join(selected_sentences)
        
        return compressed_content, selected_sentences, selected_scores
    
    def _llm_compress(self, query: str, context: str, max_length: int) -> str:
        """基于LLM的压缩"""
        if self.llm_available:
            return self.llm_compressor.compress_context(query, context, max_length)
        else:
            return context[:max_length]

class SmartReranker:
    """智能重排序器 - 结合多种信号进行重排序"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize embedder for semantic similarity
        self.embedder = SentenceTransformer(
            config.get('embedding_model', 'BAAI/bge-m3'),
            device=config.get('device', 'auto')
        )
        
        logger.info("Smart Reranker initialized")
    
    def smart_rerank(
        self, 
        query: str, 
        chunks: List[Dict], 
        top_k: int = 5,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """智能重排序"""
        
        if not chunks:
            return []
        
        # Default weights for different signals
        if weights is None:
            weights = {
                'semantic_similarity': 0.4,
                'keyword_match': 0.2,
                'chunk_quality': 0.2,
                'source_reliability': 0.1,
                'recency': 0.1
            }
        
        # Calculate multiple ranking signals
        enhanced_chunks = []
        
        for chunk in chunks:
            signals = self._calculate_ranking_signals(query, chunk)
            
            # Combine signals with weights
            final_score = sum(
                weights.get(signal_name, 0) * signal_value 
                for signal_name, signal_value in signals.items()
            )
            
            chunk_copy = chunk.copy()
            chunk_copy['final_score'] = final_score
            chunk_copy['ranking_signals'] = signals
            enhanced_chunks.append(chunk_copy)
        
        # Sort by final score
        enhanced_chunks.sort(key=lambda x: x['final_score'], reverse=True)
        
        logger.info(f"Smart reranking complete: top score = {enhanced_chunks[0]['final_score']:.3f}")
        
        return enhanced_chunks[:top_k]
    
    def _calculate_ranking_signals(self, query: str, chunk: Dict) -> Dict[str, float]:
        """计算排序信号"""
        content = chunk['content']
        
        signals = {}
        
        # 1. Semantic similarity
        try:
            query_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
            content_emb = self.embedder.encode([content], convert_to_numpy=True)[0]
            semantic_sim = float(cosine_similarity(
                query_emb.reshape(1, -1), 
                content_emb.reshape(1, -1)
            )[0][0])
            signals['semantic_similarity'] = semantic_sim
        except:
            signals['semantic_similarity'] = 0.0
        
        # 2. Keyword matching
        query_words = set(re.findall(r'\w+', query.lower()))
        content_words = set(re.findall(r'\w+', content.lower()))
        if query_words:
            keyword_overlap = len(query_words.intersection(content_words)) / len(query_words)
        else:
            keyword_overlap = 0.0
        signals['keyword_match'] = keyword_overlap
        
        # 3. Chunk quality (based on length and structure)
        content_length = len(content)
        # Optimal length around 200-600 characters
        if 200 <= content_length <= 600:
            length_score = 1.0
        elif content_length < 200:
            length_score = content_length / 200
        else:
            length_score = max(0.5, 1.0 - (content_length - 600) / 1000)
        
        # Check for structured content (lists, headers, etc.)
        structure_bonus = 0.0
        if re.search(r'^\s*[\-\*\d]+\.', content, re.MULTILINE):  # Lists
            structure_bonus += 0.1
        if re.search(r'^#{1,6}\s', content, re.MULTILINE):  # Headers
            structure_bonus += 0.1
        
        signals['chunk_quality'] = min(1.0, length_score + structure_bonus)
        
        # 4. Source reliability (based on metadata)
        source_score = 0.5  # Default
        metadata = chunk.get('metadata', {})
        if metadata.get('source_type') == 'academic_paper':
            source_score = 0.8
        elif metadata.get('source_type') == 'documentation':
            source_score = 0.7
        signals['source_reliability'] = source_score
        
        # 5. Recency (if timestamp available)
        recency_score = 0.5  # Default
        if 'timestamp' in metadata:
            # This would need actual implementation based on timestamp format
            pass
        signals['recency'] = recency_score
        
        return signals

# 使用示例
async def main():
    """测试上下文压缩和重排序"""
    config = {
        'embedding_model': 'BAAI/bge-m3',
        'llm_model': 'Qwen/Qwen2-7B-Instruct',
        'device': 'auto',
        'HUGGING_FACE_TOKEN': None
    }
    
    compressor = ContextualCompressor(config)
    reranker = SmartReranker(config)
    
    # 测试数据
    test_query = "什么是Transformer模型的注意力机制？"
    test_chunks = [
        {
            'content': 'Transformer模型是一种基于注意力机制的神经网络架构。注意力机制允许模型在处理序列时动态地关注输入的不同部分。在自注意力机制中，每个位置可以关注序列中的所有位置。',
            'chunk_id': 'chunk_1',
            'metadata': {'source_type': 'academic_paper'}
        },
        {
            'content': 'CNN卷积神经网络是深度学习中常用的架构，主要用于图像处理任务。它通过卷积层、池化层等组件来提取特征。',
            'chunk_id': 'chunk_2', 
            'metadata': {'source_type': 'tutorial'}
        },
        {
            'content': '注意力机制的核心思想是让模型学会在处理某个元素时，有选择性地关注输入序列中的相关部分。这通过计算注意力权重来实现，注意力权重反映了每个输入元素的重要性。',
            'chunk_id': 'chunk_3',
            'metadata': {'source_type': 'documentation'}
        }
    ]
    
    # 测试智能重排序
    print("=== Smart Reranking Test ===")
    reranked_chunks = reranker.smart_rerank(test_query, test_chunks, top_k=3)
    
    for i, chunk in enumerate(reranked_chunks, 1):
        print(f"{i}. Score: {chunk['final_score']:.3f}")
        print(f"   Content: {chunk['content'][:100]}...")
        print(f"   Signals: {chunk['ranking_signals']}")
        print()
    
    # 测试上下文压缩
    print("=== Context Compression Test ===")
    compressed = compressor.compress_context(
        query=test_query,
        chunks=reranked_chunks,
        max_context_length=200,
        compression_method="sentence_extraction"
    )
    
    print(f"Original length: {sum(len(c['content']) for c in reranked_chunks)} chars")
    print(f"Compressed length: {len(compressed.compressed_content)} chars")
    print(f"Compression ratio: {compressed.compression_ratio:.3f}")
    print(f"Compressed content: {compressed.compressed_content}")

if __name__ == "__main__":
    asyncio.run(main())