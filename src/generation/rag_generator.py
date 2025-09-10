# src/generation/rag_generator.py
from typing import List, Dict, Optional, Tuple
import asyncio
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from loguru import logger
import json
from pathlib import Path
import re

# Assuming these classes are in their respective files as per the project structure
from src.retrieval.vector_database import VectorDatabaseManager
from src.retrieval.reranker import AdvancedReranker
from src.retrieval.query_intelligence import QueryIntelligenceEngine
from src.retrieval.agentic_rag import AgenticRAGOrchestrator, AgenticStep
from src.retrieval.contextual_compression import ContextualCompressor, SmartReranker
from src.knowledge_graph import KnowledgeGraphIndexer, KnowledgeGraphRetriever, KGEnhancedRetriever
from src.generation.tiered_generation import TieredGenerationSystem, TaskRequest, TaskType, TaskComplexity


@dataclass
class GenerationResult:
    """生成结果结构"""
    answer: str
    source_chunks: List[Dict]
    confidence: float
    generation_time: float
    token_count: int
    query_analysis: Optional[Dict] = None
    retrieval_strategies: Optional[List[str]] = None
    agentic_steps: Optional[List[AgenticStep]] = None
    iterations_used: Optional[int] = None
    kg_entities: Optional[List[str]] = None
    kg_relations: Optional[List[Dict]] = None
    models_used: Optional[Dict[str, float]] = None  # model_name -> execution_time
    total_cost: Optional[float] = None

class EnhancedQueryProcessor:
    """增强的查询处理器 - 集成查询智能"""
    def __init__(self, embedding_model: str = "BAAI/bge-m3", config: Optional[Dict] = None):
        self.embedder = SentenceTransformer(embedding_model)
        
        # Initialize Query Intelligence Engine if config provided
        self.query_intelligence = None
        if config:
            try:
                self.query_intelligence = QueryIntelligenceEngine(config)
                logger.info("Enhanced Query Processor with Intelligence initialized.")
            except Exception as e:
                logger.warning(f"Query Intelligence initialization failed: {e}. Using basic processing.")
        else:
            logger.info("Basic Query Processor initialized.")
    
    def process_query(self, query: str) -> Dict:
        language = 'zh' if re.search(r'[\u4e00-\u9fff]', query) else 'en'
        
        result = {
            'original_query': query,
            'language': language,
            'query_vector': self.embedder.encode([query], convert_to_numpy=True)[0],
            'processed_query': query,
            'optimized_queries': [query],
            'hyde_document': '',
            'query_analysis': None
        }
        
        # Enhanced processing with Query Intelligence
        if self.query_intelligence:
            try:
                # Analyze query
                analysis = self.query_intelligence.analyze_query(query)
                result['query_analysis'] = {
                    'complexity': analysis.complexity,
                    'type': analysis.query_type,
                    'sub_questions': analysis.sub_questions,
                    'rewritten_queries': analysis.rewritten_queries
                }
                
                # Get optimized queries for retrieval
                result['optimized_queries'] = self.query_intelligence.get_optimized_queries(query)
                
                # Get HyDE document
                result['hyde_document'] = self.query_intelligence.get_hyde_document(query)
                
                logger.info(f"Enhanced query processing complete - {len(result['optimized_queries'])} variants generated")
            except Exception as e:
                logger.error(f"Enhanced query processing failed: {e}. Using basic processing.")
        
        return result

class EnhancedContextOptimizer:
    """增强的上下文优化器 - 支持智能压缩和重排序"""
    def __init__(self, max_context_length: int = 3000, config: Optional[Dict] = None):
        self.max_context_length = max_context_length
        self.config = config or {}
        
        # Initialize compression and reranking if config provided
        if config:
            try:
                self.contextual_compressor = ContextualCompressor(config)
                self.smart_reranker = SmartReranker(config)
                self.enhanced_mode = True
                logger.info("Enhanced Context Optimizer with compression initialized")
            except Exception as e:
                logger.error(f"Failed to initialize enhanced features: {e}")
                self.enhanced_mode = False
        else:
            self.enhanced_mode = False
            logger.info("Basic Context Optimizer initialized")
    
    def optimize_context(
        self, 
        retrieved_chunks: List[Dict], 
        top_k: int = 3,
        query: Optional[str] = None,
        use_compression: bool = True,
        compression_method: str = "hybrid"
    ) -> Tuple[str, List[Dict]]:
        """优化上下文"""
        
        if not retrieved_chunks:
            return "", []
        
        # Enhanced mode with smart reranking and compression
        if self.enhanced_mode and query:
            # Step 1: Smart reranking
            optimized_chunks = self.smart_reranker.smart_rerank(
                query=query,
                chunks=retrieved_chunks,
                top_k=min(top_k * 2, len(retrieved_chunks))  # Get more candidates
            )
            
            # Step 2: Context compression if enabled
            if use_compression:
                compressed_context = self.contextual_compressor.compress_context(
                    query=query,
                    chunks=optimized_chunks[:top_k],
                    max_context_length=self.max_context_length,
                    compression_method=compression_method
                )
                
                return compressed_context.compressed_content, optimized_chunks[:top_k]
            
            else:
                # Just use reranked chunks without compression
                final_chunks = optimized_chunks[:top_k]
                context_text = "\n\n".join([
                    f"[Source {i+1}]\n{chunk['content']}" 
                    for i, chunk in enumerate(final_chunks)
                ])
                return context_text[:self.max_context_length], final_chunks
        
        # Fallback to basic optimization
        else:
            final_chunks = retrieved_chunks[:top_k]
            context_text = "\n\n".join([
                f"[Source {i+1}]\n{chunk['content']}" 
                for i, chunk in enumerate(final_chunks)
            ])
            return context_text[:self.max_context_length], final_chunks

# Keep compatibility with old ContextOptimizer name
ContextOptimizer = EnhancedContextOptimizer

class LLMGenerator:
    """大语言模型生成器"""
    def __init__(
        self, 
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 1024,
        token: Optional[str] = None  # 新增: 接收token
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading LLM: {model_name} on {self.device}")
        
        # 修改: 在加载时传递token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            token=token  # 修改: 在加载时传递token
        )
        
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens, temperature=0.1, top_p=0.8, do_sample=True,
            repetition_penalty=1.1, pad_token_id=self.tokenizer.eos_token_id
        )
        logger.success("LLM loaded.")
    
    def generate_answer(self, query: str, context: str) -> Tuple[str, int]:
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, generation_config=self.generation_config)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
        return answer, len(outputs[0])

class EnhancedRAGSystem:
    """增强的RAG系统 - 集成查询智能和多策略检索"""
    def __init__(self, config: Dict, db_manager: VectorDatabaseManager, reranker: Optional[AdvancedReranker] = None):
        self.config = config
        self.db_manager = db_manager
        self.reranker = reranker
        
        # Use enhanced query processor
        self.query_processor = EnhancedQueryProcessor(
            embedding_model=config.get('embedding_model'),
            config=config
        )
        self.context_optimizer = ContextOptimizer(config.get('max_context_length'), config)
        
        # Initialize LLM Generator
        self.llm_generator = LLMGenerator(
            model_name=config.get('llm_model'),
            device=config.get('device'),
            token=config.get('HUGGING_FACE_TOKEN')
        )
        
        # Initialize Agentic RAG Orchestrator
        self.agentic_orchestrator = AgenticRAGOrchestrator(
            db_manager=self.db_manager,
            query_processor=self.query_processor,
            context_optimizer=self.context_optimizer,
            llm_generator=self.llm_generator,
            config=config
        )
        
        logger.success("Enhanced RAG System with Agentic capabilities Initialized.")

    async def generate_answer(self, user_query: str, **kwargs) -> GenerationResult:
        import time
        start_time = time.time()
        
        # Enhanced query processing
        query_info = self.query_processor.process_query(user_query)
        retrieval_strategies = []
        all_retrieved_chunks = []
        
        # Strategy 1: Multi-query retrieval
        if query_info['optimized_queries']:
            retrieval_strategies.append('multi_query')
            for i, opt_query in enumerate(query_info['optimized_queries'][:3]):  # Limit to top 3
                query_vector = self.query_processor.embedder.encode([opt_query], convert_to_numpy=True)[0]
                chunks = self.db_manager.search(
                    query_vector=query_vector,
                    query_text=opt_query,
                    top_k=kwargs.get('initial_retrieve', 15) // len(query_info['optimized_queries'])
                )
                
                # Add source information
                for chunk in chunks:
                    chunk['retrieval_source'] = f'query_variant_{i}'
                    chunk['source_query'] = opt_query
                all_retrieved_chunks.extend(chunks)
        
        # Strategy 2: HyDE retrieval if available
        if query_info['hyde_document']:
            retrieval_strategies.append('hyde')
            hyde_vector = self.query_processor.embedder.encode([query_info['hyde_document']], convert_to_numpy=True)[0]
            hyde_chunks = self.db_manager.search(
                query_vector=hyde_vector,
                query_text=query_info['hyde_document'],
                top_k=kwargs.get('initial_retrieve', 15) // 2
            )
            
            # Add source information
            for chunk in hyde_chunks:
                chunk['retrieval_source'] = 'hyde'
                chunk['source_query'] = query_info['hyde_document'][:100] + '...'
            all_retrieved_chunks.extend(hyde_chunks)
        
        # Fallback: Standard retrieval
        if not all_retrieved_chunks:
            retrieval_strategies.append('standard')
            all_retrieved_chunks = self.db_manager.search(
                query_vector=query_info['query_vector'],
                query_text=query_info['processed_query'],
                top_k=kwargs.get('initial_retrieve', 15)
            )
            for chunk in all_retrieved_chunks:
                chunk['retrieval_source'] = 'standard'
                chunk['source_query'] = user_query

        if not all_retrieved_chunks:
            return GenerationResult(
                answer="No relevant info found.", 
                source_chunks=[], 
                confidence=0.0, 
                generation_time=time.time() - start_time, 
                token_count=0,
                query_analysis=query_info.get('query_analysis'),
                retrieval_strategies=retrieval_strategies
            )
        
        # Remove duplicates based on content similarity
        unique_chunks = self._deduplicate_chunks(all_retrieved_chunks)
        logger.info(f"Retrieved {len(all_retrieved_chunks)} total chunks, {len(unique_chunks)} unique")
        
        # Reranking
        final_chunks_for_context = unique_chunks
        if kwargs.get('use_reranker') and self.reranker:
            reranked_results = self.reranker.rerank_with_mmr(
                query=user_query, 
                retrieved_chunks=unique_chunks, 
                top_k=kwargs.get('top_k', 5), 
                lambda_mult=kwargs.get('lambda_mult', 0.5)
            )
            reranked_map = {res.chunk_id: res for res in reranked_results}
            final_chunks_for_context = [chunk for chunk in unique_chunks if chunk['chunk_id'] in reranked_map]
            for chunk in final_chunks_for_context:
                chunk['scores'] = chunk.get('scores', {})
                chunk['scores']['rerank_score'] = reranked_map[chunk['chunk_id']].rerank_score
            final_chunks_for_context.sort(key=lambda c: c['scores']['rerank_score'], reverse=True)
        else:
            final_chunks_for_context = unique_chunks[:kwargs.get('top_k', 5)]
        
        # Context optimization
        optimized_context, final_chunks = self.context_optimizer.optimize_context(
            final_chunks_for_context, 
            top_k=kwargs.get('context_chunks', 3),
            query=user_query,
            use_compression=kwargs.get('use_compression', True),
            compression_method=kwargs.get('compression_method', 'hybrid')
        )
        
        # Answer generation
        answer, token_count = self.llm_generator.generate_answer(
            query=user_query, 
            context=optimized_context
        )
        
        return GenerationResult(
            answer=answer, 
            source_chunks=final_chunks, 
            confidence=self._calculate_confidence(final_chunks, query_info),
            generation_time=time.time() - start_time, 
            token_count=token_count,
            query_analysis=query_info.get('query_analysis'),
            retrieval_strategies=retrieval_strategies
        )
    
    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """去重文档块，基于内容相似度"""
        if not chunks:
            return []
        
        unique_chunks = []
        seen_contents = set()
        
        for chunk in chunks:
            content_hash = hash(chunk['content'][:200])  # Use first 200 chars for deduplication
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _calculate_confidence(self, final_chunks: List[Dict], query_info: Dict) -> float:
        """计算答案置信度"""
        if not final_chunks:
            return 0.0
        
        # Base confidence from retrieval scores
        base_confidence = 0.5
        
        # Boost confidence if we have high-scoring chunks
        if final_chunks and 'scores' in final_chunks[0]:
            max_score = max(chunk.get('scores', {}).get('hybrid_score', 0) for chunk in final_chunks)
            base_confidence += min(max_score * 0.3, 0.3)
        
        # Boost confidence if we have multiple retrieval strategies
        retrieval_sources = set(chunk.get('retrieval_source', 'standard') for chunk in final_chunks)
        if len(retrieval_sources) > 1:
            base_confidence += 0.1
        
        # Boost confidence based on query complexity analysis
        if query_info.get('query_analysis'):
            complexity = query_info['query_analysis'].get('complexity', 'simple')
            if complexity == 'simple':
                base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    async def generate_answer_agentic(self, user_query: str, **kwargs) -> GenerationResult:
        """智能体增强的生成方法"""
        import time
        start_time = time.time()
        
        # Use agentic orchestrator for enhanced retrieval and generation
        answer, final_chunks, agentic_steps, confidence = await self.agentic_orchestrator.agentic_retrieve_and_generate(
            user_query, **kwargs
        )
        
        # Get query analysis from enhanced processor
        query_info = self.query_processor.process_query(user_query)
        
        return GenerationResult(
            answer=answer,
            source_chunks=final_chunks,
            confidence=confidence,
            generation_time=time.time() - start_time,
            token_count=len(answer.split()),  # Rough estimate
            query_analysis=query_info.get('query_analysis'),
            retrieval_strategies=['agentic'],
            agentic_steps=agentic_steps,
            iterations_used=len(agentic_steps)
        )

# Keep compatibility with old RAGSystem name
RAGSystem = EnhancedRAGSystem

