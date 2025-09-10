# src/generation/ultimate_rag_system.py
from typing import List, Dict, Optional, Tuple, Any
import asyncio
import time
import uuid
from dataclasses import dataclass
from loguru import logger

from src.retrieval.vector_database import VectorDatabaseManager
from src.retrieval.reranker import AdvancedReranker
from src.retrieval.query_intelligence import QueryIntelligenceEngine
from src.retrieval.agentic_rag import AgenticRAGOrchestrator, AgenticStep
from src.retrieval.contextual_compression import ContextualCompressor, SmartReranker
from src.knowledge_graph import KnowledgeGraphIndexer, KnowledgeGraphRetriever, KGEnhancedRetriever
from src.generation.tiered_generation import TieredGenerationSystem, TaskRequest, TaskType, TaskComplexity
from src.feedback import FeedbackCollector
from .rag_generator import GenerationResult, EnhancedContextOptimizer

@dataclass
class UltimateGenerationResult:
    """终极RAG系统生成结果"""
    answer: str
    source_chunks: List[Dict]
    confidence: float
    generation_time: float
    token_count: int
    
    # 查询智能信息
    query_analysis: Optional[Dict] = None
    optimized_queries: Optional[List[str]] = None
    hyde_document: Optional[str] = None
    
    # 检索策略信息
    retrieval_strategies: Optional[List[str]] = None
    agentic_steps: Optional[List[AgenticStep]] = None
    iterations_used: Optional[int] = None
    
    # 知识图谱信息
    kg_entities: Optional[List[str]] = None
    kg_relations: Optional[List[Dict]] = None
    kg_enhanced_chunks: Optional[int] = None
    
    # 分层生成信息
    models_used: Optional[Dict[str, float]] = None
    task_breakdown: Optional[List[Dict]] = None
    total_cost: Optional[float] = None
    
    # 性能指标
    retrieval_precision: Optional[float] = None
    context_compression_ratio: Optional[float] = None
    system_version: str = "Ultimate-RAG-v1.0"

class UltimateRAGSystem:
    """终极RAG系统 - 集成所有先进功能"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.system_id = str(uuid.uuid4())[:8]
        
        logger.info(f"🚀 Initializing Ultimate RAG System [{self.system_id}]...")
        
        # 1. 初始化核心组件
        self.db_manager = VectorDatabaseManager(config)
        self.reranker = AdvancedReranker() if config.get('use_reranker', True) else None
        
        # 2. 初始化查询智能
        self.query_intelligence = QueryIntelligenceEngine(config)
        
        # 3. 初始化知识图谱（如果启用）
        self.kg_enabled = config.get('enable_knowledge_graph', False)
        if self.kg_enabled:
            try:
                self.kg_indexer = KnowledgeGraphIndexer(config)
                self.kg_retriever = KnowledgeGraphRetriever(config)
                self.kg_enhanced_retriever = KGEnhancedRetriever(
                    self.db_manager, self.kg_retriever
                )
                logger.info("✅ Knowledge Graph components initialized")
            except Exception as e:
                logger.error(f"❌ Knowledge Graph initialization failed: {e}")
                self.kg_enabled = False
        
        # 4. 初始化上下文优化器
        self.context_optimizer = EnhancedContextOptimizer(
            max_context_length=config.get('max_context_length', 3000),
            config=config
        )
        
        # 5. 初始化分层生成系统
        self.tiered_generation_enabled = config.get('enable_tiered_generation', False)
        if self.tiered_generation_enabled:
            try:
                self.tiered_generator = TieredGenerationSystem(config)
                logger.info("✅ Tiered Generation System initialized")
            except Exception as e:
                logger.error(f"❌ Tiered Generation initialization failed: {e}")
                self.tiered_generation_enabled = False
        
        # 6. 初始化智能体RAG
        self.agentic_orchestrator = AgenticRAGOrchestrator(
            db_manager=self.db_manager,
            query_processor=None,  # 将在下面设置
            context_optimizer=self.context_optimizer,
            llm_generator=None,  # 使用分层生成系统
            config=config
        )
        
        # 7. 初始化反馈收集器
        self.feedback_enabled = config.get('enable_feedback_collection', False)
        if self.feedback_enabled:
            feedback_db_path = config.get('feedback_db_path', 'data/feedback/feedback.db')
            self.feedback_collector = FeedbackCollector(feedback_db_path)
        
        # 系统统计
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'avg_response_time': 0.0,
            'total_cost': 0.0,
            'kg_enhanced_queries': 0,
            'agentic_queries': 0,
            'tiered_generation_usage': {}
        }
        
        logger.success(f"🎉 Ultimate RAG System [{self.system_id}] initialized successfully!")
        self._log_system_capabilities()
    
    def _log_system_capabilities(self):
        """记录系统能力"""
        capabilities = [
            "✓ Query Intelligence (rewriting, sub-questions, HyDE)",
            "✓ Multi-representation Indexing",
            "✓ Agentic RAG (retrieval-evaluation-correction loop)",
            "✓ Contextual Compression & Smart Reranking"
        ]
        
        if self.kg_enabled:
            capabilities.append("✓ Knowledge Graph Indexing & Retrieval")
        if self.tiered_generation_enabled:
            capabilities.append("✓ Tiered Generation with Task Routing")
        if self.feedback_enabled:
            capabilities.append("✓ Human-in-the-loop Feedback System")
        
        logger.info("🔧 System Capabilities:")
        for cap in capabilities:
            logger.info(f"   {cap}")
    
    async def generate_answer(
        self, 
        user_query: str, 
        mode: str = "ultimate",  # "basic", "enhanced", "agentic", "ultimate"
        **kwargs
    ) -> UltimateGenerationResult:
        """生成答案 - 支持多种模式"""
        
        start_time = time.time()
        session_id = str(uuid.uuid4())[:8]
        
        logger.info(f"🔍 Query [{session_id}]: {user_query[:100]}...")
        logger.info(f"🎯 Mode: {mode}")
        
        # 更新统计
        self.stats['total_queries'] += 1
        
        try:
            if mode == "basic":
                result = await self._basic_generation(user_query, session_id, **kwargs)
            elif mode == "enhanced":
                result = await self._enhanced_generation(user_query, session_id, **kwargs)
            elif mode == "agentic":
                result = await self._agentic_generation(user_query, session_id, **kwargs)
            elif mode == "ultimate":
                result = await self._ultimate_generation(user_query, session_id, **kwargs)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # 更新统计
            self.stats['successful_queries'] += 1
            self._update_stats(result)
            
            logger.success(f"✅ Query [{session_id}] completed in {result.generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Query [{session_id}] failed: {e}")
            
            # 返回错误结果
            return UltimateGenerationResult(
                answer=f"抱歉，处理您的查询时遇到错误：{str(e)}",
                source_chunks=[],
                confidence=0.0,
                generation_time=time.time() - start_time,
                token_count=0,
                system_version="Ultimate-RAG-v1.0"
            )
    
    async def _basic_generation(
        self, 
        query: str, 
        session_id: str, 
        **kwargs
    ) -> UltimateGenerationResult:
        """基础生成模式"""
        
        # 简单向量检索
        query_vector = self.query_intelligence.embedder.encode([query], convert_to_numpy=True)[0]
        chunks = self.db_manager.search(
            query_vector=query_vector,
            query_text=query,
            top_k=kwargs.get('top_k', 5)
        )
        
        # 基础上下文优化
        context, final_chunks = self.context_optimizer.optimize_context(
            chunks, top_k=kwargs.get('context_chunks', 3)
        )
        
        # 使用分层生成或回退到基础生成
        if self.tiered_generation_enabled:
            answer, models_used, cost = await self._generate_with_tiered_system(
                query, context, TaskComplexity.SIMPLE
            )
        else:
            answer = f"基于检索到的信息：{context[:500]}...\n\n回答：这是一个基础回答模式的示例。"
            models_used = {"basic_fallback": 0.1}
            cost = 0.0
        
        return UltimateGenerationResult(
            answer=answer,
            source_chunks=final_chunks,
            confidence=0.6,
            generation_time=time.time() - self.stats.get('_start_time', time.time()),
            token_count=len(answer.split()),
            retrieval_strategies=["vector_search"],
            models_used=models_used,
            total_cost=cost
        )
    
    async def _enhanced_generation(
        self, 
        query: str, 
        session_id: str, 
        **kwargs
    ) -> UltimateGenerationResult:
        """增强生成模式"""
        
        # 1. 查询智能处理
        analysis = self.query_intelligence.analyze_query(query)
        optimized_queries = self.query_intelligence.get_optimized_queries(query)
        hyde_doc = self.query_intelligence.get_hyde_document(query)
        
        # 2. 多策略检索
        all_chunks = []
        retrieval_strategies = []
        
        # 向量检索
        for opt_query in optimized_queries[:3]:
            query_vector = self.query_intelligence.embedder.encode([opt_query], convert_to_numpy=True)[0]
            chunks = self.db_manager.search(query_vector=query_vector, query_text=opt_query, top_k=5)
            all_chunks.extend(chunks)
            retrieval_strategies.append(f"vector_query_{len(retrieval_strategies)}")
        
        # HyDE检索
        if hyde_doc:
            hyde_vector = self.query_intelligence.embedder.encode([hyde_doc], convert_to_numpy=True)[0]
            hyde_chunks = self.db_manager.search(query_vector=hyde_vector, query_text=hyde_doc, top_k=5)
            all_chunks.extend(hyde_chunks)
            retrieval_strategies.append("hyde")
        
        # 知识图谱增强（如果启用）
        kg_entities = []
        kg_relations = []
        kg_enhanced_count = 0
        
        if self.kg_enabled:
            all_chunks = self.kg_retriever.enhance_chunks_with_kg(query, all_chunks)
            kg_enhanced_count = sum(1 for c in all_chunks if c.get('has_kg_enhancement', False))
            retrieval_strategies.append("kg_enhanced")
            
            # 提取KG信息
            kg_results = self.kg_retriever.retrieve_kg_context(query)
            kg_entities = [r.entity for r in kg_results if r.source_type == 'entity']
            kg_relations = [
                {'source': r.metadata.get('source_entity'), 'relation': r.metadata.get('relation_type'), 'target': r.entity}
                for r in kg_results if r.source_type == 'relation'
            ]
        
        # 3. 去重和重排序
        unique_chunks = self._deduplicate_chunks(all_chunks)
        
        # 4. 智能上下文优化
        context, final_chunks = self.context_optimizer.optimize_context(
            unique_chunks,
            top_k=kwargs.get('context_chunks', 5),
            query=query,
            use_compression=kwargs.get('use_compression', True)
        )
        
        # 5. 分层生成
        if self.tiered_generation_enabled:
            complexity = self._determine_task_complexity(query, analysis)
            answer, models_used, cost = await self._generate_with_tiered_system(
                query, context, complexity
            )
        else:
            answer = f"增强模式回答基于多策略检索和智能处理：\n\n{context[:800]}"
            models_used = {"enhanced_fallback": 0.2}
            cost = 0.0
        
        return UltimateGenerationResult(
            answer=answer,
            source_chunks=final_chunks,
            confidence=self._calculate_enhanced_confidence(final_chunks, analysis),
            generation_time=time.time() - self.stats.get('_start_time', time.time()),
            token_count=len(answer.split()),
            query_analysis=analysis.__dict__ if hasattr(analysis, '__dict__') else None,
            optimized_queries=optimized_queries,
            hyde_document=hyde_doc,
            retrieval_strategies=retrieval_strategies,
            kg_entities=kg_entities,
            kg_relations=kg_relations,
            kg_enhanced_chunks=kg_enhanced_count,
            models_used=models_used,
            total_cost=cost
        )
    
    async def _agentic_generation(
        self, 
        query: str, 
        session_id: str, 
        **kwargs
    ) -> UltimateGenerationResult:
        """智能体生成模式"""
        
        self.stats['agentic_queries'] += 1
        
        # 使用智能体RAG系统
        answer, final_chunks, agentic_steps, confidence = await self.agentic_orchestrator.agentic_retrieve_and_generate(
            query, **kwargs
        )
        
        # 分析智能体步骤
        retrieval_strategies = ["agentic"]
        models_used = {}
        total_cost = 0.0
        
        for step in agentic_steps:
            if hasattr(step, 'evaluation') and hasattr(step.evaluation, 'reasoning'):
                models_used[f"evaluator_step_{step.step_number}"] = 0.1
                total_cost += 0.001  # 估算成本
        
        return UltimateGenerationResult(
            answer=answer,
            source_chunks=final_chunks,
            confidence=confidence,
            generation_time=time.time() - self.stats.get('_start_time', time.time()),
            token_count=len(answer.split()),
            retrieval_strategies=retrieval_strategies,
            agentic_steps=agentic_steps,
            iterations_used=len(agentic_steps),
            models_used=models_used,
            total_cost=total_cost
        )
    
    async def _ultimate_generation(
        self, 
        query: str, 
        session_id: str, 
        **kwargs
    ) -> UltimateGenerationResult:
        """终极生成模式 - 集成所有功能"""
        
        # 1. 查询智能分析
        analysis = self.query_intelligence.analyze_query(query)
        complexity = self._determine_task_complexity(query, analysis)
        
        logger.info(f"📊 Query Analysis: complexity={complexity.value}, type={analysis.query_type}")
        
        # 2. 选择最佳策略
        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL] and len(analysis.sub_questions) > 1:
            # 复杂查询使用智能体模式
            logger.info("🤖 Using Agentic RAG for complex query")
            return await self._agentic_generation(query, session_id, **kwargs)
        else:
            # 中等复杂度使用增强模式 + 额外优化
            logger.info("⚡ Using Ultimate Enhanced mode")
            
            # 基于增强模式，增加额外的优化
            enhanced_result = await self._enhanced_generation(query, session_id, **kwargs)
            
            # Ultimate模式的额外处理
            if self.kg_enabled and enhanced_result.kg_enhanced_chunks == 0:
                # 如果没有KG增强，尝试更深度的KG检索
                logger.info("🔍 Attempting deeper KG retrieval")
                deeper_kg = await self._deep_kg_analysis(query)
                if deeper_kg:
                    enhanced_result.kg_entities = deeper_kg.get('entities', [])
                    enhanced_result.kg_relations = deeper_kg.get('relations', [])
            
            # 质量验证和置信度调整
            enhanced_result.confidence = self._calculate_ultimate_confidence(enhanced_result)
            
            # 添加系统性能指标
            enhanced_result.retrieval_precision = self._calculate_retrieval_precision(enhanced_result.source_chunks)
            enhanced_result.context_compression_ratio = self._calculate_compression_ratio(enhanced_result.source_chunks)
            
            return enhanced_result
    
    async def _generate_with_tiered_system(
        self, 
        query: str, 
        context: str, 
        complexity: TaskComplexity
    ) -> Tuple[str, Dict[str, float], float]:
        """使用分层生成系统生成答案"""
        
        if not self.tiered_generation_enabled:
            return f"基于上下文的回答：{context[:500]}...", {"fallback": 0.1}, 0.0
        
        # 创建生成任务
        task = TaskRequest(
            task_id=str(uuid.uuid4())[:8],
            task_type=TaskType.FINAL_GENERATION,
            complexity=complexity,
            prompt=f"基于以下上下文回答问题。\n\n上下文：{context}\n\n问题：{query}\n\n答案：",
            context={"query": query, "context": context},
            max_tokens=1024,
            priority=3 if complexity in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL] else 1
        )
        
        # 执行任务
        response = await self.tiered_generator.execute_task(task)
        
        models_used = {response.model_used: response.execution_time}
        total_cost = response.cost
        
        return response.result, models_used, total_cost
    
    def _determine_task_complexity(self, query: str, analysis) -> TaskComplexity:
        """确定任务复杂度"""
        if hasattr(analysis, 'complexity'):
            complexity_map = {
                'simple': TaskComplexity.SIMPLE,
                'medium': TaskComplexity.MEDIUM,
                'complex': TaskComplexity.COMPLEX
            }
            return complexity_map.get(analysis.complexity, TaskComplexity.MEDIUM)
        
        # 基于查询长度的简单判断
        if len(query.split()) < 5:
            return TaskComplexity.SIMPLE
        elif len(query.split()) < 15:
            return TaskComplexity.MEDIUM
        else:
            return TaskComplexity.COMPLEX
    
    async def _deep_kg_analysis(self, query: str) -> Optional[Dict]:
        """深度知识图谱分析"""
        if not self.kg_enabled:
            return None
        
        try:
            # 获取实体邻域信息
            kg_results = self.kg_retriever.retrieve_kg_context(query, top_k=20)
            
            entities = []
            relations = []
            
            for result in kg_results:
                if result.source_type == 'entity':
                    entities.append(result.entity)
                elif result.source_type == 'relation':
                    relations.append({
                        'source': result.metadata.get('source_entity'),
                        'relation': result.metadata.get('relation_type'),
                        'target': result.entity
                    })
            
            return {'entities': entities, 'relations': relations}
            
        except Exception as e:
            logger.error(f"Deep KG analysis failed: {e}")
            return None
    
    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """去重文档块"""
        if not chunks:
            return []
        
        unique_chunks = []
        seen_contents = set()
        
        for chunk in chunks:
            content_hash = hash(chunk['content'][:200])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _calculate_enhanced_confidence(self, chunks: List[Dict], analysis) -> float:
        """计算增强置信度"""
        base_confidence = 0.5
        
        # 基于检索结果数量
        if chunks:
            base_confidence += min(len(chunks) * 0.1, 0.3)
        
        # 基于查询复杂度匹配
        if hasattr(analysis, 'complexity'):
            if analysis.complexity == 'simple':
                base_confidence += 0.1
        
        # 基于知识图谱增强
        kg_enhanced = sum(1 for c in chunks if c.get('has_kg_enhancement', False))
        if kg_enhanced > 0:
            base_confidence += min(kg_enhanced * 0.05, 0.15)
        
        return min(base_confidence, 1.0)
    
    def _calculate_ultimate_confidence(self, result: UltimateGenerationResult) -> float:
        """计算终极置信度"""
        confidence = result.confidence
        
        # KG增强加分
        if result.kg_entities:
            confidence += len(result.kg_entities) * 0.02
        
        # 多策略检索加分
        if result.retrieval_strategies and len(result.retrieval_strategies) > 1:
            confidence += 0.05
        
        # 智能体迭代加分
        if result.iterations_used and result.iterations_used > 1:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_retrieval_precision(self, chunks: List[Dict]) -> float:
        """计算检索精确度"""
        if not chunks:
            return 0.0
        
        # 基于分数分布的简单精确度估算
        scores = []
        for chunk in chunks:
            chunk_scores = chunk.get('scores', {})
            if 'hybrid_score' in chunk_scores:
                scores.append(chunk_scores['hybrid_score'])
            elif 'vector_score' in chunk_scores:
                scores.append(chunk_scores['vector_score'])
        
        if scores:
            return sum(scores) / len(scores)
        
        return 0.5  # 默认值
    
    def _calculate_compression_ratio(self, chunks: List[Dict]) -> float:
        """计算压缩比"""
        if not chunks:
            return 1.0
        
        # 简单估算：基于块数量的压缩效果
        original_count = len(chunks) * 500  # 假设每块500字符
        compressed_count = sum(len(chunk.get('content', '')) for chunk in chunks[:3])  # 实际使用的前3块
        
        if original_count > 0:
            return compressed_count / original_count
        
        return 1.0
    
    def _update_stats(self, result: UltimateGenerationResult):
        """更新系统统计"""
        # 更新平均响应时间
        total_time = self.stats['avg_response_time'] * (self.stats['total_queries'] - 1)
        self.stats['avg_response_time'] = (total_time + result.generation_time) / self.stats['total_queries']
        
        # 更新成本
        if result.total_cost:
            self.stats['total_cost'] += result.total_cost
        
        # 更新KG使用统计
        if result.kg_enhanced_chunks and result.kg_enhanced_chunks > 0:
            self.stats['kg_enhanced_queries'] += 1
        
        # 更新分层生成使用统计
        if result.models_used:
            for model_name in result.models_used:
                if model_name not in self.stats['tiered_generation_usage']:
                    self.stats['tiered_generation_usage'][model_name] = 0
                self.stats['tiered_generation_usage'][model_name] += 1
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'system_id': self.system_id,
            'version': 'Ultimate-RAG-v1.0',
            'capabilities': {
                'query_intelligence': True,
                'knowledge_graph': self.kg_enabled,
                'tiered_generation': self.tiered_generation_enabled,
                'agentic_rag': True,
                'feedback_collection': self.feedback_enabled
            },
            'statistics': self.stats.copy(),
            'performance': {
                'success_rate': self.stats['successful_queries'] / max(self.stats['total_queries'], 1),
                'avg_response_time': self.stats['avg_response_time'],
                'total_cost': self.stats['total_cost']
            }
        }
    
    async def collect_feedback(
        self, 
        query: str, 
        result: UltimateGenerationResult, 
        is_positive: bool,
        comment: Optional[str] = None
    ) -> Optional[str]:
        """收集用户反馈"""
        if not self.feedback_enabled:
            return None
        
        feedback_id = self.feedback_collector.collect_thumbs_feedback(
            query=query,
            answer=result.answer,
            is_positive=is_positive,
            source_chunks=result.source_chunks,
            generation_result={
                'query_analysis': result.query_analysis,
                'retrieval_strategies': result.retrieval_strategies,
                'generation_time': result.generation_time,
                'confidence': result.confidence,
                'models_used': result.models_used,
                'total_cost': result.total_cost
            }
        )
        
        logger.info(f"📝 Feedback collected: {feedback_id} ({'positive' if is_positive else 'negative'})")
        return feedback_id

# 使用示例
async def main():
    """测试终极RAG系统"""
    config = {
        'llm_model': 'Qwen/Qwen2-7B-Instruct',
        'embedding_model': 'BAAI/bge-m3',
        'device': 'auto',
        'HUGGING_FACE_TOKEN': None,
        
        # 功能开关
        'enable_knowledge_graph': True,
        'enable_tiered_generation': True,
        'enable_feedback_collection': True,
        
        # 路径配置
        'knowledge_graph_db_path': 'data/knowledge_graph/kg.db',
        'feedback_db_path': 'data/feedback/feedback.db',
        
        # API模型配置（可选）
        'api_models': {
            # 'gpt4_api_key': 'your-key-here'
        }
    }
    
    # 初始化系统
    ultimate_rag = UltimateRAGSystem(config)
    
    # 测试不同模式
    test_queries = [
        ("什么是Transformer模型？", "basic"),
        ("请详细比较BERT和GPT模型的异同点", "enhanced"), 
        ("分析深度学习在计算机视觉和自然语言处理中的不同应用及其技术挑战", "ultimate")
    ]
    
    for query, mode in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Mode: {mode}")
        print('='*80)
        
        result = await ultimate_rag.generate_answer(query, mode=mode)
        
        print(f"Answer: {result.answer[:200]}...")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Generation Time: {result.generation_time:.2f}s")
        print(f"Retrieval Strategies: {result.retrieval_strategies}")
        print(f"Models Used: {result.models_used}")
        
        if result.kg_entities:
            print(f"KG Entities: {result.kg_entities[:5]}")
        
        if result.total_cost:
            print(f"Cost: ${result.total_cost:.4f}")
    
    # 系统状态
    status = ultimate_rag.get_system_status()
    print(f"\n📊 System Status:")
    print(f"Success Rate: {status['performance']['success_rate']:.3f}")
    print(f"Avg Response Time: {status['performance']['avg_response_time']:.2f}s")
    print(f"Total Queries: {status['statistics']['total_queries']}")

if __name__ == "__main__":
    asyncio.run(main())