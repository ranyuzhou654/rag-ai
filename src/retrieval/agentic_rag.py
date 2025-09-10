# src/retrieval/agentic_rag.py
from typing import List, Dict, Optional, Tuple, Any
import asyncio
from dataclasses import dataclass
from enum import Enum
import time
import re
from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

class RetrievalDecision(Enum):
    """检索决策类型"""
    PROCEED = "proceed"           # 继续生成答案
    RETRY = "retry"              # 重新检索
    EXPAND_QUERY = "expand_query" # 扩展查询
    SEEK_MORE = "seek_more"      # 寻找更多信息
    INSUFFICIENT = "insufficient" # 信息不足，无法回答

@dataclass 
class RetrievalEvaluation:
    """检索评估结果"""
    decision: RetrievalDecision
    confidence: float
    reasoning: str
    suggested_query: Optional[str] = None
    missing_aspects: List[str] = None
    contradictions: List[str] = None

@dataclass
class AgenticStep:
    """智能体步骤记录"""
    step_type: str
    query: str
    retrieved_chunks: List[Dict]
    evaluation: RetrievalEvaluation
    timestamp: float
    step_number: int

class RetrievalEvaluator:
    """检索质量评估器"""
    
    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        logger.info(f"Loading Retrieval Evaluator: {model_name}")
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
            max_new_tokens=400,
            temperature=0.2,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
    
    def evaluate_retrieval(
        self, 
        query: str, 
        retrieved_chunks: List[Dict],
        min_chunks_threshold: int = 2
    ) -> RetrievalEvaluation:
        """评估检索结果质量"""
        
        # Quick checks first
        if not retrieved_chunks:
            return RetrievalEvaluation(
                decision=RetrievalDecision.RETRY,
                confidence=0.0,
                reasoning="没有检索到任何相关文档",
                suggested_query=None
            )
        
        if len(retrieved_chunks) < min_chunks_threshold:
            return RetrievalEvaluation(
                decision=RetrievalDecision.SEEK_MORE,
                confidence=0.3,
                reasoning=f"检索结果太少（{len(retrieved_chunks)}个），需要更多信息",
                suggested_query=None
            )
        
        # LLM-based evaluation
        return self._llm_evaluate_retrieval(query, retrieved_chunks)
    
    def _llm_evaluate_retrieval(self, query: str, retrieved_chunks: List[Dict]) -> RetrievalEvaluation:
        """使用LLM评估检索质量"""
        
        # Prepare context
        context_summary = "\n".join([
            f"文档{i+1}: {chunk['content'][:200]}..." 
            for i, chunk in enumerate(retrieved_chunks[:5])
        ])
        
        # Detect language
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', query))
        
        if is_chinese:
            prompt = f"""请评估以下检索结果是否能够充分回答用户的问题。

用户问题：{query}

检索到的文档片段：
{context_summary}

请从以下几个角度进行评估：
1. 相关性：检索结果与问题的相关程度（高/中/低）
2. 完整性：信息是否足够完整（完整/部分完整/不完整）
3. 矛盾性：是否存在相互矛盾的信息（无矛盾/有矛盾）
4. 建议：下一步应该采取什么行动

请按以下格式回答：
相关性：[高/中/低]
完整性：[完整/部分完整/不完整]
矛盾性：[无矛盾/有矛盾]
决策：[继续/重新检索/扩展查询/寻找更多]
置信度：[0.0-1.0的数值]
理由：[详细说明]
建议查询：[如果需要重新检索，提供建议的查询词]"""
        else:
            prompt = f"""Please evaluate whether the following retrieval results can adequately answer the user's question.

User question: {query}

Retrieved document chunks:
{context_summary}

Please evaluate from the following perspectives:
1. Relevance: How relevant are the results to the question (High/Medium/Low)
2. Completeness: Is the information sufficient (Complete/Partially Complete/Incomplete)
3. Contradictions: Are there contradictory information (No Contradictions/Has Contradictions)
4. Recommendation: What action should be taken next

Please answer in the following format:
Relevance: [High/Medium/Low]
Completeness: [Complete/Partially Complete/Incomplete]  
Contradictions: [No Contradictions/Has Contradictions]
Decision: [Proceed/Retry/Expand Query/Seek More]
Confidence: [0.0-1.0 numerical value]
Reasoning: [Detailed explanation]
Suggested Query: [If retry is needed, provide suggested query terms]"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract evaluation from response
            if is_chinese:
                eval_part = response.split("请按以下格式回答：")[-1].strip()
            else:
                eval_part = response.split("Please answer in the following format:")[-1].strip()
            
            return self._parse_evaluation_response(eval_part, is_chinese)
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            # Fallback evaluation based on simple heuristics
            return self._fallback_evaluation(query, retrieved_chunks)
    
    def _parse_evaluation_response(self, response: str, is_chinese: bool) -> RetrievalEvaluation:
        """解析LLM评估响应"""
        try:
            lines = response.split('\n')
            
            decision_mapping = {
                '继续': RetrievalDecision.PROCEED,
                '重新检索': RetrievalDecision.RETRY,  
                '扩展查询': RetrievalDecision.EXPAND_QUERY,
                '寻找更多': RetrievalDecision.SEEK_MORE,
                'proceed': RetrievalDecision.PROCEED,
                'retry': RetrievalDecision.RETRY,
                'expand query': RetrievalDecision.EXPAND_QUERY,
                'seek more': RetrievalDecision.SEEK_MORE
            }
            
            decision = RetrievalDecision.PROCEED
            confidence = 0.5
            reasoning = "LLM evaluation completed"
            suggested_query = None
            
            for line in lines:
                line = line.strip()
                if is_chinese:
                    if line.startswith('决策：'):
                        decision_text = line.split('：')[1].strip().lower()
                        decision = decision_mapping.get(decision_text, RetrievalDecision.PROCEED)
                    elif line.startswith('置信度：'):
                        try:
                            confidence = float(line.split('：')[1].strip())
                        except:
                            confidence = 0.5
                    elif line.startswith('理由：'):
                        reasoning = line.split('：')[1].strip()
                    elif line.startswith('建议查询：'):
                        suggested_query = line.split('：')[1].strip()
                else:
                    if line.startswith('Decision:'):
                        decision_text = line.split(':')[1].strip().lower()
                        decision = decision_mapping.get(decision_text, RetrievalDecision.PROCEED)
                    elif line.startswith('Confidence:'):
                        try:
                            confidence = float(line.split(':')[1].strip())
                        except:
                            confidence = 0.5
                    elif line.startswith('Reasoning:'):
                        reasoning = line.split(':', 1)[1].strip()
                    elif line.startswith('Suggested Query:'):
                        suggested_query = line.split(':', 1)[1].strip()
            
            return RetrievalEvaluation(
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                suggested_query=suggested_query if suggested_query != "无" and suggested_query != "None" else None
            )
            
        except Exception as e:
            logger.error(f"Failed to parse evaluation response: {e}")
            return self._fallback_evaluation("", [])
    
    def _fallback_evaluation(self, query: str, retrieved_chunks: List[Dict]) -> RetrievalEvaluation:
        """后备评估逻辑"""
        if not retrieved_chunks:
            return RetrievalEvaluation(
                decision=RetrievalDecision.RETRY,
                confidence=0.0,
                reasoning="No retrieved chunks available"
            )
        
        # Simple heuristic based on scores
        avg_score = 0.0
        if retrieved_chunks:
            scores = []
            for chunk in retrieved_chunks:
                chunk_scores = chunk.get('scores', {})
                if 'hybrid_score' in chunk_scores:
                    scores.append(chunk_scores['hybrid_score'])
                elif 'vector_score' in chunk_scores:
                    scores.append(chunk_scores['vector_score'])
            
            if scores:
                avg_score = sum(scores) / len(scores)
        
        if avg_score > 0.7:
            decision = RetrievalDecision.PROCEED
            confidence = min(avg_score, 1.0)
        elif avg_score > 0.4:
            decision = RetrievalDecision.SEEK_MORE
            confidence = avg_score
        else:
            decision = RetrievalDecision.RETRY
            confidence = avg_score
        
        return RetrievalEvaluation(
            decision=decision,
            confidence=confidence,
            reasoning=f"Fallback evaluation based on average score: {avg_score:.3f}"
        )

class QueryRefiner:
    """查询优化器"""
    
    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        logger.info(f"Loading Query Refiner: {model_name}")
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
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
    
    def refine_query(
        self, 
        original_query: str, 
        evaluation: RetrievalEvaluation, 
        retrieved_chunks: List[Dict]
    ) -> str:
        """根据评估结果优化查询"""
        
        if evaluation.suggested_query:
            return evaluation.suggested_query
        
        # Generate refined query based on decision
        return self._generate_refined_query(original_query, evaluation, retrieved_chunks)
    
    def _generate_refined_query(
        self, 
        original_query: str, 
        evaluation: RetrievalEvaluation,
        retrieved_chunks: List[Dict]
    ) -> str:
        """生成优化查询"""
        
        # Detect language
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', original_query))
        
        context_summary = ""
        if retrieved_chunks:
            context_summary = "\n".join([
                f"文档{i+1}: {chunk['content'][:100]}..." 
                for i, chunk in enumerate(retrieved_chunks[:3])
            ])
        
        if is_chinese:
            if evaluation.decision == RetrievalDecision.EXPAND_QUERY:
                prompt = f"""原始查询无法找到足够相关的信息。请基于以下信息生成一个扩展的查询，使用更多同义词和相关术语。

原始查询：{original_query}

已找到的信息片段：
{context_summary}

评估反馈：{evaluation.reasoning}

请生成一个改进的查询（只需要输出查询内容，不要其他解释）："""
            else:
                prompt = f"""原始查询的检索效果不理想。请重新表述查询以获得更好的检索结果。

原始查询：{original_query}

问题：{evaluation.reasoning}

请生成一个重新表述的查询（只需要输出查询内容，不要其他解释）："""
        else:
            if evaluation.decision == RetrievalDecision.EXPAND_QUERY:
                prompt = f"""The original query couldn't find sufficient relevant information. Please generate an expanded query using more synonyms and related terms.

Original query: {original_query}

Retrieved information snippets:
{context_summary}

Evaluation feedback: {evaluation.reasoning}

Please generate an improved query (output only the query content, no other explanations):"""
            else:
                prompt = f"""The original query's retrieval performance was suboptimal. Please rephrase the query to get better retrieval results.

Original query: {original_query}

Issue: {evaluation.reasoning}

Please generate a rephrased query (output only the query content, no other explanations):"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the refined query
            if is_chinese:
                refined_part = response.split("请生成")[-1].strip()
                refined_query = refined_part.split('\n')[0].strip()
            else:
                refined_part = response.split("Please generate")[-1].strip()  
                refined_query = refined_part.split('\n')[0].strip()
            
            # Clean up and validate
            refined_query = refined_query.replace('"', '').replace("：", "").replace(":", "").strip()
            
            return refined_query if len(refined_query) > 5 else original_query
            
        except Exception as e:
            logger.error(f"Query refinement failed: {e}")
            return original_query

class AgenticRAGOrchestrator:
    """智能体RAG协调器"""
    
    def __init__(
        self,
        db_manager,
        query_processor, 
        context_optimizer,
        llm_generator,
        config: Dict
    ):
        self.db_manager = db_manager
        self.query_processor = query_processor
        self.context_optimizer = context_optimizer
        self.llm_generator = llm_generator
        self.config = config
        
        # Initialize agentic components
        model_name = config.get('llm_model')
        token = config.get('HUGGING_FACE_TOKEN')
        device = config.get('device', 'auto')
        
        self.max_iterations = config.get('max_agentic_iterations', 3)
        self.enable_agentic = config.get('enable_agentic_rag', True)
        
        if self.enable_agentic and model_name:
            try:
                self.evaluator = RetrievalEvaluator(
                    model_name=model_name, device=device, token=token
                )
                self.query_refiner = QueryRefiner(
                    model_name=model_name, device=device, token=token
                )
                logger.success("Agentic RAG Orchestrator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize agentic components: {e}")
                self.enable_agentic = False
        else:
            self.enable_agentic = False
            logger.warning("Agentic RAG disabled or LLM model not available")
    
    async def agentic_retrieve_and_generate(
        self, 
        user_query: str, 
        **kwargs
    ) -> Tuple[str, List[Dict], List[AgenticStep], float]:
        """智能体检索生成流程"""
        
        if not self.enable_agentic:
            # Fallback to standard retrieval
            return await self._standard_retrieve_and_generate(user_query, **kwargs)
        
        start_time = time.time()
        steps = []
        current_query = user_query
        
        for iteration in range(self.max_iterations):
            logger.info(f"Agentic iteration {iteration + 1}/{self.max_iterations}")
            
            # Step 1: Retrieve
            query_info = self.query_processor.process_query(current_query)
            retrieved_chunks = self.db_manager.search(
                query_vector=query_info['query_vector'],
                query_text=current_query,
                top_k=kwargs.get('initial_retrieve', 15)
            )
            
            # Step 2: Evaluate
            evaluation = self.evaluator.evaluate_retrieval(
                current_query, 
                retrieved_chunks,
                min_chunks_threshold=kwargs.get('min_chunks_threshold', 2)
            )
            
            # Record step
            step = AgenticStep(
                step_type=f"retrieve_evaluate_{iteration+1}",
                query=current_query,
                retrieved_chunks=retrieved_chunks,
                evaluation=evaluation,
                timestamp=time.time(),
                step_number=iteration + 1
            )
            steps.append(step)
            
            logger.info(f"Evaluation result: {evaluation.decision.value} (confidence: {evaluation.confidence:.3f})")
            
            # Step 3: Decide next action
            if evaluation.decision == RetrievalDecision.PROCEED:
                # Generate answer and return
                optimized_context, final_chunks = self.context_optimizer.optimize_context(
                    retrieved_chunks, top_k=kwargs.get('context_chunks', 3)
                )
                answer, _ = self.llm_generator.generate_answer(
                    query=user_query, context=optimized_context
                )
                
                total_time = time.time() - start_time
                logger.success(f"Agentic RAG completed in {iteration + 1} iterations ({total_time:.2f}s)")
                return answer, final_chunks, steps, evaluation.confidence
                
            elif evaluation.decision in [RetrievalDecision.RETRY, RetrievalDecision.EXPAND_QUERY]:
                # Refine query and retry
                current_query = self.query_refiner.refine_query(
                    user_query, evaluation, retrieved_chunks
                )
                logger.info(f"Refined query: {current_query}")
                
            elif evaluation.decision == RetrievalDecision.SEEK_MORE:
                # Increase search scope
                kwargs['initial_retrieve'] = kwargs.get('initial_retrieve', 15) * 2
                logger.info(f"Expanding search scope to {kwargs['initial_retrieve']}")
                
            else:  # INSUFFICIENT
                # Give up and return best attempt
                if retrieved_chunks:
                    optimized_context, final_chunks = self.context_optimizer.optimize_context(
                        retrieved_chunks, top_k=kwargs.get('context_chunks', 3)
                    )
                    answer, _ = self.llm_generator.generate_answer(
                        query=user_query, context=optimized_context
                    )
                    
                    answer = f"⚠️ 信息不足，以下是基于有限信息的回答：\n\n{answer}"
                else:
                    answer = "抱歉，未能找到足够的相关信息来回答您的问题。"
                    final_chunks = []
                
                total_time = time.time() - start_time
                return answer, final_chunks, steps, 0.2
        
        # Max iterations reached
        logger.warning(f"Agentic RAG reached max iterations ({self.max_iterations})")
        if retrieved_chunks:
            optimized_context, final_chunks = self.context_optimizer.optimize_context(
                retrieved_chunks, top_k=kwargs.get('context_chunks', 3)
            )
            answer, _ = self.llm_generator.generate_answer(
                query=user_query, context=optimized_context
            )
            answer = f"⚠️ 经过多次尝试后的回答：\n\n{answer}"
        else:
            answer = "经过多次检索尝试，仍未找到足够相关的信息。"
            final_chunks = []
        
        total_time = time.time() - start_time
        return answer, final_chunks, steps, 0.4
    
    async def _standard_retrieve_and_generate(
        self, 
        user_query: str, 
        **kwargs
    ) -> Tuple[str, List[Dict], List[AgenticStep], float]:
        """标准检索生成流程（后备）"""
        query_info = self.query_processor.process_query(user_query)
        retrieved_chunks = self.db_manager.search(
            query_vector=query_info['query_vector'],
            query_text=user_query,
            top_k=kwargs.get('initial_retrieve', 15)
        )
        
        if retrieved_chunks:
            optimized_context, final_chunks = self.context_optimizer.optimize_context(
                retrieved_chunks, top_k=kwargs.get('context_chunks', 3)
            )
            answer, _ = self.llm_generator.generate_answer(
                query=user_query, context=optimized_context
            )
            confidence = 0.6
        else:
            answer = "未找到相关信息。"
            final_chunks = []
            confidence = 0.0
        
        # Create a simple step record
        steps = [AgenticStep(
            step_type="standard_retrieve",
            query=user_query,
            retrieved_chunks=retrieved_chunks,
            evaluation=RetrievalEvaluation(
                decision=RetrievalDecision.PROCEED,
                confidence=confidence,
                reasoning="Standard retrieval without evaluation"
            ),
            timestamp=time.time(),
            step_number=1
        )]
        
        return answer, final_chunks, steps, confidence

# 使用示例
async def main():
    """测试Agentic RAG"""
    # This would be integrated into the main RAG system
    pass

if __name__ == "__main__":
    asyncio.run(main())