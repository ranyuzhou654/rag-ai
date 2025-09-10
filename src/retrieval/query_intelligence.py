# src/retrieval/query_intelligence.py
from typing import List, Dict, Optional, Tuple
import asyncio
import re
from dataclasses import dataclass
from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json
from pathlib import Path

@dataclass
class QueryAnalysisResult:
    """查询分析结果"""
    original_query: str
    language: str
    complexity: str  # 'simple', 'medium', 'complex'
    sub_questions: List[str]
    rewritten_queries: List[str]
    hypothetical_document: str
    query_type: str  # 'factual', 'comparative', 'explanatory', 'procedural'

class QueryComplexityAnalyzer:
    """查询复杂度分析器"""
    
    def __init__(self):
        self.complexity_indicators = {
            'simple': [
                r'\b(什么是|what is|define|定义)\b',
                r'\b(who|谁)\b',
                r'\b(when|什么时候)\b',
                r'\b(where|哪里)\b'
            ],
            'medium': [
                r'\b(how|如何|怎么)\b',
                r'\b(why|为什么|为何)\b',
                r'\b(which|哪个|哪种)\b',
                r'\b(compare|比较|对比)\b'
            ],
            'complex': [
                r'\b(analyze|分析|解释)\b.*\b(difference|区别|异同)\b',
                r'\b(evaluate|评估|评价)\b',
                r'\b(explain.*relationship|解释.*关系)\b',
                r'\b(pros and cons|优缺点|利弊)\b',
                r'\b(step by step|步骤|流程)\b',
                r'\band\b.*\bor\b|\b和\b.*\b或\b',  # Multiple concepts
                r'\b(综合|全面|深入|详细)\b.*\b(分析|讨论)\b'
            ]
        }
    
    def analyze_complexity(self, query: str) -> str:
        """分析查询复杂度"""
        query_lower = query.lower()
        
        # Count matches for each complexity level
        scores = {}
        for complexity, patterns in self.complexity_indicators.items():
            scores[complexity] = sum(1 for pattern in patterns if re.search(pattern, query_lower))
        
        # Determine complexity based on scores and query length
        if scores['complex'] > 0 or len(query.split()) > 20:
            return 'complex'
        elif scores['medium'] > 0 or len(query.split()) > 10:
            return 'medium'
        else:
            return 'simple'

class SubQuestionGenerator:
    """子问题生成器"""
    
    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        logger.info(f"Loading SubQuestion Generator: {model_name}")
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
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
    
    def generate_sub_questions(self, query: str, max_questions: int = 5) -> List[str]:
        """生成子问题"""
        # Detect language
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', query))
        
        if is_chinese:
            prompt = f"""请将以下复杂问题分解为3-5个更具体的子问题，每个子问题应该独立且可被单独回答：

原问题：{query}

分解后的子问题：
1."""
        else:
            prompt = f"""Break down the following complex question into 3-5 specific sub-questions that can be answered independently:

Original question: {query}

Sub-questions:
1."""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract sub-questions from response
            if is_chinese:
                response_part = response.split("分解后的子问题：")[-1].strip()
            else:
                response_part = response.split("Sub-questions:")[-1].strip()
            
            # Parse numbered questions
            sub_questions = []
            lines = response_part.split('\n')
            
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    question = re.sub(r'^\d+\.\s*', '', line).strip()
                    if question and len(question) > 10:
                        sub_questions.append(question)
                        if len(sub_questions) >= max_questions:
                            break
            
            logger.info(f"Generated {len(sub_questions)} sub-questions for: {query[:50]}...")
            return sub_questions
            
        except Exception as e:
            logger.error(f"Failed to generate sub-questions: {e}")
            return []

class QueryRewriter:
    """查询重写器 - 优化检索效果"""
    
    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        logger.info(f"Loading Query Rewriter: {model_name}")
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
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
    
    def rewrite_query(self, query: str, num_variations: int = 3) -> List[str]:
        """重写查询以提高检索效果"""
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', query))
        
        if is_chinese:
            prompt = f"""请为以下搜索查询生成{num_variations}个语义相近但表达方式不同的重写版本，使其更适合学术文献检索：

原查询：{query}

重写版本：
1."""
        else:
            prompt = f"""Generate {num_variations} rewritten versions of the following search query that are semantically similar but use different expressions, making them more suitable for academic literature retrieval:

Original query: {query}

Rewritten versions:
1."""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract rewritten queries
            if is_chinese:
                response_part = response.split("重写版本：")[-1].strip()
            else:
                response_part = response.split("Rewritten versions:")[-1].strip()
            
            rewritten_queries = []
            lines = response_part.split('\n')
            
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    rewritten = re.sub(r'^\d+\.\s*', '', line).strip()
                    if rewritten and len(rewritten) > 5:
                        rewritten_queries.append(rewritten)
                        if len(rewritten_queries) >= num_variations:
                            break
            
            # Add original query if we don't have enough variations
            if len(rewritten_queries) < num_variations:
                rewritten_queries.insert(0, query)
            
            logger.info(f"Generated {len(rewritten_queries)} query variations")
            return rewritten_queries[:num_variations]
            
        except Exception as e:
            logger.error(f"Failed to rewrite query: {e}")
            return [query]

class HyDEGenerator:
    """假设性文档嵌入生成器 (Hypothetical Document Embeddings)"""
    
    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.model_name = model_name
        
        logger.info(f"Loading HyDE Generator: {model_name}")
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
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
    
    def generate_hypothetical_document(self, query: str) -> str:
        """生成假设性文档用于改进检索"""
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', query))
        
        if is_chinese:
            prompt = f"""基于以下问题，写一段详细的学术性答案，包含相关的技术术语和概念，就像从高质量的研究论文或技术文档中摘录的内容：

问题：{query}

详细答案："""
        else:
            prompt = f"""Based on the following question, write a detailed academic answer that includes relevant technical terms and concepts, as if excerpted from a high-quality research paper or technical document:

Question: {query}

Detailed answer:"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the hypothetical document
            if is_chinese:
                answer_part = response.split("详细答案：")[-1].strip()
            else:
                answer_part = response.split("Detailed answer:")[-1].strip()
            
            logger.info(f"Generated hypothetical document ({len(answer_part)} chars)")
            return answer_part
            
        except Exception as e:
            logger.error(f"Failed to generate hypothetical document: {e}")
            return ""

class QueryIntelligenceEngine:
    """查询智能引擎 - 整合所有查询处理功能"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components
        self.complexity_analyzer = QueryComplexityAnalyzer()
        
        # Initialize LLM-based components if model is available
        model_name = config.get('llm_model')
        token = config.get('HUGGING_FACE_TOKEN')
        device = config.get('device', 'auto')
        
        if model_name:
            try:
                self.sub_question_generator = SubQuestionGenerator(
                    model_name=model_name, device=device, token=token
                )
                self.query_rewriter = QueryRewriter(
                    model_name=model_name, device=device, token=token
                )
                self.hyde_generator = HyDEGenerator(
                    model_name=model_name, device=device, token=token
                )
                self.llm_available = True
                logger.success("Query Intelligence Engine initialized with LLM support")
            except Exception as e:
                logger.error(f"Failed to initialize LLM components: {e}")
                self.llm_available = False
        else:
            self.llm_available = False
            logger.warning("Query Intelligence Engine initialized without LLM support")
    
    def analyze_query(self, query: str) -> QueryAnalysisResult:
        """全面分析查询"""
        # Basic analysis
        language = 'zh' if re.search(r'[\u4e00-\u9fff]', query) else 'en'
        complexity = self.complexity_analyzer.analyze_complexity(query)
        
        # Determine query type
        query_type = self._determine_query_type(query)
        
        # Advanced analysis if LLM is available
        sub_questions = []
        rewritten_queries = []
        hypothetical_document = ""
        
        if self.llm_available:
            try:
                # Generate sub-questions for complex queries
                if complexity in ['medium', 'complex']:
                    sub_questions = self.sub_question_generator.generate_sub_questions(query)
                
                # Generate query variations
                rewritten_queries = self.query_rewriter.rewrite_query(query, num_variations=3)
                
                # Generate hypothetical document
                hypothetical_document = self.hyde_generator.generate_hypothetical_document(query)
                
            except Exception as e:
                logger.error(f"Error in advanced query analysis: {e}")
        
        result = QueryAnalysisResult(
            original_query=query,
            language=language,
            complexity=complexity,
            sub_questions=sub_questions,
            rewritten_queries=rewritten_queries,
            hypothetical_document=hypothetical_document,
            query_type=query_type
        )
        
        logger.info(f"Query analysis complete - Language: {language}, Complexity: {complexity}, Type: {query_type}")
        return result
    
    def _determine_query_type(self, query: str) -> str:
        """确定查询类型"""
        query_lower = query.lower()
        
        # Factual questions
        if re.search(r'\b(what is|什么是|define|定义|who|谁|when|什么时候|where|哪里)\b', query_lower):
            return 'factual'
        
        # Comparative questions  
        elif re.search(r'\b(compare|比较|对比|difference|区别|异同|vs|versus)\b', query_lower):
            return 'comparative'
        
        # Procedural questions
        elif re.search(r'\b(how to|如何|怎么|step|步骤|process|流程|方法)\b', query_lower):
            return 'procedural'
        
        # Explanatory questions
        elif re.search(r'\b(why|为什么|explain|解释|analyze|分析|discuss|讨论)\b', query_lower):
            return 'explanatory'
        
        else:
            return 'general'
    
    def get_optimized_queries(self, query: str) -> List[str]:
        """获取优化后的查询列表用于检索"""
        analysis = self.analyze_query(query)
        optimized_queries = []
        
        # Always include original query
        optimized_queries.append(query)
        
        # Add rewritten queries
        optimized_queries.extend(analysis.rewritten_queries)
        
        # Add sub-questions for complex queries
        if analysis.complexity == 'complex' and analysis.sub_questions:
            optimized_queries.extend(analysis.sub_questions[:2])  # Limit to 2 sub-questions
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in optimized_queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)
        
        return unique_queries
    
    def get_hyde_document(self, query: str) -> str:
        """获取HyDE文档用于检索"""
        if self.llm_available:
            return self.hyde_generator.generate_hypothetical_document(query)
        return ""

# 使用示例
async def main():
    """测试查询智能引擎"""
    config = {
        'llm_model': 'Qwen/Qwen2-7B-Instruct',
        'device': 'auto',
        'HUGGING_FACE_TOKEN': None
    }
    
    engine = QueryIntelligenceEngine(config)
    
    test_queries = [
        "什么是Transformer模型？",
        "请对比一下LoRA和QLoRA在训练效率和模型性能上的异同",
        "How do attention mechanisms work in neural networks?",
        "Explain the differences between supervised and unsupervised learning approaches in computer vision"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        analysis = engine.analyze_query(query)
        
        print(f"Language: {analysis.language}")
        print(f"Complexity: {analysis.complexity}")
        print(f"Type: {analysis.query_type}")
        
        if analysis.sub_questions:
            print(f"\nSub-questions:")
            for i, sq in enumerate(analysis.sub_questions, 1):
                print(f"  {i}. {sq}")
        
        if analysis.rewritten_queries:
            print(f"\nRewritten queries:")
            for i, rq in enumerate(analysis.rewritten_queries, 1):
                print(f"  {i}. {rq}")
        
        if analysis.hypothetical_document:
            print(f"\nHypothetical document preview:")
            print(f"  {analysis.hypothetical_document[:200]}...")

if __name__ == "__main__":
    asyncio.run(main())