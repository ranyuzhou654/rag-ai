# src/evaluation/evaluation_pipeline.py
from typing import List, Dict, Optional, Tuple, Any
import asyncio
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import statistics
from datetime import datetime
from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import argparse
from configs.config import config as default_config
from src.generation.ultimate_rag_system import UltimateRAGSystem

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevancy as ragas_answer_relevancy,
        context_precision as ragas_context_precision,
        context_recall as ragas_context_recall,
        faithfulness as ragas_faithfulness
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:  # Optional dependency
    RAGAS_AVAILABLE = False

@dataclass
class EvaluationMetrics:
    """评估指标"""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    response_time: float
    overall_score: float

@dataclass
class TestCase:
    """测试用例"""
    query: str
    expected_answer: str
    reference_documents: List[str]
    difficulty_level: str  # 'easy', 'medium', 'hard'
    query_type: str  # 'factual', 'analytical', 'comparative'
    ground_truth_chunks: Optional[List[str]] = None

@dataclass
class EvaluationResult:
    """评估结果"""
    test_case: TestCase
    generated_answer: str
    retrieved_chunks: List[Dict]
    metrics: EvaluationMetrics
    execution_time: float
    error: Optional[str] = None


def load_test_cases(path: Path) -> List[TestCase]:
    """从JSON或JSONL文件加载测试用例"""
    if not path.exists():
        raise FileNotFoundError(f"Test case file not found: {path}")

    logger.info(f"Loading test cases from {path}")

    data: List[Dict[str, Any]] = []
    if path.suffix.lower() == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
    else:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
            if isinstance(payload, dict) and 'test_cases' in payload:
                data = payload['test_cases']
            elif isinstance(payload, list):
                data = payload
            else:
                raise ValueError("Unsupported test case format")

    test_cases: List[TestCase] = []
    for item in data:
        test_cases.append(TestCase(
            query=item['query'],
            expected_answer=item.get('expected_answer', ''),
            reference_documents=item.get('reference_documents', []),
            difficulty_level=item.get('difficulty_level', 'medium'),
            query_type=item.get('query_type', 'factual'),
            ground_truth_chunks=item.get('ground_truth_chunks')
        ))

    logger.info(f"Loaded {len(test_cases)} test cases")
    return test_cases

class LLMEvaluator:
    """基于LLM的评估器"""
    
    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        logger.info(f"Loading LLM Evaluator: {model_name}")
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
            max_new_tokens=300,
            temperature=0.1,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
    
    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """评估答案忠实度 - 答案是否基于给定上下文"""
        prompt = f"""请评估生成的答案是否忠实于给定的上下文材料。评分标准：
- 5分：答案完全基于上下文，没有添加额外信息
- 4分：答案主要基于上下文，有少量合理推理
- 3分：答案部分基于上下文，有一定推理成分
- 2分：答案与上下文有关，但包含较多外部信息
- 1分：答案与上下文关联很少或包含错误信息

上下文：
{context}

生成的答案：
{answer}

请只输出1-5的数字评分："""
        
        try:
            score = self._get_llm_score(prompt, scale=5)
            return score / 5.0  # 归一化到0-1
        except:
            return 0.5  # 默认中等得分
    
    def evaluate_answer_relevancy(self, query: str, answer: str) -> float:
        """评估答案相关性 - 答案是否直接回答了问题"""
        prompt = f"""请评估生成的答案对用户问题的相关性。评分标准：
- 5分：完全回答了问题，高度相关
- 4分：基本回答了问题，相关性好
- 3分：部分回答了问题，相关性一般
- 2分：勉强涉及问题，相关性较差
- 1分：基本没有回答问题，不相关

用户问题：
{query}

生成的答案：
{answer}

请只输出1-5的数字评分："""
        
        try:
            score = self._get_llm_score(prompt, scale=5)
            return score / 5.0
        except:
            return 0.5
    
    def evaluate_context_precision(self, query: str, retrieved_chunks: List[Dict]) -> float:
        """评估上下文精确度 - 检索到的文档中有多少是相关的"""
        if not retrieved_chunks:
            return 0.0
        
        relevant_count = 0
        for chunk in retrieved_chunks:
            relevance = self._evaluate_chunk_relevance(query, chunk['content'])
            if relevance >= 0.6:  # 阈值
                relevant_count += 1
        
        return relevant_count / len(retrieved_chunks)
    
    def evaluate_context_recall(self, query: str, retrieved_chunks: List[Dict], reference_docs: List[str]) -> float:
        """评估上下文召回率 - 应该检索到的文档中有多少被检索到了"""
        if not reference_docs:
            return 1.0  # 如果没有参考文档，假设完美召回
        
        retrieved_content = " ".join([chunk['content'] for chunk in retrieved_chunks])
        
        found_count = 0
        for ref_doc in reference_docs:
            # 简单的包含检查（可以改进为更复杂的相似度检查）
            if any(ref_phrase in retrieved_content for ref_phrase in ref_doc.split() if len(ref_phrase) > 3):
                found_count += 1
        
        return found_count / len(reference_docs)
    
    def _evaluate_chunk_relevance(self, query: str, chunk_content: str) -> float:
        """评估单个文档块的相关性"""
        prompt = f"""请评估以下文档片段对用户问题的相关性，用0-1的分数表示（0=完全不相关，1=完全相关）：

用户问题：{query}

文档片段：{chunk_content}

请只输出0-1之间的小数："""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取分数
            score_text = response.split("请只输出0-1之间的小数：")[-1].strip()
            score = float(score_text.split()[0])
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def _get_llm_score(self, prompt: str, scale: int = 5) -> float:
        """从LLM获取评分"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取数字评分
            score_part = response.split("请只输出")[-1].strip()
            score = int(score_part.split()[0])
            return max(1, min(scale, score))
        except:
            return scale // 2 + 1  # 默认中等得分

class SemanticEvaluator:
    """基于语义相似度的评估器"""
    
    def __init__(self, embedding_model: str = "BAAI/bge-m3", device: str = "auto"):
        self.embedder = SentenceTransformer(embedding_model, device=device)
        logger.info(f"Semantic Evaluator initialized with {embedding_model}")
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的语义相似度"""
        try:
            embeddings = self.embedder.encode([text1, text2], convert_to_numpy=True)
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except:
            return 0.0

class EvaluationPipeline:
    """评估流水线"""
    
    def __init__(self, config: Dict):
        self.config = config
        device_pref = config.get('device', 'auto')
        if not device_pref or device_pref == 'auto':
            device_pref = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device_pref

        # 初始化评估器
        self.semantic_evaluator = SemanticEvaluator(
            embedding_model=config.get('embedding_model', 'BAAI/bge-m3'),
            device=self.device
        )
        
        # 初始化LLM评估器（如果可用）
        model_name = config.get('llm_model')
        token = config.get('HUGGING_FACE_TOKEN')
        
        self.llm_evaluator = None
        if model_name:
            try:
                self.llm_evaluator = LLMEvaluator(
                    model_name=model_name, device=self.device, token=token
                )
                logger.success("Evaluation Pipeline with LLM evaluator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize LLM evaluator: {e}")
        else:
            logger.warning("Evaluation Pipeline initialized without LLM evaluator")
    
    async def evaluate_rag_system(
        self,
        rag_system,
        test_cases: List[TestCase],
        output_path: Optional[Path] = None,
        rag_mode: str = "ultimate",
        use_ragas: bool = False
    ) -> Dict[str, Any]:
        """评估RAG系统"""
        logger.info(f"Starting evaluation with {len(test_cases)} test cases")
        
        results = []
        start_time = time.time()
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating test case {i+1}/{len(test_cases)}: {test_case.query[:50]}...")
            
            # 执行RAG系统
            case_start_time = time.time()
            try:
                generation_result = await rag_system.generate_answer(test_case.query, mode=rag_mode)
                execution_time = time.time() - case_start_time
                
                # 计算评估指标
                metrics = self._calculate_metrics(
                    test_case=test_case,
                    generated_answer=generation_result.answer,
                    retrieved_chunks=generation_result.source_chunks,
                    response_time=generation_result.generation_time
                )
                
                result = EvaluationResult(
                    test_case=test_case,
                    generated_answer=generation_result.answer,
                    retrieved_chunks=generation_result.source_chunks,
                    metrics=metrics,
                    execution_time=execution_time
                )
                
            except Exception as e:
                logger.error(f"Error in test case {i+1}: {e}")
                result = EvaluationResult(
                    test_case=test_case,
                    generated_answer="",
                    retrieved_chunks=[],
                    metrics=EvaluationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    execution_time=0.0,
                    error=str(e)
                )
            
            results.append(result)
        
        total_time = time.time() - start_time
        
        # 汇总结果
        summary = self._summarize_results(results, total_time)

        if use_ragas:
            ragas_metrics = self._run_ragas(results)
            if ragas_metrics:
                summary['ragas'] = ragas_metrics

        # 保存结果
        if output_path:
            self._save_results(results, summary, output_path)

        logger.success(f"Evaluation completed in {total_time:.2f}s")
        return summary
    
    def _calculate_metrics(
        self,
        test_case: TestCase,
        generated_answer: str,
        retrieved_chunks: List[Dict],
        response_time: float
    ) -> EvaluationMetrics:
        """计算评估指标"""
        
        # 1. Context precision and recall
        context = " ".join([chunk['content'] for chunk in retrieved_chunks])
        
        if self.llm_evaluator:
            faithfulness = self.llm_evaluator.evaluate_faithfulness(generated_answer, context)
            answer_relevancy = self.llm_evaluator.evaluate_answer_relevancy(test_case.query, generated_answer)
            context_precision = self.llm_evaluator.evaluate_context_precision(test_case.query, retrieved_chunks)
            context_recall = self.llm_evaluator.evaluate_context_recall(
                test_case.query, retrieved_chunks, test_case.reference_documents
            )
        else:
            # 后备：基于语义相似度
            faithfulness = self.semantic_evaluator.calculate_semantic_similarity(generated_answer, context)
            answer_relevancy = self.semantic_evaluator.calculate_semantic_similarity(test_case.query, generated_answer)
            context_precision = 0.5  # 默认值
            context_recall = 0.5  # 默认值
        
        # 计算总分
        overall_score = (
            faithfulness * 0.3 +
            answer_relevancy * 0.3 +
            context_precision * 0.2 +
            context_recall * 0.2
        )
        
        return EvaluationMetrics(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
            response_time=response_time,
            overall_score=overall_score
        )
    
    def _summarize_results(self, results: List[EvaluationResult], total_time: float) -> Dict[str, Any]:
        """汇总评估结果"""
        
        successful_results = [r for r in results if r.error is None]
        
        if not successful_results:
            return {
                'total_cases': len(results),
                'successful_cases': 0,
                'error_rate': 1.0,
                'avg_metrics': {},
                'total_time': total_time
            }
        
        # 计算平均指标
        avg_metrics = {
            'faithfulness': statistics.mean([r.metrics.faithfulness for r in successful_results]),
            'answer_relevancy': statistics.mean([r.metrics.answer_relevancy for r in successful_results]),
            'context_precision': statistics.mean([r.metrics.context_precision for r in successful_results]),
            'context_recall': statistics.mean([r.metrics.context_recall for r in successful_results]),
            'response_time': statistics.mean([r.metrics.response_time for r in successful_results]),
            'overall_score': statistics.mean([r.metrics.overall_score for r in successful_results])
        }
        
        # 按难度级别分析
        difficulty_analysis = {}
        for level in ['easy', 'medium', 'hard']:
            level_results = [r for r in successful_results if r.test_case.difficulty_level == level]
            if level_results:
                difficulty_analysis[level] = {
                    'count': len(level_results),
                    'avg_score': statistics.mean([r.metrics.overall_score for r in level_results])
                }
        
        # 按查询类型分析
        type_analysis = {}
        for query_type in ['factual', 'analytical', 'comparative']:
            type_results = [r for r in successful_results if r.test_case.query_type == query_type]
            if type_results:
                type_analysis[query_type] = {
                    'count': len(type_results),
                    'avg_score': statistics.mean([r.metrics.overall_score for r in type_results])
                }
        
        return {
            'total_cases': len(results),
            'successful_cases': len(successful_results),
            'error_rate': (len(results) - len(successful_results)) / len(results),
            'avg_metrics': avg_metrics,
            'difficulty_analysis': difficulty_analysis,
            'type_analysis': type_analysis,
            'total_time': total_time,
            'avg_response_time': avg_metrics['response_time']
        }
    
    def _save_results(self, results: List[EvaluationResult], summary: Dict[str, Any], output_path: Path):
        """保存评估结果"""
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # 保存详细结果
        detailed_results = []
        for result in results:
            detailed_results.append({
                'test_case': asdict(result.test_case),
                'generated_answer': result.generated_answer,
                'retrieved_chunks': result.retrieved_chunks,
                'metrics': asdict(result.metrics),
                'execution_time': result.execution_time,
                'error': result.error
            })
        
        output_data = {
            'summary': summary,
            'detailed_results': detailed_results,
            'timestamp': datetime.now().isoformat(),
            'config': self.config
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")

    def _run_ragas(self, results: List[EvaluationResult]) -> Optional[Dict[str, float]]:
        """调用RAGAS评测，可选"""
        if not RAGAS_AVAILABLE:
            logger.warning("RAGAS not installed. Skipping RAGAS evaluation.")
            return None

        questions = []
        answers = []
        contexts = []
        ground_truth = []

        for result in results:
            if result.error is not None:
                continue
            questions.append(result.test_case.query)
            answers.append(result.generated_answer)
            contexts.append([chunk.get('content', '') for chunk in result.retrieved_chunks])
            ground_truth.append(result.test_case.expected_answer)

        if not questions:
            logger.warning("No successful results available for RAGAS evaluation")
            return None

        dataset = Dataset.from_dict({
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': ground_truth
        })

        ragas_result = ragas_evaluate(
            dataset,
            metrics=[
                ragas_faithfulness,
                ragas_answer_relevancy,
                ragas_context_precision,
                ragas_context_recall
            ]
        )

        scores: Dict[str, float] = {}
        for metric_name in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
            if metric_name in ragas_result.column_names:
                metric_values = [val for val in ragas_result[metric_name] if isinstance(val, (int, float))]
                if metric_values:
                    scores[metric_name] = float(np.mean(metric_values))

        if scores:
            logger.info("RAGAS metrics:")
            for name, value in scores.items():
                logger.info(f"  {name}: {value:.3f}")

        return scores or None

class GoldenTestSetGenerator:
    """黄金测试集生成器"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def create_ai_domain_test_set(self) -> List[TestCase]:
        """创建AI领域的测试集"""
        test_cases = [
            # 简单事实性问题
            TestCase(
                query="什么是Transformer模型？",
                expected_answer="Transformer是一种基于注意力机制的神经网络架构，由Vaswani等人在2017年提出。它完全依赖注意力机制来处理序列数据，不使用循环或卷积操作。",
                reference_documents=["transformer", "attention mechanism", "neural network"],
                difficulty_level="easy",
                query_type="factual"
            ),
            TestCase(
                query="什么是注意力机制？",
                expected_answer="注意力机制是一种允许模型在处理输入时动态关注不同部分的技术。它通过计算注意力权重来确定输入序列中每个元素的重要性。",
                reference_documents=["attention mechanism", "neural attention"],
                difficulty_level="easy",
                query_type="factual"
            ),
            
            # 中等难度分析性问题
            TestCase(
                query="Transformer模型相比于RNN有哪些优势？",
                expected_answer="Transformer相比RNN的主要优势包括：1）并行计算能力强；2）能更好地处理长距离依赖；3）训练效率更高；4）可以利用更多的计算资源。",
                reference_documents=["transformer vs rnn", "parallel processing", "long-range dependencies"],
                difficulty_level="medium",
                query_type="analytical"
            ),
            TestCase(
                query="为什么深度学习在计算机视觉领域如此成功？",
                expected_answer="深度学习在计算机视觉成功的原因包括：1）卷积神经网络能自动学习层次化特征；2）大量标注数据的可用性；3）GPU等硬件的发展；4）端到端训练的优势。",
                reference_documents=["deep learning", "computer vision", "CNN", "feature learning"],
                difficulty_level="medium",
                query_type="analytical"
            ),
            
            # 困难对比性问题
            TestCase(
                query="请详细对比LoRA和QLoRA在大模型微调中的异同点",
                expected_answer="LoRA和QLoRA都是大模型的参数高效微调方法。LoRA通过低秩矩阵分解减少可训练参数，而QLoRA在此基础上加入了量化技术。QLoRA内存占用更少，但可能在精度上有所损失。",
                reference_documents=["LoRA", "QLoRA", "parameter efficient fine-tuning", "quantization"],
                difficulty_level="hard",
                query_type="comparative"
            ),
            TestCase(
                query="比较分析Transformer、CNN和RNN三种架构在不同任务上的适用性",
                expected_answer="Transformer适合序列建模和NLP任务，具有并行计算优势；CNN擅长图像处理和局部特征提取；RNN适合时序数据处理但存在梯度消失问题。选择依赖于具体任务需求。",
                reference_documents=["transformer", "CNN", "RNN", "architecture comparison"],
                difficulty_level="hard",
                query_type="comparative"
            ),
            
            # 中英文混合测试
            TestCase(
                query="What are the key innovations in GPT models?",
                expected_answer="Key innovations in GPT models include: 1) Transformer architecture with self-attention; 2) Unsupervised pre-training on large text corpora; 3) Autoregressive language modeling; 4) Transfer learning capabilities.",
                reference_documents=["GPT", "transformer", "language modeling", "transfer learning"],
                difficulty_level="medium",
                query_type="analytical"
            ),
        ]
        
        return test_cases
    
    def save_test_set(self, test_cases: List[TestCase], output_path: Path):
        """保存测试集"""
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        test_data = {
            'test_cases': [asdict(case) for case in test_cases],
            'created_at': datetime.now().isoformat(),
            'total_cases': len(test_cases)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Test set with {len(test_cases)} cases saved to {output_path}")


def build_default_rag_config() -> Dict[str, Any]:
    """基于全局配置构建RAG系统配置字典"""
    cfg = default_config
    return {
        'llm_model': cfg.LLM_MODEL,
        'embedding_model': cfg.EMBEDDING_MODEL,
        'device': cfg.DEVICE,
        'HUGGING_FACE_TOKEN': cfg.HUGGING_FACE_TOKEN,
        'qdrant_host': cfg.QDRANT_HOST,
        'qdrant_port': cfg.QDRANT_PORT,
        'collection_name': cfg.COLLECTION_NAME,
        'vector_size': 1024,
        'enable_knowledge_graph': cfg.ENABLE_KNOWLEDGE_GRAPH,
        'knowledge_graph_db_path': str(cfg.KNOWLEDGE_GRAPH_DB_PATH),
        'enable_tiered_generation': cfg.ENABLE_TIERED_GENERATION,
        'enable_feedback_collection': cfg.ENABLE_FEEDBACK_COLLECTION,
        'feedback_db_path': cfg.FEEDBACK_DB_PATH,
        'api_models': cfg.API_MODELS,
        'enable_query_intelligence': cfg.ENABLE_QUERY_INTELLIGENCE,
        'enable_multi_representation': cfg.ENABLE_MULTI_REPRESENTATION,
        'enable_agentic_rag': cfg.ENABLE_AGENTIC_RAG,
        'enable_contextual_compression': cfg.ENABLE_CONTEXTUAL_COMPRESSION,
        'default_rag_mode': cfg.DEFAULT_RAG_MODE,
        'enable_hybrid_search': True
    }


def ensure_test_set(path: Path, config: Dict[str, Any], regenerate: bool = False) -> None:
    if path.exists() and not regenerate:
        return
    path.parent.mkdir(exist_ok=True, parents=True)
    generator = GoldenTestSetGenerator(config)
    test_cases = generator.create_ai_domain_test_set()
    generator.save_test_set(test_cases, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automated RAG evaluation pipeline")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_config.GOLDEN_TEST_SET_PATH,
        help="Path to evaluation dataset (JSON or JSONL). If missing, use --generate-testset to create a default set."
    )
    parser.add_argument(
        "--dataset-preset",
        type=str,
        choices=['golden', 'hotpotqa', 'dureader'],
        default=None,
        help="Quick-select dataset preset located in evaluation output directory."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for evaluation report (JSON). Defaults to storage evaluation directory."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=default_config.DEFAULT_RAG_MODE,
        help="RAG system mode (basic/enhanced/agentic/ultimate)"
    )
    parser.add_argument(
        "--use-ragas",
        action="store_true",
        help="Enable additional RAGAS metrics (requires ragas + datasets)"
    )
    parser.add_argument(
        "--generate-testset",
        action="store_true",
        help="Generate default golden test set when dataset path is missing"
    )
    parser.add_argument(
        "--regenerate-testset",
        action="store_true",
        help="Force regenerate the default golden test set"
    )
    return parser.parse_args()


async def run_evaluation_cli() -> None:
    args = parse_args()

    rag_config = build_default_rag_config()
    pipeline_config = {
        'embedding_model': rag_config['embedding_model'],
        'llm_model': rag_config.get('llm_model'),
        'device': rag_config.get('device', 'auto'),
        'HUGGING_FACE_TOKEN': rag_config.get('HUGGING_FACE_TOKEN'),
        'enable_ragas': args.use_ragas
    }

    if args.dataset_preset:
        preset_base = default_config.EVALUATION_OUTPUT_DIR
        preset_map = {
            'golden': default_config.GOLDEN_TEST_SET_PATH,
            'hotpotqa': preset_base / 'hotpotqa_testcases.json',
            'dureader': preset_base / 'dureader_testcases.json'
        }
        dataset_path = preset_map[args.dataset_preset]
    else:
        dataset_path: Path = args.dataset
    if not dataset_path.exists():
        if args.generate_testset or args.regenerate_testset:
            ensure_test_set(dataset_path, pipeline_config, regenerate=True)
        else:
            raise FileNotFoundError(f"Dataset not found at {dataset_path}. Use --generate-testset to create one.")

    if args.regenerate_testset:
        ensure_test_set(dataset_path, pipeline_config, regenerate=True)

    test_cases = load_test_cases(dataset_path)
    rag_system = UltimateRAGSystem(rag_config)

    pipeline = EvaluationPipeline(pipeline_config)

    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = default_config.EVALUATION_OUTPUT_DIR / f"evaluation_{timestamp}.json"

    output_path.parent.mkdir(exist_ok=True, parents=True)

    summary = await pipeline.evaluate_rag_system(
        rag_system=rag_system,
        test_cases=test_cases,
        output_path=output_path,
        rag_mode=args.mode,
        use_ragas=args.use_ragas
    )

    logger.info("\n📊 Evaluation Summary:")
    for key, value in summary.get('avg_metrics', {}).items():
        logger.info(f"  {key}: {value:.3f}")
    if 'ragas' in summary:
        logger.info("  RAGAS metrics:")
        for key, value in summary['ragas'].items():
            logger.info(f"    {key}: {value:.3f}")
    logger.info(f"Results saved to {output_path}")


def main():
    asyncio.run(run_evaluation_cli())


if __name__ == "__main__":
    main()
