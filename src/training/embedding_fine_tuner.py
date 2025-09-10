# src/training/embedding_fine_tuner.py
from typing import List, Dict, Optional, Tuple, Any
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from loguru import logger
import sqlite3
from datetime import datetime
import random
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class TrainingExample:
    """训练样本"""
    query: str
    positive_doc: str
    negative_docs: List[str]
    score: float = 1.0
    source: str = "feedback"

@dataclass
class FineTuneConfig:
    """微调配置"""
    model_name: str
    output_path: Path
    batch_size: int = 16
    num_epochs: int = 1
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    evaluation_steps: int = 1000
    max_seq_length: int = 512
    use_contrastive_loss: bool = True
    hard_negative_ratio: float = 0.3
    device: str = "auto"

class FeedbackDataExtractor:
    """从反馈数据中提取训练样本"""
    
    def __init__(self, feedback_db_path: Path):
        self.db_path = feedback_db_path
        logger.info(f"Feedback Data Extractor initialized with {feedback_db_path}")
    
    def extract_training_data(
        self,
        min_positive_score: float = 3.0,
        max_negative_score: float = 2.0,
        min_samples: int = 100
    ) -> List[TrainingExample]:
        """从反馈数据库提取训练样本"""
        
        if not self.db_path.exists():
            logger.warning("Feedback database not found")
            return []
        
        training_examples = []
        
        with sqlite3.connect(self.db_path) as conn:
            # 获取正面反馈
            positive_cursor = conn.execute('''
                SELECT user_query, system_answer, source_chunks, feedback_value
                FROM feedback_records 
                WHERE feedback_type = 'rating' AND CAST(feedback_value as REAL) >= ?
                OR feedback_type = 'thumbs_up'
                ORDER BY timestamp DESC
                LIMIT 500
            ''', (min_positive_score,))
            
            positive_samples = positive_cursor.fetchall()
            
            # 获取负面反馈
            negative_cursor = conn.execute('''
                SELECT user_query, system_answer, source_chunks, feedback_value
                FROM feedback_records 
                WHERE feedback_type = 'rating' AND CAST(feedback_value as REAL) <= ?
                OR feedback_type = 'thumbs_down'
                ORDER BY timestamp DESC
                LIMIT 500
            ''', (max_negative_score,))
            
            negative_samples = negative_cursor.fetchall()
            
            logger.info(f"Found {len(positive_samples)} positive and {len(negative_samples)} negative samples")
            
            # 处理正面样本
            for query, answer, source_chunks_json, feedback_value in positive_samples:
                try:
                    source_chunks = json.loads(source_chunks_json) if source_chunks_json else []
                    if source_chunks:
                        positive_doc = source_chunks[0].get('content', answer)
                        
                        # 随机选择一些负面文档作为困难负例
                        negative_docs = []
                        for neg_query, neg_answer, neg_chunks_json, neg_value in random.sample(negative_samples, min(3, len(negative_samples))):
                            neg_chunks = json.loads(neg_chunks_json) if neg_chunks_json else []
                            if neg_chunks:
                                negative_docs.append(neg_chunks[0].get('content', neg_answer))
                            else:
                                negative_docs.append(neg_answer)
                        
                        if negative_docs:
                            example = TrainingExample(
                                query=query,
                                positive_doc=positive_doc,
                                negative_docs=negative_docs,
                                score=float(feedback_value) if feedback_value and feedback_value != '1' else 5.0,
                                source="positive_feedback"
                            )
                            training_examples.append(example)
                            
                except Exception as e:
                    logger.error(f"Error processing positive sample: {e}")
                    continue
            
            logger.info(f"Extracted {len(training_examples)} training examples from feedback")
            return training_examples
    
    def extract_correction_pairs(self) -> List[TrainingExample]:
        """提取用户纠错数据"""
        if not self.db_path.exists():
            return []
        
        training_examples = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT user_query, system_answer, feedback_value, source_chunks
                FROM feedback_records 
                WHERE feedback_type = 'correction'
                ORDER BY timestamp DESC
                LIMIT 100
            ''')
            
            for query, wrong_answer, correction_json, source_chunks_json in cursor.fetchall():
                try:
                    correction_data = json.loads(correction_json) if correction_json else {}
                    correct_answer = correction_data.get('correct_answer', '')
                    
                    if correct_answer:
                        source_chunks = json.loads(source_chunks_json) if source_chunks_json else []
                        negative_docs = [wrong_answer]
                        
                        # 添加其他错误文档作为负例
                        if source_chunks:
                            negative_docs.extend([chunk.get('content', '') for chunk in source_chunks[:2]])
                        
                        example = TrainingExample(
                            query=query,
                            positive_doc=correct_answer,
                            negative_docs=[doc for doc in negative_docs if doc],
                            score=5.0,
                            source="correction"
                        )
                        training_examples.append(example)
                        
                except Exception as e:
                    logger.error(f"Error processing correction: {e}")
                    continue
        
        logger.info(f"Extracted {len(training_examples)} correction examples")
        return training_examples

class SyntheticDataGenerator:
    """合成训练数据生成器"""
    
    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(model_name, device=self.device)
        logger.info(f"Synthetic Data Generator initialized with {model_name}")
    
    def generate_hard_negatives(
        self,
        query: str,
        positive_docs: List[str],
        candidate_pool: List[str],
        num_negatives: int = 5
    ) -> List[str]:
        """生成困难负例"""
        
        if len(candidate_pool) < num_negatives:
            return candidate_pool
        
        # 编码查询和正例
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0]
        positive_embeddings = self.embedder.encode(positive_docs, convert_to_numpy=True)
        
        # 计算正例的平均向量
        avg_positive = np.mean(positive_embeddings, axis=0)
        
        # 编码候选负例
        candidate_embeddings = self.embedder.encode(candidate_pool, convert_to_numpy=True)
        
        # 计算与查询的相似度
        query_similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
        
        # 计算与正例平均向量的相似度
        positive_similarities = cosine_similarity([avg_positive], candidate_embeddings)[0]
        
        # 选择与查询相似但与正例不太相似的文档作为困难负例
        difficulty_scores = query_similarities - positive_similarities * 0.5
        
        # 选择得分最高的作为困难负例
        hard_negative_indices = np.argsort(difficulty_scores)[-num_negatives:]
        
        return [candidate_pool[i] for i in hard_negative_indices]
    
    def augment_training_data(
        self,
        examples: List[TrainingExample],
        document_pool: List[str],
        augment_ratio: float = 2.0
    ) -> List[TrainingExample]:
        """增强训练数据"""
        
        augmented_examples = examples.copy()
        target_size = int(len(examples) * augment_ratio)
        
        while len(augmented_examples) < target_size:
            # 随机选择一个原始样本
            original = random.choice(examples)
            
            # 生成困难负例
            hard_negatives = self.generate_hard_negatives(
                query=original.query,
                positive_docs=[original.positive_doc],
                candidate_pool=document_pool,
                num_negatives=3
            )
            
            # 创建增强样本
            augmented = TrainingExample(
                query=original.query,
                positive_doc=original.positive_doc,
                negative_docs=hard_negatives,
                score=original.score,
                source=f"{original.source}_augmented"
            )
            
            augmented_examples.append(augmented)
        
        logger.info(f"Augmented training data from {len(examples)} to {len(augmented_examples)} examples")
        return augmented_examples

class CustomDataset(Dataset):
    """自定义数据集"""
    
    def __init__(self, examples: List[TrainingExample]):
        self.examples = examples
        self.input_examples = self._prepare_input_examples()
    
    def _prepare_input_examples(self) -> List[InputExample]:
        """准备输入样本"""
        input_examples = []
        
        for example in self.examples:
            # 正例对
            input_examples.append(InputExample(
                texts=[example.query, example.positive_doc],
                label=1.0
            ))
            
            # 负例对
            for neg_doc in example.negative_docs:
                input_examples.append(InputExample(
                    texts=[example.query, neg_doc],
                    label=0.0
                ))
        
        return input_examples
    
    def __len__(self):
        return len(self.input_examples)
    
    def __getitem__(self, idx):
        return self.input_examples[idx]

class EmbeddingFineTuner:
    """Embedding模型微调器"""
    
    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载预训练模型
        logger.info(f"Loading pre-trained model: {config.model_name}")
        self.model = SentenceTransformer(config.model_name, device=self.device)
        
        # 创建输出目录
        config.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Embedding Fine-Tuner initialized")
    
    def prepare_training_data(
        self,
        feedback_db_path: Path,
        document_pool: Optional[List[str]] = None
    ) -> List[TrainingExample]:
        """准备训练数据"""
        
        # 从反馈中提取数据
        feedback_extractor = FeedbackDataExtractor(feedback_db_path)
        feedback_examples = feedback_extractor.extract_training_data()
        correction_examples = feedback_extractor.extract_correction_pairs()
        
        all_examples = feedback_examples + correction_examples
        
        if len(all_examples) < 10:
            logger.warning("Insufficient training data from feedback, using synthetic data")
            # 生成一些基础的AI领域训练样本
            all_examples = self._generate_domain_examples()
        
        # 如果有文档池，增强数据
        if document_pool and len(all_examples) > 0:
            synthetic_generator = SyntheticDataGenerator(
                self.config.model_name, device=self.config.device
            )
            all_examples = synthetic_generator.augment_training_data(
                all_examples, document_pool, augment_ratio=1.5
            )
        
        logger.info(f"Prepared {len(all_examples)} training examples")
        return all_examples
    
    def _generate_domain_examples(self) -> List[TrainingExample]:
        """生成AI领域的基础训练样本"""
        domain_examples = [
            TrainingExample(
                query="什么是Transformer模型？",
                positive_doc="Transformer是一种基于注意力机制的深度学习架构，由Vaswani等人在2017年提出。它完全依赖注意力机制来处理序列数据。",
                negative_docs=[
                    "卷积神经网络（CNN）是一种深度学习架构，主要用于图像处理。",
                    "循环神经网络（RNN）是一种处理序列数据的神经网络架构。"
                ],
                source="domain_knowledge"
            ),
            TrainingExample(
                query="BERT模型的优势是什么？",
                positive_doc="BERT（Bidirectional Encoder Representations from Transformers）通过双向训练获得更好的语言理解能力，在多个NLP任务上达到了state-of-the-art性能。",
                negative_docs=[
                    "GPT是一种生成式预训练语言模型，主要用于文本生成任务。",
                    "Word2Vec是一种词向量表示学习方法，将词语映射到连续向量空间。"
                ],
                source="domain_knowledge"
            ),
            TrainingExample(
                query="深度学习中的注意力机制",
                positive_doc="注意力机制允许模型在处理输入时动态关注不同部分，通过计算注意力权重来确定各部分的重要性。",
                negative_docs=[
                    "反向传播是深度学习中用于训练神经网络的优化算法。",
                    "梯度下降是一种优化算法，用于最小化损失函数。"
                ],
                source="domain_knowledge"
            )
        ]
        
        return domain_examples
    
    async def fine_tune(
        self,
        training_examples: List[TrainingExample],
        validation_examples: Optional[List[TrainingExample]] = None
    ) -> Dict[str, Any]:
        """执行微调"""
        
        logger.info("Starting embedding model fine-tuning...")
        
        # 准备数据集
        train_dataset = CustomDataset(training_examples)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.config.batch_size)
        
        # 选择损失函数
        if self.config.use_contrastive_loss:
            train_loss = losses.ContrastiveLoss(self.model)
        else:
            train_loss = losses.CosineSimilarityLoss(self.model)
        
        # 准备评估器
        evaluator = None
        if validation_examples:
            val_examples = []
            for example in validation_examples[:100]:  # 限制验证集大小
                val_examples.append([example.query, example.positive_doc, 1.0])
                if example.negative_docs:
                    val_examples.append([example.query, example.negative_docs[0], 0.0])
            
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                val_examples, name='validation'
            )
        
        # 训练
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.config.num_epochs,
            warmup_steps=self.config.warmup_steps,
            evaluator=evaluator,
            evaluation_steps=self.config.evaluation_steps,
            output_path=str(self.config.output_path),
            save_best_model=True,
            optimizer_params={'lr': self.config.learning_rate},
        )
        
        # 保存微调后的模型
        final_model_path = self.config.output_path / "final_model"
        self.model.save(str(final_model_path))
        
        # 评估微调效果
        evaluation_results = await self._evaluate_fine_tuned_model(validation_examples)
        
        logger.success(f"Fine-tuning completed. Model saved to {final_model_path}")
        
        return {
            'model_path': str(final_model_path),
            'training_examples': len(training_examples),
            'evaluation_results': evaluation_results,
            'config': self.config.__dict__
        }
    
    async def _evaluate_fine_tuned_model(
        self,
        test_examples: Optional[List[TrainingExample]] = None
    ) -> Dict[str, float]:
        """评估微调后的模型"""
        
        if not test_examples:
            return {}
        
        logger.info("Evaluating fine-tuned model...")
        
        # 准备测试数据
        queries = []
        positive_docs = []
        negative_docs = []
        
        for example in test_examples[:50]:  # 限制测试集大小
            queries.append(example.query)
            positive_docs.append(example.positive_doc)
            if example.negative_docs:
                negative_docs.append(example.negative_docs[0])
        
        if not queries:
            return {}
        
        # 编码
        query_embeddings = self.model.encode(queries, convert_to_numpy=True)
        positive_embeddings = self.model.encode(positive_docs, convert_to_numpy=True)
        
        # 计算正例相似度
        positive_similarities = []
        for i in range(len(queries)):
            sim = cosine_similarity([query_embeddings[i]], [positive_embeddings[i]])[0][0]
            positive_similarities.append(sim)
        
        avg_positive_similarity = np.mean(positive_similarities)
        
        # 如果有负例，计算负例相似度
        avg_negative_similarity = 0.0
        if negative_docs and len(negative_docs) == len(queries):
            negative_embeddings = self.model.encode(negative_docs, convert_to_numpy=True)
            negative_similarities = []
            for i in range(len(queries)):
                sim = cosine_similarity([query_embeddings[i]], [negative_embeddings[i]])[0][0]
                negative_similarities.append(sim)
            avg_negative_similarity = np.mean(negative_similarities)
        
        # 计算区分度
        discrimination = avg_positive_similarity - avg_negative_similarity
        
        results = {
            'avg_positive_similarity': avg_positive_similarity,
            'avg_negative_similarity': avg_negative_similarity,
            'discrimination': discrimination,
            'test_examples': len(queries)
        }
        
        logger.info(f"Evaluation results: {results}")
        return results

class FineTuningManager:
    """微调管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_model = config.get('embedding_model', 'BAAI/bge-m3')
        self.output_dir = Path(config.get('fine_tuned_models_dir', 'data/fine_tuned_models'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Fine-Tuning Manager initialized")
    
    async def start_fine_tuning(
        self,
        feedback_db_path: Path,
        document_pool: Optional[List[str]] = None,
        custom_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """启动微调过程"""
        
        # 创建微调配置
        fine_tune_config = FineTuneConfig(
            model_name=self.base_model,
            output_path=self.output_dir / f"fine_tuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            batch_size=custom_config.get('batch_size', 16) if custom_config else 16,
            num_epochs=custom_config.get('num_epochs', 1) if custom_config else 1,
            learning_rate=custom_config.get('learning_rate', 2e-5) if custom_config else 2e-5,
            device=self.config.get('device', 'auto')
        )
        
        # 初始化微调器
        fine_tuner = EmbeddingFineTuner(fine_tune_config)
        
        # 准备训练数据
        training_examples = fine_tuner.prepare_training_data(
            feedback_db_path=feedback_db_path,
            document_pool=document_pool
        )
        
        if len(training_examples) < 5:
            raise ValueError("Insufficient training data for fine-tuning")
        
        # 分割训练和验证数据
        split_idx = int(len(training_examples) * 0.8)
        train_examples = training_examples[:split_idx]
        val_examples = training_examples[split_idx:] if len(training_examples) > 10 else None
        
        # 执行微调
        results = await fine_tuner.fine_tune(train_examples, val_examples)
        
        # 记录微调历史
        self._log_fine_tuning_history(results)
        
        return results
    
    def _log_fine_tuning_history(self, results: Dict[str, Any]):
        """记录微调历史"""
        history_file = self.output_dir / "fine_tuning_history.json"
        
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        # 加载现有历史
        history = []
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        # 添加新条目
        history.append(history_entry)
        
        # 保存历史
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Fine-tuning history logged to {history_file}")
    
    def list_fine_tuned_models(self) -> List[Dict[str, Any]]:
        """列出可用的微调模型"""
        models = []
        
        for model_dir in self.output_dir.glob("fine_tuned_*"):
            if model_dir.is_dir():
                model_info = {
                    'name': model_dir.name,
                    'path': str(model_dir),
                    'created': datetime.fromtimestamp(model_dir.stat().st_ctime).isoformat()
                }
                
                # 检查是否有config文件
                config_file = model_dir / "config.json"
                if config_file.exists():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        model_info['config'] = json.load(f)
                
                models.append(model_info)
        
        return sorted(models, key=lambda x: x['created'], reverse=True)

# 使用示例
async def main():
    """测试embedding微调"""
    config = {
        'embedding_model': 'BAAI/bge-m3',
        'device': 'auto',
        'fine_tuned_models_dir': 'data/fine_tuned_models'
    }
    
    # 初始化微调管理器
    manager = FineTuningManager(config)
    
    # 模拟一些文档池
    document_pool = [
        "Transformer是一种基于注意力机制的神经网络架构",
        "BERT使用双向训练来理解上下文",
        "GPT是一种生成式预训练语言模型",
        "CNN主要用于图像处理任务",
        "RNN适合处理序列数据"
    ]
    
    feedback_db_path = Path("data/feedback/feedback.db")
    
    if feedback_db_path.exists():
        try:
            # 启动微调
            results = await manager.start_fine_tuning(
                feedback_db_path=feedback_db_path,
                document_pool=document_pool,
                custom_config={'num_epochs': 1, 'batch_size': 8}
            )
            
            print("Fine-tuning results:")
            print(json.dumps(results, indent=2, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
    
    # 列出可用模型
    models = manager.list_fine_tuned_models()
    print(f"\nAvailable fine-tuned models: {len(models)}")
    for model in models:
        print(f"- {model['name']} (created: {model['created']})")

if __name__ == "__main__":
    asyncio.run(main())