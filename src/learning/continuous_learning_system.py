# src/learning/continuous_learning_system.py
"""
持续学习系统
实现用户反馈收集、模型增量微调、知识图谱动态更新等功能
建立系统自我进化的闭环机制
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import asyncio
import time
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

# 机器学习组件
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from sentence_transformers import SentenceTransformer, losses
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, model fine-tuning disabled")

# 本地组件
from ..feedback.feedback_system import FeedbackCollector, UserFeedback
from ..retrieval.hybrid_retriever import EnhancedDocument


@dataclass
class LearningEvent:
    """学习事件"""
    event_id: str
    event_type: str  # 'feedback', 'error', 'improvement'
    timestamp: datetime
    data: Dict[str, Any]
    processed: bool = False
    learning_impact: float = 0.0


@dataclass
class ModelPerformanceHistory:
    """模型性能历史"""
    model_name: str
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    user_satisfaction: float
    cost_efficiency: float


@dataclass
class LearningInsight:
    """学习洞察"""
    insight_type: str
    description: str
    confidence: float
    suggested_action: str
    impact_estimation: float
    evidence: List[Dict[str, Any]]


class FeedbackAnalyzer:
    """反馈分析器"""
    
    def __init__(self):
        self.feedback_patterns = {}
        self.quality_trends = []
        logger.info("Feedback analyzer initialized")
    
    async def analyze_feedback_batch(
        self,
        feedbacks: List[UserFeedback]
    ) -> List[LearningInsight]:
        """批量分析用户反馈"""
        
        insights = []
        
        # 1. 分析整体满意度趋势
        satisfaction_insight = await self._analyze_satisfaction_trends(feedbacks)
        if satisfaction_insight:
            insights.append(satisfaction_insight)
        
        # 2. 分析问题模式
        problem_patterns = await self._analyze_problem_patterns(feedbacks)
        insights.extend(problem_patterns)
        
        # 3. 分析文档相关性反馈
        relevance_insights = await self._analyze_relevance_feedback(feedbacks)
        insights.extend(relevance_insights)
        
        # 4. 分析用户纠错数据
        correction_insights = await self._analyze_corrections(feedbacks)
        insights.extend(correction_insights)
        
        return insights
    
    async def _analyze_satisfaction_trends(
        self,
        feedbacks: List[UserFeedback]
    ) -> Optional[LearningInsight]:
        """分析满意度趋势"""
        
        # 按时间分组计算满意度
        time_groups = {}
        for feedback in feedbacks:
            if feedback.feedback_type == 'rating':
                day = feedback.timestamp.date()
                if day not in time_groups:
                    time_groups[day] = []
                time_groups[day].append(feedback.rating or 0)
        
        if len(time_groups) < 2:
            return None
        
        # 计算趋势
        daily_averages = []
        for day in sorted(time_groups.keys()):
            avg_rating = sum(time_groups[day]) / len(time_groups[day])
            daily_averages.append(avg_rating)
        
        # 简单趋势分析
        recent_avg = sum(daily_averages[-3:]) / min(3, len(daily_averages))
        overall_avg = sum(daily_averages) / len(daily_averages)
        
        if recent_avg < overall_avg - 0.5:
            return LearningInsight(
                insight_type="satisfaction_decline",
                description=f"用户满意度呈下降趋势：近期评分{recent_avg:.2f}，整体评分{overall_avg:.2f}",
                confidence=0.8,
                suggested_action="需要立即调查近期系统变更，优化响应质量",
                impact_estimation=0.9,
                evidence=[{"daily_averages": daily_averages}]
            )
        elif recent_avg > overall_avg + 0.3:
            return LearningInsight(
                insight_type="satisfaction_improvement", 
                description=f"用户满意度持续提升：近期评分{recent_avg:.2f}，整体评分{overall_avg:.2f}",
                confidence=0.7,
                suggested_action="保持当前优化策略，并识别成功因素",
                impact_estimation=0.6,
                evidence=[{"daily_averages": daily_averages}]
            )
        
        return None
    
    async def _analyze_problem_patterns(
        self,
        feedbacks: List[UserFeedback]
    ) -> List[LearningInsight]:
        """分析问题模式"""
        
        insights = []
        
        # 收集负面反馈
        negative_feedbacks = [
            f for f in feedbacks 
            if (f.feedback_type == 'thumbs' and not f.helpful) or
               (f.feedback_type == 'rating' and (f.rating or 0) < 3)
        ]
        
        if len(negative_feedbacks) < 5:
            return insights
        
        # 分析评论中的关键词
        comment_keywords = {}
        for feedback in negative_feedbacks:
            if feedback.comment:
                words = feedback.comment.lower().split()
                for word in words:
                    if len(word) > 2:  # 忽略短词
                        comment_keywords[word] = comment_keywords.get(word, 0) + 1
        
        # 找出高频问题词汇
        common_issues = sorted(comment_keywords.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if common_issues and common_issues[0][1] >= 3:
            most_common_issue = common_issues[0]
            insights.append(LearningInsight(
                insight_type="common_complaint",
                description=f"用户频繁提到问题：'{most_common_issue[0]}'，出现{most_common_issue[1]}次",
                confidence=0.7,
                suggested_action=f"重点优化与'{most_common_issue[0]}'相关的系统功能",
                impact_estimation=0.8,
                evidence=[{"keyword_frequency": dict(common_issues)}]
            ))
        
        return insights
    
    async def _analyze_relevance_feedback(
        self,
        feedbacks: List[UserFeedback]
    ) -> List[LearningInsight]:
        """分析文档相关性反馈"""
        
        insights = []
        
        # 收集文档相关性反馈
        doc_relevance = {}
        for feedback in feedbacks:
            if feedback.feedback_type == 'document_relevance' and feedback.document_id:
                doc_id = feedback.document_id
                if doc_id not in doc_relevance:
                    doc_relevance[doc_id] = []
                doc_relevance[doc_id].append(feedback.rating or 0)
        
        # 识别低质量文档
        low_quality_docs = []
        high_quality_docs = []
        
        for doc_id, ratings in doc_relevance.items():
            if len(ratings) >= 3:  # 至少3次评价
                avg_rating = sum(ratings) / len(ratings)
                if avg_rating < 2.0:
                    low_quality_docs.append((doc_id, avg_rating))
                elif avg_rating > 4.0:
                    high_quality_docs.append((doc_id, avg_rating))
        
        if low_quality_docs:
            insights.append(LearningInsight(
                insight_type="low_quality_documents",
                description=f"发现{len(low_quality_docs)}个低质量文档需要处理",
                confidence=0.9,
                suggested_action="审查和改进低评分文档，考虑从检索结果中降权或移除",
                impact_estimation=0.7,
                evidence=[{"low_quality_docs": low_quality_docs[:10]}]
            ))
        
        return insights
    
    async def _analyze_corrections(
        self,
        feedbacks: List[UserFeedback]
    ) -> List[LearningInsight]:
        """分析用户纠错数据"""
        
        insights = []
        
        # 收集纠错反馈
        corrections = [f for f in feedbacks if f.feedback_type == 'correction' and f.corrected_answer]
        
        if len(corrections) < 5:
            return insights
        
        # 分析纠错模式
        correction_patterns = {}
        for correction in corrections:
            query = correction.metadata.get('original_query', '')
            query_type = self._classify_query_type(query)
            
            if query_type not in correction_patterns:
                correction_patterns[query_type] = 0
            correction_patterns[query_type] += 1
        
        # 找出纠错最多的查询类型
        most_corrected = max(correction_patterns.items(), key=lambda x: x[1])
        
        if most_corrected[1] >= 3:
            insights.append(LearningInsight(
                insight_type="frequent_corrections",
                description=f"'{most_corrected[0]}'类型查询经常被用户纠错({most_corrected[1]}次)",
                confidence=0.8,
                suggested_action=f"重点改进{most_corrected[0]}类型查询的处理逻辑",
                impact_estimation=0.8,
                evidence=[{"correction_patterns": correction_patterns}]
            ))
        
        return insights
    
    def _classify_query_type(self, query: str) -> str:
        """简单的查询类型分类"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['什么是', '什么叫', '定义']):
            return 'definition'
        elif any(word in query_lower for word in ['如何', '怎样', '怎么']):
            return 'how_to'
        elif any(word in query_lower for word in ['为什么', '原因']):
            return 'why'
        elif any(word in query_lower for word in ['比较', '区别', '差异']):
            return 'comparison'
        else:
            return 'general'


class EmbeddingFineTuner:
    """嵌入模型微调器"""
    
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        self.model_name = model_name
        self.model = None
        self.fine_tuned_versions = {}
        
        if TORCH_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Embedding model loaded: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
        else:
            logger.warning("PyTorch not available, embedding fine-tuning disabled")
    
    async def prepare_training_data(
        self,
        feedbacks: List[UserFeedback]
    ) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, str]]]:
        """准备训练数据"""
        
        if not self.model:
            return [], []
        
        # 正负样本对
        triplets = []  # (query, positive_doc, negative_doc)
        pairs = []     # (query, doc, label)
        
        for feedback in feedbacks:
            query = feedback.metadata.get('original_query', '')
            if not query:
                continue
            
            # 从文档相关性反馈生成训练样本
            if feedback.feedback_type == 'document_relevance' and feedback.document_id:
                doc_content = feedback.metadata.get('document_content', '')
                if doc_content:
                    rating = feedback.rating or 0
                    # 4-5分为正样本，1-2分为负样本
                    if rating >= 4:
                        pairs.append((query, doc_content, 1))
                    elif rating <= 2:
                        pairs.append((query, doc_content, 0))
            
            # 从用户纠错生成对比样本
            elif feedback.feedback_type == 'correction':
                original_answer = feedback.metadata.get('original_answer', '')
                corrected_answer = feedback.corrected_answer or ''
                
                if original_answer and corrected_answer:
                    # 纠正后的答案为正样本
                    pairs.append((query, corrected_answer, 1))
                    pairs.append((query, original_answer, 0))
        
        # 生成困难负例三元组
        triplets = await self._generate_hard_triplets(pairs)
        
        logger.info(f"Prepared training data: {len(pairs)} pairs, {len(triplets)} triplets")
        return triplets, pairs
    
    async def _generate_hard_triplets(
        self,
        pairs: List[Tuple[str, str, int]]
    ) -> List[Tuple[str, str, str]]:
        """生成困难负例三元组"""
        
        triplets = []
        
        # 按查询分组
        query_groups = {}
        for query, doc, label in pairs:
            if query not in query_groups:
                query_groups[query] = {'positive': [], 'negative': []}
            
            if label == 1:
                query_groups[query]['positive'].append(doc)
            else:
                query_groups[query]['negative'].append(doc)
        
        # 为每个查询生成三元组
        for query, docs in query_groups.items():
            positives = docs['positive']
            negatives = docs['negative']
            
            if positives and negatives:
                # 为每个正样本选择困难负样本
                for pos_doc in positives:
                    # 选择最相似的负样本作为困难负例
                    if negatives:
                        hard_negative = negatives[0]  # 简化实现
                        triplets.append((query, pos_doc, hard_negative))
        
        return triplets
    
    async def incremental_fine_tune(
        self,
        training_data: List[Tuple[str, str, int]],
        model_version: str = None
    ) -> bool:
        """增量微调模型"""
        
        if not self.model or not training_data:
            logger.warning("Cannot fine-tune: model not available or no training data")
            return False
        
        try:
            # 准备数据集
            dataset = ContrastiveDataset(training_data)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
            
            # 定义损失函数
            train_loss = losses.ContrastiveLoss(model=self.model)
            
            # 执行微调
            logger.info(f"Starting incremental fine-tuning with {len(training_data)} samples")
            
            self.model.fit(
                train_objectives=[(dataloader, train_loss)],
                epochs=1,  # 增量训练只需要少量epochs
                warmup_steps=100,
                output_path=None  # 不保存，只在内存中更新
            )
            
            # 记录微调版本
            version_key = model_version or f"ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.fine_tuned_versions[version_key] = {
                'timestamp': datetime.now(),
                'training_samples': len(training_data),
                'performance_before': None,  # 可以添加性能评估
                'performance_after': None
            }
            
            logger.success(f"Fine-tuning completed: version {version_key}")
            return True
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'base_model': self.model_name,
            'fine_tuned_versions': len(self.fine_tuned_versions),
            'latest_version': max(self.fine_tuned_versions.keys()) if self.fine_tuned_versions else None,
            'available': self.model is not None
        }


class ContrastiveDataset(Dataset):
    """对比学习数据集"""
    
    def __init__(self, data: List[Tuple[str, str, int]]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        query, doc, label = self.data[idx]
        return {
            'texts': [query, doc],
            'label': float(label)
        }


class KnowledgeGraphUpdater:
    """知识图谱更新器"""
    
    def __init__(self, kg_path: str):
        self.kg_path = Path(kg_path)
        self.update_queue = []
        self.update_stats = {
            'entities_added': 0,
            'relations_added': 0,
            'entities_updated': 0,
            'last_update': None
        }
        logger.info(f"Knowledge graph updater initialized: {kg_path}")
    
    async def analyze_new_content(
        self,
        content: str,
        source_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """分析新内容以提取知识"""
        
        # 简化的实体和关系提取
        entities = await self._extract_entities(content)
        relations = await self._extract_relations(content, entities)
        
        updates = []
        
        # 实体更新
        for entity in entities:
            updates.append({
                'type': 'entity',
                'action': 'add_or_update',
                'data': entity,
                'source': source_metadata
            })
        
        # 关系更新  
        for relation in relations:
            updates.append({
                'type': 'relation',
                'action': 'add',
                'data': relation,
                'source': source_metadata
            })
        
        return updates
    
    async def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """提取实体（简化实现）"""
        
        # 简单的关键词匹配实体提取
        ai_terms = [
            'transformer', 'bert', 'gpt', 'llama', 'attention', 'neural network',
            'deep learning', 'machine learning', 'nlp', 'computer vision'
        ]
        
        entities = []
        content_lower = content.lower()
        
        for term in ai_terms:
            if term in content_lower:
                entities.append({
                    'name': term,
                    'type': 'ALGORITHM' if term in ['transformer', 'bert', 'gpt'] else 'CONCEPT',
                    'confidence': 0.8,
                    'context': content[:200]  # 简化上下文
                })
        
        return entities
    
    async def _extract_relations(
        self,
        content: str, 
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """提取关系（简化实现）"""
        
        relations = []
        
        # 简单的关系模式匹配
        relation_patterns = {
            'based on': 'BASED_ON',
            'improves': 'IMPROVES',
            'uses': 'USES',
            'outperforms': 'OUTPERFORMS'
        }
        
        content_lower = content.lower()
        entity_names = [e['name'] for e in entities]
        
        for pattern, relation_type in relation_patterns.items():
            if pattern in content_lower:
                # 寻找模式周围的实体
                for i, entity1 in enumerate(entity_names):
                    for entity2 in entity_names[i+1:]:
                        if entity1 in content_lower and entity2 in content_lower:
                            relations.append({
                                'subject': entity1,
                                'relation': relation_type,
                                'object': entity2,
                                'confidence': 0.6,
                                'evidence': content[:200]
                            })
        
        return relations
    
    async def apply_updates(self, updates: List[Dict[str, Any]]) -> bool:
        """应用知识图谱更新"""
        
        try:
            for update in updates:
                if update['type'] == 'entity':
                    self.update_stats['entities_added'] += 1
                elif update['type'] == 'relation':
                    self.update_stats['relations_added'] += 1
            
            self.update_stats['last_update'] = datetime.now()
            logger.info(f"Applied {len(updates)} knowledge graph updates")
            return True
            
        except Exception as e:
            logger.error(f"Knowledge graph update failed: {e}")
            return False
    
    def get_update_stats(self) -> Dict[str, Any]:
        """获取更新统计"""
        return self.update_stats.copy()


class ContinuousLearningOrchestrator:
    """持续学习协调器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化组件
        self.feedback_collector = FeedbackCollector(config.get('feedback_db_path', 'data/feedback.db'))
        self.feedback_analyzer = FeedbackAnalyzer()
        self.embedding_fine_tuner = EmbeddingFineTuner(config.get('embedding_model', 'BAAI/bge-m3'))
        self.kg_updater = KnowledgeGraphUpdater(config.get('kg_path', 'data/knowledge_graph'))
        
        # 学习状态
        self.learning_events = []
        self.performance_history = []
        self.learning_schedule = {
            'feedback_analysis_interval': config.get('feedback_analysis_interval', 3600),  # 1小时
            'model_fine_tune_interval': config.get('model_fine_tune_interval', 86400),    # 1天
            'kg_update_interval': config.get('kg_update_interval', 43200),               # 12小时
        }
        
        # 启动定期任务
        self._start_background_tasks()
        
        logger.info("Continuous learning orchestrator initialized")
    
    def _start_background_tasks(self):
        """启动后台任务"""
        
        # 定期反馈分析
        asyncio.create_task(self._periodic_feedback_analysis())
        
        # 定期模型微调
        asyncio.create_task(self._periodic_model_fine_tuning())
        
        # 定期知识图谱更新
        asyncio.create_task(self._periodic_kg_update())
        
        logger.info("Background learning tasks started")
    
    async def _periodic_feedback_analysis(self):
        """定期反馈分析任务"""
        
        while True:
            try:
                await asyncio.sleep(self.learning_schedule['feedback_analysis_interval'])
                await self.analyze_recent_feedback()
            except Exception as e:
                logger.error(f"Periodic feedback analysis failed: {e}")
    
    async def _periodic_model_fine_tuning(self):
        """定期模型微调任务"""
        
        while True:
            try:
                await asyncio.sleep(self.learning_schedule['model_fine_tune_interval'])
                await self.perform_incremental_learning()
            except Exception as e:
                logger.error(f"Periodic model fine-tuning failed: {e}")
    
    async def _periodic_kg_update(self):
        """定期知识图谱更新任务"""
        
        while True:
            try:
                await asyncio.sleep(self.learning_schedule['kg_update_interval'])
                await self.update_knowledge_graph()
            except Exception as e:
                logger.error(f"Periodic KG update failed: {e}")
    
    async def analyze_recent_feedback(self) -> List[LearningInsight]:
        """分析最近的反馈"""
        
        # 获取最近24小时的反馈
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_feedbacks = await self.feedback_collector.get_feedback_since(cutoff_time)
        
        if not recent_feedbacks:
            logger.info("No recent feedback to analyze")
            return []
        
        logger.info(f"Analyzing {len(recent_feedbacks)} recent feedback entries")
        
        # 分析反馈
        insights = await self.feedback_analyzer.analyze_feedback_batch(recent_feedbacks)
        
        # 记录学习事件
        for insight in insights:
            event = LearningEvent(
                event_id=f"insight_{int(time.time())}",
                event_type='feedback_insight',
                timestamp=datetime.now(),
                data=asdict(insight),
                learning_impact=insight.impact_estimation
            )
            self.learning_events.append(event)
        
        logger.info(f"Generated {len(insights)} learning insights")
        return insights
    
    async def perform_incremental_learning(self) -> bool:
        """执行增量学习"""
        
        logger.info("Starting incremental learning process")
        
        # 获取用于训练的反馈数据
        cutoff_time = datetime.now() - timedelta(days=7)  # 最近7天的数据
        training_feedbacks = await self.feedback_collector.get_feedback_since(cutoff_time)
        
        if len(training_feedbacks) < 10:
            logger.info("Insufficient training data for incremental learning")
            return False
        
        # 准备训练数据
        triplets, pairs = await self.embedding_fine_tuner.prepare_training_data(training_feedbacks)
        
        if len(pairs) < 5:
            logger.info("Insufficient quality training pairs for fine-tuning")
            return False
        
        # 执行微调
        success = await self.embedding_fine_tuner.incremental_fine_tune(pairs)
        
        if success:
            # 记录学习事件
            event = LearningEvent(
                event_id=f"fine_tune_{int(time.time())}",
                event_type='model_fine_tune',
                timestamp=datetime.now(),
                data={
                    'training_samples': len(pairs),
                    'model_version': f"ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                },
                learning_impact=0.7
            )
            self.learning_events.append(event)
            logger.success("Incremental learning completed successfully")
        
        return success
    
    async def update_knowledge_graph(self) -> bool:
        """更新知识图谱"""
        
        logger.info("Starting knowledge graph update")
        
        # 这里应该获取新的文档内容进行分析
        # 目前使用模拟数据
        new_content = "Transformer architecture improves upon traditional attention mechanisms..."
        source_metadata = {
            'source': 'continuous_learning',
            'timestamp': datetime.now().isoformat()
        }
        
        # 分析内容
        updates = await self.kg_updater.analyze_new_content(new_content, source_metadata)
        
        if updates:
            # 应用更新
            success = await self.kg_updater.apply_updates(updates)
            
            if success:
                # 记录学习事件
                event = LearningEvent(
                    event_id=f"kg_update_{int(time.time())}",
                    event_type='kg_update',
                    timestamp=datetime.now(),
                    data={
                        'updates_count': len(updates),
                        'content_analyzed': len(new_content)
                    },
                    learning_impact=0.5
                )
                self.learning_events.append(event)
                logger.success("Knowledge graph updated successfully")
                return True
        
        logger.info("No knowledge graph updates needed")
        return False
    
    async def process_immediate_feedback(self, feedback: UserFeedback) -> List[str]:
        """处理即时反馈"""
        
        recommendations = []
        
        # 分析单个反馈的影响
        if feedback.feedback_type == 'thumbs' and not feedback.helpful:
            recommendations.append("建议调查该查询的处理流程")
        
        elif feedback.feedback_type == 'rating' and (feedback.rating or 0) < 3:
            recommendations.append("需要改进该类型查询的响应质量")
        
        elif feedback.feedback_type == 'correction':
            recommendations.append("用户纠错数据可用于模型训练")
        
        # 记录即时学习事件
        event = LearningEvent(
            event_id=f"immediate_{int(time.time())}",
            event_type='immediate_feedback',
            timestamp=datetime.now(),
            data=asdict(feedback),
            learning_impact=0.3
        )
        self.learning_events.append(event)
        
        return recommendations
    
    def get_learning_dashboard_data(self) -> Dict[str, Any]:
        """获取学习仪表板数据"""
        
        # 最近的学习事件
        recent_events = [
            event for event in self.learning_events
            if event.timestamp > datetime.now() - timedelta(days=7)
        ]
        
        # 按类型统计事件
        event_stats = {}
        for event in recent_events:
            event_stats[event.event_type] = event_stats.get(event.event_type, 0) + 1
        
        # 计算学习影响
        total_impact = sum(event.learning_impact for event in recent_events)
        avg_impact = total_impact / len(recent_events) if recent_events else 0
        
        return {
            'learning_status': 'active' if recent_events else 'idle',
            'recent_events_count': len(recent_events),
            'event_types_distribution': event_stats,
            'total_learning_impact': total_impact,
            'average_impact': avg_impact,
            'last_learning_event': recent_events[-1].timestamp.isoformat() if recent_events else None,
            'embedding_model_info': self.embedding_fine_tuner.get_model_info(),
            'kg_update_stats': self.kg_updater.get_update_stats()
        }
    
    def export_learning_report(self, file_path: str):
        """导出学习报告"""
        
        dashboard_data = self.get_learning_dashboard_data()
        
        # 添加详细的事件历史
        dashboard_data['detailed_events'] = [
            {
                'event_id': event.event_id,
                'type': event.event_type,
                'timestamp': event.timestamp.isoformat(),
                'impact': event.learning_impact,
                'processed': event.processed
            }
            for event in self.learning_events[-50:]  # 最近50个事件
        ]
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
            logger.success(f"Learning report exported to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export learning report: {e}")
    
    def update_learning_schedule(self, new_schedule: Dict[str, int]):
        """更新学习调度"""
        self.learning_schedule.update(new_schedule)
        logger.info(f"Learning schedule updated: {self.learning_schedule}")
    
    async def manual_trigger_learning(self, learning_type: str) -> bool:
        """手动触发学习任务"""
        
        if learning_type == 'feedback_analysis':
            insights = await self.analyze_recent_feedback()
            return len(insights) > 0
        
        elif learning_type == 'model_fine_tune':
            return await self.perform_incremental_learning()
        
        elif learning_type == 'kg_update':
            return await self.update_knowledge_graph()
        
        else:
            logger.warning(f"Unknown learning type: {learning_type}")
            return False