# src/feedback/feedback_system.py
from typing import List, Dict, Optional, Any
import asyncio
import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger
import hashlib
from enum import Enum

class FeedbackType(Enum):
    """反馈类型"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"  # 1-5 stars
    TEXT_FEEDBACK = "text_feedback"
    CORRECTION = "correction"  # User provides correct answer
    RELEVANCE = "relevance"  # Rate document relevance

@dataclass
class FeedbackRecord:
    """反馈记录"""
    feedback_id: str
    session_id: str
    user_query: str
    system_answer: str
    feedback_type: str
    feedback_value: Any  # Rating number, text, etc.
    source_chunks: List[Dict]
    query_analysis: Optional[Dict]
    retrieval_strategies: Optional[List[str]]
    timestamp: str
    user_metadata: Optional[Dict] = None
    
    # Additional context
    response_time: Optional[float] = None
    iterations_used: Optional[int] = None
    confidence_score: Optional[float] = None

@dataclass
class DocumentFeedback:
    """文档反馈记录"""
    document_id: str
    chunk_id: str
    query: str
    relevance_score: float  # 1-5
    is_helpful: bool
    feedback_text: Optional[str]
    timestamp: str
    user_id: Optional[str] = None

class FeedbackDatabase:
    """反馈数据库管理器"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            # 主要反馈表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback_records (
                    feedback_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    user_query TEXT,
                    system_answer TEXT,
                    feedback_type TEXT,
                    feedback_value TEXT,
                    source_chunks TEXT,
                    query_analysis TEXT,
                    retrieval_strategies TEXT,
                    timestamp TEXT,
                    user_metadata TEXT,
                    response_time REAL,
                    iterations_used INTEGER,
                    confidence_score REAL
                )
            ''')
            
            # 文档反馈表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS document_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT,
                    chunk_id TEXT,
                    query TEXT,
                    relevance_score REAL,
                    is_helpful BOOLEAN,
                    feedback_text TEXT,
                    timestamp TEXT,
                    user_id TEXT
                )
            ''')
            
            # 用户会话表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    total_queries INTEGER DEFAULT 0,
                    positive_feedback INTEGER DEFAULT 0,
                    negative_feedback INTEGER DEFAULT 0,
                    user_metadata TEXT
                )
            ''')
            
            conn.commit()
        
        logger.info(f"Feedback database initialized at {self.db_path}")
    
    def store_feedback(self, feedback: FeedbackRecord):
        """存储反馈记录"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO feedback_records 
                (feedback_id, session_id, user_query, system_answer, feedback_type,
                 feedback_value, source_chunks, query_analysis, retrieval_strategies,
                 timestamp, user_metadata, response_time, iterations_used, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.feedback_id,
                feedback.session_id,
                feedback.user_query,
                feedback.system_answer,
                feedback.feedback_type,
                json.dumps(feedback.feedback_value) if feedback.feedback_value else None,
                json.dumps(feedback.source_chunks),
                json.dumps(feedback.query_analysis) if feedback.query_analysis else None,
                json.dumps(feedback.retrieval_strategies) if feedback.retrieval_strategies else None,
                feedback.timestamp,
                json.dumps(feedback.user_metadata) if feedback.user_metadata else None,
                feedback.response_time,
                feedback.iterations_used,
                feedback.confidence_score
            ))
            conn.commit()
    
    def store_document_feedback(self, doc_feedback: DocumentFeedback):
        """存储文档反馈"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO document_feedback
                (document_id, chunk_id, query, relevance_score, is_helpful,
                 feedback_text, timestamp, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc_feedback.document_id,
                doc_feedback.chunk_id,
                doc_feedback.query,
                doc_feedback.relevance_score,
                doc_feedback.is_helpful,
                doc_feedback.feedback_text,
                doc_feedback.timestamp,
                doc_feedback.user_id
            ))
            conn.commit()
    
    def get_feedback_stats(self, days: int = 30) -> Dict[str, Any]:
        """获取反馈统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            # 获取最近N天的统计
            cutoff_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ).isoformat()
            
            cursor = conn.cursor()
            
            # 总体统计
            cursor.execute('''
                SELECT feedback_type, COUNT(*), AVG(CAST(feedback_value as REAL))
                FROM feedback_records 
                WHERE timestamp >= datetime(?, '-{} days')
                GROUP BY feedback_type
            '''.format(days), (cutoff_date,))
            
            feedback_stats = {}
            for row in cursor.fetchall():
                feedback_stats[row[0]] = {
                    'count': row[1],
                    'avg_value': row[2] if row[2] is not None else None
                }
            
            # 获取差评的查询用于分析
            cursor.execute('''
                SELECT user_query, system_answer, feedback_value, timestamp
                FROM feedback_records
                WHERE feedback_type IN ('thumbs_down', 'rating') 
                AND (feedback_type = 'thumbs_down' OR CAST(feedback_value as REAL) <= 2)
                AND timestamp >= datetime(?, '-{} days')
                ORDER BY timestamp DESC
                LIMIT 50
            '''.format(days), (cutoff_date,))
            
            negative_feedback = []
            for row in cursor.fetchall():
                negative_feedback.append({
                    'query': row[0],
                    'answer': row[1],
                    'feedback_value': row[2],
                    'timestamp': row[3]
                })
            
            return {
                'feedback_stats': feedback_stats,
                'negative_feedback': negative_feedback,
                'analysis_period_days': days
            }
    
    def get_training_data(self, limit: int = 1000) -> List[Dict]:
        """获取用于训练的反馈数据"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_query, system_answer, feedback_type, feedback_value,
                       source_chunks, query_analysis, confidence_score
                FROM feedback_records
                WHERE feedback_type IN ('thumbs_up', 'thumbs_down', 'rating', 'correction')
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            training_data = []
            for row in cursor.fetchall():
                training_data.append({
                    'query': row[0],
                    'answer': row[1],
                    'feedback_type': row[2],
                    'feedback_value': json.loads(row[3]) if row[3] else None,
                    'source_chunks': json.loads(row[4]) if row[4] else [],
                    'query_analysis': json.loads(row[5]) if row[5] else None,
                    'confidence_score': row[6]
                })
            
            return training_data

class FeedbackCollector:
    """反馈收集器"""
    
    def __init__(self, db_path: Path):
        self.db = FeedbackDatabase(db_path)
        self.current_session_id = self._generate_session_id()
        logger.info("Feedback Collector initialized")
    
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:16]
    
    def _generate_feedback_id(self, query: str, answer: str) -> str:
        """生成反馈ID"""
        content = f"{query}:{answer}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def collect_thumbs_feedback(
        self,
        query: str,
        answer: str,
        is_positive: bool,
        source_chunks: List[Dict],
        generation_result: Optional[Dict] = None
    ) -> str:
        """收集点赞/点踩反馈"""
        
        feedback_id = self._generate_feedback_id(query, answer)
        
        feedback = FeedbackRecord(
            feedback_id=feedback_id,
            session_id=self.current_session_id,
            user_query=query,
            system_answer=answer,
            feedback_type=FeedbackType.THUMBS_UP.value if is_positive else FeedbackType.THUMBS_DOWN.value,
            feedback_value=1 if is_positive else 0,
            source_chunks=source_chunks,
            query_analysis=generation_result.get('query_analysis') if generation_result else None,
            retrieval_strategies=generation_result.get('retrieval_strategies') if generation_result else None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            response_time=generation_result.get('generation_time') if generation_result else None,
            iterations_used=generation_result.get('iterations_used') if generation_result else None,
            confidence_score=generation_result.get('confidence') if generation_result else None
        )
        
        self.db.store_feedback(feedback)
        logger.info(f"Thumbs feedback collected: {feedback_id} ({'positive' if is_positive else 'negative'})")
        
        return feedback_id
    
    def collect_rating_feedback(
        self,
        query: str,
        answer: str,
        rating: int,  # 1-5
        source_chunks: List[Dict],
        generation_result: Optional[Dict] = None,
        comment: Optional[str] = None
    ) -> str:
        """收集评分反馈"""
        
        feedback_id = self._generate_feedback_id(query, answer)
        
        feedback = FeedbackRecord(
            feedback_id=feedback_id,
            session_id=self.current_session_id,
            user_query=query,
            system_answer=answer,
            feedback_type=FeedbackType.RATING.value,
            feedback_value={"rating": rating, "comment": comment},
            source_chunks=source_chunks,
            query_analysis=generation_result.get('query_analysis') if generation_result else None,
            retrieval_strategies=generation_result.get('retrieval_strategies') if generation_result else None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            response_time=generation_result.get('generation_time') if generation_result else None,
            iterations_used=generation_result.get('iterations_used') if generation_result else None,
            confidence_score=generation_result.get('confidence') if generation_result else None
        )
        
        self.db.store_feedback(feedback)
        logger.info(f"Rating feedback collected: {feedback_id} (rating: {rating}/5)")
        
        return feedback_id
    
    def collect_correction_feedback(
        self,
        query: str,
        wrong_answer: str,
        correct_answer: str,
        source_chunks: List[Dict],
        generation_result: Optional[Dict] = None
    ) -> str:
        """收集纠错反馈"""
        
        feedback_id = self._generate_feedback_id(query, wrong_answer)
        
        feedback = FeedbackRecord(
            feedback_id=feedback_id,
            session_id=self.current_session_id,
            user_query=query,
            system_answer=wrong_answer,
            feedback_type=FeedbackType.CORRECTION.value,
            feedback_value={"correct_answer": correct_answer},
            source_chunks=source_chunks,
            query_analysis=generation_result.get('query_analysis') if generation_result else None,
            retrieval_strategies=generation_result.get('retrieval_strategies') if generation_result else None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            response_time=generation_result.get('generation_time') if generation_result else None,
            iterations_used=generation_result.get('iterations_used') if generation_result else None,
            confidence_score=generation_result.get('confidence') if generation_result else None
        )
        
        self.db.store_feedback(feedback)
        logger.info(f"Correction feedback collected: {feedback_id}")
        
        return feedback_id
    
    def collect_document_feedback(
        self,
        document_id: str,
        chunk_id: str,
        query: str,
        relevance_score: float,
        is_helpful: bool,
        feedback_text: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> None:
        """收集文档相关性反馈"""
        
        doc_feedback = DocumentFeedback(
            document_id=document_id,
            chunk_id=chunk_id,
            query=query,
            relevance_score=relevance_score,
            is_helpful=is_helpful,
            feedback_text=feedback_text,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id
        )
        
        self.db.store_document_feedback(doc_feedback)
        logger.info(f"Document feedback collected for chunk {chunk_id}")

class FeedbackAnalyzer:
    """反馈分析器"""
    
    def __init__(self, db_path: Path):
        self.db = FeedbackDatabase(db_path)
    
    def analyze_performance(self, days: int = 30) -> Dict[str, Any]:
        """分析系统性能"""
        stats = self.db.get_feedback_stats(days)
        
        # 计算总体满意度
        total_positive = 0
        total_negative = 0
        total_ratings = []
        
        for feedback_type, data in stats['feedback_stats'].items():
            if feedback_type == 'thumbs_up':
                total_positive += data['count']
            elif feedback_type == 'thumbs_down':
                total_negative += data['count']
            elif feedback_type == 'rating' and data['avg_value']:
                total_ratings.append(data['avg_value'])
        
        satisfaction_rate = 0
        if total_positive + total_negative > 0:
            satisfaction_rate = total_positive / (total_positive + total_negative)
        
        avg_rating = sum(total_ratings) / len(total_ratings) if total_ratings else 0
        
        # 分析问题模式
        problem_patterns = self._analyze_problem_patterns(stats['negative_feedback'])
        
        return {
            'satisfaction_rate': satisfaction_rate,
            'average_rating': avg_rating,
            'total_feedback': total_positive + total_negative,
            'positive_feedback': total_positive,
            'negative_feedback': total_negative,
            'problem_patterns': problem_patterns,
            'analysis_period': f"{days} days"
        }
    
    def _analyze_problem_patterns(self, negative_feedback: List[Dict]) -> Dict[str, Any]:
        """分析问题模式"""
        if not negative_feedback:
            return {}
        
        # 简单的模式分析
        common_query_words = {}
        common_issues = []
        
        for feedback in negative_feedback:
            query = feedback['query'].lower()
            words = query.split()
            
            for word in words:
                if len(word) > 3:  # 忽略短词
                    common_query_words[word] = common_query_words.get(word, 0) + 1
        
        # 找出最常见的问题词汇
        sorted_words = sorted(common_query_words.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_negative_cases': len(negative_feedback),
            'common_query_terms': sorted_words[:10],
            'sample_cases': negative_feedback[:5]  # 显示前5个案例
        }
    
    def get_improvement_suggestions(self) -> List[str]:
        """获取改进建议"""
        analysis = self.analyze_performance()
        suggestions = []
        
        if analysis['satisfaction_rate'] < 0.7:
            suggestions.append("整体满意度较低，需要改进检索和生成质量")
        
        if analysis['average_rating'] < 3.5:
            suggestions.append("平均评分偏低，建议优化答案的准确性和完整性")
        
        problem_patterns = analysis['problem_patterns']
        if problem_patterns and problem_patterns['common_query_terms']:
            top_problem_term = problem_patterns['common_query_terms'][0][0]
            suggestions.append(f"频繁出现问题的查询词汇：'{top_problem_term}'，建议针对性优化")
        
        if analysis['total_feedback'] < 10:
            suggestions.append("反馈数据不足，建议增加用户反馈收集")
        
        return suggestions

# 使用示例
def main():
    """测试反馈系统"""
    db_path = Path("data/feedback/feedback.db")
    
    # 初始化组件
    collector = FeedbackCollector(db_path)
    analyzer = FeedbackAnalyzer(db_path)
    
    # 模拟收集反馈
    test_chunks = [
        {"chunk_id": "test_1", "content": "测试内容1", "source_id": "doc_1"}
    ]
    
    # 收集正面反馈
    collector.collect_thumbs_feedback(
        query="什么是机器学习？",
        answer="机器学习是人工智能的一个分支...",
        is_positive=True,
        source_chunks=test_chunks
    )
    
    # 收集评分反馈
    collector.collect_rating_feedback(
        query="深度学习的原理是什么？",
        answer="深度学习基于神经网络...",
        rating=4,
        source_chunks=test_chunks,
        comment="回答很详细，但可以再简洁一些"
    )
    
    # 分析性能
    performance = analyzer.analyze_performance(days=7)
    print("性能分析结果:")
    print(json.dumps(performance, indent=2, ensure_ascii=False))
    
    # 获取改进建议
    suggestions = analyzer.get_improvement_suggestions()
    print("\n改进建议:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")

if __name__ == "__main__":
    main()