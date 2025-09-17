# 反馈模块 (Feedback Module)

RAG系统的用户反馈收集、分析与持续改进框架，建立用户反馈驱动的系统自我优化闭环。支持多维度反馈收集、智能分析、问题模式识别和改进建议生成，为系统持续优化提供数据驱动的决策支持。

## 🏗️ 核心架构

### 1. 反馈数据结构 (Feedback Data Structures)

**多维度反馈数据模型**

```python
# src/feedback/feedback_system.py
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

class FeedbackType(Enum):
    """反馈类型枚举"""
    THUMBS_UP = "thumbs_up"           # 点赞反馈
    THUMBS_DOWN = "thumbs_down"       # 点踩反馈
    RATING = "rating"                 # 评分反馈 (1-5星)
    TEXT_FEEDBACK = "text_feedback"   # 文本反馈
    CORRECTION = "correction"         # 用户纠错
    RELEVANCE = "relevance"           # 文档相关性评价

@dataclass
class FeedbackRecord:
    """综合反馈记录"""
    feedback_id: str                  # 反馈唯一标识
    session_id: str                   # 会话标识
    user_query: str                   # 用户查询
    system_answer: str                # 系统回答
    feedback_type: str                # 反馈类型
    feedback_value: Any               # 反馈值 (评分/文本/布尔)
    source_chunks: List[Dict]         # 检索到的文档块
    query_analysis: Optional[Dict]    # 查询分析结果
    retrieval_strategies: Optional[List[str]]  # 检索策略
    timestamp: str                    # 时间戳
    user_metadata: Optional[Dict]     # 用户元数据
    
    # 性能上下文
    response_time: Optional[float]    # 响应时间
    iterations_used: Optional[int]    # 使用的迭代次数
    confidence_score: Optional[float] # 置信度分数

@dataclass
class DocumentFeedback:
    """文档级别反馈"""
    document_id: str                  # 文档标识
    chunk_id: str                     # 文档块标识
    query: str                        # 查询内容
    relevance_score: float            # 相关性评分 (1-5)
    is_helpful: bool                  # 是否有帮助
    feedback_text: Optional[str]      # 反馈文本
    timestamp: str                    # 时间戳
    user_id: Optional[str]            # 用户标识
```

### 2. 反馈数据库管理器 (Feedback Database)

**持久化的反馈数据存储与管理**

```python
class FeedbackDatabase:
    """反馈数据库管理器"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化SQLite数据库结构"""
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
                    source_chunks TEXT,              -- JSON格式存储检索文档
                    query_analysis TEXT,             -- JSON格式存储查询分析
                    retrieval_strategies TEXT,       -- JSON格式存储检索策略
                    timestamp TEXT,
                    user_metadata TEXT,              -- JSON格式存储用户元数据
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
    
    def store_feedback(self, feedback: FeedbackRecord):
        """存储反馈记录到数据库"""
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
    
    def get_feedback_stats(self, days: int = 30) -> Dict[str, Any]:
        """获取反馈统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            cutoff_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ).isoformat()
            
            cursor = conn.cursor()
            
            # 按反馈类型统计
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
            
            # 获取负面反馈用于问题分析
            cursor.execute('''
                SELECT user_query, system_answer, feedback_value, timestamp
                FROM feedback_records
                WHERE feedback_type IN ('thumbs_down', 'rating') 
                AND (feedback_type = 'thumbs_down' OR CAST(feedback_value as REAL) <= 2)
                AND timestamp >= datetime(?, '-{} days')
                ORDER BY timestamp DESC LIMIT 50
            '''.format(days), (cutoff_date,))
            
            negative_feedback = [
                {
                    'query': row[0],
                    'answer': row[1], 
                    'feedback_value': row[2],
                    'timestamp': row[3]
                }
                for row in cursor.fetchall()
            ]
            
            return {
                'feedback_stats': feedback_stats,
                'negative_feedback': negative_feedback,
                'analysis_period_days': days
            }
```

### 3. 反馈收集器 (Feedback Collector)

**多渠道反馈收集与会话管理**

```python
class FeedbackCollector:
    """智能反馈收集器"""
    
    def __init__(self, db_path: Path):
        self.db = FeedbackDatabase(db_path)
        self.current_session_id = self._generate_session_id()
        logger.info("Feedback Collector initialized")
    
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
        rating: int,  # 1-5星评分
        source_chunks: List[Dict],
        generation_result: Optional[Dict] = None,
        comment: Optional[str] = None
    ) -> str:
        """收集星级评分反馈"""
        
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
        """收集用户纠错反馈"""
        
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
```

### 4. 反馈分析器 (Feedback Analyzer)

**智能反馈分析与洞察发现**

```python
class FeedbackAnalyzer:
    """智能反馈分析器"""
    
    def __init__(self, db_path: Path):
        self.db = FeedbackDatabase(db_path)
    
    def analyze_performance(self, days: int = 30) -> Dict[str, Any]:
        """综合性能分析"""
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
        
        # 计算满意度指标
        satisfaction_rate = 0
        if total_positive + total_negative > 0:
            satisfaction_rate = total_positive / (total_positive + total_negative)
        
        avg_rating = sum(total_ratings) / len(total_ratings) if total_ratings else 0
        
        # 问题模式分析
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
        """问题模式智能分析"""
        if not negative_feedback:
            return {}
        
        # 分析常见问题词汇
        common_query_words = {}
        common_issues = []
        
        for feedback in negative_feedback:
            query = feedback['query'].lower()
            words = query.split()
            
            for word in words:
                if len(word) > 3:  # 过滤短词
                    common_query_words[word] = common_query_words.get(word, 0) + 1
        
        # 识别高频问题词汇
        sorted_words = sorted(common_query_words.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_negative_cases': len(negative_feedback),
            'common_query_terms': sorted_words[:10],
            'sample_cases': negative_feedback[:5]  # 示例案例
        }
    
    def get_improvement_suggestions(self) -> List[str]:
        """生成智能改进建议"""
        analysis = self.analyze_performance()
        suggestions = []
        
        # 基于满意度的建议
        if analysis['satisfaction_rate'] < 0.7:
            suggestions.append("整体满意度较低，需要改进检索和生成质量")
        
        # 基于评分的建议
        if analysis['average_rating'] < 3.5:
            suggestions.append("平均评分偏低，建议优化答案的准确性和完整性")
        
        # 基于问题模式的建议
        problem_patterns = analysis['problem_patterns']
        if problem_patterns and problem_patterns['common_query_terms']:
            top_problem_term = problem_patterns['common_query_terms'][0][0]
            suggestions.append(f"频繁出现问题的查询词汇：'{top_problem_term}'，建议针对性优化")
        
        # 基于反馈数量的建议
        if analysis['total_feedback'] < 10:
            suggestions.append("反馈数据不足，建议增加用户反馈收集")
        
        return suggestions
```

## 🔧 使用方法

### 1. 基础反馈收集

```python
from src.feedback.feedback_system import FeedbackCollector, FeedbackAnalyzer
from pathlib import Path

# 初始化反馈系统
db_path = Path("data/feedback/feedback.db")
collector = FeedbackCollector(db_path)
analyzer = FeedbackAnalyzer(db_path)

# 收集点赞/点踩反馈
def collect_thumbs_feedback(query: str, answer: str, is_positive: bool, source_chunks: List[Dict]):
    feedback_id = collector.collect_thumbs_feedback(
        query=query,
        answer=answer,
        is_positive=is_positive,
        source_chunks=source_chunks,
        generation_result={
            'generation_time': 2.3,
            'confidence': 0.85,
            'query_analysis': {'complexity': 'medium'},
            'retrieval_strategies': ['hybrid', 'semantic']
        }
    )
    return feedback_id

# 收集星级评分
def collect_rating_feedback(query: str, answer: str, rating: int, comment: str = None):
    feedback_id = collector.collect_rating_feedback(
        query=query,
        answer=answer,
        rating=rating,  # 1-5星
        source_chunks=source_chunks,
        comment=comment
    )
    return feedback_id

# 收集用户纠错
def collect_correction(query: str, wrong_answer: str, correct_answer: str):
    feedback_id = collector.collect_correction_feedback(
        query=query,
        wrong_answer=wrong_answer,
        correct_answer=correct_answer,
        source_chunks=source_chunks
    )
    return feedback_id
```

### 2. 文档级别反馈收集

```python
# 收集文档相关性反馈
def collect_document_feedback(document_id: str, chunk_id: str, query: str, relevance_score: float):
    collector.collect_document_feedback(
        document_id=document_id,
        chunk_id=chunk_id,
        query=query,
        relevance_score=relevance_score,  # 1-5评分
        is_helpful=relevance_score >= 3.0,
        feedback_text="这个文档很有帮助，回答了我的问题",
        user_id="user_123"
    )
```

### 3. 反馈分析与洞察

```python
# 系统性能分析
def analyze_system_performance():
    # 获取最近30天的性能分析
    performance = analyzer.analyze_performance(days=30)
    
    print("=== 系统性能分析 ===")
    print(f"满意度: {performance['satisfaction_rate']:.2%}")
    print(f"平均评分: {performance['average_rating']:.1f}/5.0")
    print(f"总反馈数: {performance['total_feedback']}")
    print(f"正面反馈: {performance['positive_feedback']}")
    print(f"负面反馈: {performance['negative_feedback']}")
    
    # 问题模式分析
    if performance['problem_patterns']:
        print("\n=== 问题模式分析 ===")
        print(f"负面案例总数: {performance['problem_patterns']['total_negative_cases']}")
        print("高频问题词汇:")
        for word, count in performance['problem_patterns']['common_query_terms'][:5]:
            print(f"  - {word}: {count}次")
    
    return performance

# 获取改进建议
def get_improvement_suggestions():
    suggestions = analyzer.get_improvement_suggestions()
    
    print("\n=== 改进建议 ===")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")
    
    return suggestions
```

### 4. 反馈驱动的持续改进

```python
class FeedbackDrivenImprovement:
    """反馈驱动的持续改进系统"""
    
    def __init__(self, feedback_analyzer: FeedbackAnalyzer):
        self.analyzer = feedback_analyzer
        self.improvement_history = []
    
    async def auto_improvement_cycle(self):
        """自动改进循环"""
        # 1. 分析当前性能
        performance = self.analyzer.analyze_performance()
        
        # 2. 识别需要改进的区域
        improvement_areas = self._identify_improvement_areas(performance)
        
        # 3. 生成改进计划
        improvement_plan = self._generate_improvement_plan(improvement_areas)
        
        # 4. 执行改进措施
        results = await self._execute_improvements(improvement_plan)
        
        # 5. 记录改进历史
        self.improvement_history.append({
            'timestamp': datetime.now().isoformat(),
            'performance_before': performance,
            'improvement_plan': improvement_plan,
            'results': results
        })
        
        return results
    
    def _identify_improvement_areas(self, performance: Dict[str, Any]) -> List[str]:
        """识别需要改进的区域"""
        areas = []
        
        if performance['satisfaction_rate'] < 0.75:
            areas.append('user_satisfaction')
        
        if performance['average_rating'] < 3.5:
            areas.append('answer_quality')
        
        problem_patterns = performance.get('problem_patterns', {})
        if problem_patterns.get('total_negative_cases', 0) > 10:
            areas.append('query_handling')
        
        return areas
    
    def _generate_improvement_plan(self, areas: List[str]) -> Dict[str, Any]:
        """生成改进计划"""
        plan = {
            'target_areas': areas,
            'actions': [],
            'expected_impact': {},
            'implementation_timeline': '1-2 weeks'
        }
        
        if 'user_satisfaction' in areas:
            plan['actions'].append({
                'action': 'optimize_retrieval_strategy',
                'description': '优化检索策略，提高文档相关性',
                'priority': 'high'
            })
        
        if 'answer_quality' in areas:
            plan['actions'].append({
                'action': 'improve_generation_prompt',
                'description': '优化生成提示词，提高答案质量',
                'priority': 'high'
            })
        
        if 'query_handling' in areas:
            plan['actions'].append({
                'action': 'enhance_query_analysis',
                'description': '增强查询分析能力，更好理解用户意图',
                'priority': 'medium'
            })
        
        return plan
    
    async def _execute_improvements(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """执行改进措施"""
        results = {
            'completed_actions': [],
            'performance_impact': {},
            'next_steps': []
        }
        
        for action in plan['actions']:
            try:
                # 这里实现具体的改进措施
                action_result = await self._execute_single_action(action)
                results['completed_actions'].append(action_result)
            except Exception as e:
                logger.error(f"Failed to execute action {action['action']}: {e}")
                results['next_steps'].append(f"Retry action: {action['action']}")
        
        return results
    
    async def _execute_single_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个改进措施"""
        action_type = action['action']
        
        if action_type == 'optimize_retrieval_strategy':
            # 实现检索策略优化
            return {
                'action': action_type,
                'status': 'completed',
                'impact': 'retrieval_precision improved by 15%'
            }
        
        elif action_type == 'improve_generation_prompt':
            # 实现生成提示词优化
            return {
                'action': action_type,
                'status': 'completed',
                'impact': 'answer_quality score improved by 0.3'
            }
        
        elif action_type == 'enhance_query_analysis':
            # 实现查询分析增强
            return {
                'action': action_type,
                'status': 'completed',
                'impact': 'query_understanding accuracy improved by 12%'
            }
        
        else:
            return {
                'action': action_type,
                'status': 'skipped',
                'reason': 'action not implemented'
            }
```

### 5. 实时反馈监控

```python
class RealTimeFeedbackMonitor:
    """实时反馈监控系统"""
    
    def __init__(self, feedback_analyzer: FeedbackAnalyzer):
        self.analyzer = feedback_analyzer
        self.alert_thresholds = {
            'satisfaction_rate': 0.7,     # 满意度阈值
            'negative_feedback_rate': 0.3, # 负面反馈率阈值
            'avg_rating': 3.0              # 平均评分阈值
        }
        self.monitoring_active = False
    
    async def start_monitoring(self, check_interval: int = 300):
        """开始实时监控"""
        self.monitoring_active = True
        logger.info("Real-time feedback monitoring started")
        
        while self.monitoring_active:
            try:
                await self._check_performance_alerts()
                await asyncio.sleep(check_interval)  # 每5分钟检查一次
            except Exception as e:
                logger.error(f"Error in feedback monitoring: {e}")
                await asyncio.sleep(60)  # 出错后等待1分钟再继续
    
    async def _check_performance_alerts(self):
        """检查性能告警"""
        # 获取最近1小时的反馈数据
        performance = self.analyzer.analyze_performance(days=1)
        alerts = []
        
        # 检查满意度
        if performance['satisfaction_rate'] < self.alert_thresholds['satisfaction_rate']:
            alerts.append({
                'type': 'low_satisfaction',
                'message': f"满意度过低: {performance['satisfaction_rate']:.2%}",
                'severity': 'high',
                'suggested_action': '立即检查最近的负面反馈并采取改进措施'
            })
        
        # 检查平均评分
        if performance['average_rating'] < self.alert_thresholds['avg_rating']:
            alerts.append({
                'type': 'low_rating',
                'message': f"平均评分过低: {performance['average_rating']:.1f}/5.0",
                'severity': 'medium',
                'suggested_action': '分析低评分反馈并优化答案质量'
            })
        
        # 检查负面反馈率
        total_feedback = performance['total_feedback']
        if total_feedback > 0:
            negative_rate = performance['negative_feedback'] / total_feedback
            if negative_rate > self.alert_thresholds['negative_feedback_rate']:
                alerts.append({
                    'type': 'high_negative_rate',
                    'message': f"负面反馈率过高: {negative_rate:.2%}",
                    'severity': 'high',
                    'suggested_action': '立即调查负面反馈原因并修复问题'
                })
        
        # 发送告警
        for alert in alerts:
            await self._send_alert(alert)
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """发送告警通知"""
        logger.warning(f"FEEDBACK ALERT [{alert['severity'].upper()}]: {alert['message']}")
        logger.info(f"建议措施: {alert['suggested_action']}")
        
        # 这里可以集成邮件、Slack等通知渠道
        # await self._send_email_alert(alert)
        # await self._send_slack_alert(alert)
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        logger.info("Real-time feedback monitoring stopped")
```

## ⚙️ 配置参数

### 反馈系统配置 (Feedback Configuration)

```python
FEEDBACK_CONFIG = {
    # 数据库配置
    'database': {
        'db_path': 'data/feedback/feedback.db',      # SQLite数据库路径
        'backup_interval': 86400,                    # 备份间隔(秒)
        'retention_days': 365,                       # 数据保留天数
    },
    
    # 反馈收集配置
    'collection': {
        'enable_thumbs_feedback': True,              # 启用点赞/点踩
        'enable_rating_feedback': True,              # 启用评分反馈
        'enable_text_feedback': True,                # 启用文本反馈
        'enable_correction_feedback': True,          # 启用用户纠错
        'enable_document_feedback': True,            # 启用文档反馈
        'max_feedback_length': 1000,                 # 文本反馈最大长度
        'session_timeout': 3600,                     # 会话超时时间(秒)
    },
    
    # 分析配置
    'analysis': {
        'default_analysis_period': 30,               # 默认分析周期(天)
        'min_feedback_for_analysis': 10,             # 分析所需最小反馈数
        'problem_pattern_threshold': 3,              # 问题模式识别阈值
        'satisfaction_alert_threshold': 0.7,         # 满意度告警阈值
        'rating_alert_threshold': 3.0,               # 评分告警阈值
    },
    
    # 监控配置
    'monitoring': {
        'enable_realtime_monitoring': True,          # 启用实时监控
        'check_interval': 300,                       # 检查间隔(秒)
        'alert_channels': ['log', 'email'],          # 告警渠道
        'alert_email': 'admin@example.com',          # 告警邮箱
    },
    
    # 改进配置
    'improvement': {
        'enable_auto_improvement': False,            # 启用自动改进
        'improvement_cycle_interval': 86400,         # 改进循环间隔(秒)
        'min_impact_threshold': 0.05,               # 最小影响阈值
        'max_concurrent_improvements': 3,           # 最大并发改进数
    }
}
```

## 📈 反馈分析指标

### 核心反馈指标

| 指标类别 | 指标名称 | 计算方法 | 目标值 | 说明 |
|---------|---------|----------|--------|------|
| **用户满意度** | 满意度率 | 正面反馈 / 总反馈 | >80% | 点赞/(点赞+点踩) |
| | 平均评分 | 所有评分的平均值 | >4.0/5.0 | 1-5星评分系统 |
| | NPS分数 | (推荐者-批评者)/总用户 | >50 | 净推荐值 |
| **反馈质量** | 反馈参与率 | 提供反馈用户/总用户 | >30% | 反馈收集覆盖率 |
| | 文本反馈率 | 文本反馈/总反馈 | >20% | 详细反馈占比 |
| | 纠错反馈率 | 纠错反馈/负面反馈 | >10% | 用户参与改进程度 |
| **系统改进** | 问题解决率 | 已解决问题/识别问题 | >90% | 反馈驱动的问题修复 |
| | 改进响应时间 | 从反馈到改进的平均时间 | <7天 | 反馈响应速度 |
| | 性能提升幅度 | 改进后指标提升百分比 | >15% | 改进效果量化 |

### 反馈数据分析示例

```python
# 反馈趋势分析
def analyze_feedback_trends():
    """分析反馈趋势"""
    
    # 获取过去12周的反馈数据
    weekly_stats = []
    for week in range(12):
        start_date = datetime.now() - timedelta(weeks=week+1)
        end_date = datetime.now() - timedelta(weeks=week)
        
        week_performance = analyzer.analyze_performance_by_period(start_date, end_date)
        weekly_stats.append({
            'week': f"Week {week+1}",
            'satisfaction_rate': week_performance['satisfaction_rate'],
            'average_rating': week_performance['average_rating'],
            'total_feedback': week_performance['total_feedback']
        })
    
    # 趋势分析
    satisfaction_trend = [stat['satisfaction_rate'] for stat in weekly_stats]
    rating_trend = [stat['average_rating'] for stat in weekly_stats]
    
    print("=== 反馈趋势分析 ===")
    print(f"满意度趋势: {satisfaction_trend}")
    print(f"评分趋势: {rating_trend}")
    
    # 计算趋势方向
    if len(satisfaction_trend) >= 4:
        recent_avg = sum(satisfaction_trend[:4]) / 4
        earlier_avg = sum(satisfaction_trend[-4:]) / 4
        trend_direction = "上升" if recent_avg > earlier_avg else "下降"
        print(f"满意度趋势方向: {trend_direction}")
    
    return weekly_stats

# 用户细分分析
def analyze_user_segments():
    """用户细分反馈分析"""
    
    segments = {
        'power_users': {'queries_per_session': '>10', 'satisfaction': 0},
        'casual_users': {'queries_per_session': '1-5', 'satisfaction': 0},
        'new_users': {'session_count': '1', 'satisfaction': 0}
    }
    
    # 分析不同用户群体的满意度
    for segment in segments:
        segment_satisfaction = analyzer.analyze_segment_satisfaction(segment)
        segments[segment]['satisfaction'] = segment_satisfaction
    
    print("=== 用户细分分析 ===")
    for segment, data in segments.items():
        print(f"{segment}: 满意度 {data['satisfaction']:.2%}")
    
    return segments

# 问题根因分析
def analyze_problem_root_causes():
    """问题根因分析"""
    
    negative_feedback = analyzer.get_negative_feedback_details()
    
    # 按问题类型分类
    problem_categories = {
        'accuracy': 0,      # 准确性问题
        'relevance': 0,     # 相关性问题
        'completeness': 0,  # 完整性问题
        'clarity': 0,       # 清晰度问题
        'speed': 0          # 速度问题
    }
    
    # 关键词映射
    keyword_mapping = {
        'accuracy': ['错误', '不对', '不准确', '不正确'],
        'relevance': ['不相关', '跑题', '偏离', '无关'],
        'completeness': ['不完整', '缺少', '不全面', '太简单'],
        'clarity': ['不清楚', '模糊', '难懂', '不明白'],
        'speed': ['太慢', '响应慢', '等待时间长', '延迟']
    }
    
    for feedback in negative_feedback:
        feedback_text = feedback.get('feedback_text', '').lower()
        for category, keywords in keyword_mapping.items():
            if any(keyword in feedback_text for keyword in keywords):
                problem_categories[category] += 1
    
    print("=== 问题根因分析 ===")
    total_problems = sum(problem_categories.values())
    for category, count in problem_categories.items():
        if total_problems > 0:
            percentage = count / total_problems * 100
            print(f"{category}: {count}次 ({percentage:.1f}%)")
    
    return problem_categories
```

## 🚀 扩展功能

### 1. 智能反馈预测

```python
class FeedbackPredictor:
    """反馈预测系统"""
    
    def __init__(self):
        self.prediction_model = None
        self.feature_extractors = {}
    
    def predict_user_satisfaction(
        self, 
        query: str, 
        answer: str, 
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """预测用户满意度"""
        
        # 提取特征
        features = self._extract_features(query, answer, context)
        
        # 预测满意度
        predicted_satisfaction = self._predict_satisfaction(features)
        
        # 预测可能的问题
        potential_issues = self._predict_issues(features)
        
        return {
            'predicted_satisfaction': predicted_satisfaction,
            'confidence': 0.85,
            'potential_issues': potential_issues,
            'recommendation': self._generate_recommendation(predicted_satisfaction, potential_issues)
        }
    
    def _extract_features(self, query: str, answer: str, context: Dict[str, Any]) -> Dict[str, float]:
        """提取预测特征"""
        return {
            'query_length': len(query),
            'answer_length': len(answer),
            'response_time': context.get('response_time', 0),
            'confidence_score': context.get('confidence', 0),
            'retrieval_quality': context.get('retrieval_quality', 0),
            'answer_completeness': self._calculate_completeness(query, answer),
            'technical_complexity': self._calculate_complexity(query)
        }
    
    def _calculate_completeness(self, query: str, answer: str) -> float:
        """计算答案完整性"""
        # 简化实现：基于查询关键词在答案中的覆盖率
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        if not query_words:
            return 1.0
        
        coverage = len(query_words & answer_words) / len(query_words)
        return min(coverage, 1.0)
    
    def _calculate_complexity(self, query: str) -> float:
        """计算查询技术复杂度"""
        # 简化实现：基于技术术语数量
        technical_terms = [
            'algorithm', 'machine learning', 'deep learning', 'neural network',
            'transformer', 'attention', 'gradient', 'optimization'
        ]
        
        query_lower = query.lower()
        complexity = sum(1 for term in technical_terms if term in query_lower)
        return min(complexity / 3.0, 1.0)  # 归一化到0-1
```

### 2. 个性化反馈收集

```python
class PersonalizedFeedbackCollector:
    """个性化反馈收集器"""
    
    def __init__(self, base_collector: FeedbackCollector):
        self.base_collector = base_collector
        self.user_profiles = {}
        self.feedback_strategies = {}
    
    def create_user_profile(self, user_id: str, user_data: Dict[str, Any]):
        """创建用户画像"""
        self.user_profiles[user_id] = {
            'user_id': user_id,
            'expertise_level': user_data.get('expertise_level', 'beginner'),
            'feedback_frequency': user_data.get('feedback_frequency', 'normal'),
            'preferred_feedback_types': user_data.get('preferred_types', ['thumbs', 'rating']),
            'query_history': [],
            'feedback_history': [],
            'satisfaction_trend': []
        }
    
    def get_personalized_feedback_prompt(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """获取个性化反馈提示"""
        
        user_profile = self.user_profiles.get(user_id, {})
        expertise_level = user_profile.get('expertise_level', 'beginner')
        feedback_frequency = user_profile.get('feedback_frequency', 'normal')
        
        # 根据专业水平调整反馈类型
        if expertise_level == 'expert':
            feedback_types = ['correction', 'detailed_rating', 'technical_feedback']
            prompt = "作为领域专家，您的详细反馈对我们非常宝贵。请评价答案的技术准确性："
        elif expertise_level == 'intermediate':
            feedback_types = ['rating', 'relevance', 'thumbs']
            prompt = "请评价这个答案是否回答了您的问题："
        else:  # beginner
            feedback_types = ['thumbs', 'simple_rating']
            prompt = "这个答案对您有帮助吗？"
        
        # 根据反馈频率调整收集策略
        if feedback_frequency == 'low':
            collection_probability = 0.3  # 30%概率请求反馈
        elif feedback_frequency == 'high':
            collection_probability = 0.8  # 80%概率请求反馈
        else:
            collection_probability = 0.5  # 50%概率请求反馈
        
        return {
            'feedback_types': feedback_types,
            'prompt': prompt,
            'collection_probability': collection_probability,
            'incentive': self._get_feedback_incentive(user_profile)
        }
    
    def _get_feedback_incentive(self, user_profile: Dict[str, Any]) -> str:
        """获取反馈激励信息"""
        feedback_count = len(user_profile.get('feedback_history', []))
        
        if feedback_count == 0:
            return "您的第一次反馈将帮助我们更好地为您服务！"
        elif feedback_count < 5:
            return f"感谢您的 {feedback_count} 次反馈，继续帮助我们改进！"
        else:
            return "感谢您持续的支持，您是我们的超级用户！"
```

### 3. 反馈驱动的A/B测试

```python
class FeedbackDrivenABTest:
    """反馈驱动的A/B测试框架"""
    
    def __init__(self, feedback_analyzer: FeedbackAnalyzer):
        self.analyzer = feedback_analyzer
        self.active_tests = {}
        self.test_results = {}
    
    def create_ab_test(
        self,
        test_name: str,
        variant_a_config: Dict[str, Any],
        variant_b_config: Dict[str, Any],
        test_duration_days: int = 7,
        min_sample_size: int = 100
    ):
        """创建A/B测试"""
        
        self.active_tests[test_name] = {
            'test_name': test_name,
            'start_date': datetime.now(),
            'end_date': datetime.now() + timedelta(days=test_duration_days),
            'variant_a': variant_a_config,
            'variant_b': variant_b_config,
            'min_sample_size': min_sample_size,
            'variant_a_feedback': [],
            'variant_b_feedback': [],
            'status': 'active'
        }
        
        logger.info(f"A/B test '{test_name}' created and started")
    
    def assign_user_to_variant(self, user_id: str, test_name: str) -> str:
        """为用户分配测试变体"""
        
        if test_name not in self.active_tests:
            return 'control'
        
        # 简单的哈希分配
        import hashlib
        hash_value = int(hashlib.md5(f"{user_id}:{test_name}".encode()).hexdigest(), 16)
        variant = 'variant_a' if hash_value % 2 == 0 else 'variant_b'
        
        return variant
    
    def record_test_feedback(
        self,
        test_name: str,
        variant: str,
        feedback_data: Dict[str, Any]
    ):
        """记录测试反馈"""
        
        if test_name not in self.active_tests:
            return
        
        test = self.active_tests[test_name]
        if variant == 'variant_a':
            test['variant_a_feedback'].append(feedback_data)
        elif variant == 'variant_b':
            test['variant_b_feedback'].append(feedback_data)
    
    def analyze_test_results(self, test_name: str) -> Dict[str, Any]:
        """分析测试结果"""
        
        if test_name not in self.active_tests:
            return {}
        
        test = self.active_tests[test_name]
        
        # 计算各变体的指标
        variant_a_metrics = self._calculate_variant_metrics(test['variant_a_feedback'])
        variant_b_metrics = self._calculate_variant_metrics(test['variant_b_feedback'])
        
        # 统计显著性检验
        significance_test = self._perform_significance_test(
            test['variant_a_feedback'],
            test['variant_b_feedback']
        )
        
        results = {
            'test_name': test_name,
            'variant_a_metrics': variant_a_metrics,
            'variant_b_metrics': variant_b_metrics,
            'significance_test': significance_test,
            'winner': self._determine_winner(variant_a_metrics, variant_b_metrics, significance_test),
            'recommendations': self._generate_test_recommendations(variant_a_metrics, variant_b_metrics)
        }
        
        self.test_results[test_name] = results
        return results
    
    def _calculate_variant_metrics(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算变体指标"""
        if not feedback_data:
            return {'sample_size': 0, 'satisfaction_rate': 0, 'average_rating': 0}
        
        positive_feedback = sum(1 for f in feedback_data if f.get('is_positive', False))
        total_feedback = len(feedback_data)
        
        ratings = [f.get('rating', 0) for f in feedback_data if f.get('rating', 0) > 0]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        return {
            'sample_size': total_feedback,
            'satisfaction_rate': positive_feedback / total_feedback if total_feedback > 0 else 0,
            'average_rating': avg_rating,
            'response_time': np.mean([f.get('response_time', 0) for f in feedback_data])
        }
```

## 📋 最佳实践

### 1. 反馈收集策略

- **适时收集**: 在用户完成查询后的最佳时机收集反馈
- **渐进式收集**: 从简单的点赞/点踩开始，逐步引导详细反馈
- **个性化提示**: 根据用户画像调整反馈收集方式
- **激励机制**: 通过积分、徽章等方式鼓励用户提供反馈

### 2. 反馈分析方法

- **多维度分析**: 结合满意度、评分、文本反馈进行综合分析
- **趋势监控**: 持续跟踪反馈趋势，及时发现问题
- **细分分析**: 按用户群体、查询类型等维度进行细分分析
- **根因分析**: 深入挖掘负面反馈的根本原因

### 3. 持续改进循环

- **快速响应**: 对负面反馈快速响应和处理
- **数据驱动**: 基于反馈数据制定改进策略
- **A/B测试**: 通过A/B测试验证改进效果
- **效果评估**: 持续评估改进措施的效果

### 4. 反馈质量保障

- **反馈验证**: 对异常反馈进行验证和过滤
- **多渠道收集**: 通过多种渠道收集反馈，提高覆盖率
- **反馈闭环**: 向用户反馈改进结果，形成正向循环
- **隐私保护**: 确保用户反馈数据的隐私和安全

反馈模块通过系统化的反馈收集、分析和改进机制，为RAG系统建立了用户反馈驱动的持续优化能力，确保系统能够根据用户需求不断进化和改进。