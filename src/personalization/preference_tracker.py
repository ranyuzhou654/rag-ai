# src/personalization/preference_tracker.py
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger
from collections import defaultdict, deque
import uuid

from .user_profiler import UserProfiler, UserInteraction, InteractionType

@dataclass
class PreferenceEvent:
    """偏好事件"""
    event_id: str
    user_id: str
    event_type: str  # view, click, dwell, scroll, search, feedback
    timestamp: datetime
    
    # 事件数据
    document_id: Optional[str] = None
    query_text: Optional[str] = None
    duration: Optional[float] = None
    scroll_depth: Optional[float] = None
    click_position: Optional[int] = None
    rating: Optional[float] = None
    
    # 上下文信息
    session_id: Optional[str] = None
    page_context: Optional[Dict] = None
    user_agent: Optional[str] = None

@dataclass
class SessionSummary:
    """会话摘要"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime]
    
    # 会话统计
    total_duration: float = 0.0
    page_views: int = 0
    unique_documents: int = 0
    queries_count: int = 0
    clicks_count: int = 0
    
    # 行为模式
    avg_dwell_time: float = 0.0
    avg_scroll_depth: float = 0.0
    bounce_rate: float = 0.0
    
    # 兴趣分析
    dominant_categories: List[str] = None
    search_patterns: List[str] = None
    engaged_content: List[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)

class PreferenceTracker:
    """偏好追踪器 - 实时分析用户行为和偏好变化"""
    
    def __init__(self, user_profiler: UserProfiler, storage_path: Path):
        self.user_profiler = user_profiler
        self.storage_path = storage_path
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        # 实时会话追踪
        self.active_sessions: Dict[str, SessionSummary] = {}
        self.session_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # 事件队列和处理器
        self.event_queue = asyncio.Queue()
        self.event_processors: List[Callable] = []
        
        # 性能监控
        self.processing_stats = {
            'events_processed': 0,
            'processing_time_total': 0.0,
            'last_update': datetime.now()
        }
        
        # 启动后台处理任务
        self.background_task = None
        
    async def start_tracking(self):
        """启动偏好追踪"""
        if self.background_task is None:
            self.background_task = asyncio.create_task(self._process_events_background())
            logger.info("Preference tracking started")
    
    async def stop_tracking(self):
        """停止偏好追踪"""
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
            self.background_task = None
            logger.info("Preference tracking stopped")
    
    async def track_event(self, event: PreferenceEvent):
        """追踪偏好事件"""
        await self.event_queue.put(event)
    
    async def track_page_view(self, user_id: str, document_id: str, 
                            session_id: str, page_context: Optional[Dict] = None):
        """追踪页面浏览"""
        event = PreferenceEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            event_type="page_view",
            timestamp=datetime.now(timezone.utc),
            document_id=document_id,
            session_id=session_id,
            page_context=page_context
        )
        await self.track_event(event)
    
    async def track_search(self, user_id: str, query_text: str, 
                         session_id: str, search_context: Optional[Dict] = None):
        """追踪搜索行为"""
        event = PreferenceEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            event_type="search",
            timestamp=datetime.now(timezone.utc),
            query_text=query_text,
            session_id=session_id,
            page_context=search_context
        )
        await self.track_event(event)
    
    async def track_dwell_time(self, user_id: str, document_id: str, 
                             duration: float, session_id: str):
        """追踪停留时间"""
        event = PreferenceEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            event_type="dwell",
            timestamp=datetime.now(timezone.utc),
            document_id=document_id,
            duration=duration,
            session_id=session_id
        )
        await self.track_event(event)
    
    async def track_scroll_behavior(self, user_id: str, document_id: str, 
                                  scroll_depth: float, session_id: str):
        """追踪滚动行为"""
        event = PreferenceEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            event_type="scroll",
            timestamp=datetime.now(timezone.utc),
            document_id=document_id,
            scroll_depth=scroll_depth,
            session_id=session_id
        )
        await self.track_event(event)
    
    async def track_click(self, user_id: str, document_id: str, 
                        click_position: int, session_id: str):
        """追踪点击行为"""
        event = PreferenceEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            event_type="click",
            timestamp=datetime.now(timezone.utc),
            document_id=document_id,
            click_position=click_position,
            session_id=session_id
        )
        await self.track_event(event)
    
    async def track_feedback(self, user_id: str, document_id: str, 
                           rating: float, session_id: str):
        """追踪反馈行为"""
        event = PreferenceEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            event_type="feedback",
            timestamp=datetime.now(timezone.utc),
            document_id=document_id,
            rating=rating,
            session_id=session_id
        )
        await self.track_event(event)
    
    def start_session(self, user_id: str, session_id: str) -> SessionSummary:
        """开始用户会话"""
        session = SessionSummary(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(timezone.utc),
            dominant_categories=[],
            search_patterns=[],
            engaged_content=[]
        )
        
        self.active_sessions[session_id] = session
        logger.info(f"Started session {session_id} for user {user_id}")
        return session
    
    async def end_session(self, session_id: str) -> Optional[SessionSummary]:
        """结束用户会话"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        session.end_time = datetime.now(timezone.utc)
        session.total_duration = (session.end_time - session.start_time).total_seconds()
        
        # 分析会话数据
        await self._analyze_session(session)
        
        # 从活跃会话中移除
        del self.active_sessions[session_id]
        
        # 保存会话摘要
        await self._save_session_summary(session)
        
        logger.info(f"Ended session {session_id}, duration: {session.total_duration:.1f}s")
        return session
    
    async def _process_events_background(self):
        """后台事件处理"""
        while True:
            try:
                # 批量处理事件
                events = []
                try:
                    # 等待第一个事件
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                    events.append(event)
                    
                    # 收集更多事件（非阻塞）
                    while len(events) < 50:  # 批处理大小
                        try:
                            event = self.event_queue.get_nowait()
                            events.append(event)
                        except asyncio.QueueEmpty:
                            break
                    
                except asyncio.TimeoutError:
                    continue
                
                # 处理事件批次
                if events:
                    await self._process_event_batch(events)
                    
            except Exception as e:
                logger.error(f"Error in event processing: {e}")
                await asyncio.sleep(1)
    
    async def _process_event_batch(self, events: List[PreferenceEvent]):
        """处理事件批次"""
        start_time = time.time()
        
        for event in events:
            try:
                # 更新会话统计
                if event.session_id and event.session_id in self.active_sessions:
                    await self._update_session_stats(event)
                
                # 转换为用户交互记录
                interaction = await self._convert_to_interaction(event)
                if interaction:
                    self.user_profiler.record_interaction(interaction)
                
                # 实时偏好更新
                await self._update_real_time_preferences(event)
                
                # 添加到会话事件历史
                if event.session_id:
                    self.session_events[event.session_id].append(event)
                
            except Exception as e:
                logger.error(f"Error processing event {event.event_id}: {e}")
        
        # 更新性能统计
        processing_time = time.time() - start_time
        self.processing_stats['events_processed'] += len(events)
        self.processing_stats['processing_time_total'] += processing_time
        self.processing_stats['last_update'] = datetime.now()
        
        logger.debug(f"Processed {len(events)} events in {processing_time:.3f}s")
    
    async def _update_session_stats(self, event: PreferenceEvent):
        """更新会话统计"""
        session = self.active_sessions.get(event.session_id)
        if not session:
            return
        
        if event.event_type == "page_view":
            session.page_views += 1
            if event.document_id:
                if event.document_id not in session.engaged_content:
                    session.unique_documents += 1
        
        elif event.event_type == "search":
            session.queries_count += 1
        
        elif event.event_type == "click":
            session.clicks_count += 1
        
        elif event.event_type == "dwell" and event.duration:
            # 更新平均停留时间
            if session.page_views > 0:
                current_avg = session.avg_dwell_time * (session.page_views - 1)
                session.avg_dwell_time = (current_avg + event.duration) / session.page_views
        
        elif event.event_type == "scroll" and event.scroll_depth:
            # 更新平均滚动深度
            if session.page_views > 0:
                current_avg = session.avg_scroll_depth * (session.page_views - 1)
                session.avg_scroll_depth = (current_avg + event.scroll_depth) / session.page_views
    
    async def _convert_to_interaction(self, event: PreferenceEvent) -> Optional[UserInteraction]:
        """将偏好事件转换为用户交互记录"""
        interaction_type_map = {
            "page_view": InteractionType.DOCUMENT_VIEW,
            "click": InteractionType.DOCUMENT_CLICK,
            "search": InteractionType.QUERY,
            "feedback": InteractionType.FEEDBACK_POSITIVE if event.rating and event.rating > 3 else InteractionType.FEEDBACK_NEGATIVE
        }
        
        interaction_type = interaction_type_map.get(event.event_type)
        if not interaction_type:
            return None
        
        return UserInteraction(
            interaction_id=event.event_id,
            user_id=event.user_id,
            session_id=event.session_id or "unknown",
            interaction_type=interaction_type,
            timestamp=event.timestamp,
            query_text=event.query_text,
            document_id=event.document_id,
            duration_seconds=event.duration,
            scroll_depth=event.scroll_depth,
            click_position=event.click_position,
            rating=event.rating,
            user_agent=event.user_agent
        )
    
    async def _update_real_time_preferences(self, event: PreferenceEvent):
        """实时更新用户偏好"""
        # 获取用户画像
        profile = self.user_profiler.get_user_profile(event.user_id)
        if not profile:
            return
        
        # 基于事件类型进行不同的偏好更新
        if event.event_type == "dwell" and event.duration:
            # 长时间停留表示高兴趣
            if event.duration > 60:  # 超过1分钟
                await self._boost_document_preference(event.user_id, event.document_id, 0.1)
        
        elif event.event_type == "scroll" and event.scroll_depth:
            # 深度滚动表示认真阅读
            if event.scroll_depth > 0.7:  # 滚动超过70%
                await self._boost_document_preference(event.user_id, event.document_id, 0.05)
        
        elif event.event_type == "feedback" and event.rating:
            # 直接反馈
            if event.rating > 3:
                await self._boost_document_preference(event.user_id, event.document_id, 0.2)
            else:
                await self._decrease_document_preference(event.user_id, event.document_id, 0.1)
    
    async def _boost_document_preference(self, user_id: str, document_id: str, boost: float):
        """提升文档偏好"""
        # 这里可以实现更复杂的偏好更新逻辑
        # 比如基于文档内容更新用户的研究兴趣
        logger.debug(f"Boosting preference for user {user_id}, doc {document_id}, boost {boost}")
    
    async def _decrease_document_preference(self, user_id: str, document_id: str, penalty: float):
        """降低文档偏好"""
        logger.debug(f"Decreasing preference for user {user_id}, doc {document_id}, penalty {penalty}")
    
    async def _analyze_session(self, session: SessionSummary):
        """分析会话数据"""
        session_events = list(self.session_events.get(session.session_id, []))
        
        if not session_events:
            return
        
        # 分析主要内容类别
        category_counts = defaultdict(int)
        search_patterns = []
        engaged_docs = []
        
        for event in session_events:
            if event.event_type == "search" and event.query_text:
                search_patterns.append(event.query_text)
            
            elif event.event_type == "dwell" and event.duration and event.duration > 30:
                engaged_docs.append(event.document_id)
            
            # 这里可以添加更多分析逻辑
        
        # 计算跳出率
        if session.page_views > 0:
            single_page_sessions = 1 if session.page_views == 1 else 0
            session.bounce_rate = single_page_sessions
        
        session.search_patterns = search_patterns[:10]  # 限制数量
        session.engaged_content = engaged_docs[:20]
    
    async def _save_session_summary(self, session: SessionSummary):
        """保存会话摘要"""
        # 保存到文件系统
        session_file = self.storage_path / f"sessions_{session.user_id}" / f"{session.session_id}.json"
        session_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving session summary: {e}")
    
    def get_active_sessions_count(self) -> int:
        """获取活跃会话数"""
        return len(self.active_sessions)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计"""
        stats = self.processing_stats.copy()
        
        if stats['events_processed'] > 0:
            stats['avg_processing_time'] = stats['processing_time_total'] / stats['events_processed']
        else:
            stats['avg_processing_time'] = 0.0
        
        return stats
    
    async def get_user_behavior_insights(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """获取用户行为洞察"""
        # 读取用户的会话历史
        user_sessions_dir = self.storage_path / f"sessions_{user_id}"
        if not user_sessions_dir.exists():
            return {}
        
        insights = {
            'total_sessions': 0,
            'total_duration': 0.0,
            'avg_session_duration': 0.0,
            'avg_pages_per_session': 0.0,
            'preferred_time_slots': [],
            'engagement_level': 'medium',
            'content_preferences': {},
            'search_behavior': {}
        }
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            session_files = list(user_sessions_dir.glob("*.json"))
            session_data = []
            
            for session_file in session_files:
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session = json.load(f)
                        session_start = datetime.fromisoformat(session['start_time'])
                        
                        if session_start > cutoff_date:
                            session_data.append(session)
                except Exception as e:
                    logger.error(f"Error reading session file {session_file}: {e}")
            
            if session_data:
                insights['total_sessions'] = len(session_data)
                insights['total_duration'] = sum(s['total_duration'] for s in session_data)
                insights['avg_session_duration'] = insights['total_duration'] / len(session_data)
                insights['avg_pages_per_session'] = sum(s['page_views'] for s in session_data) / len(session_data)
                
                # 分析活跃时间段
                time_slots = defaultdict(int)
                for session in session_data:
                    start_time = datetime.fromisoformat(session['start_time'])
                    hour_slot = start_time.hour // 4  # 4小时时段
                    time_slots[hour_slot] += 1
                
                insights['preferred_time_slots'] = sorted(time_slots.items(), key=lambda x: x[1], reverse=True)
                
                # 判断参与度级别
                avg_dwell = sum(s.get('avg_dwell_time', 0) for s in session_data) / len(session_data)
                avg_scroll = sum(s.get('avg_scroll_depth', 0) for s in session_data) / len(session_data)
                
                if avg_dwell > 60 and avg_scroll > 0.6:
                    insights['engagement_level'] = 'high'
                elif avg_dwell > 30 and avg_scroll > 0.4:
                    insights['engagement_level'] = 'medium'
                else:
                    insights['engagement_level'] = 'low'
        
        except Exception as e:
            logger.error(f"Error analyzing user behavior insights: {e}")
        
        return insights

# 使用示例
async def main():
    """测试偏好追踪器"""
    from pathlib import Path
    from .user_profiler import UserProfiler
    
    # 初始化组件
    user_profiler = UserProfiler(
        db_path=Path("data/user_profiles/profiles.db"),
        storage_path=Path("data/user_profiles")
    )
    
    tracker = PreferenceTracker(
        user_profiler=user_profiler,
        storage_path=Path("data/preferences")
    )
    
    # 启动追踪
    await tracker.start_tracking()
    
    # 模拟用户行为
    user_id = "test_user_001"
    session_id = "session_001"
    
    # 开始会话
    tracker.start_session(user_id, session_id)
    
    # 模拟一系列用户行为
    await tracker.track_search(user_id, "transformer attention mechanism", session_id)
    await asyncio.sleep(0.1)
    
    await tracker.track_page_view(user_id, "doc_001", session_id)
    await asyncio.sleep(0.1)
    
    await tracker.track_dwell_time(user_id, "doc_001", 120.0, session_id)
    await tracker.track_scroll_behavior(user_id, "doc_001", 0.8, session_id)
    await tracker.track_feedback(user_id, "doc_001", 4.5, session_id)
    
    # 等待事件处理
    await asyncio.sleep(1)
    
    # 结束会话
    session_summary = await tracker.end_session(session_id)
    
    print("会话摘要:")
    print(json.dumps(session_summary.to_dict(), indent=2, ensure_ascii=False, default=str))
    
    # 获取处理统计
    stats = tracker.get_processing_stats()
    print("\\n处理统计:")
    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
    
    # 停止追踪
    await tracker.stop_tracking()

if __name__ == "__main__":
    asyncio.run(main())