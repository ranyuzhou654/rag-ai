# src/storage/usage_analytics.py
import asyncio
import json
import sqlite3
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from loguru import logger
from collections import defaultdict
import hashlib

@dataclass
class AccessPattern:
    """访问模式"""
    document_id: str
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    first_accessed: Optional[datetime] = None
    
    # 访问频率分析
    daily_access_avg: float = 0.0
    weekly_access_avg: float = 0.0
    monthly_access_avg: float = 0.0
    
    # 访问时间模式
    peak_hours: List[int] = field(default_factory=list)
    access_days_pattern: Dict[int, int] = field(default_factory=dict)  # weekday -> count
    
    # 用户模式
    unique_users: int = 0
    user_retention: float = 0.0  # 用户回访率
    
    # 访问深度
    avg_duration: float = 0.0
    avg_scroll_depth: float = 0.0
    download_count: int = 0
    
    # 性能指标
    cache_hit_ratio: float = 0.0
    avg_load_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AccessPattern':
        # 处理datetime字段
        if 'last_accessed' in data and data['last_accessed']:
            data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        if 'first_accessed' in data and data['first_accessed']:
            data['first_accessed'] = datetime.fromisoformat(data['first_accessed'])
        
        return cls(**data)

@dataclass
class StorageMetrics:
    """存储指标"""
    timestamp: datetime
    
    # 存储使用情况
    total_storage_gb: float = 0.0
    used_storage_gb: float = 0.0
    free_storage_gb: float = 0.0
    storage_utilization: float = 0.0
    
    # 分层存储统计
    hot_tier_gb: float = 0.0
    warm_tier_gb: float = 0.0
    cold_tier_gb: float = 0.0
    archived_tier_gb: float = 0.0
    
    # 文档统计
    total_documents: int = 0
    documents_by_tier: Dict[str, int] = field(default_factory=dict)
    documents_by_age: Dict[str, int] = field(default_factory=dict)  # age_range -> count
    
    # 性能指标
    avg_retrieval_time: float = 0.0
    cache_hit_ratio: float = 0.0
    io_operations_per_sec: float = 0.0
    
    # 访问统计
    total_access_count: int = 0
    unique_documents_accessed: int = 0
    hot_documents_ratio: float = 0.0  # 热门文档比例
    
    # 成本估算
    estimated_storage_cost: float = 0.0
    cost_per_gb: float = 0.0
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class UsageAnalytics:
    """使用分析器 - 跟踪数据访问模式和存储使用情况"""
    
    def __init__(self, db_path: Path, storage_root: Path):
        self.db_path = db_path
        self.storage_root = storage_root
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        
        # 实时统计缓存
        self.access_cache: Dict[str, AccessPattern] = {}
        self.metrics_cache: Optional[StorageMetrics] = None
        self.cache_dirty = False
        
        # 分析配置
        self.hot_threshold_days = 7  # 热门数据阈值
        self.warm_threshold_days = 30  # 温热数据阈值
        self.cold_threshold_days = 90  # 冷数据阈值
        
        # 存储层级价格配置（美元/GB/月）
        self.tier_costs = {
            'hot': 0.10,    # 高速SSD
            'warm': 0.05,   # 标准存储
            'cold': 0.02,   # 冷存储
            'archived': 0.005  # 归档存储
        }
        
        self._init_database()
        
        # 启动后台分析任务
        self.background_task = None
        
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            # 访问记录表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS access_logs (
                    log_id TEXT PRIMARY KEY,
                    document_id TEXT,
                    user_id TEXT,
                    access_time TEXT,
                    access_type TEXT,  -- view, download, search_result
                    duration_seconds REAL,
                    scroll_depth REAL,
                    load_time_ms REAL,
                    cache_hit BOOLEAN,
                    source_tier TEXT,  -- hot, warm, cold, archived
                    ip_address TEXT,
                    user_agent TEXT
                )
            ''')
            
            # 访问模式表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS access_patterns (
                    document_id TEXT PRIMARY KEY,
                    pattern_data TEXT,  -- JSON格式的AccessPattern数据
                    last_updated TEXT
                )
            ''')
            
            # 存储指标表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS storage_metrics (
                    metric_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    metrics_data TEXT,  -- JSON格式的StorageMetrics数据
                    metric_type TEXT DEFAULT 'hourly'
                )
            ''')
            
            # 文档存储信息表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS document_storage (
                    document_id TEXT PRIMARY KEY,
                    file_path TEXT,
                    file_size_bytes INTEGER,
                    storage_tier TEXT,
                    created_at TEXT,
                    last_accessed TEXT,
                    access_count INTEGER DEFAULT 0,
                    is_cached BOOLEAN DEFAULT FALSE,
                    compression_ratio REAL DEFAULT 1.0
                )
            ''')
            
            # 创建索引
            conn.execute('CREATE INDEX IF NOT EXISTS idx_access_logs_document_id ON access_logs(document_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_access_logs_time ON access_logs(access_time)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_access_logs_user_id ON access_logs(user_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_document_storage_tier ON document_storage(storage_tier)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_document_storage_last_accessed ON document_storage(last_accessed)')
            
            conn.commit()
        
        logger.info(f"Usage analytics database initialized at {self.db_path}")
    
    async def start_analytics(self):
        """启动分析服务"""
        if self.background_task is None:
            self.background_task = asyncio.create_task(self._run_background_analytics())
            logger.info("Usage analytics started")
    
    async def stop_analytics(self):
        """停止分析服务"""
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
            self.background_task = None
            logger.info("Usage analytics stopped")
    
    async def log_document_access(self, document_id: str, user_id: str, 
                                access_type: str = "view", 
                                duration: Optional[float] = None,
                                scroll_depth: Optional[float] = None,
                                load_time_ms: Optional[float] = None,
                                cache_hit: bool = False,
                                source_tier: str = "unknown",
                                ip_address: Optional[str] = None,
                                user_agent: Optional[str] = None):
        """记录文档访问"""
        log_id = hashlib.md5(f"{document_id}_{user_id}_{time.time()}".encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO access_logs 
                (log_id, document_id, user_id, access_time, access_type,
                 duration_seconds, scroll_depth, load_time_ms, cache_hit,
                 source_tier, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                log_id, document_id, user_id, datetime.now(timezone.utc).isoformat(),
                access_type, duration, scroll_depth, load_time_ms, cache_hit,
                source_tier, ip_address, user_agent
            ))
            
            # 更新文档存储信息
            conn.execute('''
                UPDATE document_storage 
                SET last_accessed = ?, access_count = access_count + 1
                WHERE document_id = ?
            ''', (datetime.now(timezone.utc).isoformat(), document_id))
            
            conn.commit()
        
        # 更新实时缓存
        await self._update_access_pattern_cache(document_id)
        
        logger.debug(f"Logged access: {document_id} by {user_id} ({access_type})")
    
    async def _update_access_pattern_cache(self, document_id: str):
        """更新访问模式缓存"""
        pattern = await self.get_access_pattern(document_id)
        if pattern:
            self.access_cache[document_id] = pattern
            self.cache_dirty = True
    
    async def get_access_pattern(self, document_id: str) -> Optional[AccessPattern]:
        """获取文档访问模式"""
        # 从缓存获取
        if document_id in self.access_cache:
            return self.access_cache[document_id]
        
        # 从数据库获取
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT pattern_data FROM access_patterns WHERE document_id = ?',
                (document_id,)
            )
            row = cursor.fetchone()
            
            if row:
                try:
                    pattern_data = json.loads(row[0])
                    pattern = AccessPattern.from_dict(pattern_data)
                    self.access_cache[document_id] = pattern
                    return pattern
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Error loading access pattern for {document_id}: {e}")
        
        # 计算新的访问模式
        return await self._calculate_access_pattern(document_id)
    
    async def _calculate_access_pattern(self, document_id: str) -> AccessPattern:
        """计算文档访问模式"""
        pattern = AccessPattern(document_id=document_id)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 基本统计
            cursor.execute('''
                SELECT COUNT(*), MIN(access_time), MAX(access_time),
                       AVG(duration_seconds), AVG(scroll_depth),
                       COUNT(DISTINCT user_id), AVG(load_time_ms),
                       AVG(CASE WHEN cache_hit THEN 1.0 ELSE 0.0 END)
                FROM access_logs 
                WHERE document_id = ?
            ''', (document_id,))
            
            stats = cursor.fetchone()
            if stats and stats[0] > 0:
                pattern.access_count = stats[0]
                pattern.first_accessed = datetime.fromisoformat(stats[1]) if stats[1] else None
                pattern.last_accessed = datetime.fromisoformat(stats[2]) if stats[2] else None
                pattern.avg_duration = stats[3] or 0.0
                pattern.avg_scroll_depth = stats[4] or 0.0
                pattern.unique_users = stats[5] or 0
                pattern.avg_load_time = stats[6] or 0.0
                pattern.cache_hit_ratio = stats[7] or 0.0
            
            # 时间模式分析
            cursor.execute('''
                SELECT strftime('%H', access_time) as hour, COUNT(*)
                FROM access_logs 
                WHERE document_id = ?
                GROUP BY hour
                ORDER BY COUNT(*) DESC
                LIMIT 5
            ''', (document_id,))
            
            pattern.peak_hours = [int(row[0]) for row in cursor.fetchall()]
            
            # 星期模式
            cursor.execute('''
                SELECT strftime('%w', access_time) as weekday, COUNT(*)
                FROM access_logs 
                WHERE document_id = ?
                GROUP BY weekday
            ''', (document_id,))
            
            pattern.access_days_pattern = {int(row[0]): row[1] for row in cursor.fetchall()}
            
            # 频率分析
            if pattern.first_accessed and pattern.last_accessed:
                total_days = (pattern.last_accessed - pattern.first_accessed).days or 1
                pattern.daily_access_avg = pattern.access_count / total_days
                pattern.weekly_access_avg = pattern.daily_access_avg * 7
                pattern.monthly_access_avg = pattern.daily_access_avg * 30
            
            # 下载统计
            cursor.execute('''
                SELECT COUNT(*)
                FROM access_logs 
                WHERE document_id = ? AND access_type = 'download'
            ''', (document_id,))
            
            download_result = cursor.fetchone()
            pattern.download_count = download_result[0] if download_result else 0
        
        # 缓存结果
        self.access_cache[document_id] = pattern
        await self._save_access_pattern(pattern)
        
        return pattern
    
    async def _save_access_pattern(self, pattern: AccessPattern):
        """保存访问模式"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO access_patterns 
                (document_id, pattern_data, last_updated)
                VALUES (?, ?, ?)
            ''', (
                pattern.document_id,
                json.dumps(pattern.to_dict(), default=str),
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
    
    async def calculate_storage_metrics(self) -> StorageMetrics:
        """计算存储指标"""
        metrics = StorageMetrics(timestamp=datetime.now(timezone.utc))
        
        try:
            # 计算存储使用情况
            total_size = 0
            tier_sizes = defaultdict(float)
            tier_counts = defaultdict(int)
            age_counts = defaultdict(int)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 文档存储统计
                cursor.execute('''
                    SELECT storage_tier, COUNT(*), SUM(file_size_bytes)
                    FROM document_storage
                    GROUP BY storage_tier
                ''')
                
                for tier, count, size_bytes in cursor.fetchall():
                    size_gb = (size_bytes or 0) / (1024**3)
                    tier_sizes[tier] = size_gb
                    tier_counts[tier] = count
                    total_size += size_gb
                
                # 总文档数
                cursor.execute('SELECT COUNT(*) FROM document_storage')
                metrics.total_documents = cursor.fetchone()[0]
                
                # 年龄分布
                now = datetime.now(timezone.utc)
                cursor.execute('SELECT created_at FROM document_storage WHERE created_at IS NOT NULL')
                
                for row in cursor.fetchall():
                    try:
                        created_date = datetime.fromisoformat(row[0])
                        age_days = (now - created_date).days
                        
                        if age_days <= 7:
                            age_range = "0-7_days"
                        elif age_days <= 30:
                            age_range = "8-30_days"
                        elif age_days <= 90:
                            age_range = "31-90_days"
                        else:
                            age_range = "90+_days"
                        
                        age_counts[age_range] += 1
                    except:
                        continue
                
                # 访问统计
                cursor.execute('''
                    SELECT COUNT(*), COUNT(DISTINCT document_id)
                    FROM access_logs 
                    WHERE access_time >= datetime('now', '-24 hours')
                ''')
                
                access_stats = cursor.fetchone()
                metrics.total_access_count = access_stats[0] or 0
                metrics.unique_documents_accessed = access_stats[1] or 0
                
                # 性能统计
                cursor.execute('''
                    SELECT AVG(load_time_ms), AVG(CASE WHEN cache_hit THEN 1.0 ELSE 0.0 END)
                    FROM access_logs 
                    WHERE access_time >= datetime('now', '-24 hours')
                    AND load_time_ms IS NOT NULL
                ''')
                
                perf_stats = cursor.fetchone()
                if perf_stats:
                    metrics.avg_retrieval_time = (perf_stats[0] or 0) / 1000  # 转为秒
                    metrics.cache_hit_ratio = perf_stats[1] or 0
            
            # 填充指标
            metrics.total_storage_gb = total_size
            metrics.used_storage_gb = total_size
            metrics.hot_tier_gb = tier_sizes.get('hot', 0)
            metrics.warm_tier_gb = tier_sizes.get('warm', 0)
            metrics.cold_tier_gb = tier_sizes.get('cold', 0)
            metrics.archived_tier_gb = tier_sizes.get('archived', 0)
            
            metrics.documents_by_tier = dict(tier_counts)
            metrics.documents_by_age = dict(age_counts)
            
            # 计算热门文档比例
            hot_documents = age_counts.get("0-7_days", 0) + age_counts.get("8-30_days", 0)
            if metrics.total_documents > 0:
                metrics.hot_documents_ratio = hot_documents / metrics.total_documents
            
            # 成本估算
            total_cost = 0
            for tier, size_gb in tier_sizes.items():
                cost_per_gb = self.tier_costs.get(tier, 0.05)
                total_cost += size_gb * cost_per_gb
            
            metrics.estimated_storage_cost = total_cost
            if total_size > 0:
                metrics.cost_per_gb = total_cost / total_size
            
            # 存储利用率（简化计算）
            if total_size > 0:
                metrics.storage_utilization = min(1.0, total_size / (total_size + 10))  # 假设还有10GB可用
                metrics.free_storage_gb = max(0, 10 - total_size)
            
        except Exception as e:
            logger.error(f"Error calculating storage metrics: {e}")
        
        self.metrics_cache = metrics
        await self._save_storage_metrics(metrics)
        
        return metrics
    
    async def _save_storage_metrics(self, metrics: StorageMetrics):
        """保存存储指标"""
        metric_id = hashlib.md5(f"storage_metrics_{metrics.timestamp.isoformat()}".encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO storage_metrics 
                (metric_id, timestamp, metrics_data, metric_type)
                VALUES (?, ?, ?, ?)
            ''', (
                metric_id,
                metrics.timestamp.isoformat(),
                json.dumps(metrics.to_dict()),
                'hourly'
            ))
            conn.commit()
    
    async def get_hot_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取热门文档"""
        hot_docs = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT ds.document_id, ds.file_size_bytes, ds.storage_tier,
                       ds.access_count, ds.last_accessed,
                       COUNT(al.log_id) as recent_access_count
                FROM document_storage ds
                LEFT JOIN access_logs al ON ds.document_id = al.document_id 
                    AND al.access_time >= datetime('now', '-7 days')
                GROUP BY ds.document_id
                HAVING recent_access_count > 0
                ORDER BY recent_access_count DESC, ds.access_count DESC
                LIMIT ?
            ''', (limit,))
            
            for row in cursor.fetchall():
                hot_docs.append({
                    'document_id': row[0],
                    'file_size_bytes': row[1],
                    'current_tier': row[2],
                    'total_access_count': row[3],
                    'last_accessed': row[4],
                    'recent_access_count': row[5],
                    'recommended_tier': 'hot' if row[5] > 10 else 'warm'
                })
        
        return hot_docs
    
    async def get_cold_documents(self, days_threshold: int = 90) -> List[Dict[str, Any]]:
        """获取冷数据文档"""
        cold_docs = []
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_threshold)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT document_id, file_size_bytes, storage_tier,
                       access_count, last_accessed
                FROM document_storage
                WHERE (last_accessed IS NULL OR last_accessed < ?)
                AND storage_tier IN ('hot', 'warm')
                ORDER BY last_accessed ASC NULLS FIRST
            ''', (cutoff_date,))
            
            for row in cursor.fetchall():
                cold_docs.append({
                    'document_id': row[0],
                    'file_size_bytes': row[1],
                    'current_tier': row[2],
                    'access_count': row[3],
                    'last_accessed': row[4],
                    'recommended_tier': 'cold'
                })
        
        return cold_docs
    
    async def get_optimization_recommendations(self) -> Dict[str, Any]:
        """获取存储优化建议"""
        recommendations = {
            'hot_tier_candidates': [],
            'cold_tier_candidates': [],
            'archive_candidates': [],
            'delete_candidates': [],
            'estimated_savings': 0.0,
            'total_recommendations': 0
        }
        
        try:
            # 热门文档推荐
            hot_docs = await self.get_hot_documents(50)
            for doc in hot_docs:
                if doc['current_tier'] != 'hot' and doc['recent_access_count'] > 5:
                    recommendations['hot_tier_candidates'].append(doc)
            
            # 冷数据推荐
            cold_docs = await self.get_cold_documents(60)
            recommendations['cold_tier_candidates'] = cold_docs[:100]
            
            # 归档推荐
            archive_candidates = await self.get_cold_documents(180)
            for doc in archive_candidates:
                if doc['access_count'] < 5:
                    recommendations['archive_candidates'].append(doc)
            
            # 计算预估节省
            total_savings = 0.0
            for doc in recommendations['cold_tier_candidates']:
                if doc['current_tier'] == 'hot':
                    size_gb = doc['file_size_bytes'] / (1024**3)
                    savings = size_gb * (self.tier_costs['hot'] - self.tier_costs['cold'])
                    total_savings += savings
            
            for doc in recommendations['archive_candidates']:
                if doc['current_tier'] in ['hot', 'warm', 'cold']:
                    size_gb = doc['file_size_bytes'] / (1024**3)
                    current_cost = self.tier_costs.get(doc['current_tier'], 0.05)
                    savings = size_gb * (current_cost - self.tier_costs['archived'])
                    total_savings += savings
            
            recommendations['estimated_savings'] = total_savings
            recommendations['total_recommendations'] = (
                len(recommendations['hot_tier_candidates']) +
                len(recommendations['cold_tier_candidates']) +
                len(recommendations['archive_candidates'])
            )
            
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {e}")
        
        return recommendations
    
    async def _run_background_analytics(self):
        """后台分析任务"""
        while True:
            try:
                # 每小时计算一次存储指标
                await self.calculate_storage_metrics()
                
                # 每天清理缓存
                if self.cache_dirty:
                    await self._flush_cache()
                    self.cache_dirty = False
                
                # 等待1小时
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in background analytics: {e}")
                await asyncio.sleep(300)  # 错误时等待5分钟
    
    async def _flush_cache(self):
        """刷新缓存到数据库"""
        for pattern in self.access_cache.values():
            await self._save_access_pattern(pattern)
        
        logger.debug("Flushed access pattern cache to database")
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """获取分析摘要"""
        summary = {
            'cache_size': len(self.access_cache),
            'cache_dirty': self.cache_dirty,
            'background_task_running': self.background_task is not None and not self.background_task.done(),
            'last_metrics_update': self.metrics_cache.timestamp.isoformat() if self.metrics_cache else None
        }
        
        if self.metrics_cache:
            summary.update({
                'total_storage_gb': self.metrics_cache.total_storage_gb,
                'total_documents': self.metrics_cache.total_documents,
                'cache_hit_ratio': self.metrics_cache.cache_hit_ratio,
                'estimated_monthly_cost': self.metrics_cache.estimated_storage_cost
            })
        
        return summary

# 使用示例
async def main():
    """测试使用分析器"""
    analytics = UsageAnalytics(
        db_path=Path("data/storage/usage_analytics.db"),
        storage_root=Path("data/storage")
    )
    
    # 启动分析
    await analytics.start_analytics()
    
    # 模拟一些访问记录
    await analytics.log_document_access(
        document_id="doc_001",
        user_id="user_001",
        access_type="view",
        duration=120.0,
        scroll_depth=0.8,
        load_time_ms=500,
        cache_hit=True,
        source_tier="hot"
    )
    
    # 计算指标
    metrics = await analytics.calculate_storage_metrics()
    print("存储指标:")
    print(json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False))
    
    # 获取优化建议
    recommendations = await analytics.get_optimization_recommendations()
    print("\\n优化建议:")
    print(json.dumps(recommendations, indent=2, ensure_ascii=False))
    
    # 停止分析
    await analytics.stop_analytics()

if __name__ == "__main__":
    asyncio.run(main())