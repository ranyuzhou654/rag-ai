# src/storage/storage_optimizer.py
import asyncio
import json
import shutil
import os
import gzip
import lzma
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from loguru import logger
from enum import Enum
import sqlite3
import hashlib
import time

from .usage_analytics import UsageAnalytics, AccessPattern, StorageMetrics

class StorageTier(Enum):
    """存储层级"""
    HOT = "hot"        # 高频访问，SSD存储
    WARM = "warm"      # 中频访问，标准存储
    COLD = "cold"      # 低频访问，冷存储
    ARCHIVED = "archived"  # 归档，压缩存储

@dataclass
class StoragePolicy:
    """存储策略"""
    tier: StorageTier
    
    # 触发条件
    min_access_frequency: float = 0.0  # 最小访问频率（次/天）
    max_days_since_access: int = 365   # 最大未访问天数
    min_file_size_mb: float = 0.0      # 最小文件大小
    max_file_size_mb: float = float('inf')  # 最大文件大小
    
    # 策略配置
    compression_enabled: bool = False   # 是否启用压缩
    compression_type: str = "gzip"     # 压缩类型：gzip, lzma
    backup_enabled: bool = True        # 是否启用备份
    replicas: int = 1                  # 副本数量
    
    # 成本配置
    cost_per_gb_per_month: float = 0.05  # 存储成本
    io_cost_per_operation: float = 0.0001  # IO操作成本
    
    def matches_criteria(self, access_pattern: AccessPattern, 
                        file_size_mb: float, days_since_access: int) -> bool:
        """检查是否匹配策略条件"""
        # 访问频率检查
        if access_pattern.daily_access_avg < self.min_access_frequency:
            return False
        
        # 未访问天数检查
        if days_since_access > self.max_days_since_access:
            return False
        
        # 文件大小检查
        if not (self.min_file_size_mb <= file_size_mb <= self.max_file_size_mb):
            return False
        
        return True

@dataclass
class MigrationTask:
    """迁移任务"""
    task_id: str
    document_id: str
    source_tier: StorageTier
    target_tier: StorageTier
    source_path: Path
    target_path: Path
    
    # 任务信息
    priority: int = 5  # 1-10，数字越小优先级越高
    estimated_time: float = 0.0  # 预计耗时（秒）
    estimated_savings: float = 0.0  # 预计节省（美元/月）
    
    # 状态
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # 进度
    progress: float = 0.0  # 0.0-1.0
    bytes_transferred: int = 0
    total_bytes: int = 0

class StorageOptimizer:
    """存储优化器 - 管理多层存储和自动优化"""
    
    def __init__(self, storage_root: Path, usage_analytics: UsageAnalytics):
        self.storage_root = storage_root
        self.usage_analytics = usage_analytics
        
        # 存储层级目录
        self.tier_paths = {
            StorageTier.HOT: storage_root / "hot",
            StorageTier.WARM: storage_root / "warm", 
            StorageTier.COLD: storage_root / "cold",
            StorageTier.ARCHIVED: storage_root / "archived"
        }
        
        # 创建存储目录
        for tier_path in self.tier_paths.values():
            tier_path.mkdir(exist_ok=True, parents=True)
        
        # 默认存储策略
        self.storage_policies = {
            StorageTier.HOT: StoragePolicy(
                tier=StorageTier.HOT,
                min_access_frequency=1.0,  # 每天至少1次访问
                max_days_since_access=7,
                cost_per_gb_per_month=0.10,
                replicas=2
            ),
            StorageTier.WARM: StoragePolicy(
                tier=StorageTier.WARM,
                min_access_frequency=0.1,  # 每10天至少1次访问
                max_days_since_access=30,
                cost_per_gb_per_month=0.05,
                replicas=1
            ),
            StorageTier.COLD: StoragePolicy(
                tier=StorageTier.COLD,
                min_access_frequency=0.01,  # 每100天至少1次访问
                max_days_since_access=90,
                compression_enabled=True,
                compression_type="gzip",
                cost_per_gb_per_month=0.02
            ),
            StorageTier.ARCHIVED: StoragePolicy(
                tier=StorageTier.ARCHIVED,
                min_access_frequency=0.0,
                max_days_since_access=365,
                compression_enabled=True,
                compression_type="lzma",
                cost_per_gb_per_month=0.005
            )
        }
        
        # 迁移队列和任务管理
        self.migration_queue: List[MigrationTask] = []
        self.active_migrations: Dict[str, MigrationTask] = {}
        self.max_concurrent_migrations = 3
        
        # 性能监控
        self.optimization_stats = {
            'total_migrations': 0,
            'successful_migrations': 0,
            'failed_migrations': 0,
            'total_bytes_migrated': 0,
            'total_time_spent': 0.0,
            'total_cost_savings': 0.0,
            'last_optimization': None
        }
        
        # 后台任务
        self.optimization_task = None
        self.migration_task = None
        
    async def start_optimizer(self):
        """启动存储优化器"""
        if self.optimization_task is None:
            self.optimization_task = asyncio.create_task(self._run_optimization_loop())
        
        if self.migration_task is None:
            self.migration_task = asyncio.create_task(self._run_migration_worker())
        
        logger.info("Storage optimizer started")
    
    async def stop_optimizer(self):
        """停止存储优化器"""
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
            self.optimization_task = None
        
        if self.migration_task:
            self.migration_task.cancel()
            try:
                await self.migration_task
            except asyncio.CancelledError:
                pass
            self.migration_task = None
        
        logger.info("Storage optimizer stopped")
    
    async def optimize_storage(self) -> Dict[str, Any]:
        """执行存储优化"""
        logger.info("Starting storage optimization...")
        start_time = time.time()
        
        optimization_result = {
            'migrations_planned': 0,
            'estimated_savings': 0.0,
            'optimization_time': 0.0,
            'tier_changes': defaultdict(int),
            'errors': []
        }
        
        try:
            # 获取所有文档的访问模式
            documents = await self._get_all_documents()
            
            for doc_info in documents:
                try:
                    # 获取访问模式
                    access_pattern = await self.usage_analytics.get_access_pattern(doc_info['document_id'])
                    if not access_pattern:
                        continue
                    
                    # 计算最佳存储层级
                    optimal_tier = await self._calculate_optimal_tier(doc_info, access_pattern)
                    current_tier = StorageTier(doc_info['storage_tier'])
                    
                    # 如果需要迁移
                    if optimal_tier != current_tier:
                        migration_task = await self._create_migration_task(
                            doc_info, current_tier, optimal_tier
                        )
                        
                        if migration_task:
                            self.migration_queue.append(migration_task)
                            optimization_result['migrations_planned'] += 1
                            optimization_result['estimated_savings'] += migration_task.estimated_savings
                            optimization_result['tier_changes'][f"{current_tier.value}_to_{optimal_tier.value}"] += 1
                
                except Exception as e:
                    error_msg = f"Error processing document {doc_info.get('document_id', 'unknown')}: {e}"
                    optimization_result['errors'].append(error_msg)
                    logger.error(error_msg)
            
            # 排序迁移队列（按优先级和预计节省）
            self.migration_queue.sort(
                key=lambda t: (t.priority, -t.estimated_savings)
            )
            
            optimization_result['optimization_time'] = time.time() - start_time
            self.optimization_stats['last_optimization'] = datetime.now(timezone.utc)
            
            logger.info(f"Storage optimization completed: {optimization_result['migrations_planned']} migrations planned")
            
        except Exception as e:
            error_msg = f"Storage optimization failed: {e}"
            optimization_result['errors'].append(error_msg)
            logger.error(error_msg)
        
        return optimization_result
    
    async def _get_all_documents(self) -> List[Dict[str, Any]]:
        """获取所有文档信息"""
        documents = []
        
        with sqlite3.connect(self.usage_analytics.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT document_id, file_path, file_size_bytes, storage_tier,
                       created_at, last_accessed, access_count
                FROM document_storage
            ''')
            
            for row in cursor.fetchall():
                documents.append({
                    'document_id': row[0],
                    'file_path': row[1],
                    'file_size_bytes': row[2] or 0,
                    'storage_tier': row[3] or 'warm',
                    'created_at': row[4],
                    'last_accessed': row[5],
                    'access_count': row[6] or 0
                })
        
        return documents
    
    async def _calculate_optimal_tier(self, doc_info: Dict[str, Any], 
                                    access_pattern: AccessPattern) -> StorageTier:
        """计算最佳存储层级"""
        file_size_mb = doc_info['file_size_bytes'] / (1024 * 1024)
        
        # 计算未访问天数
        days_since_access = 0
        if access_pattern.last_accessed:
            days_since_access = (datetime.now(timezone.utc) - access_pattern.last_accessed).days
        elif doc_info['last_accessed']:
            try:
                last_access = datetime.fromisoformat(doc_info['last_accessed'])
                days_since_access = (datetime.now(timezone.utc) - last_access).days
            except:
                days_since_access = 365  # 默认很久没访问
        else:
            days_since_access = 365
        
        # 按优先级检查每个层级
        tier_order = [StorageTier.HOT, StorageTier.WARM, StorageTier.COLD, StorageTier.ARCHIVED]
        
        for tier in tier_order:
            policy = self.storage_policies[tier]
            if policy.matches_criteria(access_pattern, file_size_mb, days_since_access):
                return tier
        
        # 默认返回归档层级
        return StorageTier.ARCHIVED
    
    async def _create_migration_task(self, doc_info: Dict[str, Any], 
                                   source_tier: StorageTier, 
                                   target_tier: StorageTier) -> Optional[MigrationTask]:
        """创建迁移任务"""
        document_id = doc_info['document_id']
        file_size_bytes = doc_info['file_size_bytes']
        
        # 构建文件路径
        filename = f"{document_id}.pdf"  # 假设都是PDF文件
        source_path = self.tier_paths[source_tier] / filename
        target_path = self.tier_paths[target_tier] / filename
        
        # 检查源文件是否存在
        if not source_path.exists():
            logger.warning(f"Source file not found: {source_path}")
            return None
        
        # 计算迁移优先级
        priority = self._calculate_migration_priority(source_tier, target_tier, file_size_bytes)
        
        # 计算预计节省
        source_policy = self.storage_policies[source_tier]
        target_policy = self.storage_policies[target_tier]
        size_gb = file_size_bytes / (1024**3)
        monthly_savings = size_gb * (source_policy.cost_per_gb_per_month - target_policy.cost_per_gb_per_month)
        
        # 预计迁移时间（基于文件大小和层级）
        estimated_time = self._estimate_migration_time(file_size_bytes, source_tier, target_tier)
        
        task_id = hashlib.md5(f"{document_id}_{source_tier.value}_{target_tier.value}_{time.time()}".encode()).hexdigest()
        
        return MigrationTask(
            task_id=task_id,
            document_id=document_id,
            source_tier=source_tier,
            target_tier=target_tier,
            source_path=source_path,
            target_path=target_path,
            priority=priority,
            estimated_time=estimated_time,
            estimated_savings=monthly_savings,
            total_bytes=file_size_bytes
        )
    
    def _calculate_migration_priority(self, source_tier: StorageTier, 
                                    target_tier: StorageTier, file_size_bytes: int) -> int:
        """计算迁移优先级"""
        # 基础优先级
        priority_map = {
            (StorageTier.HOT, StorageTier.COLD): 2,      # 高优先级：热数据到冷存储
            (StorageTier.HOT, StorageTier.ARCHIVED): 1,  # 最高优先级：热数据到归档
            (StorageTier.WARM, StorageTier.COLD): 3,
            (StorageTier.WARM, StorageTier.ARCHIVED): 2,
            (StorageTier.COLD, StorageTier.HOT): 8,      # 低优先级：冷数据到热存储
            (StorageTier.COLD, StorageTier.WARM): 7,
            (StorageTier.ARCHIVED, StorageTier.HOT): 9,  # 最低优先级：归档到热存储
            (StorageTier.ARCHIVED, StorageTier.WARM): 8,
            (StorageTier.ARCHIVED, StorageTier.COLD): 7
        }
        
        base_priority = priority_map.get((source_tier, target_tier), 5)
        
        # 大文件优先级更高（节省更多）
        size_mb = file_size_bytes / (1024 * 1024)
        if size_mb > 100:
            base_priority = max(1, base_priority - 1)
        elif size_mb < 1:
            base_priority = min(10, base_priority + 1)
        
        return base_priority
    
    def _estimate_migration_time(self, file_size_bytes: int, 
                               source_tier: StorageTier, target_tier: StorageTier) -> float:
        """估算迁移时间"""
        # 基础传输速度（字节/秒）
        transfer_speeds = {
            StorageTier.HOT: 100 * 1024 * 1024,     # 100 MB/s
            StorageTier.WARM: 50 * 1024 * 1024,     # 50 MB/s
            StorageTier.COLD: 20 * 1024 * 1024,     # 20 MB/s
            StorageTier.ARCHIVED: 10 * 1024 * 1024  # 10 MB/s
        }
        
        # 使用较慢的速度
        speed = min(transfer_speeds[source_tier], transfer_speeds[target_tier])
        
        # 基础传输时间
        transfer_time = file_size_bytes / speed
        
        # 压缩时间（如果目标层级需要压缩）
        compression_time = 0
        if self.storage_policies[target_tier].compression_enabled:
            # 压缩大约需要原始传输时间的2倍
            compression_time = transfer_time * 2
        
        # 验证时间
        verification_time = transfer_time * 0.1
        
        return transfer_time + compression_time + verification_time
    
    async def _run_optimization_loop(self):
        """运行优化循环"""
        while True:
            try:
                # 每6小时运行一次优化
                await self.optimize_storage()
                await asyncio.sleep(6 * 3600)
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(3600)  # 错误时等待1小时
    
    async def _run_migration_worker(self):
        """运行迁移工作者"""
        while True:
            try:
                # 检查是否有待处理的迁移任务
                if (len(self.active_migrations) < self.max_concurrent_migrations and 
                    self.migration_queue):
                    
                    task = self.migration_queue.pop(0)
                    self.active_migrations[task.task_id] = task
                    
                    # 在后台执行迁移
                    asyncio.create_task(self._execute_migration(task))
                
                await asyncio.sleep(10)  # 每10秒检查一次
                
            except Exception as e:
                logger.error(f"Error in migration worker: {e}")
                await asyncio.sleep(60)
    
    async def _execute_migration(self, task: MigrationTask):
        """执行迁移任务"""
        logger.info(f"Starting migration: {task.document_id} from {task.source_tier.value} to {task.target_tier.value}")
        
        task.status = "running"
        task.started_at = datetime.now(timezone.utc)
        
        try:
            # 确保目标目录存在
            task.target_path.parent.mkdir(exist_ok=True, parents=True)
            
            # 检查是否需要压缩
            target_policy = self.storage_policies[task.target_tier]
            
            if target_policy.compression_enabled:
                await self._migrate_with_compression(task)
            else:
                await self._migrate_without_compression(task)
            
            # 验证迁移
            if await self._verify_migration(task):
                # 删除源文件
                task.source_path.unlink()
                
                # 更新数据库
                await self._update_document_storage_tier(task.document_id, task.target_tier)
                
                task.status = "completed"
                task.progress = 1.0
                
                # 更新统计
                self.optimization_stats['successful_migrations'] += 1
                self.optimization_stats['total_bytes_migrated'] += task.total_bytes
                self.optimization_stats['total_cost_savings'] += task.estimated_savings
                
                logger.info(f"Migration completed successfully: {task.document_id}")
            else:
                raise Exception("Migration verification failed")
        
        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            self.optimization_stats['failed_migrations'] += 1
            logger.error(f"Migration failed: {task.document_id} - {e}")
        
        finally:
            task.completed_at = datetime.now(timezone.utc)
            if task.started_at:
                migration_time = (task.completed_at - task.started_at).total_seconds()
                self.optimization_stats['total_time_spent'] += migration_time
            
            # 从活跃迁移中移除
            self.active_migrations.pop(task.task_id, None)
            self.optimization_stats['total_migrations'] += 1
    
    async def _migrate_with_compression(self, task: MigrationTask):
        """带压缩的迁移"""
        target_policy = self.storage_policies[task.target_tier]
        
        # 选择压缩方法
        if target_policy.compression_type == "lzma":
            compressed_path = task.target_path.with_suffix(task.target_path.suffix + ".xz")
            
            with open(task.source_path, 'rb') as src_file:
                with lzma.open(compressed_path, 'wb', preset=6) as dst_file:
                    await self._copy_with_progress(src_file, dst_file, task)
                    
        else:  # gzip
            compressed_path = task.target_path.with_suffix(task.target_path.suffix + ".gz")
            
            with open(task.source_path, 'rb') as src_file:
                with gzip.open(compressed_path, 'wb', compresslevel=6) as dst_file:
                    await self._copy_with_progress(src_file, dst_file, task)
    
    async def _migrate_without_compression(self, task: MigrationTask):
        """不压缩的迁移"""
        with open(task.source_path, 'rb') as src_file:
            with open(task.target_path, 'wb') as dst_file:
                await self._copy_with_progress(src_file, dst_file, task)
    
    async def _copy_with_progress(self, src_file, dst_file, task: MigrationTask):
        """带进度的文件复制"""
        chunk_size = 1024 * 1024  # 1MB chunks
        bytes_copied = 0
        
        while True:
            chunk = src_file.read(chunk_size)
            if not chunk:
                break
            
            dst_file.write(chunk)
            bytes_copied += len(chunk)
            
            # 更新进度
            task.bytes_transferred = bytes_copied
            task.progress = bytes_copied / task.total_bytes if task.total_bytes > 0 else 0
            
            # 让出控制权
            if bytes_copied % (10 * 1024 * 1024) == 0:  # 每10MB让出一次
                await asyncio.sleep(0)
    
    async def _verify_migration(self, task: MigrationTask) -> bool:
        """验证迁移结果"""
        try:
            target_policy = self.storage_policies[task.target_tier]
            
            if target_policy.compression_enabled:
                # 验证压缩文件
                if target_policy.compression_type == "lzma":
                    compressed_path = task.target_path.with_suffix(task.target_path.suffix + ".xz")
                    with lzma.open(compressed_path, 'rb') as f:
                        # 读取前1KB验证文件可读
                        f.read(1024)
                else:
                    compressed_path = task.target_path.with_suffix(task.target_path.suffix + ".gz")
                    with gzip.open(compressed_path, 'rb') as f:
                        f.read(1024)
            else:
                # 验证普通文件
                if not task.target_path.exists():
                    return False
                
                # 检查文件大小
                if task.target_path.stat().st_size != task.total_bytes:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Migration verification failed: {e}")
            return False
    
    async def _update_document_storage_tier(self, document_id: str, new_tier: StorageTier):
        """更新文档存储层级"""
        with sqlite3.connect(self.usage_analytics.db_path) as conn:
            conn.execute('''
                UPDATE document_storage 
                SET storage_tier = ?
                WHERE document_id = ?
            ''', (new_tier.value, document_id))
            conn.commit()
    
    def get_migration_status(self) -> Dict[str, Any]:
        """获取迁移状态"""
        return {
            'queue_size': len(self.migration_queue),
            'active_migrations': len(self.active_migrations),
            'active_tasks': [
                {
                    'task_id': task.task_id,
                    'document_id': task.document_id,
                    'source_tier': task.source_tier.value,
                    'target_tier': task.target_tier.value,
                    'progress': task.progress,
                    'status': task.status
                }
                for task in self.active_migrations.values()
            ],
            'optimization_stats': self.optimization_stats.copy()
        }
    
    async def force_migration(self, document_id: str, target_tier: StorageTier) -> bool:
        """强制迁移文档到指定层级"""
        try:
            # 获取文档信息
            documents = await self._get_all_documents()
            doc_info = next((doc for doc in documents if doc['document_id'] == document_id), None)
            
            if not doc_info:
                logger.error(f"Document not found: {document_id}")
                return False
            
            current_tier = StorageTier(doc_info['storage_tier'])
            if current_tier == target_tier:
                logger.info(f"Document {document_id} already in target tier {target_tier.value}")
                return True
            
            # 创建迁移任务
            migration_task = await self._create_migration_task(doc_info, current_tier, target_tier)
            if not migration_task:
                logger.error(f"Failed to create migration task for {document_id}")
                return False
            
            # 设置高优先级
            migration_task.priority = 1
            
            # 添加到队列前端
            self.migration_queue.insert(0, migration_task)
            
            logger.info(f"Forced migration queued: {document_id} to {target_tier.value}")
            return True
            
        except Exception as e:
            logger.error(f"Force migration failed: {e}")
            return False
    
    def get_tier_statistics(self) -> Dict[str, Any]:
        """获取层级统计"""
        stats = {}
        
        for tier, path in self.tier_paths.items():
            if path.exists():
                files = list(path.iterdir())
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                
                stats[tier.value] = {
                    'file_count': len(files),
                    'total_size_gb': total_size / (1024**3),
                    'avg_file_size_mb': (total_size / len(files) / (1024**2)) if files else 0,
                    'path': str(path)
                }
        
        return stats

# 使用示例
async def main():
    """测试存储优化器"""
    from .usage_analytics import UsageAnalytics
    
    # 初始化组件
    usage_analytics = UsageAnalytics(
        db_path=Path("data/storage/usage_analytics.db"),
        storage_root=Path("data/storage")
    )
    
    optimizer = StorageOptimizer(
        storage_root=Path("data/storage"),
        usage_analytics=usage_analytics
    )
    
    # 启动优化器
    await optimizer.start_optimizer()
    
    # 运行优化
    result = await optimizer.optimize_storage()
    print("优化结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 获取迁移状态
    status = optimizer.get_migration_status()
    print("\\n迁移状态:")
    print(json.dumps(status, indent=2, ensure_ascii=False, default=str))
    
    # 获取层级统计
    tier_stats = optimizer.get_tier_statistics()
    print("\\n层级统计:")
    print(json.dumps(tier_stats, indent=2, ensure_ascii=False))
    
    # 停止优化器
    await optimizer.stop_optimizer()

if __name__ == "__main__":
    asyncio.run(main())