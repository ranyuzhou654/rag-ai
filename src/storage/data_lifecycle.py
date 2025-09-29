# src/storage/data_lifecycle.py
import asyncio
import json
import sqlite3
import shutil
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from loguru import logger
from enum import Enum
import time
import hashlib

from .usage_analytics import UsageAnalytics, AccessPattern
from .storage_optimizer import StorageOptimizer, StorageTier

class LifecycleAction(Enum):
    """生命周期动作"""
    RETAIN = "retain"          # 保留
    MIGRATE = "migrate"        # 迁移到其他层级
    COMPRESS = "compress"      # 压缩
    ARCHIVE = "archive"        # 归档
    DELETE = "delete"          # 删除
    BACKUP = "backup"          # 备份
    REPLICATE = "replicate"    # 复制

@dataclass
class LifecycleRule:
    """生命周期规则"""
    rule_id: str
    name: str
    description: str
    
    # 触发条件
    max_age_days: Optional[int] = None          # 最大年龄（天）
    min_access_frequency: Optional[float] = None  # 最小访问频率
    max_file_size_gb: Optional[float] = None    # 最大文件大小
    min_file_size_mb: Optional[float] = None    # 最小文件大小
    source_tiers: Optional[List[StorageTier]] = None  # 适用的源层级
    file_patterns: Optional[List[str]] = None   # 文件模式匹配
    
    # 动作配置
    action: LifecycleAction = LifecycleAction.RETAIN
    target_tier: Optional[StorageTier] = None   # 迁移目标层级
    retention_days: Optional[int] = None        # 保留天数
    
    # 执行配置
    enabled: bool = True
    priority: int = 5  # 1-10，数字越小优先级越高
    batch_size: int = 100  # 批处理大小
    
    # 安全设置
    require_confirmation: bool = False  # 是否需要确认
    dry_run_only: bool = False         # 仅试运行
    
    def matches_document(self, doc_info: Dict[str, Any], 
                        access_pattern: Optional[AccessPattern] = None) -> bool:
        """检查文档是否匹配规则"""
        try:
            # 年龄检查
            if self.max_age_days is not None:
                created_at = doc_info.get('created_at')
                if created_at:
                    try:
                        created_date = datetime.fromisoformat(created_at)
                        age_days = (datetime.now(timezone.utc) - created_date).days
                        if age_days < self.max_age_days:
                            return False
                    except:
                        pass  # 解析失败，跳过年龄检查
            
            # 访问频率检查
            if self.min_access_frequency is not None and access_pattern:
                if access_pattern.daily_access_avg >= self.min_access_frequency:
                    return False
            
            # 文件大小检查
            file_size_bytes = doc_info.get('file_size_bytes', 0)
            if self.max_file_size_gb is not None:
                if file_size_bytes > self.max_file_size_gb * 1024**3:
                    return False
            
            if self.min_file_size_mb is not None:
                if file_size_bytes < self.min_file_size_mb * 1024**2:
                    return False
            
            # 层级检查
            if self.source_tiers:
                current_tier = doc_info.get('storage_tier', 'warm')
                if StorageTier(current_tier) not in self.source_tiers:
                    return False
            
            # 文件模式检查
            if self.file_patterns:
                file_path = doc_info.get('file_path', '')
                import fnmatch
                if not any(fnmatch.fnmatch(file_path, pattern) for pattern in self.file_patterns):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error matching document against rule {self.rule_id}: {e}")
            return False

@dataclass
class LifecyclePolicy:
    """生命周期策略"""
    policy_id: str
    name: str
    description: str
    rules: List[LifecycleRule] = field(default_factory=list)
    
    # 策略配置
    enabled: bool = True
    schedule_cron: str = "0 2 * * *"  # 每天凌晨2点执行
    max_operations_per_run: int = 1000
    
    # 监控配置
    enable_notifications: bool = True
    notification_threshold: int = 100  # 超过多少操作发送通知
    
    def get_applicable_rules(self, doc_info: Dict[str, Any], 
                           access_pattern: Optional[AccessPattern] = None) -> List[LifecycleRule]:
        """获取适用的规则"""
        if not self.enabled:
            return []
        
        applicable_rules = []
        for rule in self.rules:
            if rule.enabled and rule.matches_document(doc_info, access_pattern):
                applicable_rules.append(rule)
        
        # 按优先级排序
        applicable_rules.sort(key=lambda r: r.priority)
        return applicable_rules

@dataclass
class LifecycleExecution:
    """生命周期执行记录"""
    execution_id: str
    policy_id: str
    rule_id: str
    document_id: str
    action: LifecycleAction
    
    # 执行信息
    status: str = "pending"  # pending, running, completed, failed, skipped
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # 执行结果
    bytes_processed: int = 0
    space_saved: int = 0
    cost_impact: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)

class DataLifecycleManager:
    """数据生命周期管理器"""
    
    def __init__(self, usage_analytics: UsageAnalytics, 
                 storage_optimizer: Optional[StorageOptimizer] = None):
        self.usage_analytics = usage_analytics
        self.storage_optimizer = storage_optimizer
        
        # 数据库路径
        self.db_path = usage_analytics.db_path.parent / "lifecycle.db"
        
        # 默认策略
        self.policies: Dict[str, LifecyclePolicy] = {}
        self._create_default_policies()
        
        # 执行状态
        self.active_executions: Dict[str, LifecycleExecution] = {}
        self.execution_history: List[LifecycleExecution] = []
        
        # 性能统计
        self.lifecycle_stats = {
            'total_executions': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_space_saved': 0,
            'total_cost_savings': 0.0,
            'last_execution': None
        }
        
        # 后台任务
        self.lifecycle_task = None
        
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            # 生命周期策略表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS lifecycle_policies (
                    policy_id TEXT PRIMARY KEY,
                    policy_data TEXT,  -- JSON格式的策略数据
                    created_at TEXT,
                    last_updated TEXT,
                    enabled BOOLEAN DEFAULT TRUE
                )
            ''')
            
            # 执行历史表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS lifecycle_executions (
                    execution_id TEXT PRIMARY KEY,
                    policy_id TEXT,
                    rule_id TEXT,
                    document_id TEXT,
                    action TEXT,
                    status TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    bytes_processed INTEGER DEFAULT 0,
                    space_saved INTEGER DEFAULT 0,
                    cost_impact REAL DEFAULT 0.0
                )
            ''')
            
            # 创建索引
            conn.execute('CREATE INDEX IF NOT EXISTS idx_executions_policy ON lifecycle_executions(policy_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_executions_status ON lifecycle_executions(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_executions_started ON lifecycle_executions(started_at)')
            
            conn.commit()
        
        logger.info(f"Lifecycle database initialized at {self.db_path}")
    
    def _create_default_policies(self):
        """创建默认生命周期策略"""
        # 旧文档清理策略
        old_docs_policy = LifecyclePolicy(
            policy_id="old_documents_cleanup",
            name="旧文档清理策略",
            description="清理超过1年未访问的旧文档",
            rules=[
                LifecycleRule(
                    rule_id="archive_old_docs",
                    name="归档旧文档",
                    description="将1年未访问的文档归档",
                    max_age_days=365,
                    min_access_frequency=0.001,  # 几乎不访问
                    action=LifecycleAction.ARCHIVE,
                    target_tier=StorageTier.ARCHIVED,
                    priority=3
                ),
                LifecycleRule(
                    rule_id="delete_ancient_docs",
                    name="删除古老文档",
                    description="删除超过3年未访问的文档",
                    max_age_days=1095,  # 3年
                    min_access_frequency=0.0,
                    action=LifecycleAction.DELETE,
                    priority=1,
                    require_confirmation=True
                )
            ]
        )
        
        # 大文件优化策略
        large_files_policy = LifecyclePolicy(
            policy_id="large_files_optimization",
            name="大文件优化策略", 
            description="优化存储中的大文件",
            rules=[
                LifecycleRule(
                    rule_id="compress_large_files",
                    name="压缩大文件",
                    description="压缩超过100MB且访问较少的文件",
                    min_file_size_mb=100,
                    min_access_frequency=0.1,  # 每10天少于1次
                    action=LifecycleAction.COMPRESS,
                    priority=4
                ),
                LifecycleRule(
                    rule_id="migrate_huge_files",
                    name="迁移巨大文件",
                    description="将超过1GB的文件迁移到冷存储",
                    min_file_size_mb=1024,  # 1GB
                    action=LifecycleAction.MIGRATE,
                    target_tier=StorageTier.COLD,
                    priority=2
                )
            ]
        )
        
        # 热数据管理策略
        hot_data_policy = LifecyclePolicy(
            policy_id="hot_data_management",
            name="热数据管理策略",
            description="管理热存储层的数据",
            rules=[
                LifecycleRule(
                    rule_id="cool_down_hot_data",
                    name="冷却热数据",
                    description="将热存储中访问较少的数据迁移到温存储",
                    source_tiers=[StorageTier.HOT],
                    max_age_days=30,
                    min_access_frequency=0.5,  # 每2天少于1次
                    action=LifecycleAction.MIGRATE,
                    target_tier=StorageTier.WARM,
                    priority=5
                )
            ]
        )
        
        self.policies = {
            old_docs_policy.policy_id: old_docs_policy,
            large_files_policy.policy_id: large_files_policy,
            hot_data_policy.policy_id: hot_data_policy
        }
    
    async def start_lifecycle_management(self):
        """启动生命周期管理"""
        if self.lifecycle_task is None:
            self.lifecycle_task = asyncio.create_task(self._run_lifecycle_loop())
            logger.info("Data lifecycle management started")
    
    async def stop_lifecycle_management(self):
        """停止生命周期管理"""
        if self.lifecycle_task:
            self.lifecycle_task.cancel()
            try:
                await self.lifecycle_task
            except asyncio.CancelledError:
                pass
            self.lifecycle_task = None
            logger.info("Data lifecycle management stopped")
    
    async def execute_policies(self, policy_ids: Optional[List[str]] = None, 
                             dry_run: bool = False) -> Dict[str, Any]:
        """执行生命周期策略"""
        logger.info("Starting lifecycle policy execution...")
        start_time = time.time()
        
        execution_result = {
            'policies_executed': 0,
            'rules_applied': 0,
            'documents_processed': 0,
            'operations_performed': 0,
            'space_saved': 0,
            'estimated_cost_savings': 0.0,
            'execution_time': 0.0,
            'errors': []
        }
        
        try:
            # 确定要执行的策略
            target_policies = []
            if policy_ids:
                target_policies = [self.policies[pid] for pid in policy_ids if pid in self.policies]
            else:
                target_policies = [p for p in self.policies.values() if p.enabled]
            
            # 获取所有文档
            documents = await self._get_all_documents()
            
            for policy in target_policies:
                try:
                    policy_result = await self._execute_policy(policy, documents, dry_run)
                    
                    execution_result['policies_executed'] += 1
                    execution_result['rules_applied'] += policy_result['rules_applied']
                    execution_result['documents_processed'] += policy_result['documents_processed']
                    execution_result['operations_performed'] += policy_result['operations_performed']
                    execution_result['space_saved'] += policy_result['space_saved']
                    execution_result['estimated_cost_savings'] += policy_result['estimated_cost_savings']
                    
                except Exception as e:
                    error_msg = f"Error executing policy {policy.policy_id}: {e}"
                    execution_result['errors'].append(error_msg)
                    logger.error(error_msg)
            
            execution_result['execution_time'] = time.time() - start_time
            self.lifecycle_stats['last_execution'] = datetime.now(timezone.utc)
            
            logger.info(f"Lifecycle execution completed in {execution_result['execution_time']:.2f}s")
            
        except Exception as e:
            error_msg = f"Lifecycle execution failed: {e}"
            execution_result['errors'].append(error_msg)
            logger.error(error_msg)
        
        return execution_result
    
    async def _execute_policy(self, policy: LifecyclePolicy, 
                            documents: List[Dict[str, Any]], 
                            dry_run: bool = False) -> Dict[str, Any]:
        """执行单个策略"""
        logger.info(f"Executing policy: {policy.name}")
        
        policy_result = {
            'rules_applied': 0,
            'documents_processed': 0,
            'operations_performed': 0,
            'space_saved': 0,
            'estimated_cost_savings': 0.0
        }
        
        operations_count = 0
        
        for document in documents:
            if operations_count >= policy.max_operations_per_run:
                logger.info(f"Reached max operations limit for policy {policy.policy_id}")
                break
            
            try:
                # 获取访问模式
                access_pattern = await self.usage_analytics.get_access_pattern(document['document_id'])
                
                # 获取适用的规则
                applicable_rules = policy.get_applicable_rules(document, access_pattern)
                
                for rule in applicable_rules:
                    if operations_count >= policy.max_operations_per_run:
                        break
                    
                    # 执行规则动作
                    execution_result = await self._execute_rule_action(
                        rule, document, access_pattern, dry_run
                    )
                    
                    if execution_result['success']:
                        policy_result['operations_performed'] += 1
                        policy_result['space_saved'] += execution_result.get('space_saved', 0)
                        policy_result['estimated_cost_savings'] += execution_result.get('cost_savings', 0.0)
                        operations_count += 1
                    
                    # 记录执行
                    await self._record_execution(policy.policy_id, rule.rule_id, 
                                               document['document_id'], rule.action, 
                                               execution_result, dry_run)
                    
                    # 一个文档只应用第一个匹配的规则
                    break
                
                if applicable_rules:
                    policy_result['documents_processed'] += 1
                    policy_result['rules_applied'] += len(applicable_rules)
            
            except Exception as e:
                logger.error(f"Error processing document {document.get('document_id', 'unknown')}: {e}")
        
        return policy_result
    
    async def _execute_rule_action(self, rule: LifecycleRule, 
                                 document: Dict[str, Any],
                                 access_pattern: Optional[AccessPattern],
                                 dry_run: bool = False) -> Dict[str, Any]:
        """执行规则动作"""
        result = {
            'success': False,
            'action_taken': rule.action.value,
            'space_saved': 0,
            'cost_savings': 0.0,
            'error': None
        }
        
        if dry_run or rule.dry_run_only:
            result['success'] = True
            result['dry_run'] = True
            return result
        
        try:
            document_id = document['document_id']
            file_size = document.get('file_size_bytes', 0)
            
            if rule.action == LifecycleAction.DELETE:
                await self._delete_document(document)
                result['space_saved'] = file_size
                result['cost_savings'] = self._calculate_cost_savings(file_size, 'delete')
            
            elif rule.action == LifecycleAction.MIGRATE and rule.target_tier:
                if self.storage_optimizer:
                    success = await self.storage_optimizer.force_migration(document_id, rule.target_tier)
                    if success:
                        result['cost_savings'] = self._calculate_migration_savings(
                            file_size, document['storage_tier'], rule.target_tier.value
                        )
                    else:
                        result['error'] = "Migration failed"
                        return result
                else:
                    result['error'] = "Storage optimizer not available"
                    return result
            
            elif rule.action == LifecycleAction.ARCHIVE:
                await self._archive_document(document)
                result['space_saved'] = int(file_size * 0.7)  # 假设压缩节省30%
                result['cost_savings'] = self._calculate_cost_savings(file_size, 'archive')
            
            elif rule.action == LifecycleAction.COMPRESS:
                compressed_size = await self._compress_document(document)
                result['space_saved'] = file_size - compressed_size
                result['cost_savings'] = self._calculate_cost_savings(result['space_saved'], 'compress')
            
            elif rule.action == LifecycleAction.BACKUP:
                await self._backup_document(document)
                # 备份不节省空间，但有数据保护价值
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error executing action {rule.action.value} for document {document.get('document_id', 'unknown')}: {e}")
        
        return result
    
    async def _delete_document(self, document: Dict[str, Any]):
        """删除文档"""
        document_id = document['document_id']
        file_path = document.get('file_path')
        
        if file_path and Path(file_path).exists():
            Path(file_path).unlink()
        
        # 从数据库中删除记录
        with sqlite3.connect(self.usage_analytics.db_path) as conn:
            conn.execute('DELETE FROM document_storage WHERE document_id = ?', (document_id,))
            conn.execute('DELETE FROM access_logs WHERE document_id = ?', (document_id,))
            conn.commit()
        
        logger.info(f"Deleted document: {document_id}")
    
    async def _archive_document(self, document: Dict[str, Any]):
        """归档文档"""
        if self.storage_optimizer:
            await self.storage_optimizer.force_migration(
                document['document_id'], StorageTier.ARCHIVED
            )
        else:
            logger.warning("Storage optimizer not available for archiving")
    
    async def _compress_document(self, document: Dict[str, Any]) -> int:
        """压缩文档，返回压缩后大小"""
        file_path = document.get('file_path')
        if not file_path or not Path(file_path).exists():
            return document.get('file_size_bytes', 0)
        
        import gzip
        original_path = Path(file_path)
        compressed_path = original_path.with_suffix(original_path.suffix + '.gz')
        
        with open(original_path, 'rb') as src:
            with gzip.open(compressed_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
        
        # 替换原文件
        original_path.unlink()
        compressed_path.rename(original_path)
        
        compressed_size = original_path.stat().st_size
        
        # 更新数据库中的文件大小
        with sqlite3.connect(self.usage_analytics.db_path) as conn:
            conn.execute(
                'UPDATE document_storage SET file_size_bytes = ? WHERE document_id = ?',
                (compressed_size, document['document_id'])
            )
            conn.commit()
        
        logger.info(f"Compressed document: {document['document_id']}")
        return compressed_size
    
    async def _backup_document(self, document: Dict[str, Any]):
        """备份文档"""
        file_path = document.get('file_path')
        if not file_path or not Path(file_path).exists():
            return
        
        # 创建备份目录
        backup_dir = Path(self.usage_analytics.storage_root) / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        # 复制文件到备份目录
        source_path = Path(file_path)
        backup_path = backup_dir / f"{document['document_id']}_{int(time.time())}{source_path.suffix}"
        
        shutil.copy2(source_path, backup_path)
        logger.info(f"Backed up document: {document['document_id']} to {backup_path}")
    
    def _calculate_cost_savings(self, bytes_saved: int, operation_type: str) -> float:
        """计算成本节省"""
        gb_saved = bytes_saved / (1024**3)
        
        # 简化的成本计算
        cost_per_gb_month = {
            'delete': 0.05,    # 删除节省存储成本
            'archive': 0.045,  # 归档节省部分成本
            'compress': 0.02   # 压缩节省部分成本
        }
        
        return gb_saved * cost_per_gb_month.get(operation_type, 0.02)
    
    def _calculate_migration_savings(self, file_size_bytes: int, 
                                   source_tier: str, target_tier: str) -> float:
        """计算迁移成本节省"""
        tier_costs = {
            'hot': 0.10,
            'warm': 0.05,
            'cold': 0.02,
            'archived': 0.005
        }
        
        gb_size = file_size_bytes / (1024**3)
        source_cost = tier_costs.get(source_tier, 0.05)
        target_cost = tier_costs.get(target_tier, 0.05)
        
        return gb_size * (source_cost - target_cost) if source_cost > target_cost else 0.0
    
    async def _record_execution(self, policy_id: str, rule_id: str, 
                               document_id: str, action: LifecycleAction,
                               execution_result: Dict[str, Any], dry_run: bool = False):
        """记录执行结果"""
        execution_id = hashlib.md5(
            f"{policy_id}_{rule_id}_{document_id}_{time.time()}".encode()
        ).hexdigest()
        
        execution = LifecycleExecution(
            execution_id=execution_id,
            policy_id=policy_id,
            rule_id=rule_id,
            document_id=document_id,
            action=action,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            status="completed" if execution_result['success'] else "failed",
            error_message=execution_result.get('error'),
            space_saved=execution_result.get('space_saved', 0),
            cost_impact=execution_result.get('cost_savings', 0.0)
        )
        
        if dry_run:
            execution.status = "dry_run"
        
        # 保存到数据库
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO lifecycle_executions 
                (execution_id, policy_id, rule_id, document_id, action, status,
                 started_at, completed_at, error_message, bytes_processed,
                 space_saved, cost_impact)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                execution.execution_id, execution.policy_id, execution.rule_id,
                execution.document_id, execution.action.value, execution.status,
                execution.started_at.isoformat(), execution.completed_at.isoformat(),
                execution.error_message, 0, execution.space_saved, execution.cost_impact
            ))
            conn.commit()
        
        # 更新统计
        self.lifecycle_stats['total_executions'] += 1
        if execution.status == "completed":
            self.lifecycle_stats['successful_operations'] += 1
            self.lifecycle_stats['total_space_saved'] += execution.space_saved
            self.lifecycle_stats['total_cost_savings'] += execution.cost_impact
        else:
            self.lifecycle_stats['failed_operations'] += 1
    
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
    
    async def _run_lifecycle_loop(self):
        """运行生命周期循环"""
        while True:
            try:
                # 每天执行一次生命周期策略
                await self.execute_policies()
                
                # 等待24小时
                await asyncio.sleep(24 * 3600)
                
            except Exception as e:
                logger.error(f"Error in lifecycle loop: {e}")
                await asyncio.sleep(3600)  # 错误时等待1小时
    
    def add_policy(self, policy: LifecyclePolicy):
        """添加生命周期策略"""
        self.policies[policy.policy_id] = policy
        
        # 保存到数据库
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO lifecycle_policies 
                (policy_id, policy_data, created_at, last_updated, enabled)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                policy.policy_id,
                json.dumps(asdict(policy), default=str),
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
                policy.enabled
            ))
            conn.commit()
        
        logger.info(f"Added lifecycle policy: {policy.name}")
    
    def remove_policy(self, policy_id: str):
        """移除生命周期策略"""
        if policy_id in self.policies:
            del self.policies[policy_id]
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM lifecycle_policies WHERE policy_id = ?', (policy_id,))
                conn.commit()
            
            logger.info(f"Removed lifecycle policy: {policy_id}")
    
    def get_execution_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """获取执行历史"""
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        
        history = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT execution_id, policy_id, rule_id, document_id, action,
                       status, started_at, completed_at, error_message,
                       bytes_processed, space_saved, cost_impact
                FROM lifecycle_executions
                WHERE started_at >= ?
                ORDER BY started_at DESC
            ''', (cutoff_date,))
            
            for row in cursor.fetchall():
                history.append({
                    'execution_id': row[0],
                    'policy_id': row[1],
                    'rule_id': row[2],
                    'document_id': row[3],
                    'action': row[4],
                    'status': row[5],
                    'started_at': row[6],
                    'completed_at': row[7],
                    'error_message': row[8],
                    'bytes_processed': row[9],
                    'space_saved': row[10],
                    'cost_impact': row[11]
                })
        
        return history
    
    def get_lifecycle_summary(self) -> Dict[str, Any]:
        """获取生命周期管理摘要"""
        return {
            'policies_count': len(self.policies),
            'enabled_policies': len([p for p in self.policies.values() if p.enabled]),
            'lifecycle_stats': self.lifecycle_stats.copy(),
            'background_task_running': self.lifecycle_task is not None and not self.lifecycle_task.done()
        }

# 使用示例
async def main():
    """测试数据生命周期管理器"""
    from .usage_analytics import UsageAnalytics
    from .storage_optimizer import StorageOptimizer
    
    # 初始化组件
    usage_analytics = UsageAnalytics(
        db_path=Path("data/storage/usage_analytics.db"),
        storage_root=Path("data/storage")
    )
    
    storage_optimizer = StorageOptimizer(
        storage_root=Path("data/storage"),
        usage_analytics=usage_analytics
    )
    
    lifecycle_manager = DataLifecycleManager(
        usage_analytics=usage_analytics,
        storage_optimizer=storage_optimizer
    )
    
    # 启动生命周期管理
    await lifecycle_manager.start_lifecycle_management()
    
    # 执行策略（试运行）
    result = await lifecycle_manager.execute_policies(dry_run=True)
    print("生命周期执行结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 获取执行历史
    history = lifecycle_manager.get_execution_history(7)
    print(f"\\n最近7天执行历史 ({len(history)}条):")
    for record in history[:5]:  # 显示前5条
        print(f"- {record['action']} {record['document_id']} ({record['status']})")
    
    # 获取摘要
    summary = lifecycle_manager.get_lifecycle_summary()
    print("\\n生命周期管理摘要:")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    
    # 停止生命周期管理
    await lifecycle_manager.stop_lifecycle_management()

if __name__ == "__main__":
    asyncio.run(main())