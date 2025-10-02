# src/utils/progress_tracker.py
import time
import threading
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import sys
from loguru import logger


@dataclass
class ProgressStage:
    """进度阶段定义"""
    name: str
    total_items: int
    completed_items: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    weight: float = 1.0  # 阶段权重
    status: str = "pending"  # pending, running, completed, failed
    
    @property
    def progress_percent(self) -> float:
        if self.total_items == 0:
            return 100.0
        return (self.completed_items / self.total_items) * 100
    
    @property
    def elapsed_time(self) -> float:
        if not self.start_time:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def eta_seconds(self) -> float:
        if self.completed_items == 0 or not self.start_time:
            return 0.0
        elapsed = self.elapsed_time
        rate = self.completed_items / elapsed
        remaining = self.total_items - self.completed_items
        return remaining / rate if rate > 0 else 0.0


class MultiStageProgressTracker:
    """多阶段进度跟踪器"""
    
    def __init__(self, description: str = "处理进度"):
        self.description = description
        self.stages: Dict[str, ProgressStage] = {}
        self.current_stage: Optional[str] = None
        self.start_time = time.time()
        self.display_thread: Optional[threading.Thread] = None
        self.stop_display = threading.Event()
        self._lock = threading.Lock()
        
    def add_stage(self, stage_id: str, name: str, total_items: int, weight: float = 1.0):
        """添加处理阶段"""
        with self._lock:
            self.stages[stage_id] = ProgressStage(
                name=name,
                total_items=total_items,
                weight=weight
            )
    
    def start_stage(self, stage_id: str):
        """开始处理阶段"""
        with self._lock:
            if stage_id in self.stages:
                self.current_stage = stage_id
                self.stages[stage_id].status = "running"
                self.stages[stage_id].start_time = time.time()
                logger.info(f"开始阶段: {self.stages[stage_id].name}")
    
    def update_stage(self, stage_id: str, completed: int, details: str = ""):
        """更新阶段进度"""
        with self._lock:
            if stage_id in self.stages:
                stage = self.stages[stage_id]
                stage.completed_items = max(0, min(completed, stage.total_items))

    def increment_stage(self, stage_id: str, amount: int = 1):
        """增量更新阶段进度，适合并发场景"""
        if amount == 0:
            return
        with self._lock:
            if stage_id in self.stages:
                stage = self.stages[stage_id]
                stage.completed_items = max(
                    0,
                    min(stage.completed_items + amount, stage.total_items)
                )
                
    def complete_stage(self, stage_id: str):
        """完成阶段"""
        with self._lock:
            if stage_id in self.stages:
                stage = self.stages[stage_id]
                stage.status = "completed"
                stage.completed_items = stage.total_items
                stage.end_time = time.time()
                logger.success(f"完成阶段: {stage.name} ({stage.elapsed_time:.1f}秒)")
    
    def fail_stage(self, stage_id: str, error: str = ""):
        """标记阶段失败"""
        with self._lock:
            if stage_id in self.stages:
                self.stages[stage_id].status = "failed"
                self.stages[stage_id].end_time = time.time()
                logger.error(f"阶段失败: {self.stages[stage_id].name} - {error}")
    
    def get_overall_progress(self) -> float:
        """获取总体进度百分比"""
        if not self.stages:
            return 0.0
            
        total_weight = sum(stage.weight for stage in self.stages.values())
        completed_weight = sum(
            stage.weight * (stage.progress_percent / 100)
            for stage in self.stages.values()
        )
        
        return (completed_weight / total_weight) * 100 if total_weight > 0 else 0.0
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        with self._lock:
            current_stage_info = None
            if self.current_stage and self.current_stage in self.stages:
                stage = self.stages[self.current_stage]
                current_stage_info = {
                    "name": stage.name,
                    "progress": stage.progress_percent,
                    "completed": stage.completed_items,
                    "total": stage.total_items,
                    "eta": stage.eta_seconds,
                    "elapsed": stage.elapsed_time
                }
            
            return {
                "overall_progress": self.get_overall_progress(),
                "current_stage": current_stage_info,
                "total_elapsed": time.time() - self.start_time,
                "stages_summary": {
                    stage_id: {
                        "name": stage.name,
                        "status": stage.status,
                        "progress": stage.progress_percent
                    }
                    for stage_id, stage in self.stages.items()
                }
            }
    
    def start_display(self, update_interval: float = 1.0):
        """开始显示进度条"""
        def display_loop():
            while not self.stop_display.wait(update_interval):
                self._update_display()
        
        self.display_thread = threading.Thread(target=display_loop, daemon=True)
        self.display_thread.start()
    
    def stop_display_thread(self):
        """停止显示进度条"""
        if self.display_thread:
            self.stop_display.set()
            self.display_thread.join()
            self._update_display(final=True)
    
    def _update_display(self, final: bool = False):
        """更新进度显示"""
        status = self.get_current_status()
        
        # 清除当前行
        if not final:
            sys.stdout.write('\r' + ' ' * 100 + '\r')
        
        # 总体进度条
        overall = status["overall_progress"]
        bar_length = 30
        filled_length = int(bar_length * overall / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        output = f"🚀 {self.description}: [{bar}] {overall:.1f}%"
        
        # 当前阶段信息
        if status["current_stage"]:
            current = status["current_stage"]
            eta_str = f"{current['eta']:.0f}s" if current['eta'] > 0 else "未知"
            output += f" | {current['name']}: {current['completed']}/{current['total']} (ETA: {eta_str})"
        
        # 总用时
        output += f" | 用时: {status['total_elapsed']:.1f}s"
        
        if final:
            print(output)
            print()  # 添加空行
        else:
            sys.stdout.write(output)
            sys.stdout.flush()
    
    def __enter__(self):
        self.start_display()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_display_thread()


class SimpleProgressBar:
    """简单进度条（用于单阶段处理）"""
    
    def __init__(self, total: int, description: str = "处理中", unit: str = "items"):
        self.total = total
        self.description = description
        self.unit = unit
        self.completed = 0
        self.start_time = time.time()
        
    def update(self, count: int = 1, postfix: str = ""):
        """更新进度"""
        self.completed = min(self.completed + count, self.total)
        self._display(postfix)
    
    def set_progress(self, completed: int, postfix: str = ""):
        """设置当前进度"""
        self.completed = min(completed, self.total)
        self._display(postfix)
    
    def _display(self, postfix: str = ""):
        """显示进度条"""
        percent = (self.completed / self.total) * 100 if self.total > 0 else 0
        elapsed = time.time() - self.start_time
        
        if self.completed > 0:
            rate = self.completed / elapsed
            eta = (self.total - self.completed) / rate if rate > 0 else 0
            eta_str = f"{eta:.0f}s"
        else:
            rate = 0
            eta_str = "未知"
        
        # 进度条
        bar_length = 25
        filled_length = int(bar_length * percent / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        output = (f"\r{self.description}: [{bar}] {percent:.1f}% "
                 f"({self.completed}/{self.total} {self.unit}) "
                 f"[{rate:.1f} {self.unit}/s, ETA: {eta_str}]")
        
        if postfix:
            output += f" | {postfix}"
        
        sys.stdout.write(output)
        sys.stdout.flush()
        
        if self.completed >= self.total:
            print()  # 完成时换行
    
    def close(self):
        """关闭进度条"""
        if self.completed < self.total:
            self.completed = self.total
            self._display("完成")


# 进度回调装饰器
def progress_callback(tracker: MultiStageProgressTracker, stage_id: str):
    """为函数添加进度回调的装饰器"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            tracker.start_stage(stage_id)
            try:
                result = func(*args, **kwargs)
                tracker.complete_stage(stage_id)
                return result
            except Exception as e:
                tracker.fail_stage(stage_id, str(e))
                raise
        return wrapper
    return decorator


# 使用示例
if __name__ == "__main__":
    import asyncio
    
    async def demo_multi_stage_progress():
        """演示多阶段进度跟踪"""
        
        with MultiStageProgressTracker("多表示处理演示") as tracker:
            # 添加处理阶段
            tracker.add_stage("load", "加载模型", 3, weight=1.0)
            tracker.add_stage("generate", "生成表示", 731, weight=3.0)
            tracker.add_stage("embed", "嵌入向量化", 2193, weight=2.0)  # 731 * 3 representations
            
            # 阶段1: 加载模型
            tracker.start_stage("load")
            for i in range(3):
                await asyncio.sleep(0.5)  # 模拟加载时间
                tracker.update_stage("load", i + 1)
            tracker.complete_stage("load")
            
            # 阶段2: 生成表示
            tracker.start_stage("generate")
            for i in range(731):
                await asyncio.sleep(0.01)  # 模拟生成时间
                tracker.update_stage("generate", i + 1)
            tracker.complete_stage("generate")
            
            # 阶段3: 嵌入向量化
            tracker.start_stage("embed")
            for i in range(2193):
                await asyncio.sleep(0.005)  # 模拟嵌入时间
                tracker.update_stage("embed", i + 1)
            tracker.complete_stage("embed")
    
    # 运行演示
    asyncio.run(demo_multi_stage_progress())
