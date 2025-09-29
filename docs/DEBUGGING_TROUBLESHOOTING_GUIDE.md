# RAG-AI 调试与故障排除指南

> **完整的系统调试、故障诊断和问题解决指南**

本指南提供 RAG-AI 系统的全面调试方法、常见问题诊断步骤和故障排除流程，帮助开发者和运维人员快速定位和解决问题。

## 📋 目录

1. [调试工具和方法](#1-调试工具和方法)
2. [常见问题分类诊断](#2-常见问题分类诊断)
3. [性能问题排查](#3-性能问题排查)
4. [组件故障诊断](#4-组件故障诊断)
5. [日志分析和监控](#5-日志分析和监控)
6. [紧急故障处理](#6-紧急故障处理)
7. [预防性维护](#7-预防性维护)

## 1. 调试工具和方法

### 1.1 系统级调试工具

#### 1.1.1 Docker 调试命令

```bash
# ====== 容器状态诊断 ======

# 查看所有容器状态
docker-compose ps

# 查看详细容器信息
docker-compose ps --services
docker-compose config

# 检查容器资源使用
docker stats
docker-compose top

# 查看容器网络
docker network ls
docker network inspect rag-ai_rag-ai-network

# ====== 容器日志分析 ======

# 查看实时日志
docker-compose logs -f
docker-compose logs -f api
docker-compose logs -f frontend
docker-compose logs -f qdrant
docker-compose logs -f redis

# 查看特定时间段日志
docker-compose logs --since="2024-01-01T10:00:00" api
docker-compose logs --until="2024-01-01T12:00:00" api

# 过滤错误日志
docker-compose logs api | grep -i error
docker-compose logs api | grep -i exception
docker-compose logs api | grep -i failed

# ====== 容器内部调试 ======

# 进入容器进行调试
docker-compose exec api bash
docker-compose exec frontend sh
docker-compose exec qdrant bash

# 在容器内执行命令
docker-compose exec api python -c "import sys; print(sys.path)"
docker-compose exec api pip list
docker-compose exec api ls -la /app

# 检查容器文件系统
docker-compose exec api df -h
docker-compose exec api free -h
docker-compose exec api ps aux
```

#### 1.1.2 系统资源监控

```bash
#!/bin/bash
# system_diagnostics.sh - 系统诊断脚本

echo "====== RAG-AI 系统诊断报告 ======"
echo "时间: $(date)"
echo ""

echo "====== 系统资源使用 ======"
echo "CPU 使用率:"
top -bn1 | grep "Cpu(s)" | awk '{print $2 $3 $4 $5 $6 $7 $8}'

echo ""
echo "内存使用情况:"
free -h

echo ""
echo "磁盘使用情况:"
df -h

echo ""
echo "网络连接:"
netstat -tlnp | grep -E ":(3000|8000|6333|6379)" || echo "未发现关键端口监听"

echo ""
echo "====== Docker 容器状态 ======"
if command -v docker-compose &> /dev/null; then
    docker-compose ps
else
    echo "Docker Compose 未安装或不可用"
fi

echo ""
echo "====== 进程检查 ======"
echo "Python 进程:"
ps aux | grep python | grep -v grep || echo "未发现 Python 进程"

echo ""
echo "Node.js 进程:"
ps aux | grep node | grep -v grep || echo "未发现 Node.js 进程"

echo ""
echo "====== 日志文件检查 ======"
if [ -d "./logs" ]; then
    echo "日志目录大小:"
    du -sh ./logs/*
    echo ""
    echo "最新错误日志:"
    find ./logs -name "*.log" -exec grep -l "ERROR\|CRITICAL\|Exception" {} \; | head -5
else
    echo "日志目录不存在"
fi

echo ""
echo "====== 网络连通性测试 ======"
echo "本地服务连通性:"
curl -s -o /dev/null -w "API Server: %{http_code}\n" http://localhost:8000/health || echo "API Server: 无法连接"
curl -s -o /dev/null -w "Frontend: %{http_code}\n" http://localhost:3000 || echo "Frontend: 无法连接"
curl -s -o /dev/null -w "Qdrant: %{http_code}\n" http://localhost:6333/health || echo "Qdrant: 无法连接"

echo ""
echo "====== 配置文件检查 ======"
if [ -f ".env" ]; then
    echo ".env 文件存在"
    echo "关键配置项:"
    grep -E "^(QDRANT_|REDIS_|STORAGE_|EMBEDDING_|LLM_)" .env | sed 's/=.*/=***/' || echo "未找到关键配置"
else
    echo ".env 文件不存在 - 这可能是问题所在"
fi

echo ""
echo "====== 诊断完成 ======"
```

### 1.2 应用级调试工具

#### 1.2.1 Python 调试工具

```python
# debug_tools.py - Python 调试工具集

import asyncio
import inspect
import logging
import time
import traceback
import gc
import psutil
import sys
from typing import Any, Dict, List, Callable
from functools import wraps
from contextlib import asynccontextmanager

class DebugManager:
    """调试管理器"""
    
    def __init__(self, enable_verbose: bool = False):
        self.enable_verbose = enable_verbose
        self.call_stack = []
        self.performance_data = {}
        self.error_history = []
        
    def debug_decorator(self, func_name: str = None):
        """调试装饰器"""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._debug_async_function(func, func_name, *args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._debug_sync_function(func, func_name, *args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    async def _debug_async_function(self, func: Callable, func_name: str, *args, **kwargs):
        """异步函数调试"""
        name = func_name or func.__name__
        start_time = time.time()
        
        # 记录调用栈
        self.call_stack.append({
            'function': name,
            'start_time': start_time,
            'args_count': len(args),
            'kwargs_count': len(kwargs)
        })
        
        if self.enable_verbose:
            print(f"🔍 [DEBUG] 调用函数: {name}")
            print(f"   参数: args={len(args)}, kwargs={list(kwargs.keys())}")
        
        try:
            result = await func(*args, **kwargs)
            
            duration = time.time() - start_time
            self.performance_data[name] = self.performance_data.get(name, [])
            self.performance_data[name].append(duration)
            
            if self.enable_verbose:
                print(f"✅ [DEBUG] 函数完成: {name} ({duration:.3f}s)")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_info = {
                'function': name,
                'error': str(e),
                'error_type': type(e).__name__,
                'duration': duration,
                'traceback': traceback.format_exc(),
                'timestamp': time.time()
            }
            
            self.error_history.append(error_info)
            
            if self.enable_verbose:
                print(f"❌ [DEBUG] 函数错误: {name} - {str(e)}")
            
            raise
        finally:
            self.call_stack.pop()
    
    def _debug_sync_function(self, func: Callable, func_name: str, *args, **kwargs):
        """同步函数调试（类似异步版本的实现）"""
        # 实现逻辑类似异步版本
        pass
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {}
        for func_name, durations in self.performance_data.items():
            summary[func_name] = {
                'call_count': len(durations),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'total_duration': sum(durations)
            }
        return summary
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        if not self.error_history:
            return {"total_errors": 0}
        
        error_by_type = {}
        error_by_function = {}
        
        for error in self.error_history:
            error_type = error['error_type']
            function = error['function']
            
            error_by_type[error_type] = error_by_type.get(error_type, 0) + 1
            error_by_function[function] = error_by_function.get(function, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'errors_by_type': error_by_type,
            'errors_by_function': error_by_function,
            'recent_errors': self.error_history[-5:]  # 最近 5 个错误
        }

# 全局调试管理器
debug_manager = DebugManager(enable_verbose=True)

class SystemDiagnostics:
    """系统诊断工具"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent
            },
            'process_count': len(psutil.pids())
        }
    
    @staticmethod
    def check_dependencies() -> Dict[str, Any]:
        """检查依赖项"""
        required_packages = [
            'fastapi', 'uvicorn', 'qdrant-client', 'redis', 
            'torch', 'transformers', 'sentence-transformers',
            'numpy', 'pandas', 'asyncio'
        ]
        
        results = {}
        for package in required_packages:
            try:
                __import__(package)
                results[package] = "✅ 已安装"
            except ImportError:
                results[package] = "❌ 未安装"
        
        return results
    
    @staticmethod
    def diagnose_imports() -> Dict[str, Any]:
        """诊断导入问题"""
        import_tests = {}
        
        # 测试核心模块导入
        test_imports = [
            ('src.retrieval.vector_database', 'VectorDatabaseManager'),
            ('src.generation.rag_generator', 'RAGSystem'),
            ('src.data_ingestion.multi_source_collector', 'MultiSourceCollector'),
            ('src.caching.multilayer_cache', 'MultiLayerCache'),
            ('src.citation.citation_manager', 'CitationManager')
        ]
        
        for module_name, class_name in test_imports:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
                import_tests[f"{module_name}.{class_name}"] = "✅ 成功"
            except ImportError as e:
                import_tests[f"{module_name}.{class_name}"] = f"❌ 导入错误: {str(e)}"
            except AttributeError as e:
                import_tests[f"{module_name}.{class_name}"] = f"❌ 属性错误: {str(e)}"
        
        return import_tests

# 使用示例
@debug_manager.debug_decorator("test_function")
async def test_function(param1: str, param2: int = 10):
    """测试函数"""
    await asyncio.sleep(0.1)  # 模拟异步操作
    return f"Result: {param1} - {param2}"

class MemoryProfiler:
    """内存分析器"""
    
    def __init__(self):
        self.snapshots = []
    
    def take_snapshot(self, label: str = None):
        """获取内存快照"""
        import tracemalloc
        
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        memory_info = {
            'label': label or f"snapshot_{len(self.snapshots)}",
            'timestamp': time.time(),
            'total_memory': sum(stat.size for stat in top_stats),
            'top_allocations': [
                {
                    'file': stat.traceback.format()[-1],
                    'size_mb': stat.size / 1024 / 1024,
                    'count': stat.count
                }
                for stat in top_stats[:10]
            ]
        }
        
        self.snapshots.append(memory_info)
        return memory_info
    
    def compare_snapshots(self, snapshot1_idx: int = -2, snapshot2_idx: int = -1):
        """比较内存快照"""
        if len(self.snapshots) < 2:
            return {"error": "需要至少两个快照进行比较"}
        
        snap1 = self.snapshots[snapshot1_idx]
        snap2 = self.snapshots[snapshot2_idx]
        
        memory_diff = snap2['total_memory'] - snap1['total_memory']
        
        return {
            'snapshot1': snap1['label'],
            'snapshot2': snap2['label'],
            'memory_difference_mb': memory_diff / 1024 / 1024,
            'time_difference': snap2['timestamp'] - snap1['timestamp']
        }

# 创建全局实例
memory_profiler = MemoryProfiler()
system_diagnostics = SystemDiagnostics()
```

#### 1.2.2 FastAPI 调试中间件

```python
# debug_middleware.py - FastAPI 调试中间件

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import time
import json
import uuid
from typing import Dict, Any
import logging

class DebugMiddleware(BaseHTTPMiddleware):
    """调试中间件"""
    
    def __init__(self, app, enable_request_logging: bool = True, 
                 enable_performance_tracking: bool = True):
        super().__init__(app)
        self.enable_request_logging = enable_request_logging
        self.enable_performance_tracking = enable_performance_tracking
        self.request_data = {}
        
    async def dispatch(self, request: Request, call_next):
        # 生成请求 ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 记录请求开始
        start_time = time.time()
        
        if self.enable_request_logging:
            await self._log_request_start(request, request_id)
        
        # 执行请求
        try:
            response = await call_next(request)
            
            # 记录成功响应
            duration = time.time() - start_time
            if self.enable_performance_tracking:
                await self._log_request_success(request, response, request_id, duration)
            
            # 添加调试头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{duration:.3f}"
            
            return response
            
        except Exception as e:
            # 记录异常
            duration = time.time() - start_time
            await self._log_request_error(request, e, request_id, duration)
            
            # 重新抛出异常
            raise
    
    async def _log_request_start(self, request: Request, request_id: str):
        """记录请求开始"""
        # 读取请求体（小心处理，避免影响后续处理）
        body = b""
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                # 重新设置请求体以供后续使用
                async def receive():
                    return {"type": "http.request", "body": body}
                request._receive = receive
            except:
                pass
        
        log_data = {
            "event": "request_start",
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "query_params": dict(request.query_params),
            "body_size": len(body),
            "client": request.client.host if request.client else None
        }
        
        logging.info(json.dumps(log_data))
    
    async def _log_request_success(self, request: Request, response: Response, 
                                 request_id: str, duration: float):
        """记录成功响应"""
        log_data = {
            "event": "request_success",
            "request_id": request_id,
            "status_code": response.status_code,
            "duration": duration,
            "response_headers": dict(response.headers)
        }
        
        logging.info(json.dumps(log_data))
    
    async def _log_request_error(self, request: Request, error: Exception, 
                                request_id: str, duration: float):
        """记录错误响应"""
        log_data = {
            "event": "request_error",
            "request_id": request_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "duration": duration,
            "traceback": traceback.format_exc()
        }
        
        logging.error(json.dumps(log_data))

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """性能监控中间件"""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_stats = {}
        self.slow_request_threshold = 5.0  # 5 秒
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # 获取内存使用（请求前）
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        try:
            response = await call_next(request)
            
            # 计算性能指标
            duration = time.time() - start_time
            memory_after = process.memory_info().rss
            memory_used = memory_after - memory_before
            
            # 记录统计信息
            endpoint = f"{request.method} {request.url.path}"
            if endpoint not in self.request_stats:
                self.request_stats[endpoint] = {
                    'count': 0,
                    'total_time': 0,
                    'max_time': 0,
                    'min_time': float('inf'),
                    'total_memory': 0,
                    'errors': 0
                }
            
            stats = self.request_stats[endpoint]
            stats['count'] += 1
            stats['total_time'] += duration
            stats['max_time'] = max(stats['max_time'], duration)
            stats['min_time'] = min(stats['min_time'], duration)
            stats['total_memory'] += memory_used
            
            # 记录慢请求
            if duration > self.slow_request_threshold:
                logging.warning(f"慢请求检测: {endpoint} - {duration:.3f}s")
            
            # 添加性能头
            response.headers["X-Memory-Used"] = str(memory_used)
            response.headers["X-Endpoint-Avg-Time"] = f"{stats['total_time'] / stats['count']:.3f}"
            
            return response
            
        except Exception as e:
            # 记录错误统计
            endpoint = f"{request.method} {request.url.path}"
            if endpoint in self.request_stats:
                self.request_stats[endpoint]['errors'] += 1
            
            raise
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {}
        for endpoint, stats in self.request_stats.items():
            if stats['count'] > 0:
                report[endpoint] = {
                    'total_requests': stats['count'],
                    'avg_response_time': stats['total_time'] / stats['count'],
                    'max_response_time': stats['max_time'],
                    'min_response_time': stats['min_time'],
                    'avg_memory_usage': stats['total_memory'] / stats['count'],
                    'error_rate': stats['errors'] / stats['count'] * 100,
                    'requests_per_second': stats['count'] / stats['total_time'] if stats['total_time'] > 0 else 0
                }
        
        return report

# 创建中间件实例
debug_middleware = DebugMiddleware
performance_middleware = PerformanceMonitoringMiddleware
```

## 2. 常见问题分类诊断

### 2.1 启动问题

#### 2.1.1 容器启动失败

```bash
# 问题诊断脚本
#!/bin/bash
# diagnose_startup.sh

echo "🔍 诊断容器启动问题..."

# 检查 Docker 环境
echo "====== Docker 环境检查 ======"
docker version || echo "❌ Docker 未安装或不可用"
docker-compose version || echo "❌ Docker Compose 未安装或不可用"

# 检查端口占用
echo "====== 端口占用检查 ======"
for port in 3000 8000 6333 6379 9090 3001; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "⚠️  端口 $port 已被占用:"
        lsof -i :$port
    else
        echo "✅ 端口 $port 可用"
    fi
done

# 检查配置文件
echo "====== 配置文件检查 ======"
if [ -f "docker-compose.yml" ]; then
    echo "✅ docker-compose.yml 存在"
    # 验证配置文件语法
    docker-compose config > /dev/null 2>&1 && echo "✅ 配置文件语法正确" || echo "❌ 配置文件语法错误"
else
    echo "❌ docker-compose.yml 不存在"
fi

if [ -f ".env" ]; then
    echo "✅ .env 文件存在"
    echo "关键配置项检查:"
    grep -E "^[A-Z_]+=" .env | head -10
else
    echo "⚠️  .env 文件不存在，使用默认配置"
fi

# 检查磁盘空间
echo "====== 磁盘空间检查 ======"
df -h | grep -E "/$|/var" | while read line; do
    usage=$(echo $line | awk '{print $5}' | sed 's/%//')
    if [ $usage -gt 90 ]; then
        echo "❌ 磁盘空间不足: $line"
    else
        echo "✅ 磁盘空间充足: $line"
    fi
done

# 检查内存
echo "====== 内存检查 ======"
total_mem=$(free -m | awk 'NR==2{print $2}')
available_mem=$(free -m | awk 'NR==2{print $7}')
if [ $available_mem -lt 2048 ]; then
    echo "⚠️  可用内存不足 ($available_mem MB)，建议至少 2GB"
else
    echo "✅ 内存充足 ($available_mem MB 可用)"
fi

# 检查 Docker 权限
echo "====== Docker 权限检查 ======"
if docker ps > /dev/null 2>&1; then
    echo "✅ Docker 权限正常"
else
    echo "❌ Docker 权限问题，可能需要 sudo 或将用户添加到 docker 组"
    echo "解决方案: sudo usermod -aG docker $USER"
fi

echo ""
echo "🔧 常见解决方案:"
echo "1. 端口冲突: 修改 docker-compose.yml 中的端口映射"
echo "2. 权限问题: sudo usermod -aG docker \$USER && newgrp docker"
echo "3. 磁盘空间: docker system prune -a"
echo "4. 内存不足: 关闭其他应用或增加系统内存"
echo "5. 配置错误: 检查 .env 文件和 docker-compose.yml"
```

#### 2.1.2 Python 应用启动失败

```python
# startup_diagnostics.py - Python 应用启动诊断

import sys
import os
import subprocess
import importlib
from pathlib import Path

class StartupDiagnostics:
    """启动诊断工具"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.success = []
    
    def run_full_diagnostics(self):
        """运行完整诊断"""
        print("🔍 Python 应用启动诊断...")
        
        self.check_python_version()
        self.check_working_directory()
        self.check_environment_variables()
        self.check_dependencies()
        self.check_file_permissions()
        self.check_model_access()
        self.check_database_connections()
        
        self.print_summary()
    
    def check_python_version(self):
        """检查 Python 版本"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 10:
            self.success.append(f"✅ Python 版本合适: {version.major}.{version.minor}.{version.micro}")
        elif version.major == 3 and version.minor >= 8:
            self.warnings.append(f"⚠️  Python 版本较旧: {version.major}.{version.minor}.{version.micro}，建议使用 3.10+")
        else:
            self.issues.append(f"❌ Python 版本不支持: {version.major}.{version.minor}.{version.micro}，需要 3.8+")
    
    def check_working_directory(self):
        """检查工作目录"""
        cwd = Path.cwd()
        required_files = [
            'requirements.txt',
            'configs/config.py',
            'src/__init__.py',
            'api/main.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (cwd / file_path).exists():
                missing_files.append(file_path)
        
        if not missing_files:
            self.success.append("✅ 工作目录结构正确")
        else:
            self.issues.append(f"❌ 缺少必要文件: {', '.join(missing_files)}")
    
    def check_environment_variables(self):
        """检查环境变量"""
        required_env_vars = [
            'STORAGE_ROOT',
            'EMBEDDING_MODEL', 
            'LLM_MODEL',
            'QDRANT_HOST',
            'QDRANT_PORT'
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if not missing_vars:
            self.success.append("✅ 环境变量配置完整")
        else:
            self.warnings.append(f"⚠️  缺少环境变量: {', '.join(missing_vars)}")
    
    def check_dependencies(self):
        """检查依赖项"""
        critical_packages = [
            'fastapi',
            'uvicorn', 
            'qdrant-client',
            'redis',
            'torch',
            'transformers',
            'sentence-transformers'
        ]
        
        missing_packages = []
        for package in critical_packages:
            try:
                importlib.import_module(package.replace('-', '_'))
                self.success.append(f"✅ {package} 已安装")
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.issues.append(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
            self.issues.append("解决方案: pip install -r requirements.txt")
    
    def check_file_permissions(self):
        """检查文件权限"""
        critical_dirs = ['data', 'logs', 'configs']
        permission_issues = []
        
        for dir_name in critical_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                if not os.access(dir_path, os.R_OK | os.W_OK):
                    permission_issues.append(dir_name)
            else:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.success.append(f"✅ 创建目录: {dir_name}")
                except PermissionError:
                    permission_issues.append(dir_name)
        
        if permission_issues:
            self.issues.append(f"❌ 权限问题: {', '.join(permission_issues)}")
            self.issues.append("解决方案: sudo chown -R $USER:$USER .")
        else:
            self.success.append("✅ 文件权限正常")
    
    def check_model_access(self):
        """检查模型访问"""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # 测试 HuggingFace 连接
            model_info = api.model_info("sentence-transformers/all-MiniLM-L6-v2")
            self.success.append("✅ HuggingFace Hub 连接正常")
            
        except Exception as e:
            self.warnings.append(f"⚠️  HuggingFace Hub 连接问题: {str(e)}")
            self.warnings.append("检查网络连接和 HUGGING_FACE_TOKEN")
    
    def check_database_connections(self):
        """检查数据库连接"""
        # 检查 Qdrant
        try:
            import requests
            qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
            qdrant_port = os.getenv('QDRANT_PORT', '6333')
            
            response = requests.get(f"http://{qdrant_host}:{qdrant_port}/health", timeout=5)
            if response.status_code == 200:
                self.success.append("✅ Qdrant 连接正常")
            else:
                self.issues.append(f"❌ Qdrant 响应异常: {response.status_code}")
        except Exception as e:
            self.issues.append(f"❌ Qdrant 连接失败: {str(e)}")
            self.issues.append("解决方案: 启动 Qdrant 服务或检查配置")
        
        # 检查 Redis
        try:
            import redis
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', '6379'))
            
            r = redis.Redis(host=redis_host, port=redis_port, socket_timeout=5)
            r.ping()
            self.success.append("✅ Redis 连接正常")
        except Exception as e:
            self.warnings.append(f"⚠️  Redis 连接失败: {str(e)}")
            self.warnings.append("Redis 是可选组件，不影响基本功能")
    
    def print_summary(self):
        """打印诊断摘要"""
        print("\n" + "="*50)
        print("📊 诊断摘要")
        print("="*50)
        
        if self.success:
            print(f"\n✅ 成功项目 ({len(self.success)}):")
            for item in self.success:
                print(f"   {item}")
        
        if self.warnings:
            print(f"\n⚠️  警告项目 ({len(self.warnings)}):")
            for item in self.warnings:
                print(f"   {item}")
        
        if self.issues:
            print(f"\n❌ 问题项目 ({len(self.issues)}):")
            for item in self.issues:
                print(f"   {item}")
            print("\n🔧 需要解决上述问题才能正常启动")
        else:
            print("\n🎉 所有检查通过，应用可以正常启动！")

if __name__ == "__main__":
    diagnostics = StartupDiagnostics()
    diagnostics.run_full_diagnostics()
```

### 2.2 API 响应问题

#### 2.2.1 API 超时诊断

```python
# api_timeout_diagnostics.py

import asyncio
import aiohttp
import time
from typing import Dict, List, Any

class APITimeoutDiagnostics:
    """API 超时诊断工具"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.timeout_threshold = 30.0  # 30 秒超时阈值
        
    async def diagnose_api_performance(self) -> Dict[str, Any]:
        """诊断 API 性能"""
        diagnostics = {
            "timestamp": time.time(),
            "base_url": self.base_url,
            "tests": {}
        }
        
        # 测试用例
        test_cases = [
            {
                "name": "health_check",
                "method": "GET",
                "endpoint": "/health",
                "expected_time": 1.0
            },
            {
                "name": "simple_ask", 
                "method": "POST",
                "endpoint": "/ask",
                "data": {
                    "query": "什么是人工智能？",
                    "max_results": 3
                },
                "expected_time": 10.0
            },
            {
                "name": "complex_ask",
                "method": "POST", 
                "endpoint": "/ask",
                "data": {
                    "query": "详细解释Transformer架构的注意力机制原理，包括多头注意力、位置编码和残差连接的作用",
                    "max_results": 10,
                    "rag_mode": "ultimate"
                },
                "expected_time": 30.0
            },
            {
                "name": "search_test",
                "method": "POST",
                "endpoint": "/search", 
                "data": {
                    "query": "机器学习",
                    "limit": 20
                },
                "expected_time": 5.0
            }
        ]
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            for test_case in test_cases:
                test_result = await self._run_test_case(session, test_case)
                diagnostics["tests"][test_case["name"]] = test_result
        
        # 分析结果
        diagnostics["analysis"] = self._analyze_results(diagnostics["tests"])
        
        return diagnostics
    
    async def _run_test_case(self, session: aiohttp.ClientSession, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个测试用例"""
        start_time = time.time()
        
        try:
            url = f"{self.base_url}{test_case['endpoint']}"
            
            if test_case["method"] == "GET":
                async with session.get(url) as response:
                    duration = time.time() - start_time
                    content = await response.text()
                    
                    return {
                        "status": "success",
                        "status_code": response.status,
                        "duration": duration,
                        "expected_time": test_case["expected_time"],
                        "response_size": len(content),
                        "is_slow": duration > test_case["expected_time"]
                    }
            
            elif test_case["method"] == "POST":
                async with session.post(url, json=test_case["data"]) as response:
                    duration = time.time() - start_time
                    content = await response.text()
                    
                    return {
                        "status": "success",
                        "status_code": response.status,
                        "duration": duration,
                        "expected_time": test_case["expected_time"],
                        "response_size": len(content),
                        "is_slow": duration > test_case["expected_time"]
                    }
                    
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            return {
                "status": "timeout",
                "duration": duration,
                "expected_time": test_case["expected_time"],
                "error": "Request timed out"
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                "status": "error",
                "duration": duration,
                "expected_time": test_case["expected_time"],
                "error": str(e)
            }
    
    def _analyze_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析测试结果"""
        analysis = {
            "total_tests": len(test_results),
            "successful_tests": 0,
            "failed_tests": 0,
            "slow_tests": 0,
            "timeout_tests": 0,
            "recommendations": []
        }
        
        for test_name, result in test_results.items():
            if result["status"] == "success":
                analysis["successful_tests"] += 1
                if result["is_slow"]:
                    analysis["slow_tests"] += 1
            elif result["status"] == "timeout":
                analysis["timeout_tests"] += 1
                analysis["failed_tests"] += 1
            else:
                analysis["failed_tests"] += 1
        
        # 生成建议
        if analysis["timeout_tests"] > 0:
            analysis["recommendations"].append("存在超时请求，检查网络连接和服务器负载")
        
        if analysis["slow_tests"] > 0:
            analysis["recommendations"].append("存在慢请求，考虑优化查询或增加缓存")
        
        if analysis["failed_tests"] > analysis["successful_tests"]:
            analysis["recommendations"].append("大部分请求失败，检查API服务状态")
        
        return analysis

    async def diagnose_streaming_performance(self) -> Dict[str, Any]:
        """诊断流式响应性能"""
        url = f"{self.base_url}/ask/stream"
        
        test_data = {
            "query": "解释深度学习的基本概念",
            "stream_response": True
        }
        
        diagnostics = {
            "test_type": "streaming",
            "start_time": time.time(),
            "chunks_received": 0,
            "total_content_length": 0,
            "first_chunk_time": None,
            "last_chunk_time": None,
            "chunk_intervals": [],
            "errors": []
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=test_data) as response:
                    if response.status != 200:
                        diagnostics["errors"].append(f"HTTP {response.status}: {await response.text()}")
                        return diagnostics
                    
                    last_chunk_time = time.time()
                    
                    async for line in response.content:
                        current_time = time.time()
                        
                        if diagnostics["first_chunk_time"] is None:
                            diagnostics["first_chunk_time"] = current_time - diagnostics["start_time"]
                        
                        # 计算块间隔
                        interval = current_time - last_chunk_time
                        diagnostics["chunk_intervals"].append(interval)
                        
                        diagnostics["chunks_received"] += 1
                        diagnostics["total_content_length"] += len(line)
                        diagnostics["last_chunk_time"] = current_time - diagnostics["start_time"]
                        
                        last_chunk_time = current_time
                        
        except Exception as e:
            diagnostics["errors"].append(str(e))
        
        # 计算统计信息
        if diagnostics["chunk_intervals"]:
            diagnostics["avg_chunk_interval"] = sum(diagnostics["chunk_intervals"]) / len(diagnostics["chunk_intervals"])
            diagnostics["max_chunk_interval"] = max(diagnostics["chunk_intervals"])
            diagnostics["min_chunk_interval"] = min(diagnostics["chunk_intervals"])
        
        diagnostics["total_duration"] = diagnostics.get("last_chunk_time", 0)
        
        return diagnostics

# 使用示例
async def run_api_diagnostics():
    """运行 API 诊断"""
    diagnostics = APITimeoutDiagnostics()
    
    print("🔍 开始 API 性能诊断...")
    
    # 常规 API 测试
    api_results = await diagnostics.diagnose_api_performance()
    print("\n📊 API 性能测试结果:")
    for test_name, result in api_results["tests"].items():
        status_emoji = "✅" if result["status"] == "success" else "❌"
        slow_emoji = "🐌" if result.get("is_slow", False) else "⚡"
        print(f"   {status_emoji} {slow_emoji} {test_name}: {result['duration']:.2f}s (期望: {result['expected_time']}s)")
    
    print(f"\n📈 分析结果:")
    analysis = api_results["analysis"]
    print(f"   成功: {analysis['successful_tests']}/{analysis['total_tests']}")
    print(f"   超时: {analysis['timeout_tests']}")
    print(f"   慢请求: {analysis['slow_tests']}")
    
    if analysis["recommendations"]:
        print(f"\n💡 建议:")
        for rec in analysis["recommendations"]:
            print(f"   • {rec}")
    
    # 流式响应测试
    print("\n🔍 测试流式响应...")
    streaming_results = await diagnostics.diagnose_streaming_performance()
    
    if streaming_results["errors"]:
        print("❌ 流式响应测试失败:")
        for error in streaming_results["errors"]:
            print(f"   {error}")
    else:
        print(f"✅ 流式响应测试成功:")
        print(f"   首个数据块延迟: {streaming_results.get('first_chunk_time', 0):.2f}s")
        print(f"   总接收数据块: {streaming_results['chunks_received']}")
        print(f"   平均块间隔: {streaming_results.get('avg_chunk_interval', 0):.3f}s")
        print(f"   总响应时间: {streaming_results.get('total_duration', 0):.2f}s")

if __name__ == "__main__":
    asyncio.run(run_api_diagnostics())
```

#### 2.2.2 API 错误分析

```python
# api_error_analyzer.py

import re
import json
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class APIErrorAnalyzer:
    """API 错误分析器"""
    
    def __init__(self, log_file_path: str = None):
        self.log_file_path = log_file_path
        self.error_patterns = {
            'connection_timeout': [
                r'connection.*timeout',
                r'timeout.*connection',
                r'ConnectTimeoutError',
                r'TimeoutError'
            ],
            'database_error': [
                r'qdrant.*error',
                r'redis.*error', 
                r'connection.*refused.*6333',
                r'connection.*refused.*6379'
            ],
            'model_error': [
                r'model.*not.*found',
                r'CUDA.*out.*of.*memory',
                r'RuntimeError.*tensor',
                r'transformers.*error'
            ],
            'validation_error': [
                r'ValidationError',
                r'pydantic.*error',
                r'invalid.*input',
                r'422.*Unprocessable'
            ],
            'permission_error': [
                r'PermissionError',
                r'403.*Forbidden',
                r'unauthorized',
                r'access.*denied'
            ],
            'internal_server_error': [
                r'500.*Internal.*Server.*Error',
                r'HTTPException.*500',
                r'Internal.*Server.*Error'
            ]
        }
    
    def analyze_log_file(self, hours_back: int = 24) -> Dict[str, Any]:
        """分析日志文件"""
        if not self.log_file_path:
            return {"error": "No log file specified"}
        
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                logs = f.readlines()
        except FileNotFoundError:
            return {"error": f"Log file not found: {self.log_file_path}"}
        
        # 过滤时间范围
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        filtered_logs = self._filter_logs_by_time(logs, cutoff_time)
        
        # 分析错误
        error_analysis = self._analyze_errors(filtered_logs)
        
        # 分析趋势
        trend_analysis = self._analyze_error_trends(filtered_logs)
        
        return {
            "analysis_period_hours": hours_back,
            "total_log_lines": len(filtered_logs),
            "error_analysis": error_analysis,
            "trend_analysis": trend_analysis,
            "recommendations": self._generate_recommendations(error_analysis)
        }
    
    def _filter_logs_by_time(self, logs: List[str], cutoff_time: datetime) -> List[str]:
        """按时间过滤日志"""
        filtered_logs = []
        
        for log_line in logs:
            # 尝试从日志行中提取时间戳
            timestamp = self._extract_timestamp(log_line)
            if timestamp and timestamp >= cutoff_time:
                filtered_logs.append(log_line)
            elif not timestamp:
                # 如果无法提取时间戳，保留该行
                filtered_logs.append(log_line)
        
        return filtered_logs
    
    def _extract_timestamp(self, log_line: str) -> Optional[datetime]:
        """从日志行中提取时间戳"""
        # 常见的时间戳格式
        timestamp_patterns = [
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',  # ISO format
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',   # Standard format
            r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})',   # US format
        ]
        
        for pattern in timestamp_patterns:
            match = re.search(pattern, log_line)
            if match:
                timestamp_str = match.group(1)
                try:
                    if 'T' in timestamp_str:
                        return datetime.fromisoformat(timestamp_str)
                    else:
                        return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue
        
        return None
    
    def _analyze_errors(self, logs: List[str]) -> Dict[str, Any]:
        """分析错误"""
        error_counts = defaultdict(int)
        error_examples = defaultdict(list)
        total_errors = 0
        
        for log_line in logs:
            # 检查是否为错误日志
            if any(keyword in log_line.lower() for keyword in ['error', 'exception', 'failed', 'critical']):
                total_errors += 1
                
                # 分类错误
                error_classified = False
                for error_type, patterns in self.error_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, log_line, re.IGNORECASE):
                            error_counts[error_type] += 1
                            if len(error_examples[error_type]) < 3:  # 保存前3个例子
                                error_examples[error_type].append(log_line.strip())
                            error_classified = True
                            break
                    if error_classified:
                        break
                
                # 未分类的错误
                if not error_classified:
                    error_counts['unclassified'] += 1
                    if len(error_examples['unclassified']) < 3:
                        error_examples['unclassified'].append(log_line.strip())
        
        return {
            "total_errors": total_errors,
            "error_counts": dict(error_counts),
            "error_examples": dict(error_examples),
            "error_rate": total_errors / len(logs) * 100 if logs else 0
        }
    
    def _analyze_error_trends(self, logs: List[str]) -> Dict[str, Any]:
        """分析错误趋势"""
        hourly_errors = defaultdict(int)
        error_keywords = Counter()
        
        for log_line in logs:
            if any(keyword in log_line.lower() for keyword in ['error', 'exception', 'failed']):
                # 提取小时
                timestamp = self._extract_timestamp(log_line)
                if timestamp:
                    hour_key = timestamp.strftime('%Y-%m-%d %H:00')
                    hourly_errors[hour_key] += 1
                
                # 提取关键词
                words = re.findall(r'\b\w+\b', log_line.lower())
                for word in words:
                    if len(word) > 3 and word not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use']:
                        error_keywords[word] += 1
        
        return {
            "hourly_error_distribution": dict(hourly_errors),
            "peak_error_hour": max(hourly_errors.items(), key=lambda x: x[1])[0] if hourly_errors else None,
            "top_error_keywords": error_keywords.most_common(10)
        }
    
    def _generate_recommendations(self, error_analysis: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        error_counts = error_analysis.get("error_counts", {})
        
        if error_counts.get("connection_timeout", 0) > 5:
            recommendations.append("检测到多个连接超时错误，建议检查网络连接和超时配置")
        
        if error_counts.get("database_error", 0) > 3:
            recommendations.append("数据库连接错误较多，建议检查 Qdrant 和 Redis 服务状态")
        
        if error_counts.get("model_error", 0) > 2:
            recommendations.append("模型相关错误，建议检查 GPU 内存和模型文件")
        
        if error_counts.get("validation_error", 0) > 10:
            recommendations.append("输入验证错误较多，建议加强前端输入验证")
        
        if error_analysis.get("error_rate", 0) > 10:
            recommendations.append("错误率过高 (>10%)，建议全面检查系统状态")
        
        if not recommendations:
            recommendations.append("系统错误率在可接受范围内")
        
        return recommendations

    def analyze_real_time_errors(self, error_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析实时错误数据"""
        if not error_data:
            return {"message": "No error data to analyze"}
        
        # 错误类型统计
        error_types = Counter()
        endpoint_errors = Counter()
        user_errors = Counter()
        
        for error in error_data:
            error_types[error.get('error_type', 'unknown')] += 1
            endpoint_errors[error.get('endpoint', 'unknown')] += 1
            if error.get('user_id'):
                user_errors[error.get('user_id')] += 1
        
        # 计算错误模式
        recent_errors = sorted(error_data, key=lambda x: x.get('timestamp', 0), reverse=True)[:10]
        
        return {
            "total_errors": len(error_data),
            "error_types": dict(error_types.most_common(10)),
            "problematic_endpoints": dict(endpoint_errors.most_common(5)),
            "users_with_most_errors": dict(user_errors.most_common(5)),
            "recent_errors": recent_errors,
            "error_spike_detected": len(error_data) > 50,  # 简单的错误激增检测
        }

# 使用示例
def analyze_api_errors():
    """分析 API 错误"""
    analyzer = APIErrorAnalyzer("/app/logs/api.log")
    
    print("🔍 分析 API 错误日志...")
    
    # 分析最近24小时的日志
    results = analyzer.analyze_log_file(hours_back=24)
    
    if "error" in results:
        print(f"❌ 分析失败: {results['error']}")
        return
    
    print(f"\n📊 错误分析结果 (最近 {results['analysis_period_hours']} 小时):")
    print(f"   总日志行数: {results['total_log_lines']}")
    print(f"   错误总数: {results['error_analysis']['total_errors']}")
    print(f"   错误率: {results['error_analysis']['error_rate']:.2f}%")
    
    print(f"\n🏷️  错误分类:")
    for error_type, count in results['error_analysis']['error_counts'].items():
        print(f"   {error_type}: {count}")
    
    print(f"\n📈 错误趋势:")
    trend = results['trend_analysis']
    if trend['peak_error_hour']:
        print(f"   错误高峰时段: {trend['peak_error_hour']}")
    
    print(f"   高频错误关键词:")
    for keyword, count in trend['top_error_keywords'][:5]:
        print(f"     {keyword}: {count}")
    
    print(f"\n💡 建议:")
    for recommendation in results['recommendations']:
        print(f"   • {recommendation}")

if __name__ == "__main__":
    analyze_api_errors()
```

## 3. 性能问题排查

### 3.1 内存泄漏诊断

```python
# memory_leak_detector.py

import gc
import tracemalloc
import time
import psutil
import threading
from collections import defaultdict
from typing import Dict, List, Any, Optional
import weakref

class MemoryLeakDetector:
    """内存泄漏检测器"""
    
    def __init__(self, sampling_interval: int = 60):
        self.sampling_interval = sampling_interval
        self.memory_samples = []
        self.object_tracking = defaultdict(list)
        self.monitoring = False
        self.monitor_thread = None
        
        # 启用内存追踪
        tracemalloc.start()
    
    def start_monitoring(self):
        """开始内存监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 内存监控已启动")
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("⏹️  内存监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring:
            sample = self._take_memory_sample()
            self.memory_samples.append(sample)
            
            # 只保留最近100个样本
            if len(self.memory_samples) > 100:
                self.memory_samples.pop(0)
            
            time.sleep(self.sampling_interval)
    
    def _take_memory_sample(self) -> Dict[str, Any]:
        """获取内存样本"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Python 对象统计
        gc.collect()
        object_counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        # 内存追踪信息
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        return {
            'timestamp': time.time(),
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'object_counts': object_counts,
            'top_memory_allocations': [
                {
                    'file': stat.traceback.format()[-1] if stat.traceback else 'unknown',
                    'size_mb': stat.size / 1024 / 1024,
                    'count': stat.count
                }
                for stat in top_stats[:5]
            ]
        }
    
    def detect_memory_leaks(self) -> Dict[str, Any]:
        """检测内存泄漏"""
        if len(self.memory_samples) < 3:
            return {"error": "需要至少3个内存样本进行分析"}
        
        analysis = {
            "monitoring_duration_minutes": len(self.memory_samples) * self.sampling_interval / 60,
            "sample_count": len(self.memory_samples),
            "memory_trend": self._analyze_memory_trend(),
            "object_growth": self._analyze_object_growth(),
            "potential_leaks": [],
            "recommendations": []
        }
        
        # 检测内存泄漏模式
        memory_trend = analysis["memory_trend"]
        if memory_trend["growth_rate_mb_per_hour"] > 50:  # 每小时增长超过50MB
            analysis["potential_leaks"].append({
                "type": "continuous_memory_growth",
                "severity": "high",
                "description": f"内存持续增长，增长率: {memory_trend['growth_rate_mb_per_hour']:.2f} MB/小时"
            })
        
        # 检测对象泄漏
        object_growth = analysis["object_growth"]
        for obj_type, growth_rate in object_growth["high_growth_objects"].items():
            if growth_rate > 100:  # 每小时增长超过100个对象
                analysis["potential_leaks"].append({
                    "type": "object_leak",
                    "severity": "medium",
                    "object_type": obj_type,
                    "growth_rate": growth_rate,
                    "description": f"{obj_type} 对象数量快速增长: {growth_rate:.0f} 个/小时"
                })
        
        # 生成建议
        analysis["recommendations"] = self._generate_memory_recommendations(analysis)
        
        return analysis
    
    def _analyze_memory_trend(self) -> Dict[str, Any]:
        """分析内存趋势"""
        if len(self.memory_samples) < 2:
            return {}
        
        memory_values = [sample['rss_mb'] for sample in self.memory_samples]
        time_values = [sample['timestamp'] for sample in self.memory_samples]
        
        # 简单线性回归计算增长率
        n = len(memory_values)
        sum_x = sum(range(n))
        sum_y = sum(memory_values)
        sum_xy = sum(i * memory_values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # 转换为每小时增长率
        time_interval_hours = (time_values[-1] - time_values[0]) / 3600
        growth_rate_mb_per_hour = slope * (3600 / self.sampling_interval) if time_interval_hours > 0 else 0
        
        return {
            "initial_memory_mb": memory_values[0],
            "current_memory_mb": memory_values[-1],
            "peak_memory_mb": max(memory_values),
            "growth_rate_mb_per_hour": growth_rate_mb_per_hour,
            "total_growth_mb": memory_values[-1] - memory_values[0],
            "is_growing": growth_rate_mb_per_hour > 1  # 每小时增长超过1MB认为在增长
        }
    
    def _analyze_object_growth(self) -> Dict[str, Any]:
        """分析对象增长"""
        if len(self.memory_samples) < 2:
            return {}
        
        first_sample = self.memory_samples[0]
        last_sample = self.memory_samples[-1]
        
        time_diff_hours = (last_sample['timestamp'] - first_sample['timestamp']) / 3600
        
        object_growth_rates = {}
        high_growth_objects = {}
        
        first_counts = first_sample['object_counts']
        last_counts = last_sample['object_counts']
        
        for obj_type in set(list(first_counts.keys()) + list(last_counts.keys())):
            first_count = first_counts.get(obj_type, 0)
            last_count = last_counts.get(obj_type, 0)
            
            if time_diff_hours > 0:
                growth_rate = (last_count - first_count) / time_diff_hours
                object_growth_rates[obj_type] = growth_rate
                
                if growth_rate > 10:  # 每小时增长超过10个对象
                    high_growth_objects[obj_type] = growth_rate
        
        return {
            "object_growth_rates": object_growth_rates,
            "high_growth_objects": high_growth_objects,
            "total_objects_first": sum(first_counts.values()),
            "total_objects_last": sum(last_counts.values())
        }
    
    def _generate_memory_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """生成内存建议"""
        recommendations = []
        
        if analysis.get("memory_trend", {}).get("is_growing", False):
            recommendations.append("检测到内存持续增长，建议检查是否存在未释放的资源")
        
        if analysis.get("potential_leaks"):
            recommendations.append("发现潜在内存泄漏，建议详细检查相关代码")
        
        high_growth = analysis.get("object_growth", {}).get("high_growth_objects", {})
        if high_growth:
            top_growth = max(high_growth.items(), key=lambda x: x[1])
            recommendations.append(f"注意 {top_growth[0]} 对象的快速增长，可能存在泄漏")
        
        # 内存使用建议
        current_memory = analysis.get("memory_trend", {}).get("current_memory_mb", 0)
        if current_memory > 2048:  # 超过2GB
            recommendations.append("内存使用量较高，考虑优化内存使用或增加系统内存")
        
        if not recommendations:
            recommendations.append("内存使用情况正常")
        
        return recommendations
    
    def force_garbage_collection(self) -> Dict[str, Any]:
        """强制垃圾回收"""
        before_objects = len(gc.get_objects())
        before_memory = psutil.Process().memory_info().rss
        
        # 执行垃圾回收
        collected = gc.collect()
        
        after_objects = len(gc.get_objects())
        after_memory = psutil.Process().memory_info().rss
        
        return {
            "objects_before": before_objects,
            "objects_after": after_objects,
            "objects_collected": collected,
            "objects_freed": before_objects - after_objects,
            "memory_before_mb": before_memory / 1024 / 1024,
            "memory_after_mb": after_memory / 1024 / 1024,
            "memory_freed_mb": (before_memory - after_memory) / 1024 / 1024
        }
    
    def get_detailed_memory_report(self) -> Dict[str, Any]:
        """获取详细内存报告"""
        if not self.memory_samples:
            return {"error": "No memory samples available"}
        
        current_sample = self.memory_samples[-1]
        
        # 获取大对象
        large_objects = []
        for obj in gc.get_objects():
            try:
                size = obj.__sizeof__()
                if size > 1024 * 1024:  # 大于1MB的对象
                    large_objects.append({
                        "type": type(obj).__name__,
                        "size_mb": size / 1024 / 1024,
                        "id": id(obj)
                    })
            except:
                pass
        
        large_objects.sort(key=lambda x: x["size_mb"], reverse=True)
        
        return {
            "current_memory_mb": current_sample["rss_mb"],
            "memory_percent": current_sample["percent"],
            "total_objects": sum(current_sample["object_counts"].values()),
            "top_object_types": sorted(
                current_sample["object_counts"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "large_objects": large_objects[:10],
            "gc_stats": {
                "collections": gc.get_stats(),
                "garbage_count": len(gc.garbage),
                "thresholds": gc.get_threshold()
            }
        }

# 全局内存检测器实例
memory_detector = MemoryLeakDetector()

# 使用示例和工具函数
def start_memory_monitoring():
    """启动内存监控"""
    memory_detector.start_monitoring()

def check_memory_leaks():
    """检查内存泄漏"""
    print("🔍 分析内存泄漏...")
    
    results = memory_detector.detect_memory_leaks()
    
    if "error" in results:
        print(f"❌ 分析失败: {results['error']}")
        return
    
    print(f"📊 监控时长: {results['monitoring_duration_minutes']:.1f} 分钟")
    print(f"📊 样本数量: {results['sample_count']}")
    
    memory_trend = results["memory_trend"]
    print(f"\n📈 内存趋势:")
    print(f"   当前内存: {memory_trend.get('current_memory_mb', 0):.1f} MB")
    print(f"   增长率: {memory_trend.get('growth_rate_mb_per_hour', 0):.2f} MB/小时")
    print(f"   总增长: {memory_trend.get('total_growth_mb', 0):.1f} MB")
    
    if results["potential_leaks"]:
        print(f"\n⚠️  发现潜在泄漏:")
        for leak in results["potential_leaks"]:
            print(f"   {leak['severity'].upper()}: {leak['description']}")
    else:
        print(f"\n✅ 未发现明显内存泄漏")
    
    print(f"\n💡 建议:")
    for rec in results["recommendations"]:
        print(f"   • {rec}")

def get_memory_snapshot():
    """获取内存快照"""
    report = memory_detector.get_detailed_memory_report()
    
    if "error" in report:
        print(f"❌ 获取报告失败: {report['error']}")
        return
    
    print(f"📊 内存快照:")
    print(f"   当前内存: {report['current_memory_mb']:.1f} MB ({report['memory_percent']:.1f}%)")
    print(f"   总对象数: {report['total_objects']:,}")
    
    print(f"\n🏷️  主要对象类型:")
    for obj_type, count in report['top_object_types'][:5]:
        print(f"   {obj_type}: {count:,}")
    
    if report['large_objects']:
        print(f"\n📦 大对象 (>1MB):")
        for obj in report['large_objects'][:3]:
            print(f"   {obj['type']}: {obj['size_mb']:.2f} MB")

def force_cleanup():
    """强制清理内存"""
    print("🧹 执行强制内存清理...")
    
    results = memory_detector.force_garbage_collection()
    
    print(f"📊 清理结果:")
    print(f"   回收对象: {results['objects_collected']}")
    print(f"   释放对象: {results['objects_freed']:,}")
    print(f"   释放内存: {results['memory_freed_mb']:.2f} MB")
    print(f"   当前内存: {results['memory_after_mb']:.1f} MB")

if __name__ == "__main__":
    # 演示用法
    print("启动内存监控...")
    start_memory_monitoring()
    
    print("等待收集数据...")
    time.sleep(10)  # 等待一段时间收集数据
    
    check_memory_leaks()
    get_memory_snapshot()
    force_cleanup()
```

### 3.2 数据库性能诊断

```python
# database_performance_diagnostics.py

import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
import redis
import logging

class DatabasePerformanceDiagnostics:
    """数据库性能诊断工具"""
    
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333,
                 redis_host: str = "localhost", redis_port: int = 6379):
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.performance_data = {
            'qdrant': [],
            'redis': []
        }
    
    async def run_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """运行综合数据库诊断"""
        print("🔍 开始数据库性能诊断...")
        
        diagnostics = {
            "timestamp": time.time(),
            "qdrant_diagnostics": await self.diagnose_qdrant_performance(),
            "redis_diagnostics": await self.diagnose_redis_performance(),
            "connection_tests": await self.test_connections(),
            "load_tests": await self.run_load_tests(),
            "recommendations": []
        }
        
        # 生成综合建议
        diagnostics["recommendations"] = self._generate_comprehensive_recommendations(diagnostics)
        
        return diagnostics
    
    async def diagnose_qdrant_performance(self) -> Dict[str, Any]:
        """诊断 Qdrant 性能"""
        try:
            # 基本连接测试
            collections = self.qdrant_client.get_collections()
            
            if not collections.collections:
                return {
                    "status": "warning",
                    "message": "No collections found in Qdrant",
                    "collections": []
                }
            
            collection_diagnostics = {}
            
            for collection in collections.collections:
                collection_name = collection.name
                print(f"   诊断集合: {collection_name}")
                
                # 获取集合信息
                collection_info = self.qdrant_client.get_collection(collection_name)
                
                # 性能测试
                search_performance = await self._test_qdrant_search_performance(collection_name)
                
                collection_diagnostics[collection_name] = {
                    "info": {
                        "status": collection_info.status,
                        "vectors_count": collection_info.vectors_count,
                        "indexed_vectors_count": collection_info.indexed_vectors_count,
                        "points_count": collection_info.points_count,
                        "segments_count": len(collection_info.payload_schema),
                        "config": {
                            "vector_size": collection_info.config.params.vectors.size,
                            "distance": collection_info.config.params.vectors.distance,
                        }
                    },
                    "performance": search_performance,
                    "health_score": self._calculate_collection_health_score(collection_info, search_performance)
                }
            
            return {
                "status": "success",
                "collections": collection_diagnostics,
                "overall_health": self._calculate_overall_qdrant_health(collection_diagnostics)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to connect to Qdrant or perform diagnostics"
            }
    
    async def _test_qdrant_search_performance(self, collection_name: str) -> Dict[str, Any]:
        """测试 Qdrant 搜索性能"""
        try:
            # 获取集合信息以确定向量维度
            collection_info = self.qdrant_client.get_collection(collection_name)
            vector_size = collection_info.config.params.vectors.size
            
            # 生成测试向量
            test_vectors = [np.random.random(vector_size).tolist() for _ in range(10)]
            
            # 测试不同的搜索参数
            test_cases = [
                {"limit": 5, "name": "small_search"},
                {"limit": 20, "name": "medium_search"},
                {"limit": 50, "name": "large_search"},
            ]
            
            performance_results = {}
            
            for test_case in test_cases:
                times = []
                for test_vector in test_vectors:
                    start_time = time.time()
                    
                    results = self.qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=test_vector,
                        limit=test_case["limit"]
                    )
                    
                    search_time = time.time() - start_time
                    times.append(search_time)
                
                performance_results[test_case["name"]] = {
                    "avg_time": statistics.mean(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "median_time": statistics.median(times),
                    "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
                    "limit": test_case["limit"]
                }
            
            return {
                "status": "success",
                "test_results": performance_results,
                "vector_size": vector_size,
                "tests_performed": len(test_vectors) * len(test_cases)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def diagnose_redis_performance(self) -> Dict[str, Any]:
        """诊断 Redis 性能"""
        try:
            # 基本连接测试
            ping_result = self.redis_client.ping()
            if not ping_result:
                return {
                    "status": "error",
                    "message": "Redis ping failed"
                }
            
            # 获取 Redis 信息
            info = self.redis_client.info()
            
            # 性能测试
            performance_tests = await self._test_redis_performance()
            
            # 内存分析
            memory_analysis = self._analyze_redis_memory(info)
            
            return {
                "status": "success",
                "connection": "healthy",
                "server_info": {
                    "version": info.get("redis_version"),
                    "uptime_seconds": info.get("uptime_in_seconds"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human"),
                    "used_memory_peak_human": info.get("used_memory_peak_human"),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                },
                "performance": performance_tests,
                "memory_analysis": memory_analysis,
                "health_score": self._calculate_redis_health_score(info, performance_tests)
            }
            
        except redis.ConnectionError:
            return {
                "status": "error",
                "message": "Cannot connect to Redis server"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _test_redis_performance(self) -> Dict[str, Any]:
        """测试 Redis 性能"""
        # SET 操作性能测试
        set_times = []
        test_data = {"test_key": "test_value_with_some_data" * 10}
        
        for i in range(100):
            start_time = time.time()
            self.redis_client.set(f"perf_test_key_{i}", f"test_value_{i}")
            set_times.append(time.time() - start_time)
        
        # GET 操作性能测试
        get_times = []
        for i in range(100):
            start_time = time.time()
            self.redis_client.get(f"perf_test_key_{i}")
            get_times.append(time.time() - start_time)
        
        # DELETE 操作性能测试
        del_times = []
        for i in range(100):
            start_time = time.time()
            self.redis_client.delete(f"perf_test_key_{i}")
            del_times.append(time.time() - start_time)
        
        # PING 延迟测试
        ping_times = []
        for _ in range(50):
            start_time = time.time()
            self.redis_client.ping()
            ping_times.append(time.time() - start_time)
        
        return {
            "set_operations": {
                "avg_time_ms": statistics.mean(set_times) * 1000,
                "min_time_ms": min(set_times) * 1000,
                "max_time_ms": max(set_times) * 1000,
                "operations_per_second": 1 / statistics.mean(set_times)
            },
            "get_operations": {
                "avg_time_ms": statistics.mean(get_times) * 1000,
                "min_time_ms": min(get_times) * 1000,
                "max_time_ms": max(get_times) * 1000,
                "operations_per_second": 1 / statistics.mean(get_times)
            },
            "delete_operations": {
                "avg_time_ms": statistics.mean(del_times) * 1000,
                "operations_per_second": 1 / statistics.mean(del_times)
            },
            "ping_latency": {
                "avg_latency_ms": statistics.mean(ping_times) * 1000,
                "min_latency_ms": min(ping_times) * 1000,
                "max_latency_ms": max(ping_times) * 1000
            }
        }
    
    def _analyze_redis_memory(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """分析 Redis 内存使用"""
        used_memory = info.get("used_memory", 0)
        max_memory = info.get("maxmemory", 0)
        
        memory_analysis = {
            "used_memory_bytes": used_memory,
            "used_memory_mb": used_memory / 1024 / 1024,
            "memory_utilization_percent": 0
        }
        
        if max_memory > 0:
            memory_analysis["memory_utilization_percent"] = (used_memory / max_memory) * 100
            memory_analysis["max_memory_mb"] = max_memory / 1024 / 1024
        
        # 计算命中率
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total_requests = hits + misses
        
        if total_requests > 0:
            memory_analysis["hit_rate_percent"] = (hits / total_requests) * 100
        else:
            memory_analysis["hit_rate_percent"] = 0
        
        memory_analysis["total_requests"] = total_requests
        
        return memory_analysis
    
    def _calculate_collection_health_score(self, collection_info, performance_data) -> int:
        """计算集合健康分数 (0-100)"""
        score = 100
        
        # 检查索引状态
        if collection_info.status != "green":
            score -= 30
        
        # 检查索引覆盖率
        if collection_info.vectors_count > 0:
            index_coverage = collection_info.indexed_vectors_count / collection_info.vectors_count
            if index_coverage < 0.9:
                score -= 20
        
        # 检查搜索性能
        if performance_data.get("status") == "success":
            test_results = performance_data.get("test_results", {})
            avg_search_time = test_results.get("small_search", {}).get("avg_time", 0)
            if avg_search_time > 1.0:  # 超过1秒
                score -= 25
            elif avg_search_time > 0.5:  # 超过0.5秒
                score -= 10
        else:
            score -= 40
        
        return max(0, score)
    
    def _calculate_redis_health_score(self, info: Dict[str, Any], performance_data: Dict[str, Any]) -> int:
        """计算 Redis 健康分数 (0-100)"""
        score = 100
        
        # 检查内存使用
        max_memory = info.get("maxmemory", 0)
        used_memory = info.get("used_memory", 0)
        
        if max_memory > 0:
            memory_utilization = (used_memory / max_memory) * 100
            if memory_utilization > 90:
                score -= 30
            elif memory_utilization > 80:
                score -= 15
        
        # 检查命中率
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        if hits + misses > 0:
            hit_rate = hits / (hits + misses)
            if hit_rate < 0.5:
                score -= 25
            elif hit_rate < 0.8:
                score -= 10
        
        # 检查延迟
        ping_latency = performance_data.get("ping_latency", {}).get("avg_latency_ms", 0)
        if ping_latency > 10:  # 超过10ms
            score -= 20
        elif ping_latency > 5:  # 超过5ms
            score -= 10
        
        return max(0, score)
    
    def _calculate_overall_qdrant_health(self, collection_diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """计算 Qdrant 整体健康状况"""
        if not collection_diagnostics:
            return {"score": 0, "status": "critical", "message": "No collections found"}
        
        health_scores = [col_data["health_score"] for col_data in collection_diagnostics.values()]
        avg_score = statistics.mean(health_scores)
        
        if avg_score >= 80:
            status = "healthy"
        elif avg_score >= 60:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "score": avg_score,
            "status": status,
            "collection_count": len(collection_diagnostics),
            "healthy_collections": sum(1 for score in health_scores if score >= 80)
        }
    
    async def test_connections(self) -> Dict[str, Any]:
        """测试数据库连接"""
        connection_tests = {}
        
        # 测试 Qdrant 连接
        try:
            start_time = time.time()
            collections = self.qdrant_client.get_collections()
            qdrant_latency = time.time() - start_time
            
            connection_tests["qdrant"] = {
                "status": "connected",
                "latency_ms": qdrant_latency * 1000,
                "collections_count": len(collections.collections)
            }
        except Exception as e:
            connection_tests["qdrant"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # 测试 Redis 连接
        try:
            start_time = time.time()
            self.redis_client.ping()
            redis_latency = time.time() - start_time
            
            connection_tests["redis"] = {
                "status": "connected",
                "latency_ms": redis_latency * 1000,
                "db_size": self.redis_client.dbsize()
            }
        except Exception as e:
            connection_tests["redis"] = {
                "status": "failed", 
                "error": str(e)
            }
        
        return connection_tests
    
    async def run_load_tests(self) -> Dict[str, Any]:
        """运行负载测试"""
        print("   执行负载测试...")
        
        load_test_results = {}
        
        # Qdrant 负载测试
        try:
            collections = self.qdrant_client.get_collections()
            if collections.collections:
                collection_name = collections.collections[0].name
                qdrant_load_test = await self._qdrant_load_test(collection_name)
                load_test_results["qdrant"] = qdrant_load_test
        except Exception as e:
            load_test_results["qdrant"] = {"status": "error", "error": str(e)}
        
        # Redis 负载测试
        try:
            redis_load_test = await self._redis_load_test()
            load_test_results["redis"] = redis_load_test
        except Exception as e:
            load_test_results["redis"] = {"status": "error", "error": str(e)}
        
        return load_test_results
    
    async def _qdrant_load_test(self, collection_name: str, concurrent_requests: int = 10) -> Dict[str, Any]:
        """Qdrant 负载测试"""
        try:
            collection_info = self.qdrant_client.get_collection(collection_name)
            vector_size = collection_info.config.params.vectors.size
            
            # 生成测试向量
            test_vectors = [np.random.random(vector_size).tolist() for _ in range(concurrent_requests)]
            
            # 并发搜索测试
            async def single_search(vector):
                start_time = time.time()
                results = self.qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=vector,
                    limit=10
                )
                return time.time() - start_time
            
            start_time = time.time()
            search_times = await asyncio.gather(*[single_search(v) for v in test_vectors])
            total_time = time.time() - start_time
            
            return {
                "status": "success",
                "concurrent_requests": concurrent_requests,
                "total_time": total_time,
                "avg_response_time": statistics.mean(search_times),
                "max_response_time": max(search_times),
                "min_response_time": min(search_times),
                "requests_per_second": concurrent_requests / total_time,
                "collection_name": collection_name
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _redis_load_test(self, concurrent_operations: int = 100) -> Dict[str, Any]:
        """Redis 负载测试"""
        try:
            # 并发 SET 操作
            async def single_set_operation(i):
                start_time = time.time()
                self.redis_client.set(f"load_test_key_{i}", f"load_test_value_{i}")
                return time.time() - start_time
            
            start_time = time.time()
            set_times = await asyncio.gather(*[single_set_operation(i) for i in range(concurrent_operations)])
            set_total_time = time.time() - start_time
            
            # 并发 GET 操作
            async def single_get_operation(i):
                start_time = time.time()
                self.redis_client.get(f"load_test_key_{i}")
                return time.time() - start_time
            
            start_time = time.time()
            get_times = await asyncio.gather(*[single_get_operation(i) for i in range(concurrent_operations)])
            get_total_time = time.time() - start_time
            
            # 清理测试数据
            for i in range(concurrent_operations):
                self.redis_client.delete(f"load_test_key_{i}")
            
            return {
                "status": "success",
                "concurrent_operations": concurrent_operations,
                "set_operations": {
                    "total_time": set_total_time,
                    "avg_time": statistics.mean(set_times),
                    "operations_per_second": concurrent_operations / set_total_time
                },
                "get_operations": {
                    "total_time": get_total_time,
                    "avg_time": statistics.mean(get_times),
                    "operations_per_second": concurrent_operations / get_total_time
                }
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _generate_comprehensive_recommendations(self, diagnostics: Dict[str, Any]) -> List[str]:
        """生成综合建议"""
        recommendations = []
        
        # Qdrant 建议
        qdrant_diag = diagnostics.get("qdrant_diagnostics", {})
        if qdrant_diag.get("status") == "error":
            recommendations.append("Qdrant 连接失败，检查服务状态和网络连接")
        else:
            overall_health = qdrant_diag.get("overall_health", {})
            if overall_health.get("score", 0) < 60:
                recommendations.append("Qdrant 整体健康分数较低，需要优化集合配置或重建索引")
        
        # Redis 建议
        redis_diag = diagnostics.get("redis_diagnostics", {})
        if redis_diag.get("status") == "error":
            recommendations.append("Redis 连接失败，检查服务状态")
        else:
            health_score = redis_diag.get("health_score", 0)
            if health_score < 60:
                recommendations.append("Redis 性能较差，检查内存使用和网络延迟")
            
            memory_analysis = redis_diag.get("memory_analysis", {})
            hit_rate = memory_analysis.get("hit_rate_percent", 0)
            if hit_rate < 50:
                recommendations.append(f"Redis 缓存命中率较低 ({hit_rate:.1f}%)，考虑调整缓存策略")
        
        # 负载测试建议
        load_tests = diagnostics.get("load_tests", {})
        qdrant_load = load_tests.get("qdrant", {})
        if qdrant_load.get("requests_per_second", 0) < 10:
            recommendations.append("Qdrant 并发性能较低，考虑优化硬件或配置")
        
        redis_load = load_tests.get("redis", {})
        redis_set_ops = redis_load.get("set_operations", {}).get("operations_per_second", 0)
        if redis_set_ops < 1000:
            recommendations.append("Redis 写入性能较低，检查硬件配置")
        
        if not recommendations:
            recommendations.append("数据库性能良好，无需特殊优化")
        
        return recommendations

# 使用示例
async def run_database_diagnostics():
    """运行数据库诊断"""
    diagnostics = DatabasePerformanceDiagnostics()
    
    results = await diagnostics.run_comprehensive_diagnostics()
    
    print("📊 数据库诊断结果:")
    
    # Qdrant 结果
    qdrant_diag = results["qdrant_diagnostics"]
    if qdrant_diag["status"] == "success":
        overall_health = qdrant_diag["overall_health"]
        print(f"   Qdrant: {overall_health['status'].upper()} (分数: {overall_health['score']:.1f})")
        print(f"   集合数量: {overall_health['collection_count']}")
    else:
        print(f"   Qdrant: ERROR - {qdrant_diag.get('message', 'Unknown error')}")
    
    # Redis 结果
    redis_diag = results["redis_diagnostics"]
    if redis_diag["status"] == "success":
        print(f"   Redis: HEALTHY (分数: {redis_diag['health_score']:.1f})")
        memory_analysis = redis_diag["memory_analysis"]
        print(f"   内存使用: {memory_analysis['used_memory_mb']:.1f} MB")
        print(f"   缓存命中率: {memory_analysis['hit_rate_percent']:.1f}%")
    else:
        print(f"   Redis: ERROR - {redis_diag.get('message', 'Unknown error')}")
    
    print("\n💡 建议:")
    for rec in results["recommendations"]:
        print(f"   • {rec}")

if __name__ == "__main__":
    asyncio.run(run_database_diagnostics())
```

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create comprehensive system technical documentation", "status": "completed", "activeForm": "Creating comprehensive system technical documentation"}, {"content": "Generate debugging and troubleshooting guide", "status": "completed", "activeForm": "Generating debugging and troubleshooting guide"}, {"content": "Update and enhance existing documentation", "status": "in_progress", "activeForm": "Updating and enhancing existing documentation"}, {"content": "Create deployment and operations guide", "status": "pending", "activeForm": "Creating deployment and operations guide"}]