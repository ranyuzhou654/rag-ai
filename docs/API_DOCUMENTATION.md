# RAG-AI Enhanced API Documentation

本文档详细介绍了RAG-AI系统的增强API端点，包括个性化功能、存储优化和用户管理等功能。

## 📋 目录

- [API概述](#api概述)
- [认证与安全](#认证与安全)
- [个性化端点](#个性化端点)
- [存储优化端点](#存储优化端点)
- [用户管理端点](#用户管理端点)
- [系统监控端点](#系统监控端点)
- [错误处理](#错误处理)
- [SDK和客户端](#sdk和客户端)

## 🌐 API概述

### 基础信息

- **Base URL**: `http://localhost:8000` (开发环境)
- **API 版本**: v2.0
- **协议**: HTTP/HTTPS
- **数据格式**: JSON
- **文档**: OpenAPI 3.0 规范

### 核心特性

- ✅ **异步处理**: 所有端点支持异步操作
- ✅ **流式响应**: 支持Server-Sent Events (SSE)
- ✅ **个性化**: 基于用户画像的智能响应
- ✅ **存储优化**: 自动化存储管理
- ✅ **实时监控**: Prometheus指标集成
- ✅ **错误恢复**: 自动重试和故障转移

### 快速开始

```bash
# 安装依赖
pip install httpx

# 测试API连接
curl http://localhost:8000/health

# 查看API文档
curl http://localhost:8000/docs
```

## 🔐 认证与安全

### API Key认证 (可选)

```python
import httpx

headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v2/ask",
        headers=headers,
        json={"query": "你的问题"}
    )
```

### 安全配置

```python
# 环境变量
API_KEY_REQUIRED=false  # 开发环境
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8501"]
MAX_REQUEST_SIZE=10MB
RATE_LIMIT_PER_MINUTE=60
```

## 🎯 个性化端点

### 1. 增强问答 - Enhanced Q&A

**端点**: `POST /api/v2/ask`

智能问答系统，支持个性化回答和推荐。

#### 请求参数

```python
from pydantic import BaseModel
from typing import Optional, Dict, Any

class PersonalizedQuestionRequest(BaseModel):
    query: str                              # 用户问题
    user_id: Optional[str] = None          # 用户ID（用于个性化）
    context: Optional[str] = None          # 额外上下文
    max_results: int = 5                   # 检索结果数量
    include_sources: bool = True           # 是否包含来源
    include_recommendations: bool = True    # 是否包含推荐
    preferences: Optional[Dict[str, Any]] = None  # 用户偏好设置
    rag_mode: str = "enhanced"             # RAG模式
```

#### 响应格式

```python
class EnhancedRAGResponse(BaseModel):
    answer: str                            # 生成的答案
    confidence: float                      # 置信度分数
    sources: List[SourceInfo]             # 参考来源
    recommendations: List[RecommendationItem]  # 个性化推荐
    personalization_score: float          # 个性化程度
    generation_time: float               # 生成时间
    token_count: int                      # Token使用量
    user_profile_updated: bool            # 用户画像是否更新
```

#### 代码示例

```python
import httpx
import asyncio

async def ask_personalized_question():
    """个性化问答示例"""
    
    request_data = {
        "query": "最新的Transformer架构改进有哪些？",
        "user_id": "user_123",
        "max_results": 8,
        "include_sources": True,
        "include_recommendations": True,
        "preferences": {
            "response_length": "detailed",
            "technical_level": "advanced",
            "preferred_sources": ["arxiv", "huggingface"]
        },
        "rag_mode": "enhanced"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v2/ask",
            json=request_data,
            timeout=60.0
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"答案: {result['answer']}")
            print(f"置信度: {result['confidence']:.3f}")
            print(f"个性化分数: {result['personalization_score']:.3f}")
            print(f"来源数量: {len(result['sources'])}")
            print(f"推荐数量: {len(result['recommendations'])}")
            
            # 显示推荐
            for i, rec in enumerate(result['recommendations'][:3]):
                print(f"推荐 {i+1}: {rec['title']}")
                print(f"  理由: {rec['recommendation_reason']}")
        else:
            print(f"请求失败: {response.status_code}")

# 运行示例
asyncio.run(ask_personalized_question())
```

### 2. 流式问答 - Streaming Q&A

**端点**: `POST /api/v2/ask/stream`

支持实时流式响应的问答端点。

#### 使用Server-Sent Events

```python
import httpx
import json

async def stream_answer():
    """流式答案生成示例"""
    
    request_data = {
        "query": "解释深度学习中的注意力机制原理",
        "user_id": "user_123",
        "stream": True
    }
    
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/api/v2/ask/stream",
            json=request_data
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    
                    if data["type"] == "content":
                        print(data["content"], end="", flush=True)
                    elif data["type"] == "sources":
                        print(f"\n\n来源: {len(data['sources'])}个")
                    elif data["type"] == "recommendations":
                        print(f"推荐: {len(data['recommendations'])}个")
                    elif data["type"] == "complete":
                        print("\n\n生成完成")
                        break

asyncio.run(stream_answer())
```

### 3. 获取用户推荐 - Get User Recommendations

**端点**: `GET /api/v2/recommendations/{user_id}`

获取用户的个性化推荐内容。

#### 查询参数

- `limit`: 推荐数量 (默认: 10, 范围: 1-50)
- `days_back`: 分析天数 (默认: 7, 范围: 1-30)
- `recommendation_type`: 推荐类型 ("content_based", "collaborative", "trending", "all")

#### 代码示例

```python
async def get_user_recommendations():
    """获取用户推荐示例"""
    
    user_id = "user_123"
    params = {
        "limit": 15,
        "days_back": 14,
        "recommendation_type": "all"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://localhost:8000/api/v2/recommendations/{user_id}",
            params=params
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"为用户 {user_id} 生成了 {data['total_count']} 个推荐")
            
            for rec in data['recommendations']:
                print(f"\n标题: {rec['title']}")
                print(f"类型: {rec['recommendation_type']}")
                print(f"分数: {rec['score']:.3f}")
                print(f"理由: {rec['recommendation_reason']}")
                print(f"摘要: {rec['summary'][:100]}...")

asyncio.run(get_user_recommendations())
```

### 4. 用户仪表板 - User Dashboard

**端点**: `GET /api/v2/user/{user_id}/dashboard`

获取用户的完整仪表板数据。

#### 响应内容

```python
class UserDashboardResponse(BaseModel):
    user_profile: UserProfile              # 用户画像
    usage_statistics: UsageStatistics      # 使用统计
    recent_recommendations: List[RecommendationItem]  # 最近推荐
    interaction_history: List[UserInteraction]       # 交互历史
    research_insights: ResearchInsights    # 研究洞察
    dashboard_generated_at: datetime       # 生成时间
```

#### 代码示例

```python
async def get_user_dashboard():
    """获取用户仪表板示例"""
    
    user_id = "user_123"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://localhost:8000/api/v2/user/{user_id}/dashboard"
        )
        
        if response.status_code == 200:
            dashboard = response.json()
            
            profile = dashboard['user_profile']
            stats = dashboard['usage_statistics']
            
            print(f"用户ID: {profile['user_id']}")
            print(f"总查询数: {profile['total_queries']}")
            print(f"研究兴趣: {len(profile['research_interests'])}个")
            print(f"平均会话时长: {profile['avg_session_duration']:.1f}分钟")
            
            print(f"\n使用统计:")
            print(f"本月查询: {stats['monthly_queries']}")
            print(f"最活跃时间: {stats['most_active_hours']}")
            print(f"偏好主题: {stats['top_topics'][:3]}")

asyncio.run(get_user_dashboard())
```

## 💾 存储优化端点

### 1. 触发存储优化 - Trigger Storage Optimization

**端点**: `POST /api/v2/storage/optimize`

启动自动化存储优化任务。

#### 请求参数

```python
class StorageOptimizationRequest(BaseModel):
    target_hot_ratio: float = 0.1         # 热存储目标比例
    target_warm_ratio: float = 0.3        # 温存储目标比例
    target_cold_ratio: float = 0.5        # 冷存储目标比例
    dry_run: bool = False                 # 是否为试运行
    force_optimization: bool = False      # 是否强制优化
    optimization_strategy: str = "balanced"  # 优化策略
```

#### 代码示例

```python
async def optimize_storage():
    """存储优化示例"""
    
    optimization_request = {
        "target_hot_ratio": 0.15,
        "target_warm_ratio": 0.35,
        "target_cold_ratio": 0.45,
        "dry_run": False,
        "optimization_strategy": "performance"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v2/storage/optimize",
            json=optimization_request
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"优化任务已启动")
            print(f"任务ID: {result['job_id']}")
            print(f"预估耗时: {result['estimated_duration']}")
            
            # 轮询任务状态
            await poll_optimization_status(result['job_id'])

async def poll_optimization_status(job_id: str):
    """轮询优化状态"""
    
    async with httpx.AsyncClient() as client:
        while True:
            response = await client.get(
                f"http://localhost:8000/api/v2/storage/optimize/{job_id}/status"
            )
            
            if response.status_code == 200:
                status = response.json()
                print(f"优化进度: {status['progress']:.1%}")
                
                if status['status'] == 'completed':
                    print("存储优化完成!")
                    print(f"节省空间: {status['storage_saved']} GB")
                    print(f"迁移文档: {status['documents_moved']} 个")
                    break
                elif status['status'] == 'failed':
                    print(f"优化失败: {status['error_message']}")
                    break
            
            await asyncio.sleep(10)  # 10秒检查一次

asyncio.run(optimize_storage())
```

### 2. 存储分析 - Storage Analytics

**端点**: `GET /api/v2/storage/analytics`

获取详细的存储使用分析。

#### 查询参数

- `days`: 分析天数 (默认: 30, 范围: 1-90)
- `include_predictions`: 是否包含预测数据
- `granularity`: 数据粒度 ("daily", "weekly", "monthly")

#### 代码示例

```python
async def get_storage_analytics():
    """获取存储分析示例"""
    
    params = {
        "days": 60,
        "include_predictions": True,
        "granularity": "weekly"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8000/api/v2/storage/analytics",
            params=params
        )
        
        if response.status_code == 200:
            analytics = response.json()
            
            distribution = analytics['storage_distribution']
            print("存储分布:")
            print(f"  热存储: {distribution['hot']:.1%}")
            print(f"  温存储: {distribution['warm']:.1%}")
            print(f"  冷存储: {distribution['cold']:.1%}")
            print(f"  归档: {distribution['archived']:.1%}")
            
            patterns = analytics['access_patterns']
            print(f"\n访问模式分析:")
            print(f"  高频文档: {patterns['high_frequency_docs']} 个")
            print(f"  平均访问间隔: {patterns['avg_access_interval']} 小时")
            print(f"  热点时段: {patterns['peak_hours']}")
            
            recommendations = analytics['optimization_recommendations']
            print(f"\n优化建议:")
            for rec in recommendations:
                print(f"  - {rec['recommendation']}")
                print(f"    预期收益: {rec['expected_benefit']}")

asyncio.run(get_storage_analytics())
```

## 👤 用户管理端点

### 1. 创建用户画像 - Create User Profile

**端点**: `POST /api/v2/users/{user_id}/profile`

创建或更新用户画像。

#### 请求参数

```python
class UserProfileRequest(BaseModel):
    research_interests: List[str]          # 研究兴趣
    preferred_response_length: str = "medium"  # 偏好响应长度
    technical_level: str = "intermediate"  # 技术水平
    preferred_sources: List[str] = []      # 偏好数据源
    language_preference: str = "zh"        # 语言偏好
    notification_settings: Dict[str, bool] = {}  # 通知设置
```

#### 代码示例

```python
async def create_user_profile():
    """创建用户画像示例"""
    
    user_id = "user_456"
    profile_data = {
        "research_interests": [
            "transformer模型", "注意力机制", "多模态学习", 
            "强化学习", "知识图谱"
        ],
        "preferred_response_length": "detailed",
        "technical_level": "advanced",
        "preferred_sources": ["arxiv", "huggingface", "acl"],
        "language_preference": "zh",
        "notification_settings": {
            "daily_recommendations": True,
            "trending_topics": True,
            "research_updates": False
        }
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://localhost:8000/api/v2/users/{user_id}/profile",
            json=profile_data
        )
        
        if response.status_code == 201:
            result = response.json()
            print(f"用户画像创建成功")
            print(f"用户ID: {result['user_id']}")
            print(f"兴趣数量: {len(result['research_interests'])}")
            print(f"画像创建时间: {result['created_at']}")

asyncio.run(create_user_profile())
```

### 2. 用户交互记录 - User Interaction Logging

**端点**: `POST /api/v2/users/{user_id}/interactions`

记录用户交互，用于画像学习。

#### 代码示例

```python
async def log_user_interaction():
    """记录用户交互示例"""
    
    user_id = "user_123"
    interaction_data = {
        "query": "如何提高Transformer模型的效率？",
        "response_content": "可以通过以下几种方法提高Transformer效率...",
        "interaction_type": "question_answer",
        "response_quality_rating": 4.5,
        "sources_used": [
            {"source_id": "arxiv_001", "relevance_score": 0.9},
            {"source_id": "arxiv_002", "relevance_score": 0.8}
        ],
        "session_context": {
            "session_id": "session_789",
            "previous_queries": 3,
            "session_duration": 1200  # 秒
        }
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://localhost:8000/api/v2/users/{user_id}/interactions",
            json=interaction_data
        )
        
        if response.status_code == 201:
            print("交互记录成功")
            print("用户画像将在后台更新")

asyncio.run(log_user_interaction())
```

### 3. 获取用户统计 - Get User Statistics

**端点**: `GET /api/v2/users/{user_id}/statistics`

获取用户的详细使用统计。

#### 代码示例

```python
async def get_user_statistics():
    """获取用户统计示例"""
    
    user_id = "user_123"
    params = {
        "period": "30d",
        "include_trends": True,
        "include_comparisons": True
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://localhost:8000/api/v2/users/{user_id}/statistics",
            params=params
        )
        
        if response.status_code == 200:
            stats = response.json()
            
            print(f"用户统计 (最近30天):")
            print(f"总查询数: {stats['total_queries']}")
            print(f"平均每日查询: {stats['avg_daily_queries']:.1f}")
            print(f"最长会话: {stats['longest_session']} 分钟")
            print(f"查询成功率: {stats['success_rate']:.1%}")
            
            trends = stats.get('trends', {})
            print(f"\n趋势分析:")
            print(f"查询量变化: {trends.get('query_growth', 0):+.1%}")
            print(f"满意度变化: {trends.get('satisfaction_change', 0):+.2f}")
            
            comparisons = stats.get('comparisons', {})
            print(f"\n与平均用户对比:")
            print(f"活跃度: {comparisons.get('activity_percentile', 0):.0f}%")
            print(f"专业度: {comparisons.get('expertise_level', 'medium')}")

asyncio.run(get_user_statistics())
```

## 📊 系统监控端点

### 1. 系统健康检查 - Health Check

**端点**: `GET /health`

检查系统整体健康状态。

#### 代码示例

```python
async def check_system_health():
    """系统健康检查示例"""
    
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/health")
        
        if response.status_code == 200:
            health = response.json()
            
            print(f"系统状态: {health['status']}")
            print(f"检查时间: {health['timestamp']}")
            
            components = health['components']
            for component, status in components.items():
                indicator = "✅" if status['healthy'] else "❌"
                print(f"{indicator} {component}: {status['status']}")
                
                if not status['healthy']:
                    print(f"   错误: {status.get('error', 'Unknown')}")

asyncio.run(check_system_health())
```

### 2. 系统指标 - System Metrics

**端点**: `GET /api/v2/metrics`

获取详细的系统性能指标。

#### 代码示例

```python
async def get_system_metrics():
    """获取系统指标示例"""
    
    params = {
        "period": "1h",
        "metrics": ["cpu", "memory", "requests", "cache_hits"]
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8000/api/v2/metrics",
            params=params
        )
        
        if response.status_code == 200:
            metrics = response.json()
            
            print("系统性能指标 (最近1小时):")
            
            cpu_usage = metrics.get('cpu_usage', {})
            print(f"CPU使用率: {cpu_usage.get('average', 0):.1f}%")
            print(f"CPU峰值: {cpu_usage.get('peak', 0):.1f}%")
            
            memory = metrics.get('memory_usage', {})
            print(f"内存使用: {memory.get('used_gb', 0):.1f}GB / {memory.get('total_gb', 0):.1f}GB")
            print(f"内存使用率: {memory.get('percentage', 0):.1f}%")
            
            requests = metrics.get('request_metrics', {})
            print(f"总请求数: {requests.get('total_requests', 0)}")
            print(f"平均响应时间: {requests.get('avg_response_time', 0):.2f}s")
            print(f"错误率: {requests.get('error_rate', 0):.2%}")
            
            cache = metrics.get('cache_metrics', {})
            print(f"缓存命中率: {cache.get('hit_rate', 0):.1%}")
            print(f"缓存大小: {cache.get('size_mb', 0):.1f}MB")

asyncio.run(get_system_metrics())
```

### 3. 实时监控 - Real-time Monitoring

**端点**: `GET /api/v2/monitor/stream`

通过WebSocket获取实时系统监控数据。

#### 代码示例

```python
import websockets
import json

async def real_time_monitoring():
    """实时监控示例"""
    
    uri = "ws://localhost:8000/api/v2/monitor/stream"
    
    async with websockets.connect(uri) as websocket:
        # 订阅监控数据
        subscription = {
            "type": "subscribe",
            "metrics": ["requests", "users", "cache", "storage"],
            "interval": 5  # 5秒间隔
        }
        
        await websocket.send(json.dumps(subscription))
        
        print("开始实时监控...")
        
        async for message in websocket:
            data = json.loads(message)
            
            if data['type'] == 'metrics_update':
                metrics = data['metrics']
                timestamp = data['timestamp']
                
                print(f"\n--- {timestamp} ---")
                print(f"活跃用户: {metrics.get('active_users', 0)}")
                print(f"每分钟请求: {metrics.get('requests_per_minute', 0)}")
                print(f"平均响应时间: {metrics.get('avg_response_time', 0):.2f}s")
                print(f"缓存命中率: {metrics.get('cache_hit_rate', 0):.1%}")
                print(f"存储使用: {metrics.get('storage_usage_gb', 0):.1f}GB")
            
            elif data['type'] == 'alert':
                alert = data['alert']
                print(f"\n🚨 警报: {alert['severity']} - {alert['message']}")

# 运行实时监控
asyncio.run(real_time_monitoring())
```

## ⚠️ 错误处理

### 错误响应格式

```python
class ErrorResponse(BaseModel):
    error_code: str                        # 错误代码
    error_message: str                     # 错误描述
    details: Optional[Dict[str, Any]]      # 详细信息
    timestamp: datetime                    # 错误时间
    request_id: str                        # 请求ID
    suggestions: List[str] = []            # 解决建议
```

### 常见错误代码

| 错误代码 | HTTP状态码 | 描述 | 解决方案 |
|----------|------------|------|----------|
| `USER_NOT_FOUND` | 404 | 用户未找到 | 检查user_id是否正确 |
| `PROFILE_NOT_INITIALIZED` | 400 | 用户画像未初始化 | 先创建用户画像 |
| `INVALID_QUERY` | 400 | 查询参数无效 | 检查查询格式和参数 |
| `STORAGE_OPTIMIZATION_RUNNING` | 409 | 存储优化进行中 | 等待当前优化完成 |
| `RATE_LIMIT_EXCEEDED` | 429 | 请求频率超限 | 降低请求频率 |
| `INTERNAL_ERROR` | 500 | 服务器内部错误 | 检查服务器日志 |

### 错误处理示例

```python
import httpx
from typing import Optional

class APIError(Exception):
    def __init__(self, error_code: str, message: str, details: Optional[dict] = None):
        self.error_code = error_code
        self.message = message
        self.details = details
        super().__init__(f"{error_code}: {message}")

async def api_request_with_error_handling(url: str, method: str = "GET", **kwargs):
    """带错误处理的API请求"""
    
    async with httpx.AsyncClient() as client:
        try:
            if method.upper() == "GET":
                response = await client.get(url, **kwargs)
            elif method.upper() == "POST":
                response = await client.post(url, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # 检查HTTP状态码
            if response.status_code >= 400:
                error_data = response.json()
                raise APIError(
                    error_code=error_data.get('error_code', 'UNKNOWN_ERROR'),
                    message=error_data.get('error_message', 'Unknown error occurred'),
                    details=error_data.get('details')
                )
            
            return response.json()
            
        except httpx.RequestError as e:
            raise APIError(
                error_code="REQUEST_ERROR",
                message=f"Request failed: {str(e)}",
                details={"original_error": str(e)}
            )
        except httpx.TimeoutException:
            raise APIError(
                error_code="TIMEOUT_ERROR",
                message="Request timed out",
                details={"timeout": kwargs.get('timeout', 'default')}
            )

# 使用示例
async def safe_api_call():
    try:
        result = await api_request_with_error_handling(
            "http://localhost:8000/api/v2/ask",
            method="POST",
            json={"query": "测试问题"},
            timeout=30.0
        )
        print("请求成功:", result)
        
    except APIError as e:
        print(f"API错误 [{e.error_code}]: {e.message}")
        if e.details:
            print(f"详细信息: {e.details}")
        
        # 根据错误代码采取不同处理策略
        if e.error_code == "RATE_LIMIT_EXCEEDED":
            print("请求频率过高，等待30秒后重试...")
            await asyncio.sleep(30)
        elif e.error_code == "USER_NOT_FOUND":
            print("用户不存在，需要先创建用户画像")
        elif e.error_code == "TIMEOUT_ERROR":
            print("请求超时，可能需要增加超时时间")

asyncio.run(safe_api_call())
```

## 🛠️ SDK和客户端

### Python SDK

```python
class RAGAIClient:
    """RAG-AI Python SDK"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = httpx.AsyncClient()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.aclose()
    
    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def ask_question(
        self,
        query: str,
        user_id: Optional[str] = None,
        **kwargs
    ) -> dict:
        """提问"""
        data = {"query": query, "user_id": user_id, **kwargs}
        
        response = await self.session.post(
            f"{self.base_url}/api/v2/ask",
            json=data,
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()
    
    async def get_recommendations(
        self,
        user_id: str,
        limit: int = 10,
        days_back: int = 7
    ) -> dict:
        """获取推荐"""
        params = {"limit": limit, "days_back": days_back}
        
        response = await self.session.get(
            f"{self.base_url}/api/v2/recommendations/{user_id}",
            params=params,
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()
    
    async def optimize_storage(self, **kwargs) -> dict:
        """存储优化"""
        response = await self.session.post(
            f"{self.base_url}/api/v2/storage/optimize",
            json=kwargs,
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()
    
    async def get_user_dashboard(self, user_id: str) -> dict:
        """获取用户仪表板"""
        response = await self.session.get(
            f"{self.base_url}/api/v2/user/{user_id}/dashboard",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()

# 使用SDK示例
async def sdk_example():
    async with RAGAIClient() as client:
        # 提问
        answer = await client.ask_question(
            query="什么是注意力机制？",
            user_id="user_123",
            include_recommendations=True
        )
        print(f"答案: {answer['answer']}")
        
        # 获取推荐
        recommendations = await client.get_recommendations("user_123")
        print(f"推荐数量: {recommendations['total_count']}")
        
        # 获取仪表板
        dashboard = await client.get_user_dashboard("user_123")
        print(f"用户总查询数: {dashboard['user_profile']['total_queries']}")

asyncio.run(sdk_example())
```

### JavaScript/TypeScript客户端

```typescript
interface RAGAIClientConfig {
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
}

interface QuestionRequest {
  query: string;
  userId?: string;
  maxResults?: number;
  includeSources?: boolean;
  includeRecommendations?: boolean;
  ragMode?: string;
}

class RAGAIClient {
  private baseUrl: string;
  private apiKey?: string;
  private timeout: number;

  constructor(config: RAGAIClientConfig = {}) {
    this.baseUrl = config.baseUrl || 'http://localhost:8000';
    this.apiKey = config.apiKey;
    this.timeout = config.timeout || 30000;
  }

  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
    
    return headers;
  }

  async askQuestion(request: QuestionRequest): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/v2/ask`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(request),
      signal: AbortSignal.timeout(this.timeout),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getRecommendations(
    userId: string,
    limit: number = 10,
    daysBack: number = 7
  ): Promise<any> {
    const params = new URLSearchParams({
      limit: limit.toString(),
      days_back: daysBack.toString(),
    });

    const response = await fetch(
      `${this.baseUrl}/api/v2/recommendations/${userId}?${params}`,
      {
        headers: this.getHeaders(),
        signal: AbortSignal.timeout(this.timeout),
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async streamAnswer(request: QuestionRequest): Promise<ReadableStream> {
    const response = await fetch(`${this.baseUrl}/api/v2/ask/stream`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ ...request, stream: true }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.body!;
  }
}

// 使用示例
const client = new RAGAIClient({
  baseUrl: 'http://localhost:8000',
  timeout: 60000,
});

// 基础问答
client.askQuestion({
  query: '什么是Transformer模型？',
  userId: 'user_123',
  includeRecommendations: true,
}).then(result => {
  console.log('答案:', result.answer);
  console.log('推荐数量:', result.recommendations.length);
});

// 流式响应
client.streamAnswer({
  query: '解释深度学习的基本原理',
  userId: 'user_123',
}).then(stream => {
  const reader = stream.getReader();
  const decoder = new TextDecoder();

  function readStream() {
    reader.read().then(({ done, value }) => {
      if (done) {
        console.log('流式响应完成');
        return;
      }

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      lines.forEach(line => {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'content') {
              process.stdout.write(data.content);
            }
          } catch (e) {
            // 忽略解析错误
          }
        }
      });

      readStream();
    });
  }

  readStream();
});
```

## 📝 总结

RAG-AI的增强API提供了完整的个性化和存储优化功能：

### 🎯 核心功能
- **个性化问答**: 基于用户画像的智能回答
- **实时推荐**: 多策略混合推荐引擎
- **存储优化**: 自动化多层存储管理
- **用户分析**: 详细的使用统计和洞察

### 🚀 技术特色
- **异步处理**: 所有端点支持高并发
- **流式响应**: 实时答案生成
- **智能缓存**: 多层缓存提升性能
- **监控集成**: 全面的指标和健康检查

### 📊 监控和分析
- **实时指标**: Prometheus集成
- **用户分析**: 详细的行为分析
- **存储洞察**: 自动化优化建议
- **性能监控**: 全链路性能跟踪

通过这些API，开发者可以构建具有个性化能力的智能应用，同时享受自动化的存储优化和全面的系统监控。