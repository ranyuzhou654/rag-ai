# RAG-AI: 企业级检索增强生成系统

> **🎉 版本 2.0 - 重大架构升级完成！**

专为学术论文研究和技术文档设计的先进RAG系统。本项目实现了**个性化推荐**、**用户画像**、**存储优化**、元数据优先数据收集、混合搜索、多层缓存、全面引用管理和生产就绪部署基础设施等高级功能。

## 🌟 核心特性

### 🔬 **学术研究专用RAG系统**
- **多源数据收集**: ArXiv、Hugging Face Papers、AI研究博客
- **元数据优先策略**: 优化存储，按需全文检索
- **学术引用系统**: APA、MLA、BibTeX、IEEE格式生成
- **来源可追溯**: 完整的学术诚信和引用跟踪
- **个性化推荐**: 基于用户兴趣的每日AI策划内容
- **用户画像**: 智能跟踪研究兴趣和偏好

### 🚀 **企业级架构** 
- **增强Streamlit前端**: 个性化界面和用户仪表板
- **增强FastAPI后端**: 异步API，支持个性化端点
- **多层存储优化**: 热/温/冷/归档数据生命周期
- **多层缓存**: 内存、Redis、文件和向量缓存
- **混合搜索**: 语义 + 关键词(BM25) + 元数据过滤
- **微服务部署**: Docker Compose + Nginx + 监控

### 🧠 **先进AI能力**
- **智能查询处理**: 查询重写、子问题生成
- **智能体RAG**: 自评估检索与质量反馈循环
- **分层生成**: 成本优化的模型路由(本地→API)
- **知识图谱增强**: 实体抽取和图谱检索
- **推荐引擎**: 内容过滤和协同过滤
- **使用分析**: 高级存储和访问模式分析

### 📊 **生产就绪运维**
- **全面监控**: Prometheus + Grafana仪表板
- **性能优化**: 缓存、异步处理、模型共享
- **水平扩展**: 容器编排和负载均衡
- **CI/CD就绪**: 全环境Docker配置

## 📦 系统架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG-AI 系统架构                              │
├─────────────────────────────────────────────────────────────────┤
│  前端 (Next.js)        │  API网关 (Nginx)                       │
├─────────────────────────┼────────────────────────────────────────┤
│  FastAPI 后端           │  多层缓存                               │
│  ├─ 异步RAG端点         │  ├─ 内存缓存 (LRU)                     │
│  ├─ 流式响应           │  ├─ Redis (分布式)                      │
│  └─ 引用管理           │  ├─ 文件缓存 (持久化)                   │
│                         │  └─ 向量缓存 (专用)                     │
├─────────────────────────┼────────────────────────────────────────┤
│  向量数据库             │  知识图谱                               │
│  ├─ Qdrant (混合)       │  ├─ 实体抽取                           │
│  ├─ 语义搜索           │  ├─ 关系映射                           │
│  ├─ BM25 + 过滤        │  └─ 图谱增强检索                       │
│  └─ 学术元数据         │                                        │
├─────────────────────────┼────────────────────────────────────────┤
│  数据收集               │  监控与可观测性                         │
│  ├─ 元数据优先         │  ├─ Prometheus指标                      │
│  ├─ 按需PDF            │  ├─ Grafana仪表板                      │
│  ├─ 每日增量           │  ├─ 性能跟踪                           │
│  └─ 多源异步           │  └─ 错误分析                           │
└─────────────────────────────────────────────────────────────────┘
```

## 🗂️ 项目结构

```
rag-ai/
├── 🚀 部署与配置
│   ├── docker-compose.yml         # 生产环境部署
│   ├── docker-compose.dev.yml     # 开发环境  
│   ├── Dockerfile.*               # 容器配置
│   ├── nginx/nginx.conf           # 反向代理设置
│   └── monitoring/                # Prometheus & Grafana配置
│
├── 🔧 核心应用
│   ├── api/enhanced_main.py       # 增强版FastAPI后端（含个性化功能）
│   ├── enhanced_app.py            # 增强版Streamlit界面（含个性化功能）
│   ├── run_rag_system.py          # 系统编排器
│   └── configs/config.py          # 集中配置
│
├── 📚 源代码模块
│   └── src/
│       ├── 📥 data_ingestion/     # 多源数据收集
│       │   └── multi_source_collector.py  # 元数据优先收集器
│       ├── 🏗️ processing/          # 文本处理与索引
│       │   ├── text_processor.py          # 增强文本处理
│       │   └── multi_representation_indexer.py
│       ├── 🔍 retrieval/          # 混合搜索与检索
│       │   ├── vector_database.py         # 增强Qdrant集成
│       │   ├── query_intelligence.py      # 查询处理
│       │   └── agentic_rag.py            # 自评估检索
│       ├── 🤖 generation/         # 答案生成
│       │   ├── enhanced_rag_system.py    # 增强RAG系统（含个性化）
│       │   ├── ultimate_rag_system.py    # 主RAG编排器
│       │   └── tiered_generation.py      # 成本优化路由
│       ├── 👤 personalization/    # 🆕 用户个性化
│       │   ├── user_profiler.py          # 用户画像管理
│       │   ├── recommendation_engine.py  # 每日推荐
│       │   └── preference_tracker.py     # 兴趣跟踪
│       ├── 💾 storage/            # 🆕 存储优化
│       │   ├── storage_optimizer.py      # 多层优化
│       │   ├── usage_analytics.py        # 访问模式分析
│       │   └── data_lifecycle.py         # 自动生命周期管理
│       ├── 📖 citation/           # 引用管理
│       │   └── citation_manager.py       # 学术引用系统
│       ├── 💾 caching/            # 多层缓存
│       │   └── multilayer_cache.py       # 高级缓存系统
│       ├── 📊 monitoring/         # 性能监控
│       │   └── metrics_collector.py      # Prometheus集成
│       ├── 🧠 knowledge_graph/    # 知识增强
│       ├── 📈 evaluation/         # 系统评估
│       └── ⚡ optimization/       # 性能优化
│
└── 📖 文档
    ├── README.md                  # 英文版综合指南
    ├── README_zh.md               # 中文版综合指南(本文档)
    └── docs/                      # 详细模块文档
```

## 🚀 快速开始

### 选项1: Docker Compose (推荐)

```bash
# 克隆仓库
git clone https://github.com/your-username/rag-ai.git
cd rag-ai

# 启动所有服务
docker-compose up -d

# 检查服务状态
docker-compose ps

# 访问系统
# - API文档: http://localhost/docs
# - 前端: http://localhost
# - Grafana监控: http://localhost:3001
```

### 选项2: 开发环境搭建

```bash
# 1. 环境搭建
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# 2. 启动向量数据库
docker run -d -p 6333:6333 qdrant/qdrant:v1.7.0

# 3. 启动Redis (可选，用于缓存)
docker run -d -p 6379:6379 redis:7.2-alpine

# 4. 配置环境
cp .env.example .env
# 编辑.env文件设置您的配置

# 5. 初始化系统
python run_rag_system.py

# 6. 启动增强API服务器（一个终端）
uvicorn api.enhanced_main:app --host 0.0.0.0 --port 8000 --reload

# 7. 启动增强Streamlit界面（另一个终端）
streamlit run enhanced_app.py
```

### 选项3: Docker开发环境

```bash
# 使用开发版compose配置
docker-compose -f docker-compose.dev.yml up -d

# 提供以下功能:
# - 后端和前端热重载
# - 开发卷挂载
# - 调试日志开启
```

## 🔧 配置说明

### 环境变量 (.env)

```bash
# 存储配置
STORAGE_ROOT=./project_data
HF_HOME=./project_data/models
HUGGING_FACE_TOKEN=your_hf_token_here

# 向量数据库
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=ai_papers

# 缓存配置  
REDIS_HOST=localhost
REDIS_PORT=6379
ENABLE_CACHE=true

# 模型配置
EMBEDDING_MODEL=BAAI/bge-m3
LLM_MODEL=Qwen/Qwen2-7B-Instruct
DEVICE=auto

# API密钥 (用于分层生成)
GPT4_API_KEY=your_openai_key
CLAUDE_API_KEY=your_claude_key

# 功能开关
ENABLE_HYBRID_SEARCH=true
ENABLE_AGENTIC_RAG=true
ENABLE_KNOWLEDGE_GRAPH=true
ENABLE_TIERED_GENERATION=true
```

## 🌟 2.0版本增强功能

### 👤 **个性化系统**
- **用户画像**: 智能跟踪研究兴趣和交互模式
- **每日推荐**: 基于用户偏好的AI策划内容
- **内容过滤**: 混合推荐引擎（协同和内容过滤）
- **用户仪表板**: 个性化分析和推荐管理

```python
# 示例: 用户画像和推荐
user_profiler = UserProfiler()
profile = await user_profiler.get_or_create_user_profile(user_id)

recommendation_engine = RecommendationEngine()
recommendations = await recommendation_engine.generate_daily_recommendations(
    user_id=user_id,
    limit=10,
    days_back=7
)
```

### 💾 **存储优化系统**
- **多层存储**: 热/温/冷/归档数据生命周期管理
- **使用分析**: 高级访问模式分析和优化
- **自动迁移**: 基于访问模式的智能数据移动
- **成本优化**: 高效存储利用和性能监控

```python
# 示例: 存储优化
storage_optimizer = StorageOptimizer()
analytics = UsageAnalytics()

# 分析访问模式
patterns = await analytics.analyze_access_patterns(days=30)

# 基于模式优化存储
optimization_result = await storage_optimizer.optimize_storage(
    target_hot_ratio=0.1,
    target_warm_ratio=0.3
)
```

### 📄 **元数据优先数据收集**
- **智能缓存**: 仅在需要时获取完整PDF
- **每日增量更新**: 高效数据管道
- **多源异步收集**: ArXiv、HuggingFace、博客
- **引用就绪元数据**: 内置学术合规性

```python
# 示例: 按需全文检索
collector = MultiSourceCollector(data_dir, metadata_only=True)
full_text = await collector.fetch_full_text_on_demand(document_id)
```

### 🔄 **多层缓存系统**
- **4层架构**: 内存 → Redis → 文件 → 向量缓存
- **智能缓存提升**: 频繁访问的数据自动上移
- **专用缓存**: 不同数据类型的分别处理
- **缓存分析**: 命中率和性能监控

```python
# 示例: 使用缓存系统
cache = create_multilayer_cache(config)
await cache.cache_query_embedding(query, embedding)
results = await cache.get_search_results(query_hash)
```

### 📚 **学术引用管理**
- **多种格式**: APA、MLA、BibTeX、IEEE、Chicago
- **来源验证**: 链接验证和可访问性检查
- **使用跟踪**: 引用流行度和趋势
- **参考文献导出**: 自动化参考列表生成

```python
# 示例: 生成引用
citation_manager = CitationManager(data_dir)
apa_citation = citation_manager.generate_citation(source_id, "apa")
source_links = citation_manager.generate_source_links(source_id)
```

### 🔍 **增强混合搜索**
- **语义+关键词**: 向量相似度 + BM25评分
- **学术过滤**: 作者、年份、期刊、分类过滤
- **查询智能**: 自动查询重写和扩展
- **性能优化**: 分布式索引和缓存

```python
# 示例: 高级学术搜索
results = await db.advanced_academic_search(
    query_vector=embedding,
    query_text=query,
    authors=["Bengio", "LeCun"],
    year_range=(2020, 2024),
    sources=["arxiv"],
    categories=["cs.AI", "cs.LG"]
)
```

### 🚀 **生产就绪API**
- **FastAPI后端**: 高性能异步API
- **流式响应**: 实时答案生成
- **WebSocket支持**: 实时更新和通知
- **OpenAPI文档**: 交互式API探索器

```bash
# API端点
POST /ask              # 主要问答端点
POST /ask/stream       # 流式响应
POST /search           # 文档搜索
GET  /document/{id}    # 文档检索
POST /feedback         # 用户反馈
GET  /stats           # 系统统计
```

## 📊 监控与可观测性

### Prometheus指标
- **请求指标**: 延迟、吞吐量、错误率
- **系统指标**: CPU、内存、缓存性能
- **业务指标**: 查询类型、引用使用
- **自定义仪表板**: Grafana可视化

### 健康检查
```bash
# 服务健康检查端点
curl http://localhost/health           # 整体系统
curl http://localhost/api/health       # API服务器
curl http://localhost:6333/health      # 向量数据库
```

## 🎯 使用示例

### 基础问答
```python
import aiohttp
import asyncio

async def ask_question():
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost/api/ask",
            json={
                "query": "Transformer模型的最新发展是什么？",
                "max_results": 5,
                "include_sources": True,
                "rag_mode": "ultimate"
            }
        ) as response:
            result = await response.json()
            print(f"答案: {result['answer']}")
            print(f"来源数量: {len(result['sources'])}")
            for source in result['sources']:
                print(f"- {source['citation']}")
```

### 流式响应
```javascript
// 前端流式示例
const eventSource = new EventSource('http://localhost/api/ask/stream');
eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'content') {
        document.getElementById('answer').innerHTML += data.content;
    }
};
```

### 高级搜索
```python
# 带学术过滤的搜索
response = await client.post("/api/search", json={
    "query": "神经网络中的注意力机制",
    "search_type": "academic",
    "filters": {
        "authors": ["Vaswani"],
        "year_range": [2017, 2024],
        "sources": ["arxiv"],
        "has_full_text": True
    },
    "limit": 10
})
```

## 🔧 开发指南

### 添加新功能

1. **创建功能分支**
```bash
git checkout -b feature/new-enhancement
```

2. **实现并监控**
```python
from src.monitoring import monitor_async_performance

@monitor_async_performance("component", "operation")
async def new_feature():
    # 自动监控的实现
    pass
```

3. **添加测试**
```bash
pytest tests/test_new_feature.py -v
```

4. **更新文档**
```markdown
## 新功能
描述和使用示例...
```

### 性能优化

系统包含多个优化层级:

1. **模型缓存**: 跨请求共享模型实例
2. **查询缓存**: 相似查询的LRU缓存  
3. **结果缓存**: Redis支持的响应缓存
4. **向量缓存**: 专用嵌入存储
5. **连接池**: 高效数据库连接

## 🚀 部署

### 生产环境部署

```bash
# 1. 克隆并配置
git clone https://github.com/your-username/rag-ai.git
cd rag-ai
cp .env.example .env
# 编辑.env设置生产配置

# 2. 部署并监控
docker-compose up -d

# 3. 初始化数据
docker-compose exec api python run_rag_system.py --setup

# 4. 验证部署
curl http://your-domain/health
```

### 扩展考虑

- **API实例**: 在负载均衡器后水平扩展FastAPI
- **向量数据库**: 大数据集使用Qdrant集群
- **缓存层**: 高可用Redis集群
- **后台任务**: 独立的collector服务实例

## 🎮 系统管理

### 常用管理命令

```bash
# 查看系统状态
docker-compose ps

# 查看日志
docker-compose logs -f api
docker-compose logs -f qdrant

# 重启服务
docker-compose restart api

# 更新系统
git pull
docker-compose build
docker-compose up -d

# 数据备份
docker-compose exec qdrant tar -czf /tmp/qdrant_backup.tar.gz /qdrant/storage
docker cp rag-ai-qdrant:/tmp/qdrant_backup.tar.gz ./backup/

# 性能监控
curl http://localhost/stats | jq
```

### 故障排除

```bash
# 检查服务连接
docker-compose exec api python -c "from qdrant_client import QdrantClient; print(QdrantClient('qdrant', 6333).get_collections())"

# 检查缓存
docker-compose exec redis redis-cli ping

# 查看系统资源
docker-compose exec api python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, 内存: {psutil.virtual_memory().percent}%')"

# 重建索引
docker-compose exec api python run_rag_system.py --rebuild-index
```

## 🤝 贡献

我们欢迎贡献！请查看我们的[贡献指南](CONTRIBUTING.md)了解详情。

### 开发工作流

1. Fork仓库
2. 创建功能分支
3. 实现变更并添加测试
4. 提交带有清晰描述的PR
5. 确保CI/CD通过

## 📝 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 🔗 资源链接

- **文档**: [完整文档](docs/)
- **API参考**: [OpenAPI文档](http://localhost/docs)
- **监控**: [Grafana仪表板](http://localhost:3001)
- **问题**: [GitHub Issues](https://github.com/your-username/rag-ai/issues)
- **讨论**: [GitHub Discussions](https://github.com/your-username/rag-ai/discussions)

## 📈 开发路线图

### 版本 2.1 (计划中)
- [ ] Next.js前端实现
- [ ] 高级知识图谱功能
- [ ] 多模态文档支持(图片、表格)
- [ ] 实时协作功能

### 版本 2.2 (未来)
- [ ] 微调管道集成
- [ ] 高级评估框架
- [ ] 多语言支持扩展
- [ ] 企业SSO集成

## ❓ 常见问题

### Q: 如何更换嵌入模型？
A: 在`.env`文件中修改`EMBEDDING_MODEL`，然后重新构建索引:
```bash
docker-compose exec api python run_rag_system.py --rebuild-index
```

### Q: 如何添加新的数据源？
A: 修改`src/data_ingestion/multi_source_collector.py`，在`blog_feeds`字典中添加新的RSS源。

### Q: 如何优化检索性能？
A: 
1. 调整缓存配置提高命中率
2. 使用过滤器减少搜索范围
3. 增加Qdrant集群节点
4. 优化查询向量化

### Q: 如何处理中文查询？
A: 系统已支持中文，使用BGE-M3多语言模型。确保查询文本编码为UTF-8。

---

## 🎉 致谢

使用现代AI和Web技术构建:
- **FastAPI** 高性能API框架
- **Qdrant** 向量相似度搜索  
- **Redis** 分布式缓存
- **Prometheus & Grafana** 监控分析
- **Docker** 容器化技术
- **Nginx** 反向代理

**📧 联系**: 如有问题或需要支持，请提交issue或联系维护者。

---

*RAG-AI v2.0 - 为下一代学术研究和知识发现提供动力* 🚀