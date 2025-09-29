# RAG-AI: Enterprise-Grade Retrieval-Augmented Generation System

> **🎉 Version 2.0 - Major Architecture Upgrade Complete!**

A state-of-the-art RAG system built specifically for academic paper research and technical documentation. This project implements advanced features including **personalized recommendations**, **user profiling**, **storage optimization**, **metadata-first data collection**, **hybrid search**, **multi-layer caching**, comprehensive citation management, modern frontend interfaces, and production-ready deployment infrastructure.

## 🌟 Key Features

### 🔬 **Academic-Focused RAG System**
- **Multi-source data collection**: ArXiv, Hugging Face Papers, AI research blogs
- **Metadata-first strategy**: Optimized storage with on-demand full-text retrieval
- **Academic citation system**: APA, MLA, BibTeX, IEEE format generation
- **Source traceability**: Full academic integrity and citation tracking
- **Personalized recommendations**: Daily AI-curated content based on user interests
- **User profiling**: Intelligent tracking of research interests and preferences

### 🚀 **Enterprise Architecture** 
- **Enhanced Streamlit frontend**: Personalized interface with user dashboards
- **Enhanced FastAPI backend**: Async API with personalization endpoints
- **Multi-tier storage optimization**: Hot/warm/cold/archived data lifecycle
- **Multi-layer caching**: Memory, Redis, file, and vector caches
- **Hybrid search**: Semantic + keyword (BM25) + metadata filtering
- **Microservices deployment**: Docker Compose with Nginx, monitoring

### 🧠 **Advanced AI Capabilities**
- **Intelligent query processing**: Query rewriting, sub-question generation
- **Agentic RAG**: Self-evaluating retrieval with quality feedback loops
- **Tiered generation**: Cost-optimized model routing (local → API)
- **Knowledge graph enhancement**: Entity extraction and graph-based retrieval
- **Recommendation engine**: Content-based and collaborative filtering
- **Usage analytics**: Advanced storage and access pattern analysis

### 📊 **Production-Ready Operations**
- **Comprehensive monitoring**: Prometheus + Grafana dashboards
- **Performance optimization**: Caching, async processing, model sharing
- **Horizontal scaling**: Container orchestration and load balancing
- **CI/CD ready**: Docker configurations for all environments

## 📦 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG-AI System Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (Next.js)     │  API Gateway (Nginx)                  │
├─────────────────────────┼────────────────────────────────────────┤
│  FastAPI Backend        │  Multi-Layer Cache                     │
│  ├─ Async RAG Endpoints │  ├─ Memory (LRU)                      │
│  ├─ Streaming Responses │  ├─ Redis (Distributed)               │
│  └─ Citation Management │  ├─ File (Persistent)                 │
│                         │  └─ Vector (Specialized)              │
├─────────────────────────┼────────────────────────────────────────┤
│  Vector Database        │  Knowledge Graph                      │
│  ├─ Qdrant (Hybrid)     │  ├─ Entity Extraction                 │
│  ├─ Semantic Search     │  ├─ Relationship Mapping              │
│  ├─ BM25 + Filtering    │  └─ Graph-Enhanced Retrieval          │
│  └─ Academic Metadata   │                                        │
├─────────────────────────┼────────────────────────────────────────┤
│  Data Collection        │  Monitoring & Observability           │
│  ├─ Metadata-First      │  ├─ Prometheus Metrics                │
│  ├─ On-Demand PDF       │  ├─ Grafana Dashboards                │
│  ├─ Daily Incremental   │  ├─ Performance Tracking              │
│  └─ Multi-Source Async  │  └─ Error Analytics                   │
└─────────────────────────────────────────────────────────────────┘
```

## 🗂️ Project Structure

```
rag-ai/
├── 🚀 Deployment & Configuration
│   ├── docker-compose.yml         # Production deployment
│   ├── docker-compose.dev.yml     # Development environment  
│   ├── Dockerfile.*               # Container configurations
│   ├── nginx/nginx.conf           # Reverse proxy setup
│   └── monitoring/                # Prometheus & Grafana configs
│
├── 🔧 Core Application
│   ├── api/enhanced_main.py       # Enhanced FastAPI backend with personalization
│   ├── enhanced_app.py            # Enhanced Streamlit interface with personalization
│   ├── frontend/                  # Next.js frontend application
│   │   ├── src/app/               # Next.js app router pages
│   │   ├── src/components/        # React components
│   │   ├── src/lib/               # Utilities and API client
│   │   └── src/store/             # Zustand state management
│   ├── run_rag_system.py          # System orchestrator
│   └── configs/config.py          # Centralized configuration
│
├── 📚 Source Code Modules
│   └── src/
│       ├── 📥 data_ingestion/     # Multi-source data collection
│       │   └── multi_source_collector.py  # Metadata-first collector
│       ├── 🏗️ processing/          # Text processing & indexing
│       │   ├── text_processor.py          # Enhanced text processing
│       │   └── multi_representation_indexer.py
│       ├── 🔍 retrieval/          # Hybrid search & retrieval
│       │   ├── vector_database.py         # Enhanced Qdrant integration
│       │   ├── query_intelligence.py      # Query processing
│       │   └── agentic_rag.py            # Self-evaluating retrieval
│       ├── 🤖 generation/         # Answer generation
│       │   ├── enhanced_rag_system.py    # Enhanced RAG with personalization
│       │   ├── ultimate_rag_system.py    # Main RAG orchestrator
│       │   └── tiered_generation.py      # Cost-optimized routing
│       ├── 👤 personalization/    # 🆕 User personalization
│       │   ├── user_profiler.py          # User profile management
│       │   ├── recommendation_engine.py  # Daily recommendations
│       │   └── preference_tracker.py     # Interest tracking
│       ├── 💾 storage/            # 🆕 Storage optimization
│       │   ├── storage_optimizer.py      # Multi-tier optimization
│       │   ├── usage_analytics.py        # Access pattern analysis
│       │   └── data_lifecycle.py         # Automated lifecycle management
│       ├── 📖 citation/           # Citation management
│       │   └── citation_manager.py       # Academic citation system
│       ├── 💾 caching/            # Multi-layer caching
│       │   └── multilayer_cache.py       # Advanced caching system
│       ├── 📊 monitoring/         # Performance monitoring
│       │   └── metrics_collector.py      # Prometheus integration
│       ├── 🧠 knowledge_graph/    # Knowledge enhancement
│       ├── 📈 evaluation/         # System evaluation
│       └── ⚡ optimization/       # Performance optimization
│
└── 📖 Documentation
    ├── README.md                  # This comprehensive guide
    └── docs/                      # Detailed module documentation
```

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/rag-ai.git
cd rag-ai

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# Access the system
# - Frontend Interface: http://localhost
# - API Documentation: http://localhost/docs  
# - Grafana Monitoring: http://localhost:3001
```

### Option 2: Development Setup

```bash
# 1. Environment setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# 2. Start vector database
docker run -d -p 6333:6333 qdrant/qdrant:v1.7.0

# 3. Start Redis (optional, for caching)
docker run -d -p 6379:6379 redis:7.2-alpine

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Initialize system
python run_rag_system.py

# 6. Start enhanced API server (in one terminal)
uvicorn api.enhanced_main:app --host 0.0.0.0 --port 8000 --reload

# 7. Start enhanced Streamlit interface (in another terminal)
streamlit run enhanced_app.py

# Alternative: Start frontend development server
cd frontend
npm install
npm run dev
```

### Option 3: Development with Docker

```bash
# Use development compose configuration
docker-compose -f docker-compose.dev.yml up -d

# This provides:
# - Hot reload for backend and frontend
# - Volume mounts for development
# - Debug logging enabled
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Storage Configuration
STORAGE_ROOT=./project_data
HF_HOME=./project_data/models
HUGGING_FACE_TOKEN=your_hf_token_here

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=ai_papers

# Cache Configuration  
REDIS_HOST=localhost
REDIS_PORT=6379
ENABLE_CACHE=true

# Model Configuration
EMBEDDING_MODEL=BAAI/bge-m3
LLM_MODEL=Qwen/Qwen2-7B-Instruct
DEVICE=auto

# API Keys (for tiered generation)
GPT4_API_KEY=your_openai_key
CLAUDE_API_KEY=your_claude_key

# Feature Toggles
ENABLE_HYBRID_SEARCH=true
ENABLE_AGENTIC_RAG=true
ENABLE_KNOWLEDGE_GRAPH=true
ENABLE_TIERED_GENERATION=true
```

## 🌟 Enhanced Features in Version 2.0

### 👤 **Personalization System**
- **User profiling**: Intelligent tracking of research interests and interaction patterns
- **Daily recommendations**: AI-curated content based on user preferences
- **Content filtering**: Hybrid recommendation engine with collaborative and content-based filtering
- **User dashboard**: Personalized analytics and recommendation management

```python
# Example: User profiling and recommendations
user_profiler = UserProfiler()
profile = await user_profiler.get_or_create_user_profile(user_id)

recommendation_engine = RecommendationEngine()
recommendations = await recommendation_engine.generate_daily_recommendations(
    user_id=user_id,
    limit=10,
    days_back=7
)
```

### 💾 **Storage Optimization System**
- **Multi-tier storage**: Hot/warm/cold/archived data lifecycle management
- **Usage analytics**: Advanced access pattern analysis and optimization
- **Automated migration**: Smart data movement based on access patterns
- **Cost optimization**: Efficient storage utilization with performance monitoring

```python
# Example: Storage optimization
storage_optimizer = StorageOptimizer()
analytics = UsageAnalytics()

# Analyze access patterns
patterns = await analytics.analyze_access_patterns(days=30)

# Optimize storage based on patterns
optimization_result = await storage_optimizer.optimize_storage(
    target_hot_ratio=0.1,
    target_warm_ratio=0.3
)
```

### 📄 **Metadata-First Data Collection**
- **Intelligent caching**: Only fetch full PDFs when needed
- **Daily incremental updates**: Efficient data pipeline
- **Multi-source async collection**: ArXiv, HuggingFace, blogs
- **Citation-ready metadata**: Academic compliance built-in

```python
# Example: On-demand full-text retrieval
collector = MultiSourceCollector(data_dir, metadata_only=True)
full_text = await collector.fetch_full_text_on_demand(document_id)
```

### 🔄 **Multi-Layer Caching System**
- **4-tier architecture**: Memory → Redis → File → Vector caches
- **Intelligent cache promotion**: Frequently accessed data moves up
- **Specialized caches**: Separate handling for different data types
- **Cache analytics**: Hit rates and performance monitoring

```python
# Example: Using the caching system
cache = create_multilayer_cache(config)
await cache.cache_query_embedding(query, embedding)
results = await cache.get_search_results(query_hash)
```

### 📚 **Academic Citation Management**
- **Multiple formats**: APA, MLA, BibTeX, IEEE, Chicago
- **Source verification**: Link validation and accessibility checks
- **Usage tracking**: Citation popularity and trends
- **Bibliography export**: Automated reference list generation

```python
# Example: Generate citations
citation_manager = CitationManager(data_dir)
apa_citation = citation_manager.generate_citation(source_id, "apa")
source_links = citation_manager.generate_source_links(source_id)
```

### 🔍 **Enhanced Hybrid Search**
- **Semantic + Keyword**: Vector similarity + BM25 scoring
- **Academic filtering**: Author, year, journal, category filters
- **Query intelligence**: Automatic query rewriting and expansion
- **Performance optimized**: Distributed indexing and caching

```python
# Example: Advanced academic search
results = await db.advanced_academic_search(
    query_vector=embedding,
    query_text=query,
    authors=["Bengio", "LeCun"],
    year_range=(2020, 2024),
    sources=["arxiv"],
    categories=["cs.AI", "cs.LG"]
)
```

### 🚀 **Production-Ready API**
- **FastAPI backend**: High-performance async API
- **Streaming responses**: Real-time answer generation
- **WebSocket support**: Live updates and notifications
- **OpenAPI documentation**: Interactive API explorer

```bash
# API Endpoints
POST /ask              # Main Q&A endpoint
POST /ask/stream       # Streaming responses
POST /search           # Document search
GET  /document/{id}    # Document retrieval
POST /feedback         # User feedback
GET  /stats           # System statistics
```

### 🖥️ **Modern Frontend Interface**
- **Next.js 14**: App Router with React Server Components
- **TypeScript**: Full type safety and developer experience
- **Tailwind CSS**: Responsive design and modern UI components
- **Real-time chat**: Streaming responses with Server-Sent Events
- **State management**: Zustand for efficient client state
- **Comprehensive search**: Advanced filtering and citation export

```tsx
// Example: Real-time streaming chat
const stream = apiClient.askQuestionStream({
  query: "Explain transformer architecture",
  rag_mode: "ultimate",
  include_sources: true
});

for await (const chunk of stream) {
  if (chunk.type === 'content') {
    updateMessage(chunk.content);
  }
}
```

## 📊 Monitoring & Observability

### Prometheus Metrics
- **Request metrics**: Latency, throughput, error rates
- **System metrics**: CPU, memory, cache performance
- **Business metrics**: Query types, citation usage
- **Custom dashboards**: Grafana visualization

### Health Checks
```bash
# Service health endpoints
curl http://localhost/health           # Overall system
curl http://localhost/api/health       # API server
curl http://localhost:6333/health      # Vector database
```

## 🎯 Usage Examples

### Basic Q&A
```python
import aiohttp
import asyncio

async def ask_question():
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost/api/ask",
            json={
                "query": "What are the latest developments in transformer models?",
                "max_results": 5,
                "include_sources": True,
                "rag_mode": "ultimate"
            }
        ) as response:
            result = await response.json()
            print(f"Answer: {result['answer']}")
            print(f"Sources: {len(result['sources'])}")
            for source in result['sources']:
                print(f"- {source['citation']}")
```

### Streaming Responses
```javascript
// Frontend streaming example
const eventSource = new EventSource('http://localhost/api/ask/stream');
eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'content') {
        document.getElementById('answer').innerHTML += data.content;
    }
};
```

### Advanced Search
```python
# Search with academic filters
response = await client.post("/api/search", json={
    "query": "attention mechanism in neural networks",
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

## 🔧 Development Guide

### Adding New Features

1. **Create feature branch**
```bash
git checkout -b feature/new-enhancement
```

2. **Implement with monitoring**
```python
from src.monitoring import monitor_async_performance

@monitor_async_performance("component", "operation")
async def new_feature():
    # Implementation with automatic metrics
    pass
```

3. **Add tests**
```bash
pytest tests/test_new_feature.py -v
```

4. **Update documentation**
```markdown
## New Feature
Description and usage examples...
```

### Performance Optimization

The system includes several optimization layers:

1. **Model caching**: Shared model instances across requests
2. **Query caching**: LRU cache for similar queries  
3. **Result caching**: Redis-backed response caching
4. **Vector caching**: Specialized embedding storage
5. **Connection pooling**: Efficient database connections

## 🚀 Deployment

### Production Deployment

```bash
# 1. Clone and configure
git clone https://github.com/your-username/rag-ai.git
cd rag-ai
cp .env.example .env
# Edit .env with production settings

# 2. Deploy with monitoring
docker-compose up -d

# 3. Initialize data
docker-compose exec api python run_rag_system.py --setup

# 4. Verify deployment
curl http://your-domain/health
```

### Scaling Considerations

- **API instances**: Scale FastAPI horizontally behind load balancer
- **Vector database**: Use Qdrant clusters for large datasets
- **Cache layer**: Redis cluster for high availability
- **Background jobs**: Separate collector service instances

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request with clear description
5. Ensure CI/CD passes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

- **Documentation**: [Full documentation](docs/)
- **API Reference**: [OpenAPI Docs](http://localhost/docs)
- **Monitoring**: [Grafana Dashboards](http://localhost:3001)
- **Issues**: [GitHub Issues](https://github.com/your-username/rag-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/rag-ai/discussions)

## 📈 Roadmap

### Version 2.1 (Planned)
- [x] **Next.js frontend implementation** ✅ **COMPLETED**
  - Modern React interface with TypeScript
  - Real-time chat with Server-Sent Events streaming
  - Document search and citation management
  - System monitoring dashboard
  - Responsive design with Tailwind CSS
- [ ] Advanced knowledge graph features
- [ ] Multi-modal document support (images, tables)
- [ ] Real-time collaboration features

### Version 2.2 (Future)
- [ ] Fine-tuning pipeline integration
- [ ] Advanced evaluation frameworks
- [ ] Multi-language support expansion
- [ ] Enterprise SSO integration

---

## 🎉 Acknowledgments

Built with modern AI and web technologies:
- **FastAPI** for high-performance API
- **Qdrant** for vector similarity search  
- **Redis** for distributed caching
- **Prometheus & Grafana** for monitoring
- **Docker** for containerization
- **Nginx** for reverse proxy

**📧 Contact**: For questions or support, please open an issue or reach out to the maintainers.

---

*RAG-AI v2.0 - Powering the next generation of academic research and knowledge discovery* 🚀