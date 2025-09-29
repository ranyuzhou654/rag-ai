# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Plan & Review

### Before Starting Work
- Always in plan mode to make a plan.
- After get the plan, make sure you Write the plan to .claude/tasks/TASK_NAME.md
- The plan should be a detailed implementation plan and the reasoning behind them. As well as tasks broken down.
- If the task requires external knowledge or certain packages, also research to get latest knowledge. (Use Task tool for research)
- Don't over plan it, always think MVP.
- Once the plan is done, ask for approval from the user before starting work.

### While implementing
- You should update the plan as you work.
- After you complete tasks in the plan, you should update and append detailed descriptions of the change you made, so following tasks can be easily hand over to other engineers.

## Project Overview

This is an **Ultimate RAG (Retrieval-Augmented Generation) System** - a Chinese-language AI question-answering system designed for technical documentation and research papers. The system features multi-modal retrieval, knowledge graphs, agentic workflows, and tiered generation capabilities.

## Common Development Commands

### Environment Setup
```bash
# Initial setup (interactive configuration)
python setup.py

# Install dependencies
pip install -r requirements.txt
```

### Running the System
```bash
# Data collection and indexing
python run_rag_system.py

# Launch enhanced Streamlit web interface with personalization
streamlit run enhanced_app.py

# Background Qdrant vector database (required)
./qdrant --storage-path ./storage
```

### Development and Testing
```bash
# No formal test framework configured
# Testing is done through run_rag_system.py and app.py
# Evaluation pipeline: python -c "from src.evaluation.evaluation_pipeline import EvaluationPipeline; EvaluationPipeline().run_evaluation()"

# Code compilation check
python -m compileall src

# Qdrant vector database (run in background, required for system operation)
./qdrant --storage-path ./storage
```

## High-Level Architecture

The system follows a 7-layer architecture:

1. **Query Intelligence Layer** (`src/retrieval/query_intelligence.py`)
   - Query complexity analysis and rewriting
   - Sub-question generation and HyDE document generation
   - Multi-language support (Chinese/English)

2. **Knowledge Graph Layer** (`src/knowledge_graph/`)
   - Entity and relationship extraction for AI domain
   - Graph-based retrieval enhancement
   - NetworkX + SQLite storage

3. **Multi-Representation Indexing** (`src/processing/multi_representation_indexer.py`)
   - Original content, summaries, and hypothetical questions
   - Vector storage in Qdrant database

4. **Agentic RAG Layer** (`src/retrieval/agentic_rag.py`)
   - Self-evaluating retrieval with retry loops
   - Quality assessment and query refinement

5. **Context Optimization** (`src/retrieval/contextual_compression.py`, `src/retrieval/reranker.py`)
   - Intelligent reranking with multiple signals
   - Context compression for LLM efficiency

6. **Tiered Generation** (`src/generation/tiered_generation.py`)
   - Local models (Qwen2-7B) for simple tasks
   - API models (GPT-4, Claude) for complex reasoning
   - Cost-optimized model routing

7. **Feedback & Learning** (`src/feedback/feedback_system.py`, `src/training/embedding_fine_tuner.py`)
   - User feedback collection and analysis
   - Embedding model fine-tuning pipeline

## Key Configuration Files

- **`configs/config.py`**: Main configuration with environment variable loading
- **`.env`**: Environment variables (storage paths, API keys, model settings)
- **`requirements.txt`**: Python dependencies including transformers, qdrant-client, streamlit

## Entry Points

- **`run_rag_system.py`**: Data collection, indexing, and system initialization
- **`enhanced_app.py`**: Enhanced Streamlit web interface with personalization features  
- **`api/enhanced_main.py`**: Enhanced FastAPI backend with personalization endpoints
- **`setup.py`**: Interactive environment setup script

## Enhanced Features (New in v2.0)

### 🎯 Personalization System
- **User Profiling** (`src/personalization/user_profiler.py`): Intelligent tracking of research interests and interaction patterns
- **Recommendation Engine** (`src/personalization/recommendation_engine.py`): Daily AI-curated content with hybrid filtering  
- **Preference Tracking** (`src/personalization/preference_tracker.py`): Automated learning from user behavior

### 💾 Storage Optimization  
- **Multi-tier Storage** (`src/storage/storage_optimizer.py`): Hot/warm/cold/archived data lifecycle management
- **Usage Analytics** (`src/storage/usage_analytics.py`): Advanced access pattern analysis and optimization
- **Data Lifecycle** (`src/storage/data_lifecycle.py`): Automated data migration and compression

### 🚀 Enhanced RAG Engine
- **Enhanced RAG System** (`src/generation/enhanced_rag_system.py`): Integrated personalization with traditional RAG
- **Personalized Query Processing**: Context-aware query enhancement based on user profile
- **Smart Retrieval**: User preference-weighted document ranking and filtering

## Key Implementation Details

### User Profiling Architecture
```python
# Core data structures
@dataclass
class UserProfile:
    user_id: str
    research_interests: List[ResearchInterest]
    interaction_history: List[UserInteraction]
    preferences: Dict[str, Any]
    total_queries: int = 0
    avg_session_duration: float = 0.0

# Automatic interest extraction
async def extract_research_interests(query: str, response: str) -> List[ResearchInterest]:
    # NLP-based keyword and concept extraction
    # Weighted scoring based on interaction context
```

### Storage Optimization Strategy
```python  
# Multi-tier storage classification
class StorageTier(Enum):
    HOT = "hot"        # High-frequency access, SSD storage
    WARM = "warm"      # Medium-frequency access, hybrid storage  
    COLD = "cold"      # Low-frequency access, HDD storage
    ARCHIVED = "archived"  # Compressed long-term storage

# Intelligent migration based on access patterns
async def optimize_storage(target_hot_ratio: float = 0.1):
    patterns = await analyze_access_patterns(days=30)
    migration_plan = generate_migration_plan(patterns, target_hot_ratio)
    return await execute_migration(migration_plan)
```

### Recommendation Algorithm
```python
# Hybrid recommendation approach
async def generate_daily_recommendations(user_id: str, limit: int = 10):
    profile = await get_user_profile(user_id)
    
    # Content-based filtering
    content_recs = await content_based_recommendations(profile, limit // 2)
    
    # Collaborative filtering  
    collaborative_recs = await collaborative_filtering(profile, limit // 2)
    
    # Trending content
    trending_recs = await trending_recommendations(limit // 4)
    
    # Score and rank combined results
    return await rank_recommendations(content_recs + collaborative_recs + trending_recs)
```

## Important Implementation Notes

- **Chinese Language Focus**: Most UI text, prompts, and documentation are in Chinese
- **Environment Variables**: System heavily relies on `.env` configuration file (auto-loaded by `configs/config.py`)
- **Qdrant Dependency**: Requires running Qdrant vector database instance before system operations
- **HuggingFace Integration**: Uses HF Hub for model downloads, requires token for some models
- **Modular Design**: Each component can operate independently via `src/generation/ultimate_rag_system.py`
- **Model Caching**: All models are cached via `src/optimization/model_registry.py` to avoid duplicate loading
- **Feature Toggles**: Most advanced features can be disabled via environment variables (e.g., `ENABLE_AGENTIC_RAG`, `ENABLE_KNOWLEDGE_GRAPH`)
- **AutoDL Support**: `setup.py` provides interactive configuration optimized for AutoDL cloud environments
- **Async Architecture**: Data collection uses async/await patterns with `aiohttp` for improved performance

## Data Flow

1. Documents ingested via `src/data_ingestion/multi_source_collector.py`
2. Text processed through `src/processing/text_processor.py`
3. Multiple representations created and indexed
4. Queries processed through intelligence layer
5. Retrieval via vector search + knowledge graph enhancement
6. Context optimization and reranking
7. Tiered generation based on query complexity
8. Feedback collection for continuous improvement

## Storage Structure

```
STORAGE_ROOT/
├── data/
│   ├── raw/        # Raw collected documents
│   └── processed/  # Processed text chunks
├── models/         # HuggingFace model cache (controlled by HF_HOME)
├── logs/           # System logs with rotation
├── evaluation/     # Evaluation results and golden test sets
├── feedback/       # User feedback database
├── knowledge_graph/ # SQLite KG database
└── qdrant_storage/ # Vector database storage
```

## Command Line Arguments

The `run_rag_system.py` script supports several arguments for flexible execution:

```bash
python run_rag_system.py                     # Full pipeline execution
python run_rag_system.py --quick             # Skip data collection, use existing data
python run_rag_system.py --frontend-only     # Launch only Streamlit interface
python run_rag_system.py --skip-check        # Skip dependency and Qdrant checks
python run_rag_system.py --skip-collect      # Skip data collection phase
python run_rag_system.py --skip-process      # Skip text processing phase
python run_rag_system.py --skip-build        # Skip vector database building
python run_rag_system.py --test              # Run regression tests after building
python run_rag_system.py --no-frontend       # Run offline stages only
python run_rag_system.py --port 8501         # Specify Streamlit port
```

## Docker Development

### Production Deployment
```bash
# Start all services (includes Qdrant, Redis, API, Frontend, Nginx, Monitoring)
docker-compose up -d

# Check service status
docker-compose ps

# Access services:
# - Frontend Interface: http://localhost
# - API Documentation: http://localhost/docs  
# - Grafana Monitoring: http://localhost:3001
```

### Development Environment
```bash
# Development with hot reload
docker-compose -f docker-compose.dev.yml up -d

# Start individual services
docker-compose up qdrant redis  # Dependencies only
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev        # Development server
npm run build      # Production build
npm run lint       # Linting
npm run type-check # TypeScript checking
```

## Production Architecture

The system supports full containerized deployment with:

- **Enhanced FastAPI Backend** (`api/enhanced_main.py`): Async API with personalization and streaming responses
- **Next.js Frontend** (`frontend/`): Modern React interface with TypeScript
- **Nginx Reverse Proxy**: Load balancing and SSL termination
- **Monitoring Stack**: Prometheus + Grafana for observability
- **Multi-layer Caching**: Memory, Redis, file, and vector caches
- **Background Data Collection**: Automated daily updates

## System Monitoring

- **Health Checks**: Available at `/health` endpoints
- **Metrics**: Prometheus metrics collection
- **Logging**: Structured logging with Loguru
- **Performance**: Real-time monitoring via Grafana dashboards