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

# Launch Streamlit web interface
streamlit run app.py

# Background Qdrant vector database (required)
./qdrant --storage-path ./storage
```

### Development and Testing
```bash
# No formal test framework configured
# Testing is done through run_rag_system.py and app.py
# Evaluation pipeline: python -c "from src.evaluation.evaluation_pipeline import EvaluationPipeline; EvaluationPipeline().run_evaluation()"
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
- **`app.py`**: Streamlit web interface for end-user interaction
- **`setup.py`**: Interactive environment setup script

## Important Implementation Notes

- **Chinese Language Focus**: Most UI text, prompts, and documentation are in Chinese
- **Environment Variables**: System heavily relies on `.env` configuration file
- **Qdrant Dependency**: Requires running Qdrant vector database instance
- **HuggingFace Integration**: Uses HF Hub for model downloads, requires token for some models
- **Modular Design**: Each component can operate independently via `src/generation/ultimate_rag_system.py`

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
├── data/           # Raw and processed documents
├── models/         # HuggingFace model cache
├── logs/           # System logs
└── qdrant_storage/ # Vector database storage
```