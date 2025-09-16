# RAG-AI 项目总览

现代化的 Retrieval-Augmented Generation (RAG) 系统，面向 AI 技术资讯检索与问答场景。
整个项目已经围绕真实的工程代码实现了从数据采集、处理、索引构建、检索生成、知识图谱
增强、评估监控到前端呈现的完整链路。本 README 将帮助你理解核心流程、快速运行项目，并
指向更深入的模块文档与源码位置。

## 📦 仓库结构速览

```
├── app.py                     # Streamlit 前端
├── run_rag_system.py          # 一键式管线入口
├── configs/config.py          # 配置中心，统一读取 .env
├── docs/                      # 各子系统说明文档
└── src/
    ├── data_ingestion/        # 多源异步采集器
    ├── processing/            # 文本切分、多表示索引
    ├── retrieval/             # 混合检索、查询智能、Agentic
    ├── generation/            # RAG 主流程与分层生成
    ├── knowledge_graph/       # 知识图谱抽取与检索增强
    ├── evaluation/            # RAGAS / TruLens 综合评估
    └── optimization/          # 模型共享与性能优化工具
```

## 🚀 快速上手

### 1. 准备环境

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
pip install -r requirements.txt
```

建议在项目根目录创建 `.env` 并至少填入：

```
STORAGE_ROOT=./project_data        # 数据、日志、模型等默认落地位置
HUGGING_FACE_TOKEN=你的HF令牌      # 需要访问受限模型或API时配置
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

> `configs/config.py` 会在导入时自动加载 `.env`，并把 `STORAGE_ROOT` 下的 data、logs 等目录补齐。

### 2. 初始化与运行

项目提供 `run_rag_system.py` 作为流水线总控：

```bash
python run_rag_system.py                # 完整执行：环境检查 → 采集 → 处理 → 建库 → 测试 → 前端
python run_rag_system.py --quick        # 复用已有数据，跳过采集
python run_rag_system.py --frontend-only --port 8501
```

常用参数：

| 参数 | 作用 |
| ---- | ---- |
| `--skip-check` | 跳过依赖和 Qdrant 自检 |
| `--skip-collect` / `--skip-process` / `--skip-build` | 分别跳过数据采集、文本处理、向量建库 |
| `--test` | 在完成建库后用 `RAGSystem.generate_answer` 做回归测试 |
| `--no-frontend` | 仅执行离线阶段，不拉起 Streamlit |

### 3. 启动前端

流水线执行完成后，可单独启动 Web 界面：

```bash
streamlit run app.py
```

前端会通过 `VectorDatabaseManager` 检查 Qdrant 集合是否已填充，再缓存初始化好的 `RAGSystem`
以提供对话式体验。界面会展示生成结果、置信度、引用片段等指标。

## 🔄 端到端工作流

### 1. 数据采集：`MultiSourceCollector`

`src/data_ingestion/multi_source_collector.py` 内的 `MultiSourceCollector.collect_all` 异步抓取三类来源：

- ArXiv：解析 Atom Feed、下载 PDF 并用 `pymupdf4llm`/PyMuPDF 抽取正文。
- Hugging Face Papers：通过 `HfApi.list_papers` 获取热门论文元信息。
- 主流 AI 博客：配置 RSS 列表（Google AI、OpenAI、BAIR、DeepMind 等）。

采集结果统一序列化为 `raw_collected_data.json`，同时缓存已处理 ID、复用 `aiohttp` 会话以提高吞吐。

### 2. 文本处理：`EnhancedTextProcessor`

`src/processing/text_processor.py` 将原始文档转换为可检索的文本块：

1. `HierarchicalTextSplitter` 先按章节正则再递归切分，保留章节名和序号元信息。
2. `MultilingualEmbedder` 通过 `ModelRegistry.get_sentence_transformer` 共享加载 BGE-M3，批量生成向量。
3. 若配置启用多表示（默认开启），`MultiRepresentationIndexer` 会并发调用共享 LLM 生成摘要与假设问题，
   并再次编码，形成 `original/summary/hypothetical_question` 多条索引记录。
4. 结果写入 `processed_chunks.json`，并携带 embedding、语义类型、原始元数据等字段。

### 3. 向量知识库：`VectorDatabaseManager`

`src/retrieval/vector_database.py` 封装了 Qdrant 连接、集合创建与批量写入：

- `QdrantVectorDB.add_chunks` 按批 upsert，附带 `text_tokens` 字段实现密集向量 + 简易中文滑窗分词的混合检索。
- `hybrid_search` 将向量得分与关键词命中进行加权融合，同时支持按 `semantic_type` 等字段过滤。
- 构建完成后可通过 `get_collection_stats` 查看向量维度、数量、索引状态。

### 4. 查询理解与检索

- `EnhancedQueryProcessor`（`src/generation/rag_generator.py`）使用共享嵌入模型缓存查询向量，
  并在配置允许时委托 `QueryIntelligenceEngine`（`src/retrieval/query_intelligence.py`）分析语言、复杂度、查询类型，
  生成重写版本、子问题以及 HyDE 假设文档。
- `VectorDatabaseManager.search` 对每个优化查询执行混合检索，并记录来源、权重。系统会进行去重，
  避免重复片段污染上下文。

### 5. 上下文优化与生成

- `ContextOptimizer`（`EnhancedContextOptimizer`）在启用时结合 `SmartReranker`、`ContextualCompressor`
  做多信号重排序与上下文压缩，控制最终拼接长度。
- `LLMGenerator` 复用 `ModelRegistry.get_llm` 加载 Hugging Face 上的指令模型，按照 prompt 模版生成答案。
- `EnhancedRAGSystem.generate_answer` 汇总过程耗时、检索策略、置信度等信息返回给前端或调用方。

### 6. Agentic 检索循环（可选）

`src/retrieval/agentic_rag.py` 提供 `AgenticRAGOrchestrator`：

- `RetrievalEvaluator` 通过共享 LLM 评估检索片段的相关性、完整性、矛盾点并给出下一步动作建议。
- 根据决策自动扩展查询、追加检索或终止循环，并记录 `AgenticStep` 供分析。
- `generate_answer_agentic` 将多轮检索结果与生成输出整合，适合难题或需要多跳推理的场景。

### 7. 知识图谱增强

- `KnowledgeGraphIndexer`（`src/knowledge_graph/knowledge_extractor.py`）抽取实体、关系，落地到 SQLite + NetworkX。
- `KnowledgeGraphRetriever` 和 `KGEnhancedRetriever` 在检索阶段补充结构化上下文，
  并向前端返回识别到的实体、关系以提升可解释性。

### 8. 评估与持续优化

`src/evaluation/comprehensive_evaluation.py` 构建统一评估入口：

- `RAGASEvaluator`、`TruLensEvaluator` 会检测依赖是否安装，缺失时记录模拟分数并给出提醒。
- `ComprehensiveEvaluator.evaluate_single_case / evaluate_golden_dataset` 支持单例及黄金集合评估，
  输出 faithfulness、relevancy、groundedness 等指标、建议与汇总报告。

性能层面，`src/optimization/model_registry.py` 通过全局缓存复用所有向量模型与大模型，
避免重复加载导致的 GPU / 显存浪费。

## ⚙️ 配置与存储

- 默认数据目录位于 `STORAGE_ROOT` 下：`data/raw`、`data/processed`、`logs`、`evaluation`、`knowledge_graph` 等。
- 可在 `.env` 中覆盖嵌入模型、LLM、Qdrant 参数、功能开关（例如 `ENABLE_MULTI_REPRESENTATION`、
  `ENABLE_AGENTIC_RAG`、`ENABLE_CONTEXTUAL_COMPRESSION`）。
- Hugging Face 模型缓存目录可通过 `HF_HOME` 指定，`config.py` 会同步设置 `TRANSFORMERS_CACHE` 等环境变量。

## 📚 深入阅读

项目为每个子系统准备了独立文档，位于 `docs/` 或对应模块目录：

- `docs/API_REFERENCE.md` – 主要 Python API 与调用示例。
- `docs/pipeline/README.md` – `run_rag_system.py` 全流程解析。
- `src/data_ingestion/README.md`、`src/processing/README.md`、`src/retrieval/README.md` 等 – 深入说明实现细节。
- `docs/LEARNING_GUIDE.md` – 针对初学者的完整学习路径与工程优化建议。

## 🧪 开发与测试

建议在提交前运行：

```bash
python -m compileall src
```

如果需要检查 Streamlit 前端或 Agentic 流程，可结合 README 中的步骤拉起服务并进行手工验收。

---

欢迎通过 Issue / PR 贡献更多改进。若本项目对你有帮助，别忘了点亮 ⭐️！

