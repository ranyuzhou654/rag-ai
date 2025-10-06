# RAG-AI 学习指南（2025 版）

欢迎来到 RAG-AI！这份指南面向第一次接触项目的同学，帮你在一周内迅速上手：先弄清系统做什么，再逐层阅读代码、跑通流程、尝试扩展。阅读时请配合源码与 README 一起对照。

---

## 1. 我们要实现什么？

RAG-AI 是一套面向学术论文的检索增强生成系统。它提供：

- **多源采集**：ArXiv、OpenAlex、Hugging Face Papers、AI 博客，支持历史回溯 + 每日增量。
- **元数据优先**：先存标题/摘要/概念标签，按需下载 PDF 减少成本；自动对接 Semantic Scholar TLDR。
- **混合检索**：向量 + BM25 + 概念匹配，配合 Cross-Encoder 精排、上下文压缩。
- **个性化与推荐**：根据用户画像、兴趣向量生成每日主题推荐。
- **多层缓存与监控**：内存/Redis/文件缓存 + Prometheus 指标，适合真实上线。

### 架构速览

```
数据源 → MultiSourceCollector → TextProcessor / MultiRepresentationIndexer
       → VectorDatabaseManager (Qdrant / Milvus) → Retrieval Pipeline
       → EnhancedContextOptimizer → TieredGenerationSystem → 前端与反馈
```

建议先阅读 `README.md` 的“Architecture Overview”，再返回本指南逐节深入。

---

## 2. 第一天下来的任务：跑通流程

1. **准备环境**
   - Python ≥ 3.9，Node.js ≥ 18，Docker（可选）。
   - `cp .env.example .env` 或执行 `python setup.py` 自动生成 `.env`。
   - 按实际情况填写 `.env` 中的 `HUGGING_FACE_TOKEN`、`ENABLE_SEMANTIC_SCHOLAR` 等。

2. **安装依赖**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **启动 Qdrant（单机）**
   ```bash
   docker run -d -p 6333:6333 -v $(pwd)/qdrant:/qdrant/storage qdrant/qdrant:v1.7.3
   ```

4. **一键脚本**
   ```bash
   python run_rag_system.py --days-back 7 --max-papers 200
   ```

   该脚本会依次：环境检查 → 采集 → 处理 → 写入向量库 → 运行自测 → 启动 Streamlit。

5. **验证**
   - 浏览 `storage/data/raw/metadata_index.json`，确认抓到的论文条目。
   - 访问 `http://localhost:8501`（Streamlit）或 `http://localhost:8000/docs`（FastAPI）。

做到这里说明链路已经跑通。接下来开始逐个模块学习。

---

## 3. 配置中心与常量

文件：[`configs/config.py`](../configs/config.py)

- `Config` 类负责读取 `.env`，设置存储根目录、模型名称、向量库参数等。
- 所有 **功能开关** 都集中在这里：`ENABLE_MULTI_REPRESENTATION`、`ENABLE_CONTEXTUAL_COMPRESSION`、`ENABLE_SEMANTIC_SCHOLAR` 等。你在其他模块看到的 `config.ENABLE_...` 就从这里取值。
- 建议边阅读边顺手在 `.env` 修改一些字段（比如把 `STORAGE_CONTENT_MODE` 改为 `summary`），看流程运行时有何不同。

**练习**：在 `.env` 里把 `ENABLE_SEMANTIC_SCHOLAR=false`，重新采集一次，比较 TLDR 字段出现的差异。

---

## 4. 数据采集层

文件：[`src/data_ingestion/multi_source_collector.py`](../src/data_ingestion/multi_source_collector.py)

关键点：

- `Document` 数据类定义统一结构：`id`, `source`, `title`, `content`, `metadata`, `tldr`, `concepts` 等。
- `MultiSourceCollector` 的职责：
  1. **抓取 ArXiv**：通过 Atom API 获取论文元数据，支持窗口滑动、增量更新。
  2. **抓取 OpenAlex**：分页调用 `works` API，并解析概念标签、作者、出版信息。
  3. **Semantic Scholar 增强**：对每个文档尝试补齐 TLDR、完善摘要，结果写入 metadata。
  4. **按需 PDF**：`fetch_full_text_on_demand` 会下载 PDF → 提取文本 → 缓存。
  5. **每日增量**：`daily_incremental_update` 只拉取最近一天数据并更新索引。

- 学习节点：
  - 看 `_collect_from_sources` 如何切换 `metadata_only` 模式。
  - 了解 `_prepare_metadata_enrichment` 生成 metadata 摘要的方式。

**练习**：阅读 `_enrich_with_semantic_scholar` 的实现，尝试调低 `max_total`，观察开销变化。

---

## 5. 文本处理层

文件：
- [`src/processing/text_processor.py`](../src/processing/text_processor.py)
- [`src/processing/multi_representation_indexer.py`](../src/processing/multi_representation_indexer.py)

流程：

1. `EnhancedTextProcessor.process_documents` 依次执行：
   - **切分**：
     - `HierarchicalTextSplitter` 依据章节标题 + 递归字符分割。
     - 或 `SemanticGraphSplitter` 根据句向量相似度切分，适合主题跨度大的文本。
   - **向量化**：复用 BGE-M3 embedder 为 chunk 生成向量。
   - **多表示**（可选）：调用 `MultiRepresentationIndexer` 生成摘要、关键词、假设性问题等，多视角入库。
   - **保存**：输出 JSON，可用于调试或断点续跑。

2. `MultiRepresentationIndexer` 值得重点阅读：
   - 使用共享 LLM（配置可换成 OpenAI/本地模型）。
   - `create_multi_representations` 会批量生成摘要、假设问题，提升检索覆盖面。
   - `generate_index_entries` 负责把不同表示统一打包，写入向量库。

**练习**：在 `.env` 中把 `ENABLE_MULTI_REPRESENTATION=false`，观察向量库中 chunk 数量和检索效果的变化。

---

## 6. 向量知识库

文件：[`src/retrieval/vector_database.py`](../src/retrieval/vector_database.py)

- `QdrantVectorDB`：
  - `_ensure_collection_exists` 创建集合、设置 HNSW 参数、payload 索引。
  - `_prepare_metadata_enrichment` 将 TLDR 与概念词整合到 payload。
  - `add_chunks` 添加向量时会自动把标题/TLDR/概念拼入 `text_tokens`，便于 BM25。
  - `hybrid_search` 计算向量分数 + 文本匹配 + 概念重叠，输出综合得分。

- `MilvusVectorDB`：可切换 backend，只要安装 `pymilvus` 并配置连接。

- `VectorDatabaseManager` 是外部统一入口，负责根据配置选择 backend，并提供 `search`、`get_trending_papers` 等功能。

**练习**：修改 `QdrantVectorDB.hybrid_search` 中 `vector_weight` / `text_weight`，看看哪种组合对召回影响更大。

---

## 7. 查询理解与检索策略

文件：
- [`src/retrieval/query_intelligence.py`](../src/retrieval/query_intelligence.py)
- [`src/generation/rag_generator.py`](../src/generation/rag_generator.py)（`EnhancedQueryProcessor` 部分）
- [`src/retrieval/reranker.py`](../src/retrieval/reranker.py)
- [`src/retrieval/contextual_compression.py`](../src/retrieval/contextual_compression.py)

核心组件：

1. **查询智能**：
   - `QueryIntelligenceEngine` 分析问题类别、生成子问题、改写和 HyDE 文档。
   - `EnhancedQueryProcessor` 缓存查询向量，避免频繁编码；返回优化查询列表供检索使用。

2. **混合检索**：
   - `VectorDatabaseManager.search` 先做语义检索，再融合关键词/概念匹配。
   - 重排阶段 `AdvancedReranker` 通过 Cross-Encoder + MMR 确保结果质量与多样性。

3. **上下文压缩**：
   - `SmartReranker` 根据语义相似度、chunk 质量、概念覆盖进行筛选。
   - `ContextualCompressor` 提供句子抽取、LLM 压缩、混合三套策略。
   - `EnhancedContextOptimizer` 把压缩结果整理成最终 prompt。

**练习**：在 `.env` 中关闭 `ENABLE_CONTEXTUAL_COMPRESSION`，比较回答中引用的段落数量和准确性。

---

## 8. 答案生成与分层模型

文件：
- [`src/generation/rag_generator.py`](../src/generation/rag_generator.py)
- [`src/generation/ultimate_rag_system.py`](../src/generation/ultimate_rag_system.py)
- [`src/generation/tiered_generation.py`](../src/generation/tiered_generation.py)

重点：

- `RAGSystem`（`rag_generator.py` 底部）实现基础 RAG 流程：检索 → 重排 → 压缩 → 生成 → 汇总指标。
- `TieredGenerationSystem` 定义任务路由。若 `.env` 提供多种模型 Key（如 OpenAI/Claude/Qwen），可根据任务复杂度自动选择最优模型组合。
- `UltimateRAGSystem` 把所有模块串联：
  - 支持 basic/enhanced/agentic/ultimate 四种模式。
  - 在 agentic 模式下，配合 `agentic_rag.py` 进行多轮检索评估。
  - 记录每次回答的耗时、模型成本、缓存命中等统计信息。

**练习**：在 `run_rag_system.py` 中调用 `generate_answer(..., mode="agentic")`，对比普通模式的回答质量和耗时。

---

## 9. 个性化与推荐

文件：
- [`src/personalization/user_profiler.py`](../src/personalization/user_profiler.py)
- [`src/personalization/recommendation_engine.py`](../src/personalization/recommendation_engine.py)
- [`src/personalization/preference_tracker.py`](../src/personalization/preference_tracker.py)
- [`api/enhanced_main.py`](../api/enhanced_main.py)（API 端点）

功能：

- `UserProfiler` 管理用户画像（兴趣标签、常看的论文、偏好作者）。
- `PreferenceTracker` 记录页面浏览、停留时间、搜索行为，异步写入会话统计。
- `RecommendationEngine` 基于内容 + 协同过滤生成“今日推荐”主题，利用 OpenAlex 概念和 TLDR 加速理解。
- FastAPI 端点 `/api/v2/ask`、`/api/v2/recommendations` 将个性化结果返回前端。

**练习**：在 `api/enhanced_main.py` 中阅读 `ask_question` 流程，思考如何把用户画像影响检索权重。

---

## 10. 监控与缓存

文件：
- [`src/monitoring/metrics_collector.py`](../src/monitoring/metrics_collector.py)
- [`src/caching/multilayer_cache.py`](../src/caching/multilayer_cache.py)

亮点：

- `MetricsCollector` 暴露 Prometheus 指标：请求总数、响应时间、缓存命中率、向量库延迟等。
- 多层缓存：内存 LRU、Redis、文件缓存、向量缓存。`MultiLayerCache` 支持按策略决定是否写入下一层，显著降低重复请求成本。

**练习**：运行 `scripts/collector-health.sh`（若有权限），观察缓存命中率随时间的变化。

---

## 11. 评估与持续学习

文件：
- [`src/evaluation/evaluation_pipeline.py`](../src/evaluation/evaluation_pipeline.py)
- [`src/learning/continuous_learning_system.py`](../src/learning/continuous_learning_system.py)

- 评估管线可加载固定问题集合，自动执行检索与生成，输出 BLEU/F1/引用完整性等指标。
- 持续学习模块示例化了如何将用户反馈转化为重新排序权重或微调训练数据。

**练习**：在 `evaluation_pipeline.py` 中找到 `run_full_evaluation`，替换自己的测试集看看得分如何。

---

## 12. 前端与部署

- **Streamlit**：`enhanced_app.py` 用于内部运营调试，展示问答、数据统计、缓存状态。
- **Next.js**：`frontend/` 提供用户界面，支持实时流式回答与引用展示。`README.md`、`docs/数据库与前端部署指南.md` 详细介绍部署方案。
- **部署脚本**：`setup.py` 支持云主机快速初始化；`scripts/smoke_test_sources.py` 验证数据源联通性。

**练习**：阅读 `docs/数据库与前端部署指南.md`，在自己的服务器上尝试用 Docker Compose 一键部署。

---

## 13. 工程实践清单

- 使用 `pyenv`/`conda` 管理 Python 版本；`poetry` 版依赖可自行迁移。
- 上线时务必启用 HTTPS、接入 WAF/速率限制。
- 定期备份 Qdrant/Milvus 数据，建议配合对象存储。
- 监控成本：开启语义缓存、分层路由，减少大模型调用。

---

## 14. 自学路线建议

| 周次 | 目标 | 建议任务 |
|------|------|----------|
| 第 1 周 | 跑通流程 | 按本指南完成采集→回答→推荐的全流程；阅读核心文件结构 |
| 第 2 周 | 深入检索 | 调整 `vector_weight`、`text_weight`，尝试改写重排策略 |
| 第 3 周 | 优化生成 | 实现自定义 prompt 或新增模型路由策略 |
| 第 4 周 | 部署上线 | 用 Docker Compose 或 Kubernetes 部署到云服务器，接入监控 |

完成以上步骤后，你已经具备独立扩展 RAG-AI 的能力。祝学习顺利！
