# 全流程运行脚本说明

`run_rag_system.py` 将整个 RAG 系统的生命周期封装到一个可配置的一键脚本中，负责从环境检测、数据采集到前端启动的所有阶段。

## 入口结构

- `main()` 负责解析命令行参数、触发 Hugging Face 登录并根据便捷开关决定是否仅启动前端或执行全流程。核心逻辑位于 [`main`](../../run_rag_system.py#L402-L480)，最终委托 `RAGSystemRunner` 来执行任务。
- `RAGSystemRunner.__init__` 会在 [`run_rag_system.py`](../../run_rag_system.py#L45-L71) 中检查并补齐 `data/raw`、`data/processed` 等目录，保证项目结构与配置一致。

```python
# run_rag_system.py
class RAGSystemRunner:
    def __init__(self):
        self.project_root = project_root
        self.data_dir = config.DATA_DIR
        self._check_project_structure()
```

## 管线阶段
| 阶段 | 关联方法 | 功能要点 |
| ---- | -------- | -------- |

| 环境检测 | [`check_environment`](../../run_rag_system.py#L72-L115) | 校验 Python 版本、关键依赖与 Qdrant 服务是否就绪，失败会中止后续步骤。 |

| 数据采集 | [`collect_data`](../../run_rag_system.py#L117-L161) | 通过 `MultiSourceCollector` 异步抓取 ArXiv、Hugging Face Papers 与 RSS 博客，并在采集完成后将 `Document` 序列保存到 `data/raw/raw_collected_data.json`。 |
| 文本处理 | [`process_data`](../../run_rag_system.py#L163-L221) | 使用 `TextProcessor` 调用分层切分器、向量化器以及多表示索引器，将原始文档转换成可写入向量库的索引条目。 |
| 知识库构建 | [`build_knowledge_base`](../../run_rag_system.py#L223-L276) | 复用 `VectorDatabaseManager` 的 `build_knowledge_base`，若调用方已传入内存中的 chunks 列表，会跳过二次读盘。 |
| 功能测试 | [`test_system`](../../run_rag_system.py#L278-L344) | 初始化 `RAGSystem` 与向量库后，对多条中英问题进行端到端问答，统计成功率。 |
| 前端启动 | [`launch_frontend`](../../run_rag_system.py#L346-L401) | 使用子进程执行 `streamlit run app.py`，并在日志中输出访问地址。 |

## 串联逻辑

核心的串联过程由 `run_full_pipeline` 驱动，根据命令行参数决定是否跳过各阶段，并在最后可选地启动前端：

```python
async def run_full_pipeline(self, args):
    if not args.skip_check and not self.check_environment():
        return False
    if not args.skip_collect:
        await self.collect_data(args.max_papers, args.days_back)
    if not args.skip_process:
        await self.process_data()
    if not args.skip_build:
        self.build_knowledge_base()
    if args.test:
        self.test_system()
    if not args.no_frontend:
        self.launch_frontend(args.port)
```


- 该方法位于 [`run_rag_system.py`](../../run_rag_system.py#L359-L401)，通过事件循环顺序执行异步阶段并记录总耗时。

- `--quick`、`--frontend-only` 等参数会在 `main()` 中预先设置 `skip_*` 标记，使 `run_full_pipeline` 能够复用统一的流程控制代码。

## 配置依赖
所有阶段均从 `configs/config.py` 读取统一的模型、设备与 Qdrant 信息，避免脚本与前端重复拼装配置。运行时生成的日志会写入 `logs/rag_system_YYYYmmdd_HHMMSS.log` 以便追踪每一步的状态。
