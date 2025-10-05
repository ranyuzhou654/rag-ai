# RAG-AI Scaling & Storage Strategies

This document summarises recommended practices for operating RAG-AI with
large-scale scholarly corpora (≥10 years of publications) while balancing
retrieval quality, storage footprint, and inference throughput.

## 1. Vector Database Selection

| Backend   | Strengths | Considerations |
|-----------|-----------|----------------|
| **Qdrant** (default) | Simple deployment, good hybrid search, can shard via Enterprise edition | Community version limited to single node, plan sharding if corpus grows beyond hundreds of millions of vectors |
| **Milvus** | Horizontal scalability, multi-channel ingestion, built-in tiered storage | Requires separate etcd + Meta storage; best suited for >1B vectors |
| **Weaviate / Pinecone** | Managed options, minimal ops overhead | Vendor lock-in + usage costs |

**Recommendation**: For immediate needs stick with Qdrant, but expose the
`VECTOR_DB_BACKEND` setting. When the corpus exceeds a single-node Qdrant,
evaluate Milvus (standalone or managed) and reuse the same storage schema.

## 2. Storage Modes

Large corpora can be stored with minimal content while retaining high recall.

`STORAGE_CONTENT_MODE` (env var, default `full`):

| Mode | Stored content | Use case |
|------|----------------|----------|
| `full` | Entire chunk text | Highest fidelity, default during experimentation |
| `summary` | LLM-generated summary (fallback to chunk) | Medium footprint, sufficient for answer grounding |
| `title_abstract` | Combines metadata title + summary/abstract preview | Minimal storage, relies on metadata-rich documents |

The original chunk preview (`content_preview`) is retained in metadata when a
compressed mode is active, enabling downstream audit without rehydrating full
documents.

## 3. Minimal Corpus Strategy

1. Store embedded vectors for `title + abstract` only.
2. Keep full text in cold storage (object store or compressed files).
3. During retrieval, fetch cold storage lazily for top-k documents when high
precision answers are required.

## 4. Inference Backends

`INFERENCE_BACKEND` (env var, default `local`):

| Backend | Description | When to use |
|---------|-------------|-------------|
| `local` | On-device inference (current behaviour) | Development, low-latency local GPU |
| `api`   | Offload to managed APIs (OpenAI/Azure/Anthropic) | Scale-out, fallback when local capacity overloads |
| `vllm`  | Serve local models via [vLLM](https://vllm.ai/) or TensorRT-LLM | Production with high throughput GPU clusters |

Suggested deployment pattern:

1. Serve quantised models (4/8bit) via vLLM for bulk workloads.
2. Route high-importance or complex queries to managed APIs as fallback.
3. Track latency & cost metrics to adjust routing in TieredGenerationSystem.

## 5. Action Plan

1. **Configure Environment**
   ```bash
   export VECTOR_DB_BACKEND=milvus
   export STORAGE_CONTENT_MODE=summary
   export INFERENCE_BACKEND=vllm
   ```

2. **Provision Vector DB**
   - Qdrant Enterprise or Milvus with shared storage (MinIO / S3).
   - Enable hybrid search and metadata filters.

3. **Adjust Processing Pipeline**
   - Run `STORAGE_CONTENT_MODE=summary` to shrink persisted payload.
   - Keep cold storage for full PDFs if future audit is required.

4. **Deploy Inference Backend**
   - Launch vLLM server for base model (e.g., Qwen2 7B) with batch size tuned.
   - Configure API credentials for fallback provider.

5. **Continuous Evaluation**
   - Generate datasets via `convert_hotpotqa.py` and `convert_dureader.py`.
   - Run `python -m src.evaluation.evaluation_pipeline --dataset-preset hotpotqa --use-ragas` on each release.

By combining the above configuration levers, RAG-AI can scale to tens of
millions of academic documents while keeping latency and storage under control.
