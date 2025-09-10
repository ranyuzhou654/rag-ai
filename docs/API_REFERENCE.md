# API 参考文档

## 目录
- [核心 API](#核心-api)
- [检索组件 API](#检索组件-api)
- [生成组件 API](#生成组件-api)
- [评估组件 API](#评估组件-api)
- [优化组件 API](#优化组件-api)
- [配置管理 API](#配置管理-api)
- [错误处理](#错误处理)

## 核心 API

### RAGSystem

主要的 RAG 系统入口类。

```python
from src.generation.langchain_rag_system import LangChainRAGSystem

class RAGSystem:
    def __init__(self, config_path: str = None)
    async def query(self, question: str, **kwargs) -> Dict[str, Any]
    async def add_documents(self, documents: List[str], **kwargs) -> bool
    def get_system_stats(self) -> Dict[str, Any]
```

#### 初始化
```python
# 使用默认配置
rag_system = RAGSystem()

# 使用自定义配置
rag_system = RAGSystem(config_path="./config.yaml")

# 使用代码配置
config = {
    "retrieval": {"top_k": 10},
    "generation": {"temperature": 0.1}
}
rag_system = RAGSystem(config=config)
```

#### 查询接口
```python
async def query(self, 
                question: str,
                top_k: int = 10,
                temperature: float = 0.1,
                enable_rerank: bool = True,
                return_sources: bool = True,
                stream: bool = False) -> Dict[str, Any]:
    """
    执行 RAG 查询
    
    Args:
        question: 用户问题
        top_k: 返回的文档数量
        temperature: 生成温度
        enable_rerank: 是否启用重排序
        return_sources: 是否返回来源文档
        stream: 是否流式返回
        
    Returns:
        {
            "answer": str,           # 生成的答案
            "sources": List[Dict],   # 来源文档
            "confidence": float,     # 置信度评分
            "latency": float,        # 响应延迟
            "evaluation": Dict       # 评估结果
        }
    """
```

#### 使用示例
```python
import asyncio

async def main():
    rag_system = RAGSystem()
    
    # 基础查询
    result = await rag_system.query("什么是深度学习？")
    print(f"答案: {result['answer']}")
    print(f"置信度: {result['confidence']}")
    
    # 高级查询
    result = await rag_system.query(
        question="解释Transformer架构的工作原理",
        top_k=15,
        temperature=0.2,
        enable_rerank=True,
        return_sources=True
    )
    
    # 流式查询
    async for chunk in rag_system.query_stream("机器学习的发展历史"):
        print(chunk, end="")

asyncio.run(main())
```

## 检索组件 API

### HybridRetriever

混合检索器，结合多种检索策略。

```python
from src.retrieval.hybrid_retriever import HybridRetriever

class HybridRetriever:
    def __init__(self, 
                 dense_retriever_config: Dict = None,
                 sparse_retriever_config: Dict = None,
                 kg_retriever_config: Dict = None)
    
    async def retrieve(self, 
                       query: str, 
                       top_k: int = 10,
                       fusion_weights: Dict[str, float] = None) -> List[Document]
    
    def update_weights(self, weights: Dict[str, float]) -> None
    def add_documents(self, documents: List[Document]) -> None
    def get_retrieval_stats(self) -> Dict[str, Any]
```

#### 使用示例
```python
# 初始化混合检索器
retriever = HybridRetriever(
    dense_retriever_config={
        "model_name": "text-embedding-ada-002",
        "dimension": 1536
    },
    sparse_retriever_config={
        "k1": 1.2,
        "b": 0.75
    },
    kg_retriever_config={
        "hop_limit": 2,
        "relation_weights": {"related_to": 0.8}
    }
)

# 检索文档
documents = await retriever.retrieve(
    query="深度学习在计算机视觉中的应用",
    top_k=10,
    fusion_weights={
        "dense": 0.6,
        "sparse": 0.3,
        "kg": 0.1
    }
)

# 动态调整权重
retriever.update_weights({
    "dense": 0.5,
    "sparse": 0.4,
    "kg": 0.1
})
```

### WeightedEmbeddingFusion

加权嵌入融合组件。

```python
from src.processing.weighted_embedding_fusion import WeightedEmbeddingFusion

class WeightedEmbeddingFusion:
    def __init__(self, 
                 embedding_model: str = "text-embedding-ada-002",
                 weights: Dict[str, float] = None)
    
    async def create_multi_representation(self, 
                                          document: str) -> Dict[str, np.ndarray]
    
    def fuse_embeddings(self, 
                        embeddings: Dict[str, np.ndarray]) -> np.ndarray
    
    async def process_documents(self, 
                                documents: List[str]) -> List[Dict[str, Any]]
```

#### 使用示例
```python
# 初始化嵌入融合器
fusion = WeightedEmbeddingFusion(
    embedding_model="text-embedding-ada-002",
    weights={
        "original": 0.6,
        "summary": 0.3,
        "questions": 0.1
    }
)

# 处理单个文档
document = "这是一篇关于机器学习的文章..."
multi_rep = await fusion.create_multi_representation(document)

# 批量处理文档
documents = ["文档1", "文档2", "文档3"]
processed_docs = await fusion.process_documents(documents)

for doc in processed_docs:
    print(f"文档ID: {doc['id']}")
    print(f"融合嵌入维度: {doc['fused_embedding'].shape}")
```

## 生成组件 API

### LangChainRAGSystem

基于 LangChain LCEL 的 RAG 生成系统。

```python
from src.generation.langchain_rag_system import LangChainRAGSystem

class LangChainRAGSystem:
    def __init__(self, 
                 llm_config: Dict = None,
                 retriever_config: Dict = None)
    
    async def generate_answer(self, 
                              query: str, 
                              context: List[Document],
                              **kwargs) -> Dict[str, Any]
    
    def create_rag_chain(self) -> RunnableSequence
    async def stream_answer(self, query: str, context: List[Document])
```

#### 使用示例
```python
# 初始化生成系统
rag_gen = LangChainRAGSystem(
    llm_config={
        "model": "gpt-4",
        "temperature": 0.1,
        "max_tokens": 1000
    }
)

# 生成答案
context_docs = await retriever.retrieve("机器学习基础")
result = await rag_gen.generate_answer(
    query="什么是监督学习？",
    context=context_docs,
    include_sources=True
)

# 流式生成
async for chunk in rag_gen.stream_answer("解释神经网络", context_docs):
    print(chunk, end="", flush=True)
```

### IntelligentTaskRouter

智能任务路由器，根据查询复杂度选择合适的模型。

```python
from src.generation.intelligent_task_routing import IntelligentTaskRouter

class IntelligentTaskRouter:
    def __init__(self, 
                 model_configs: Dict[str, Dict] = None,
                 complexity_thresholds: Dict[str, float] = None)
    
    def analyze_query_complexity(self, query: str) -> Dict[str, float]
    def route_to_model(self, query: str) -> str
    async def execute_with_routing(self, query: str, context: List[Document]) -> Dict[str, Any]
    def get_routing_stats(self) -> Dict[str, Any]
```

#### 使用示例
```python
# 配置路由器
router = IntelligentTaskRouter(
    model_configs={
        "simple": {"model": "gpt-3.5-turbo", "cost_per_token": 0.001},
        "complex": {"model": "gpt-4", "cost_per_token": 0.01}
    },
    complexity_thresholds={
        "simple": 0.3,
        "medium": 0.7,
        "complex": 1.0
    }
)

# 分析查询复杂度
complexity = router.analyze_query_complexity("什么是AI？")
print(f"复杂度分析: {complexity}")

# 自动路由执行
result = await router.execute_with_routing(
    query="解释量子计算在机器学习中的应用前景",
    context=context_docs
)
```

## 评估组件 API

### EnhancedAgenticRAG

增强的代理式 RAG，包含四维评估。

```python
from src.retrieval.enhanced_agentic_rag import EnhancedAgenticRAG

class EnhancedAgenticRAG:
    def __init__(self, 
                 retriever: BaseRetriever,
                 generator: BaseGenerator,
                 evaluator: BaseEvaluator)
    
    async def agentic_query(self, 
                            query: str,
                            max_iterations: int = 3,
                            quality_threshold: float = 0.8) -> Dict[str, Any]
    
    async def four_dimensional_evaluation(self, 
                                          query: str, 
                                          answer: str, 
                                          sources: List[Document]) -> Dict[str, float]
    
    def self_reflection(self, evaluation_results: Dict) -> bool
```

#### 使用示例
```python
# 初始化代理式RAG
agentic_rag = EnhancedAgenticRAG(
    retriever=hybrid_retriever,
    generator=rag_generator,
    evaluator=four_dim_evaluator
)

# 执行代理式查询
result = await agentic_rag.agentic_query(
    query="分析深度学习在医疗诊断中的应用现状和挑战",
    max_iterations=3,
    quality_threshold=0.85
)

print(f"最终答案: {result['final_answer']}")
print(f"迭代次数: {result['iterations']}")
print(f"评估分数: {result['evaluation_scores']}")
```

### ComprehensiveEvaluation

综合评估系统，集成 RAGAS 和 TruLens。

```python
from src.evaluation.comprehensive_evaluation import ComprehensiveEvaluation

class ComprehensiveEvaluation:
    def __init__(self, 
                 enable_ragas: bool = True,
                 enable_trulens: bool = True,
                 custom_metrics: List[str] = None)
    
    async def evaluate_response(self, 
                                query: str, 
                                answer: str, 
                                context: List[str],
                                ground_truth: str = None) -> Dict[str, float]
    
    async def batch_evaluate(self, 
                             evaluation_data: List[Dict]) -> Dict[str, Any]
    
    def generate_evaluation_report(self, 
                                   results: Dict) -> str
```

#### 使用示例
```python
# 初始化评估系统
evaluator = ComprehensiveEvaluation(
    enable_ragas=True,
    enable_trulens=True,
    custom_metrics=["domain_accuracy", "factual_consistency"]
)

# 单次评估
eval_result = await evaluator.evaluate_response(
    query="什么是机器学习？",
    answer="机器学习是一种人工智能技术...",
    context=["机器学习定义文档1", "机器学习应用文档2"],
    ground_truth="标准答案..."
)

# 批量评估
eval_data = [
    {"query": "问题1", "answer": "答案1", "context": ["上下文1"]},
    {"query": "问题2", "answer": "答案2", "context": ["上下文2"]}
]
batch_results = await evaluator.batch_evaluate(eval_data)

# 生成评估报告
report = evaluator.generate_evaluation_report(batch_results)
print(report)
```

## 优化组件 API

### PerformanceOptimizer

性能优化组件，包含缓存和GPU加速。

```python
from src.optimization.performance_optimizer import PerformanceOptimizer

class PerformanceOptimizer:
    def __init__(self, 
                 enable_memory_cache: bool = True,
                 enable_redis_cache: bool = True,
                 enable_gpu_acceleration: bool = True)
    
    async def cached_embedding(self, 
                               texts: List[str], 
                               model_name: str) -> List[np.ndarray]
    
    def setup_gpu_acceleration(self) -> None
    async def batch_process(self, 
                            items: List[Any], 
                            process_func: Callable,
                            batch_size: int = 32) -> List[Any]
    
    def get_performance_metrics(self) -> Dict[str, Any]
```

#### 使用示例
```python
# 初始化性能优化器
optimizer = PerformanceOptimizer(
    enable_memory_cache=True,
    enable_redis_cache=True,
    enable_gpu_acceleration=True
)

# 缓存嵌入计算
texts = ["文本1", "文本2", "文本3"]
embeddings = await optimizer.cached_embedding(
    texts=texts,
    model_name="text-embedding-ada-002"
)

# 批量处理
def process_item(item):
    return {"processed": item, "timestamp": time.time()}

results = await optimizer.batch_process(
    items=["item1", "item2", "item3"],
    process_func=process_item,
    batch_size=2
)

# 获取性能指标
metrics = optimizer.get_performance_metrics()
print(f"缓存命中率: {metrics['cache_hit_rate']}")
print(f"平均处理时间: {metrics['avg_processing_time']}")
```

### ContinuousLearningSystem

持续学习系统，支持反馈驱动的模型优化。

```python
from src.learning.continuous_learning_system import ContinuousLearningSystem

class ContinuousLearningSystem:
    def __init__(self, 
                 feedback_threshold: float = 0.1,
                 update_frequency: int = 100)
    
    def collect_feedback(self, 
                         query: str, 
                         answer: str, 
                         rating: float,
                         user_feedback: str = None) -> None
    
    async def analyze_feedback_patterns(self) -> Dict[str, Any]
    async def update_model_parameters(self) -> bool
    def get_learning_statistics(self) -> Dict[str, Any]
```

#### 使用示例
```python
# 初始化持续学习系统
learning_system = ContinuousLearningSystem(
    feedback_threshold=0.1,
    update_frequency=50
)

# 收集用户反馈
learning_system.collect_feedback(
    query="什么是深度学习？",
    answer="深度学习是机器学习的一个分支...",
    rating=4.5,
    user_feedback="回答很好，但可以更详细"
)

# 分析反馈模式
patterns = await learning_system.analyze_feedback_patterns()
print(f"识别的改进点: {patterns['improvement_areas']}")

# 更新模型参数
updated = await learning_system.update_model_parameters()
if updated:
    print("模型参数已更新")
```

## 配置管理 API

### ConfigManager

配置管理器，统一管理系统配置。

```python
from src.config.config_manager import ConfigManager

class ConfigManager:
    def __init__(self, config_path: str = "./config.yaml")
    
    def load_config(self, config_path: str = None) -> Dict[str, Any]
    def save_config(self, config: Dict[str, Any], path: str = None) -> None
    def get(self, key: str, default: Any = None) -> Any
    def set(self, key: str, value: Any) -> None
    def validate_config(self) -> List[str]
```

#### 使用示例
```python
# 初始化配置管理器
config_manager = ConfigManager("./config.yaml")

# 获取配置值
embedding_model = config_manager.get("retrieval.embedding_model")
temperature = config_manager.get("generation.temperature", 0.1)

# 设置配置值
config_manager.set("retrieval.top_k", 15)
config_manager.set("generation.max_tokens", 1200)

# 验证配置
validation_errors = config_manager.validate_config()
if validation_errors:
    print(f"配置错误: {validation_errors}")

# 保存配置
config_manager.save_config()
```

## 错误处理

### 异常类型

```python
class RAGException(Exception):
    """RAG系统基础异常"""
    pass

class RetrievalException(RAGException):
    """检索相关异常"""
    pass

class GenerationException(RAGException):
    """生成相关异常"""
    pass

class EvaluationException(RAGException):
    """评估相关异常"""
    pass

class ConfigurationException(RAGException):
    """配置相关异常"""
    pass
```

### 错误处理示例

```python
try:
    result = await rag_system.query("什么是AI？")
except RetrievalException as e:
    logger.error(f"检索失败: {e}")
    # 使用备用检索策略
    result = await fallback_retrieval(query)
except GenerationException as e:
    logger.error(f"生成失败: {e}")
    # 降级到简单回答
    result = {"answer": "抱歉，无法生成详细回答"}
except RAGException as e:
    logger.error(f"系统错误: {e}")
    result = {"error": str(e)}
```

### 重试机制

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def robust_query(rag_system, query):
    try:
        return await rag_system.query(query)
    except Exception as e:
        logger.warning(f"查询失败，准备重试: {e}")
        raise
```

## 性能监控 API

### PerformanceMonitor

性能监控组件。

```python
from src.monitoring.performance_monitor import PerformanceMonitor

class PerformanceMonitor:
    def __init__(self, enable_detailed_metrics: bool = True)
    
    @contextmanager
    def measure_time(self, operation: str)
    def record_metric(self, name: str, value: float, tags: Dict = None)
    def get_metrics_summary(self) -> Dict[str, Any]
    def export_metrics(self, format: str = "json") -> str
```

#### 使用示例
```python
monitor = PerformanceMonitor()

# 测量操作时间
with monitor.measure_time("retrieval"):
    documents = await retriever.retrieve(query)

# 记录自定义指标
monitor.record_metric("cache_hit_rate", 0.85, {"component": "embedding"})

# 获取指标摘要
summary = monitor.get_metrics_summary()
print(f"平均检索时间: {summary['retrieval']['avg_time']}")
```

## 批量操作 API

### BatchProcessor

批量处理组件，支持大规模数据处理。

```python
from src.processing.batch_processor import BatchProcessor

class BatchProcessor:
    def __init__(self, 
                 batch_size: int = 32,
                 max_workers: int = 4)
    
    async def batch_embed_documents(self, 
                                    documents: List[str]) -> List[np.ndarray]
    async def batch_query(self, 
                          queries: List[str]) -> List[Dict[str, Any]]
    async def batch_evaluate(self, 
                             eval_data: List[Dict]) -> List[Dict[str, float]]
```

#### 使用示例
```python
processor = BatchProcessor(batch_size=64, max_workers=8)

# 批量嵌入文档
documents = ["文档1", "文档2", ...]  # 1000个文档
embeddings = await processor.batch_embed_documents(documents)

# 批量查询
queries = ["问题1", "问题2", ...]  # 100个问题
results = await processor.batch_query(queries)

# 批量评估
eval_data = [{"query": "q1", "answer": "a1"}, ...]
evaluations = await processor.batch_evaluate(eval_data)
```

## SDK 快速集成

### Python SDK

```python
from rag_system import RAGClient

# 初始化客户端
client = RAGClient(api_key="your_api_key", base_url="http://localhost:8000")

# 简单查询
response = client.query("什么是机器学习？")
print(response.answer)

# 流式查询
for chunk in client.query_stream("解释深度学习"):
    print(chunk, end="")

# 添加文档
client.add_documents(["新文档内容1", "新文档内容2"])

# 获取系统状态
status = client.get_status()
print(f"系统健康状态: {status.health}")
```

### REST API

```bash
# 查询接口
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是人工智能？",
    "top_k": 10,
    "temperature": 0.1
  }'

# 添加文档
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["文档内容1", "文档内容2"],
    "metadata": [{"source": "web"}, {"source": "pdf"}]
  }'

# 获取系统状态
curl -X GET "http://localhost:8000/api/v1/health"
```