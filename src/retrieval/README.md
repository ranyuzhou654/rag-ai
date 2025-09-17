# 检索与上下文优化模块 (Retrieval Module)

## 概述 (Overview)

检索模块是 RAG 系统的核心智能层，负责将用户查询转换为高质量的上下文信息。该模块整合了先进的向量检索、查询智能分析、上下文压缩优化和自适应智能体检索流程，通过多层次的智能处理确保为答案生成提供最相关、最准确的信息基础。

**核心特性：**
- 🔍 **混合检索引擎**: 向量相似度 + 全文检索的融合搜索
- 🧠 **查询智能分析**: 复杂度评估、子问题分解、查询重写
- 🗜️ **上下文压缩优化**: 智能句子提取和内容压缩
- 🤖 **Agentic RAG 流程**: 自评估、自适应的多轮检索
- 📊 **多信号重排序**: 基于相关性、新颖性、多样性的智能排序

## 核心架构

### 1. 向量数据库层

#### Qdrant 向量数据库管理器
```python
class QdrantVectorDB:
    """
    Qdrant向量数据库管理器
    核心功能：高效的向量检索 + 混合搜索能力
    """
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "ai_papers",
        vector_size: int = 1024,  # BGE-M3的向量维度
        timeout: int = 60
    ):
        self.client = QdrantClient(host=host, port=port, timeout=timeout)
        self._ensure_collection_exists()
```

**核心设计**: 位于 [`vector_database.py:14-44`](./vector_database.py#L14-L44)
- **高性能连接**: 配置化的超时和连接池管理
- **自动初始化**: 智能检测和创建向量集合
- **优化配置**: HNSW索引和分段优化参数

#### 集合创建与优化
```python
def _ensure_collection_exists(self):
    """确保集合存在，不存在则创建"""
    self.client.create_collection(
        collection_name=self.collection_name,
        vectors_config=VectorParams(
            size=self.vector_size,
            distance=Distance.COSINE  # 余弦相似度
        ),
        # 性能优化配置
        optimizers_config=models.OptimizersConfig(
            default_segment_number=2,      # 分段数
            max_segment_size=20000,        # 最大分段大小
            memmap_threshold=20000,        # 内存映射阈值
            indexing_threshold=10000       # 索引阈值
        ),
        # HNSW索引参数优化
        hnsw_config=models.HnswConfig(
            m=16,                         # 每层连接数
            ef_construct=200,             # 构建时搜索宽度
            full_scan_threshold=10000     # 全扫描阈值
        )
    )
```

**优化策略**: 位于 [`vector_database.py:45-77`](./vector_database.py#L45-L77)
- **HNSW索引**: 平衡检索速度和准确率的图索引
- **内存管理**: 智能的内存映射和分段策略
- **余弦距离**: 适合文本语义相似度计算

#### 混合检索引擎
```python
def hybrid_search(
    self, 
    query_vector: np.ndarray, 
    query_text: str, 
    limit: int = 10,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    **filters
) -> List[Dict]:
    """
    混合检索：向量相似度 + 全文检索
    """
    # 1. 向量检索
    vector_results = self.client.search(
        collection_name=self.collection_name,
        query_vector=query_vector,
        limit=limit * 2,  # 获取更多候选
        score_threshold=0.5,
        query_filter=self._build_filter(filters) if filters else None
    )
    
    # 2. 文本分词和匹配
    query_tokens = self._tokenize_for_search(query_text)
    
    # 3. 混合评分
    hybrid_results = []
    for result in vector_results:
        vector_score = result.score
        text_score = self._calculate_text_score(
            result.payload.get('text_tokens', []), 
            query_tokens
        )
        
        # 加权融合
        hybrid_score = (vector_weight * vector_score + 
                       text_weight * text_score)
        
        hybrid_results.append({
            'id': result.id,
            'content': result.payload.get('content', ''),
            'vector_score': vector_score,
            'text_score': text_score,
            'hybrid_score': hybrid_score,
            'metadata': result.payload.get('metadata', {}),
            'semantic_type': result.payload.get('semantic_type', 'original')
        })
    
    # 4. 按混合分数排序
    hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return hybrid_results[:limit]
```

**技术创新**: 位于 [`vector_database.py:172-251`](./vector_database.py#L172-L251)
- **双重检索**: 结合语义相似度和关键词匹配
- **动态权重**: 可配置的向量和文本权重比例
- **智能分词**: 中英文自适应的分词策略
- **过滤增强**: 支持语义类型、时间等多维度过滤

#### 中英文混合分词器
```python
def _tokenize_for_search(self, text: str) -> List[str]:
    """智能分词：处理中英文混合文本"""
    import jieba
    
    # 英文部分：词汇级分词
    english_tokens = []
    english_pattern = r'[a-zA-Z]+(?:\.[a-zA-Z]+)*'
    english_words = re.findall(english_pattern, text.lower())
    english_tokens.extend(english_words)
    
    # 中文部分：jieba分词
    chinese_text = re.sub(r'[a-zA-Z0-9\s\.,;:!?()]+', ' ', text)
    chinese_tokens = jieba.lcut(chinese_text)
    chinese_tokens = [token.strip() for token in chinese_tokens 
                     if len(token.strip()) > 1]
    
    # 合并和去重
    all_tokens = list(set(english_tokens + chinese_tokens))
    return [token for token in all_tokens if len(token) > 1]
```

**语言处理**: 位于 [`vector_database.py:125-170`](./vector_database.py#L125-L170)
- **双语支持**: 分别处理中英文的分词需求
- **正则优化**: 精确匹配英文单词和中文词汇
- **去重过滤**: 智能去除停用词和短词

### 2. 查询智能分析层

#### 查询分析结果结构
```python
@dataclass
class QueryAnalysisResult:
    """查询分析结果 - 多维度查询理解"""
    original_query: str           # 原始查询
    language: str                 # 查询语言
    complexity: str               # 复杂度: simple/medium/complex
    sub_questions: List[str]      # 子问题分解
    rewritten_queries: List[str]  # 重写查询
    hypothetical_document: str    # HyDE假设文档
    query_type: str              # 查询类型: factual/comparative/explanatory/procedural
```

#### 查询复杂度分析器
```python
class QueryComplexityAnalyzer:
    """查询复杂度分析器 - 基于模式匹配的智能分析"""
    
    def __init__(self):
        self.complexity_indicators = {
            'simple': [
                r'\b(什么是|what is|define|定义)\b',
                r'\b(who|谁)\b',
                r'\b(when|什么时候)\b',
                r'\b(where|哪里)\b'
            ],
            'medium': [
                r'\b(how|如何|怎么)\b',
                r'\b(why|为什么|为何)\b',
                r'\b(which|哪个|哪种)\b',
                r'\b(compare|比较|对比)\b'
            ],
            'complex': [
                r'\b(analyze|分析|解释)\b.*\b(difference|区别|异同)\b',
                r'\b(evaluate|评估|评价)\b',
                r'\b(explain.*relationship|解释.*关系)\b',
                r'\b(pros and cons|优缺点|利弊)\b',
                r'\b(step by step|步骤|流程)\b',
                r'\band\b.*\bor\b|\b和\b.*\b或\b',  # 多概念
                r'\b(综合|全面|深入|详细)\b.*\b(分析|讨论)\b'
            ]
        }
    
    def analyze_complexity(self, query: str) -> str:
        """基于正则模式和启发式规则分析复杂度"""
        query_lower = query.lower()
        
        # 计算各复杂度级别的匹配分数
        scores = {}
        for complexity, patterns in self.complexity_indicators.items():
            scores[complexity] = sum(1 for pattern in patterns 
                                   if re.search(pattern, query_lower))
        
        # 长度调整
        if len(query) > 100:
            scores['complex'] += 1
        elif len(query) < 20:
            scores['simple'] += 1
        
        # 返回最高分的复杂度
        return max(scores.items(), key=lambda x: x[1])[0] if any(scores.values()) else 'medium'
```

**智能特性**: 位于 [`query_intelligence.py:24-60`](./query_intelligence.py#L24-L60)
- **模式识别**: 基于语言学特征的复杂度判断
- **多语言支持**: 中英文混合的模式匹配
- **启发式增强**: 结合查询长度等额外特征
- **自适应阈值**: 动态调整复杂度边界

#### 子问题生成器
```python
class SubQuestionGenerator(_SharedLLMComponent):
    """子问题生成器 - 复杂查询分解"""
    
    def generate_sub_questions(self, query: str, max_questions: int = 3) -> List[str]:
        """将复杂查询分解为子问题"""
        prompt = f"""将以下复杂问题分解为{max_questions}个更简单的子问题。
每个子问题应该独立可回答，且组合起来能完整回答原问题。

原问题：{query}

请生成子问题（每行一个）："""

        inputs = self.tokenizer(prompt, return_tensors="pt", 
                               truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=200,
                    temperature=0.5,
                    do_sample=True,
                    top_p=0.9
                )
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_questions(response, max_questions)
```

#### HyDE 文档生成器
```python
class HydeDocumentGenerator(_SharedLLMComponent):
    """HyDE假设文档生成器 - 增强检索的关键技术"""
    
    def generate_hypothetical_document(self, query: str) -> str:
        """生成假设文档来改善检索"""
        prompt = f"""基于以下问题，请生成一个假设的、详细的文档段落，
这个段落应该包含回答该问题所需的信息。
不要回答问题，而是生成一个可能包含答案的文档片段。

问题：{query}

假设文档段落："""

        inputs = self.tokenizer(prompt, return_tensors="pt", 
                               truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=300,
                    temperature=0.4,  # 较低温度保证质量
                    do_sample=True,
                    top_p=0.8
                )
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("假设文档段落：")[-1].strip()
```

**HyDE原理**: HyDE (Hypothetical Document Embeddings) 通过生成假设文档来改善检索性能
- **问题转换**: 将查询转换为可能的答案文档
- **语义对齐**: 假设文档与真实文档在向量空间中更相似
- **检索增强**: 使用假设文档的向量进行检索，提升召回率

### 3. 上下文压缩与优化层

#### 句子级相关性提取
```python
class SentenceExtractor:
    """句子提取器 - 从文档块中提取最相关的句子"""
    
    def extract_relevant_sentences(
        self, 
        query: str, 
        chunks: List[Dict], 
        top_k_sentences: int = 10,
        min_sentence_length: int = 20
    ) -> List[Tuple[str, float, int]]:
        """基于语义相似度提取关键句子"""
        
        # 1. 句子分割
        all_sentences = []
        sentence_to_chunk_map = []
        
        for chunk_idx, chunk in enumerate(chunks):
            sentences = self._split_into_sentences(chunk['content'])
            for sentence in sentences:
                if len(sentence.strip()) >= min_sentence_length:
                    all_sentences.append(sentence.strip())
                    sentence_to_chunk_map.append(chunk_idx)
        
        # 2. 语义相似度计算
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0]
        sentence_embeddings = self.embedder.encode(
            all_sentences, 
            batch_size=32, 
            convert_to_numpy=True
        )
        
        # 3. 计算相似度分数
        similarities = cosine_similarity([query_embedding], sentence_embeddings)[0]
        
        # 4. 排序并选择top-k
        sentence_scores = list(zip(all_sentences, similarities, sentence_to_chunk_map))
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        return sentence_scores[:top_k_sentences]
```

**技术特点**: 位于 [`contextual_compression.py:22-78`](./contextual_compression.py#L22-L78)
- **细粒度提取**: 句子级别的精确相关性计算
- **语义理解**: 基于嵌入向量的深度语义匹配
- **质量过滤**: 长度阈值和相似度阈值双重过滤
- **上下文保持**: 保留句子与原始chunk的映射关系

#### LLM驱动的上下文压缩
```python
class LLMCompressor(_SharedLLMComponent):
    """LLM压缩器 - 使用语言模型进行智能压缩"""
    
    def compress_context(
        self, 
        query: str, 
        context: str, 
        target_length: int = 1000
    ) -> str:
        """智能压缩上下文，保留关键信息"""
        
        if len(context) <= target_length:
            return context
        
        prompt = f"""请将以下上下文压缩到大约{target_length}个字符，
保留与问题最相关的核心信息，确保压缩后的内容能够回答问题。

问题：{query}

原始上下文：
{context}

压缩后的上下文："""

        inputs = self.tokenizer(prompt, return_tensors="pt", 
                               truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=target_length // 2,  # 控制输出长度
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.8
                )
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        compressed = response.split("压缩后的上下文：")[-1].strip()
        
        return compressed if compressed else context[:target_length]
```

**压缩策略**: 位于 [`contextual_compression.py:97-150`](./contextual_compression.py#L97-L150)
- **智能理解**: LLM理解查询意图和上下文关系
- **信息保持**: 优先保留与查询最相关的信息
- **长度控制**: 精确控制压缩后的文本长度
- **质量保证**: 低温度生成确保压缩质量

#### 多信号智能重排序
```python
class SmartReranker:
    """智能重排序器 - 基于多个信号的文档重排"""
    
    def smart_rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 10,
        diversity_weight: float = 0.3,
        recency_weight: float = 0.2,
        relevance_weight: float = 0.5
    ) -> List[Dict]:
        """多信号重排序算法"""
        
        if not chunks:
            return []
        
        # 1. 计算相关性分数（基础分数）
        relevance_scores = self._calculate_relevance_scores(query, chunks)
        
        # 2. 计算多样性分数
        diversity_scores = self._calculate_diversity_scores(chunks)
        
        # 3. 计算时效性分数
        recency_scores = self._calculate_recency_scores(chunks)
        
        # 4. 综合评分
        final_scores = []
        for i, chunk in enumerate(chunks):
            composite_score = (
                relevance_weight * relevance_scores[i] +
                diversity_weight * diversity_scores[i] +
                recency_weight * recency_scores[i]
            )
            final_scores.append((chunk, composite_score))
        
        # 5. 排序和选择
        final_scores.sort(key=lambda x: x[1], reverse=True)
        reranked_chunks = [chunk for chunk, score in final_scores[:top_k]]
        
        return reranked_chunks
```

**重排信号**: 位于 [`contextual_compression.py:152-239`](./contextual_compression.py#L152-L239)
- **相关性信号**: 基于语义相似度的核心相关性
- **多样性信号**: 避免信息重复，增加内容多样性
- **时效性信号**: 优先考虑较新的文档内容
- **权重平衡**: 可配置的多信号权重组合

### 4. Agentic RAG 智能体层

#### 检索评估器
```python
class RetrievalEvaluator(_SharedLLMComponent):
    """检索评估器 - 自评估检索质量"""
    
    def evaluate_retrieval(
        self, 
        query: str, 
        retrieved_chunks: List[Dict], 
        min_chunks: int = 2
    ) -> Tuple[str, Dict]:
        """评估检索结果质量并决定下一步行动"""
        
        # 1. 快速检查
        if len(retrieved_chunks) < min_chunks:
            return "EXPAND_SEARCH", {
                "reason": "检索到的文档数量不足",
                "retrieved_count": len(retrieved_chunks),
                "min_required": min_chunks
            }
        
        # 2. LLM深度评估
        evaluation_result = self._llm_evaluate_retrieval(query, retrieved_chunks)
        
        # 3. 决策逻辑
        if evaluation_result["relevance_score"] >= 0.8:
            return "PROCEED", evaluation_result
        elif evaluation_result["relevance_score"] >= 0.5:
            return "REWRITE_QUERY", evaluation_result
        else:
            return "RETRY", evaluation_result
    
    def _llm_evaluate_retrieval(self, query: str, chunks: List[Dict]) -> Dict:
        """使用LLM评估检索结果"""
        context = "\n\n".join([f"文档{i+1}: {chunk['content'][:200]}..." 
                              for i, chunk in enumerate(chunks[:3])])
        
        prompt = f"""请评估以下检索结果对于回答用户问题的质量。

用户问题: {query}

检索到的文档:
{context}

请从以下几个方面评估（0-1分）：
1. 相关性：文档内容与问题的相关程度
2. 完整性：文档是否包含回答问题所需的充分信息
3. 一致性：多个文档之间是否存在矛盾

请返回JSON格式的评估结果。"""

        # LLM推理和解析...
        return evaluation_result
```

**智能评估**: 位于 [`agentic_rag.py:24-150`](./agentic_rag.py#L24-L150)
- **多维评估**: 相关性、完整性、一致性三重评估
- **决策引擎**: 基于评估结果的智能行动决策
- **自适应阈值**: 动态调整评估标准

#### Agentic RAG 协调器
```python
class AgenticRAGOrchestrator:
    """Agentic RAG协调器 - 自适应多轮检索"""
    
    async def agentic_retrieve_and_generate(
        self, 
        original_query: str, 
        max_iterations: int = 3
    ) -> Dict:
        """智能体驱动的检索和生成流程"""
        
        agentic_steps = []
        current_query = original_query
        accumulated_context = []
        
        for iteration in range(max_iterations):
            logger.info(f"🤖 Agentic iteration {iteration + 1}")
            
            # 1. 查询分析和优化
            if iteration == 0:
                query_analysis = await self.query_intelligence.analyze_query(current_query)
            else:
                # 基于历史结果优化查询
                optimized_queries = await self.query_intelligence.get_optimized_queries(
                    current_query, context=accumulated_context
                )
                current_query = optimized_queries[0] if optimized_queries else current_query
            
            # 2. 执行检索
            retrieved_chunks = await self.vector_db.hybrid_search(
                query_text=current_query,
                limit=10
            )
            
            # 3. 评估检索结果
            action, evaluation = self.retrieval_evaluator.evaluate_retrieval(
                current_query, retrieved_chunks
            )
            
            # 4. 记录步骤
            step = AgenticStep(
                iteration=iteration,
                query=current_query,
                action=action,
                retrieved_chunks=retrieved_chunks,
                evaluation=evaluation,
                timestamp=time.time()
            )
            agentic_steps.append(step)
            
            # 5. 决策分支
            if action == "PROCEED":
                accumulated_context.extend(retrieved_chunks)
                break
            elif action == "REWRITE_QUERY":
                current_query = await self._rewrite_query_based_on_feedback(
                    current_query, evaluation
                )
                accumulated_context.extend(retrieved_chunks)
            elif action == "RETRY":
                current_query = await self._expand_query(current_query)
            elif action == "EXPAND_SEARCH":
                # 扩展搜索参数
                retrieved_chunks = await self.vector_db.hybrid_search(
                    query_text=current_query,
                    limit=20,  # 增加检索数量
                    vector_weight=0.5  # 调整权重
                )
                accumulated_context.extend(retrieved_chunks)
        
        # 6. 最终上下文优化
        optimized_context = await self.context_optimizer.optimize_context(
            original_query, accumulated_context
        )
        
        # 7. 生成最终答案
        final_answer = await self.generator.generate_answer(
            original_query, optimized_context
        )
        
        return {
            "answer": final_answer,
            "agentic_steps": agentic_steps,
            "final_context": optimized_context,
            "iterations_used": len(agentic_steps)
        }
```

**智能特性**: 位于 [`agentic_rag.py:229-360`](./agentic_rag.py#L229-L360)
- **自适应迭代**: 基于评估结果动态调整检索策略
- **上下文累积**: 多轮检索结果的智能积累
- **查询进化**: 基于反馈的查询重写和优化
- **完整追踪**: 记录每个决策步骤便于分析

## 性能与优化

### 1. 检索性能优化
- **HNSW索引**: 近似最近邻搜索，平衡速度与精度
- **批量处理**: 向量编码和相似度计算的批量优化
- **缓存机制**: 查询分析和重写结果的智能缓存
- **分层过滤**: 多阶段过滤减少计算开销

### 2. 内存管理
- **模型共享**: 通过 `ModelRegistry` 避免重复加载
- **渐进式加载**: 按需加载检索组件
- **内存映射**: Qdrant 的内存映射优化大数据集
- **垃圾回收**: 及时清理临时数据和中间结果

### 3. 质量保证
- **多维评估**: 相关性、完整性、一致性三重质量检查
- **阈值控制**: 动态调整各类评估阈值
- **降级策略**: 检索失败时的多层降级机制
- **结果验证**: 生成结果的后处理验证

## 使用示例

```python
from src.retrieval.vector_database import VectorDatabaseManager
from src.retrieval.query_intelligence import QueryIntelligenceEngine
from src.retrieval.agentic_rag import AgenticRAGOrchestrator

# 初始化检索系统
vector_db = VectorDatabaseManager(config)
query_intelligence = QueryIntelligenceEngine(config)
agentic_orchestrator = AgenticRAGOrchestrator(config)

# 简单检索
query = "什么是Transformer架构"
results = await vector_db.hybrid_search(
    query_text=query,
    limit=10,
    vector_weight=0.7,
    text_weight=0.3
)

# 智能查询分析
analysis = await query_intelligence.analyze_query(query)
print(f"查询复杂度: {analysis.complexity}")
print(f"子问题: {analysis.sub_questions}")

# Agentic RAG 自适应检索
agentic_result = await agentic_orchestrator.agentic_retrieve_and_generate(
    query, max_iterations=3
)
print(f"最终答案: {agentic_result['answer']}")
print(f"使用迭代次数: {agentic_result['iterations_used']}")
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vector_weight` | 0.7 | 混合检索中向量权重 |
| `text_weight` | 0.3 | 混合检索中文本权重 |
| `max_agentic_iterations` | 3 | Agentic RAG最大迭代次数 |
| `relevance_threshold` | 0.8 | 相关性评估阈值 |
| `compression_ratio` | 0.5 | 上下文压缩比例 |
| `diversity_weight` | 0.3 | 重排序中多样性权重 |

## 扩展指南

### 添加新的检索策略
1. 继承 `_SharedLLMComponent` 创建新的检索器
2. 实现 `search` 方法返回标准化结果
3. 在 `AgenticRAGOrchestrator` 中注册新策略
4. 添加相应的评估逻辑

### 优化建议
- **索引调优**: 根据数据规模调整 HNSW 参数
- **权重调优**: 基于业务场景调整混合检索权重
- **阈值优化**: 根据评估反馈动态调整质量阈值
- **缓存策略**: 为高频查询添加结果缓存

检索模块通过多层次的智能处理和自适应优化，为 RAG 系统提供了高质量、高效率的信息检索能力，是整个问答系统的核心引擎。