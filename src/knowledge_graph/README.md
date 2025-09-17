# 知识图谱抽取与检索模块 (Knowledge Graph Module)

## 概述 (Overview)

知识图谱模块是 RAG 系统的结构化知识增强层，负责从文本中抽取实体和关系，构建领域知识图谱，并提供基于图结构的检索和推理能力。该模块通过将非结构化文本转换为结构化知识，显著增强了系统的推理能力和答案可解释性。

**核心特性：**
- 🧠 **智能实体识别**: 基于LLM的领域实体抽取和类型分类
- 🔗 **关系抽取**: 自动识别实体间的语义关系
- 🗃️ **混合存储**: SQLite持久化 + NetworkX内存图的高效存储
- 🔍 **图谱检索**: 基于实体匹配和路径搜索的知识检索
- 📊 **上下文增强**: 结构化知识与文本检索的融合增强

## 核心架构

### 1. 数据模型层

#### 实体数据结构
```python
@dataclass
class Entity:
    """实体结构 - 知识图谱的基本节点"""
    name: str                               # 实体名称
    entity_type: str                        # 实体类型
    aliases: Set[str] = field(default_factory=set)  # 别名集合
    properties: Dict[str, Any] = field(default_factory=dict)  # 属性字典
    confidence: float = 1.0                 # 置信度分数
    source_chunks: List[str] = field(default_factory=list)  # 来源文档块
```

**设计原理**: 位于 [`knowledge_extractor.py:17-25`](./knowledge_extractor.py#L17-L25)
- **类型化实体**: 支持AI领域的专业实体分类
- **别名支持**: 处理同一实体的不同表述方式
- **来源追踪**: 保持实体与源文档的关联关系
- **置信度机制**: 支持实体抽取质量评估

#### 关系数据结构
```python
@dataclass
class Relation:
    """关系结构 - 知识图谱的边"""
    head_entity: str                        # 头实体
    relation_type: str                      # 关系类型
    tail_entity: str                        # 尾实体
    confidence: float = 1.0                 # 置信度分数
    source_text: str = ""                   # 源文本片段
    source_chunk_id: str = ""               # 源文档块ID
    properties: Dict[str, Any] = field(default_factory=dict)  # 关系属性
```

#### 知识三元组
```python
@dataclass
class KnowledgeTriplet:
    """知识三元组 - RDF风格的知识表示"""
    subject: str                            # 主语实体
    predicate: str                          # 谓语关系
    object: str                             # 宾语实体
    confidence: float = 1.0                 # 置信度
    source: str = ""                        # 来源信息
```

**关系建模**: 位于 [`knowledge_extractor.py:27-45`](./knowledge_extractor.py#L27-L45)
- **三元组结构**: 标准的RDF三元组知识表示
- **源文本保持**: 保留关系的原始文本证据
- **属性扩展**: 支持关系的附加属性信息
- **置信度评估**: 关系抽取质量的量化评估

### 2. 知识抽取层

#### 共享LLM组件基类
```python
class _SharedLLMComponent:
    """Utility mixin to reuse cached LLM resources."""
    
    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None, component_name: str = "KG component"):
        resource = ModelRegistry.get_llm(model_name, device=device, token=token)
        self.device = resource.device
        self.model_name = model_name
        self.tokenizer = resource.tokenizer
        self.model = resource.model
        logger.info(f"{component_name} using shared LLM: {model_name}")
```

**资源共享**: 位于 [`knowledge_extractor.py:47-56`](./knowledge_extractor.py#L47-L56)
- **模型复用**: 所有抽取组件共享同一LLM实例
- **统一接口**: 标准化的模型访问方式
- **内存优化**: 避免重复加载大型语言模型

#### 实体抽取器
```python
class EntityExtractor(_SharedLLMComponent):
    """实体抽取器 - 基于LLM的智能实体识别"""
    
    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None):
        super().__init__(model_name=model_name, device=device, token=token, component_name="Entity Extractor")
        
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.2,        # 低温度确保稳定性
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 定义AI领域的实体类型
        self.entity_types = {
            "ALGORITHM": "算法和模型",
            "CONCEPT": "技术概念",
            "PERSON": "人名",
            "ORGANIZATION": "机构组织",
            "DATASET": "数据集",
            "FRAMEWORK": "框架和工具",
            "METRIC": "评估指标",
            "APPLICATION": "应用场景"
        }
    
    def extract_entities(self, text: str, language: str = "zh") -> List[Entity]:
        """从文本中抽取实体"""
        
        # 构建提示模板
        if language == "zh":
            prompt = f"""请从以下AI技术文本中识别和抽取重要的实体，并按照JSON格式输出。

实体类型包括：
- ALGORITHM: 算法和模型 (如: Transformer, BERT, GPT)
- CONCEPT: 技术概念 (如: 注意力机制, 反向传播)
- PERSON: 人名 (如: Geoffrey Hinton, Yann LeCun)
- ORGANIZATION: 机构组织 (如: OpenAI, Google)
- DATASET: 数据集 (如: ImageNet, GLUE)
- FRAMEWORK: 框架和工具 (如: PyTorch, TensorFlow)
- METRIC: 评估指标 (如: BLEU, F1-score)
- APPLICATION: 应用场景 (如: 机器翻译, 图像识别)

文本内容：
{text}

请输出JSON格式，包含entities数组，每个实体包含name, type, aliases(可选)字段："""
        else:
            prompt = f"""Extract important entities from the following AI technical text and output in JSON format.

Entity types include:
- ALGORITHM: Algorithms and models (e.g., Transformer, BERT, GPT)
- CONCEPT: Technical concepts (e.g., attention mechanism, backpropagation)
- PERSON: Person names (e.g., Geoffrey Hinton, Yann LeCun)
- ORGANIZATION: Organizations (e.g., OpenAI, Google)
- DATASET: Datasets (e.g., ImageNet, GLUE)
- FRAMEWORK: Frameworks and tools (e.g., PyTorch, TensorFlow)
- METRIC: Evaluation metrics (e.g., BLEU, F1-score)
- APPLICATION: Application scenarios (e.g., machine translation, image recognition)

Text content:
{text}

Please output JSON format with entities array, each entity contains name, type, aliases(optional) fields:"""
        
        try:
            # LLM生成
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            entities_json = self._extract_json_from_response(response)
            
            # 解析JSON并创建Entity对象
            entities = []
            if entities_json and 'entities' in entities_json:
                for entity_data in entities_json['entities']:
                    if 'name' in entity_data and 'type' in entity_data:
                        entity = Entity(
                            name=entity_data['name'].strip(),
                            entity_type=entity_data['type'].upper(),
                            aliases=set(entity_data.get('aliases', [])),
                            confidence=0.9  # LLM抽取的基础置信度
                        )
                        entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}, falling back to rule-based")
            return self._rule_based_extraction(text)
    
    def _rule_based_extraction(self, text: str) -> List[Entity]:
        """规则基础的实体抽取降级方案"""
        entities = []
        
        # AI算法关键词
        algorithm_patterns = [
            r'\b(Transformer|BERT|GPT|ResNet|LSTM|CNN|RNN|GAN)\b',
            r'\b(transformer|bert|gpt|resnet|lstm|cnn|rnn|gan)\b'
        ]
        
        # 技术概念关键词
        concept_patterns = [
            r'\b(attention|注意力)\b',
            r'\b(neural network|神经网络)\b',
            r'\b(deep learning|深度学习)\b',
            r'\b(machine learning|机器学习)\b'
        ]
        
        # 抽取算法实体
        for pattern in algorithm_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_name = match.group().strip()
                if len(entity_name) > 2:
                    entities.append(Entity(
                        name=entity_name,
                        entity_type="ALGORITHM",
                        confidence=0.7  # 规则抽取的较低置信度
                    ))
        
        # 抽取概念实体
        for pattern in concept_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_name = match.group().strip()
                if len(entity_name) > 2:
                    entities.append(Entity(
                        name=entity_name,
                        entity_type="CONCEPT",
                        confidence=0.6
                    ))
        
        return list({entity.name: entity for entity in entities}.values())  # 去重
```

**抽取策略**: 位于 [`knowledge_extractor.py:59-247`](./knowledge_extractor.py#L59-L247)
- **双重抽取**: LLM智能抽取 + 规则降级抽取
- **领域特化**: 针对AI技术领域的专业实体类型
- **多语言支持**: 中英文实体抽取能力
- **置信度评估**: 不同抽取方法的置信度区分

#### 关系抽取器
```python
class RelationExtractor(_SharedLLMComponent):
    """关系抽取器 - 识别实体间的语义关系"""
    
    def extract_relations(self, text: str, entities: List[Entity], language: str = "zh") -> List[Relation]:
        """从文本和实体列表中抽取关系"""
        
        if len(entities) < 2:
            return []
        
        entity_names = [entity.name for entity in entities]
        
        # 构建关系抽取提示
        if language == "zh":
            prompt = f"""请分析以下文本中实体之间的关系，并以JSON格式输出。

已识别的实体：{', '.join(entity_names)}

关系类型包括：
- IS_A: 是一种关系 (如: BERT是一种Transformer模型)
- PART_OF: 部分关系 (如: 注意力机制是Transformer的一部分)
- CREATED_BY: 创造关系 (如: GPT由OpenAI创造)
- APPLIED_TO: 应用关系 (如: BERT应用于自然语言处理)
- IMPROVES: 改进关系 (如: Transformer改进了RNN)
- EVALUATES: 评估关系 (如: BLEU评估机器翻译)
- USES: 使用关系 (如: 模型使用数据集训练)

文本内容：
{text}

请输出JSON格式的relations数组，每个关系包含head_entity, relation_type, tail_entity字段："""
        else:
            prompt = f"""Analyze the relationships between entities in the following text and output in JSON format.

Identified entities: {', '.join(entity_names)}

Relation types include:
- IS_A: is-a relationship (e.g., BERT is a Transformer model)
- PART_OF: part-of relationship (e.g., attention mechanism is part of Transformer)
- CREATED_BY: creation relationship (e.g., GPT created by OpenAI)
- APPLIED_TO: application relationship (e.g., BERT applied to NLP)
- IMPROVES: improvement relationship (e.g., Transformer improves RNN)
- EVALUATES: evaluation relationship (e.g., BLEU evaluates machine translation)
- USES: usage relationship (e.g., model uses dataset for training)

Text content:
{text}

Please output JSON format relations array, each relation contains head_entity, relation_type, tail_entity fields:"""
        
        try:
            # LLM关系抽取
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            relations_json = self._extract_json_from_response(response)
            
            # 解析关系
            relations = []
            if relations_json and 'relations' in relations_json:
                for rel_data in relations_json['relations']:
                    if all(key in rel_data for key in ['head_entity', 'relation_type', 'tail_entity']):
                        relation = Relation(
                            head_entity=rel_data['head_entity'].strip(),
                            relation_type=rel_data['relation_type'].upper(),
                            tail_entity=rel_data['tail_entity'].strip(),
                            confidence=0.8,
                            source_text=text[:200] + "..."  # 保留源文本片段
                        )
                        relations.append(relation)
            
            return relations
            
        except Exception as e:
            logger.warning(f"LLM relation extraction failed: {e}")
            return []
```

**关系识别**: 位于 [`knowledge_extractor.py:150+`](./knowledge_extractor.py#L150)
- **语义关系**: 定义AI领域的核心关系类型
- **上下文敏感**: 基于文本上下文识别实体关系
- **质量控制**: JSON格式验证和错误处理
- **源文本保持**: 保留关系的文本证据

### 3. 图数据库管理层

#### 知识图谱数据库
```python
class KnowledgeGraphDatabase:
    """知识图谱数据库 - SQLite持久化 + NetworkX内存图"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # NetworkX图用于内存操作
        self.graph = nx.DiGraph()
        
        # 初始化SQLite数据库
        self._init_database()
        
        # 实体和关系的内存索引
        self.entities_index = {}  # name -> Entity
        self.relations_index = []  # List[Relation]
        
        logger.info(f"Knowledge Graph Database initialized at {db_path}")
    
    def _init_database(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建实体表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    entity_type TEXT NOT NULL,
                    aliases TEXT,  -- JSON格式存储别名
                    properties TEXT,  -- JSON格式存储属性
                    confidence REAL DEFAULT 1.0,
                    source_chunks TEXT,  -- JSON格式存储来源块
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建关系表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    head_entity TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    tail_entity TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    source_text TEXT,
                    source_chunk_id TEXT,
                    properties TEXT,  -- JSON格式存储属性
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (head_entity) REFERENCES entities (name),
                    FOREIGN KEY (tail_entity) REFERENCES entities (name)
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_name ON entities (name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (entity_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_relations_head ON relations (head_entity)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_relations_tail ON relations (tail_entity)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_relations_type ON relations (relation_type)')
            
            conn.commit()
    
    def store_entity(self, entity: Entity) -> bool:
        """存储实体到数据库和内存图"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 准备数据
                aliases_json = json.dumps(list(entity.aliases)) if entity.aliases else "[]"
                properties_json = json.dumps(entity.properties) if entity.properties else "{}"
                source_chunks_json = json.dumps(entity.source_chunks) if entity.source_chunks else "[]"
                
                # 使用UPSERT语法更新或插入
                cursor.execute('''
                    INSERT OR REPLACE INTO entities 
                    (name, entity_type, aliases, properties, confidence, source_chunks, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    entity.name, entity.entity_type, aliases_json, 
                    properties_json, entity.confidence, source_chunks_json
                ))
                
                conn.commit()
            
            # 更新内存图和索引
            self.graph.add_node(entity.name, 
                               entity_type=entity.entity_type,
                               confidence=entity.confidence,
                               aliases=entity.aliases)
            self.entities_index[entity.name] = entity
            
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Failed to store entity {entity.name}: {e}")
            return False
    
    def store_relation(self, relation: Relation) -> bool:
        """存储关系到数据库和内存图"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                properties_json = json.dumps(relation.properties) if relation.properties else "{}"
                
                cursor.execute('''
                    INSERT INTO relations 
                    (head_entity, relation_type, tail_entity, confidence, source_text, source_chunk_id, properties)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    relation.head_entity, relation.relation_type, relation.tail_entity,
                    relation.confidence, relation.source_text, relation.source_chunk_id, properties_json
                ))
                
                conn.commit()
            
            # 更新内存图
            self.graph.add_edge(relation.head_entity, relation.tail_entity,
                               relation_type=relation.relation_type,
                               confidence=relation.confidence,
                               source_text=relation.source_text)
            self.relations_index.append(relation)
            
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Failed to store relation {relation.head_entity}->{relation.tail_entity}: {e}")
            return False
    
    def load_graph_from_db(self):
        """从数据库加载图到内存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 加载实体
                cursor.execute('SELECT name, entity_type, aliases, confidence FROM entities')
                for row in cursor.fetchall():
                    name, entity_type, aliases_json, confidence = row
                    aliases = set(json.loads(aliases_json)) if aliases_json else set()
                    
                    self.graph.add_node(name, 
                                       entity_type=entity_type,
                                       confidence=confidence,
                                       aliases=aliases)
                
                # 加载关系
                cursor.execute('''
                    SELECT head_entity, tail_entity, relation_type, confidence, source_text 
                    FROM relations
                ''')
                for row in cursor.fetchall():
                    head, tail, rel_type, confidence, source_text = row
                    
                    self.graph.add_edge(head, tail,
                                       relation_type=rel_type,
                                       confidence=confidence,
                                       source_text=source_text)
            
            logger.info(f"Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            
        except sqlite3.Error as e:
            logger.error(f"Failed to load graph from database: {e}")
```

**存储架构**: 位于 [`knowledge_extractor.py:409-622`](./knowledge_extractor.py#L409-L622)
- **双重存储**: SQLite持久化 + NetworkX内存图高效访问
- **ACID保证**: SQLite事务确保数据一致性
- **索引优化**: 多维度索引支持快速查询
- **JSON扩展**: 灵活的属性和元数据存储

### 4. 知识图谱索引器

#### 知识图谱构建
```python
class KnowledgeGraphIndexer:
    """知识图谱索引器 - 协调抽取和存储"""
    
    def __init__(self, config: Dict):
        model_name = config.get('llm_model', 'Qwen/Qwen2-7B-Instruct')
        device = config.get('device', 'auto')
        token = config.get('hugging_face_token')
        
        # 初始化抽取器
        self.entity_extractor = EntityExtractor(model_name, device, token)
        self.relation_extractor = RelationExtractor(model_name, device, token)
        
        # 初始化知识图谱数据库
        kg_db_path = config.get('knowledge_graph_db_path', './kg.db')
        self.kg_db = KnowledgeGraphDatabase(kg_db_path)
        
        self.config = config
        logger.info("Knowledge Graph Indexer initialized")
    
    async def build_knowledge_graph(self, chunks: List[Dict], max_concurrent: int = 5) -> Dict:
        """构建知识图谱 - 支持异步并发处理"""
        
        if not chunks:
            return {"entities_count": 0, "relations_count": 0}
        
        logger.info(f"Building knowledge graph from {len(chunks)} chunks")
        
        # 检查是否在事件循环中
        try:
            loop = asyncio.get_running_loop()
            return await self._build_knowledge_graph_async(chunks, max_concurrent)
        except RuntimeError:
            # 不在事件循环中，使用同步处理
            return self._build_knowledge_graph_sync(chunks)
    
    async def _build_knowledge_graph_async(self, chunks: List[Dict], max_concurrent: int) -> Dict:
        """异步构建知识图谱"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # 并发处理所有chunks
        tasks = [
            self._extract_chunk_async(chunk, semaphore) 
            for chunk in chunks
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 汇总统计
        total_entities = 0
        total_relations = 0
        
        for result in results:
            if isinstance(result, dict):
                total_entities += result.get('entities_count', 0)
                total_relations += result.get('relations_count', 0)
            elif isinstance(result, Exception):
                logger.error(f"Chunk processing failed: {result}")
        
        # 最终加载图到内存
        self.kg_db.load_graph_from_db()
        
        logger.success(f"Knowledge graph built: {total_entities} entities, {total_relations} relations")
        
        return {
            "entities_count": total_entities,
            "relations_count": total_relations,
            "graph_nodes": self.kg_db.graph.number_of_nodes(),
            "graph_edges": self.kg_db.graph.number_of_edges()
        }
    
    async def _extract_chunk_async(self, chunk: Dict, semaphore: asyncio.Semaphore) -> Dict:
        """异步处理单个文档块"""
        
        async with semaphore:
            # 使用线程池执行同步的抽取操作
            return await asyncio.to_thread(self._extract_from_chunk, chunk)
    
    def _extract_from_chunk(self, chunk: Dict) -> Dict:
        """从单个文档块中抽取知识"""
        
        content = chunk.get('content', '')
        chunk_id = chunk.get('chunk_id', '')
        
        if not content:
            return {"entities_count": 0, "relations_count": 0}
        
        try:
            # 1. 抽取实体
            entities = self.entity_extractor.extract_entities(content)
            
            # 2. 存储实体
            stored_entities = 0
            for entity in entities:
                entity.source_chunks.append(chunk_id)
                if self.kg_db.store_entity(entity):
                    stored_entities += 1
            
            # 3. 抽取关系
            relations = self.relation_extractor.extract_relations(content, entities)
            
            # 4. 存储关系
            stored_relations = 0
            for relation in relations:
                relation.source_chunk_id = chunk_id
                if self.kg_db.store_relation(relation):
                    stored_relations += 1
            
            return {
                "entities_count": stored_entities,
                "relations_count": stored_relations
            }
            
        except Exception as e:
            logger.error(f"Failed to extract from chunk {chunk_id}: {e}")
            return {"entities_count": 0, "relations_count": 0}
```

**构建流程**: 位于 [`knowledge_extractor.py:624-733`](./knowledge_extractor.py#L624-L733)
- **异步并发**: 基于信号量的并发控制
- **线程池**: 同步抽取操作的异步化
- **错误隔离**: 单个chunk失败不影响整体流程
- **统计追踪**: 详细的构建统计信息

### 5. 知识图谱检索层

#### 检索结果结构
```python
@dataclass
class KGRetrievalResult:
    """知识图谱检索结果"""
    entity: str                     # 实体名称
    entity_type: str               # 实体类型
    related_info: str              # 相关信息
    confidence: float              # 置信度
    source_type: str               # 来源类型: 'entity', 'relation', 'path'
    metadata: Dict[str, Any]       # 附加元数据
```

#### 知识图谱检索器
```python
class KnowledgeGraphRetriever:
    """知识图谱检索器 - 基于图结构的知识检索"""
    
    def retrieve_kg_context(
        self,
        query: str,
        top_k: int = 10,
        include_relations: bool = True,
        include_paths: bool = True,
        max_hops: int = 2
    ) -> List[KGRetrievalResult]:
        """从知识图谱检索相关上下文"""
        
        try:
            # 确保图已加载到内存
            if self.kg_indexer.kg_db.graph.number_of_nodes() == 0:
                self.kg_indexer.kg_db.load_graph_from_db()
            
            # 查询知识图谱
            kg_results = self.kg_indexer.query_knowledge_graph(
                query=query,
                max_entities=top_k,
                max_hops=max_hops
            )
            
            # 转换为检索结果格式
            retrieval_results = []
            
            # 处理实体结果
            for entity_name, entity_info in kg_results.get('entities', {}).items():
                result = KGRetrievalResult(
                    entity=entity_name,
                    entity_type=entity_info.get('type', 'UNKNOWN'),
                    related_info=f"实体类型: {entity_info.get('type', 'UNKNOWN')}",
                    confidence=entity_info.get('confidence', 1.0),
                    source_type='entity',
                    metadata={'aliases': entity_info.get('aliases', [])}
                )
                retrieval_results.append(result)
            
            # 处理关系结果
            if include_relations:
                for relation in kg_results.get('relations', []):
                    result = KGRetrievalResult(
                        entity=f"{relation['head']} -> {relation['tail']}",
                        entity_type='RELATION',
                        related_info=f"关系: {relation['head']} {relation['type']} {relation['tail']}",
                        confidence=relation.get('confidence', 1.0),
                        source_type='relation',
                        metadata={'relation_type': relation['type']}
                    )
                    retrieval_results.append(result)
            
            # 处理路径结果
            if include_paths:
                for path in kg_results.get('paths', []):
                    path_str = " -> ".join(path['nodes'])
                    result = KGRetrievalResult(
                        entity=path_str,
                        entity_type='PATH',
                        related_info=f"推理路径: {path_str}",
                        confidence=path.get('confidence', 0.8),
                        source_type='path',
                        metadata={'path_length': len(path['nodes'])}
                    )
                    retrieval_results.append(result)
            
            return retrieval_results[:top_k]
            
        except Exception as e:
            logger.error(f"KG retrieval failed: {e}")
            return []
    
    def enhance_chunks_with_kg(self, chunks: List[Dict], query: str) -> List[Dict]:
        """使用知识图谱增强文档块"""
        
        # 获取查询相关的知识图谱上下文
        kg_context = self.retrieve_kg_context(query, top_k=20)
        
        enhanced_chunks = []
        
        for chunk in chunks:
            content = chunk.get('content', '')
            enhanced_content = content
            enhancement_count = 0
            
            # 查找内容中包含的实体
            for kg_result in kg_context:
                entity_name = kg_result.entity
                
                # 实体匹配（考虑别名）
                if (entity_name.lower() in content.lower() or
                    any(alias.lower() in content.lower() 
                        for alias in kg_result.metadata.get('aliases', []))):
                    
                    # 添加知识图谱增强信息
                    kg_info = f"\n[KG] {kg_result.related_info}"
                    enhanced_content += kg_info
                    enhancement_count += 1
            
            # 创建增强后的chunk
            enhanced_chunk = chunk.copy()
            enhanced_chunk['content'] = enhanced_content
            enhanced_chunk['kg_enhancement_count'] = enhancement_count
            
            # 提升有知识图谱增强的chunk的分数
            if enhancement_count > 0 and 'score' in enhanced_chunk:
                enhanced_chunk['score'] *= (1 + 0.1 * enhancement_count)
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def generate_kg_summary(self, entities: List[str], relations: List[Dict]) -> str:
        """生成知识图谱摘要"""
        
        if not entities and not relations:
            return ""
        
        summary_parts = []
        
        if entities:
            entities_str = ", ".join(entities[:5])  # 最多显示5个实体
            if len(entities) > 5:
                entities_str += f" 等{len(entities)}个实体"
            summary_parts.append(f"相关实体: {entities_str}")
        
        if relations:
            relation_types = list(set(rel.get('type', 'UNKNOWN') for rel in relations))
            relations_str = ", ".join(relation_types[:3])  # 最多显示3种关系类型
            summary_parts.append(f"关系类型: {relations_str}")
        
        return "; ".join(summary_parts)
```

**检索能力**: 位于 [`kg_retriever.py:40-154`](./kg_retriever.py#L40-L154)
- **多模式检索**: 实体匹配、关系查找、路径搜索
- **上下文增强**: 为文档块添加结构化知识
- **相关性计算**: 基于图结构的相关性评分
- **摘要生成**: 结构化的知识图谱摘要

## 性能与优化

### 1. 抽取性能优化
- **模型共享**: 通过 `ModelRegistry` 避免重复加载LLM
- **并发处理**: 异步信号量控制的并发抽取
- **降级策略**: LLM失败时自动降级到规则抽取
- **批量存储**: 事务批量提交减少I/O开销

### 2. 存储优化
- **混合存储**: SQLite持久化 + NetworkX内存操作
- **索引优化**: 多维度索引加速查询
- **JSON扩展**: 灵活的属性和元数据存储
- **增量更新**: 支持实体和关系的增量构建

### 3. 检索优化
- **图算法**: NetworkX高效图遍历和路径搜索
- **缓存机制**: 图结构的内存缓存
- **相关性排序**: 基于置信度和路径长度的排序
- **结果过滤**: 多层次的结果过滤和去重

## 使用示例

```python
from src.knowledge_graph.knowledge_extractor import KnowledgeGraphIndexer
from src.knowledge_graph.kg_retriever import KnowledgeGraphRetriever

# 初始化知识图谱系统
config = {
    'llm_model': 'Qwen/Qwen2-7B-Instruct',
    'embedding_model': 'BAAI/bge-m3',
    'knowledge_graph_db_path': './kg.db',
    'device': 'auto'
}

# 构建知识图谱
kg_indexer = KnowledgeGraphIndexer(config)
chunks = [
    {'chunk_id': 'doc1_chunk1', 'content': 'Transformer是一种基于注意力机制的神经网络架构...'},
    {'chunk_id': 'doc1_chunk2', 'content': 'BERT是Google开发的预训练语言模型...'}
]

# 异步构建
build_result = await kg_indexer.build_knowledge_graph(chunks)
print(f"构建完成: {build_result['entities_count']} 个实体, {build_result['relations_count']} 个关系")

# 知识图谱检索
kg_retriever = KnowledgeGraphRetriever(config)
kg_results = kg_retriever.retrieve_kg_context(
    query="什么是Transformer架构",
    top_k=10,
    include_relations=True
)

# 查看检索结果
for result in kg_results:
    print(f"实体: {result.entity}")
    print(f"类型: {result.entity_type}")
    print(f"信息: {result.related_info}")
    print(f"置信度: {result.confidence}")
    print("---")

# 文档块增强
enhanced_chunks = kg_retriever.enhance_chunks_with_kg(chunks, query)
print(f"增强后文档数: {len(enhanced_chunks)}")
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `knowledge_graph_db_path` | "./kg.db" | 知识图谱数据库路径 |
| `max_concurrent` | 5 | 最大并发抽取数 |
| `entity_confidence_threshold` | 0.6 | 实体置信度阈值 |
| `relation_confidence_threshold` | 0.7 | 关系置信度阈值 |
| `max_hops` | 2 | 图遍历最大跳数 |
| `enable_rule_fallback` | true | 是否启用规则降级 |

## 扩展指南

### 添加新的实体类型
1. 在 `EntityExtractor.entity_types` 中添加新类型
2. 更新抽取提示模板包含新类型
3. 添加对应的规则匹配模式
4. 更新数据库索引支持新类型

### 优化建议
- **领域定制**: 根据具体领域调整实体类型和关系类型
- **模型选择**: 使用更强的LLM提升抽取质量
- **图算法**: 根据查询模式优化图遍历算法
- **存储优化**: 对大规模图考虑使用专业图数据库

知识图谱模块通过结构化知识的抽取、存储和检索，为 RAG 系统提供了强大的推理能力和可解释性支持，是系统智能化的重要组成部分。