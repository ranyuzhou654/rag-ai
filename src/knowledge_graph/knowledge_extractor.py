# src/knowledge_graph/knowledge_extractor.py
from typing import List, Dict, Optional, Tuple, Set, Any
import asyncio
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from loguru import logger
import sqlite3
from datetime import datetime
import hashlib

@dataclass
class Entity:
    """实体结构"""
    name: str
    entity_type: str
    aliases: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_chunks: List[str] = field(default_factory=list)

@dataclass
class Relation:
    """关系结构"""
    head_entity: str
    relation_type: str
    tail_entity: str
    confidence: float = 1.0
    source_text: str = ""
    source_chunk_id: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeTriplet:
    """知识三元组"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = ""

class EntityExtractor:
    """实体抽取器"""
    
    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        logger.info(f"Loading Entity Extractor: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            token=token
        )
        
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 定义AI领域的实体类型
        self.entity_types = {
            "MODEL": "机器学习模型",
            "ALGORITHM": "算法",
            "TECHNIQUE": "技术方法", 
            "DATASET": "数据集",
            "METRIC": "评估指标",
            "PERSON": "人名",
            "ORGANIZATION": "机构",
            "PAPER": "论文",
            "CONCEPT": "概念",
            "TOOL": "工具",
            "FRAMEWORK": "框架"
        }
    
    def extract_entities(self, text: str, chunk_id: str) -> List[Entity]:
        """从文本中抽取实体"""
        # 检测语言
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        
        if is_chinese:
            prompt = f"""请从以下AI技术文本中识别和抽取重要实体。实体类型包括：
- MODEL: 机器学习模型（如GPT、BERT、Transformer等）
- ALGORITHM: 算法（如Adam、SGD、Attention等）
- TECHNIQUE: 技术方法（如微调、量化、蒸馏等）
- DATASET: 数据集（如ImageNet、GLUE等）
- METRIC: 评估指标（如BLEU、ROUGE、Accuracy等）
- PERSON: 研究者姓名
- ORGANIZATION: 机构名称
- CONCEPT: 重要概念（如注意力机制、反向传播等）

文本：{text}

请按照以下JSON格式输出：
{{
    "entities": [
        {{"name": "实体名称", "type": "实体类型", "confidence": 0.9}},
        ...
    ]
}}"""
        else:
            prompt = f"""Please identify and extract important entities from the following AI technology text. Entity types include:
- MODEL: Machine learning models (e.g., GPT, BERT, Transformer)
- ALGORITHM: Algorithms (e.g., Adam, SGD, Attention)
- TECHNIQUE: Techniques (e.g., fine-tuning, quantization, distillation)
- DATASET: Datasets (e.g., ImageNet, GLUE)
- METRIC: Evaluation metrics (e.g., BLEU, ROUGE, Accuracy)
- PERSON: Researcher names
- ORGANIZATION: Institution names  
- CONCEPT: Important concepts (e.g., attention mechanism, backpropagation)

Text: {text}

Please output in the following JSON format:
{{
    "entities": [
        {{"name": "entity name", "type": "entity type", "confidence": 0.9}},
        ...
    ]
}}"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取JSON部分
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                entities = []
                for ent_data in data.get('entities', []):
                    entity = Entity(
                        name=ent_data['name'],
                        entity_type=ent_data.get('type', 'CONCEPT'),
                        confidence=ent_data.get('confidence', 0.8),
                        source_chunks=[chunk_id]
                    )
                    entities.append(entity)
                
                logger.info(f"Extracted {len(entities)} entities from chunk {chunk_id}")
                return entities
                
        except Exception as e:
            logger.error(f"Entity extraction failed for chunk {chunk_id}: {e}")
        
        # 后备：使用规则提取
        return self._rule_based_extraction(text, chunk_id)
    
    def _rule_based_extraction(self, text: str, chunk_id: str) -> List[Entity]:
        """基于规则的实体抽取（后备方案）"""
        entities = []
        
        # 常见AI模型名称
        model_patterns = [
            r'\b(GPT-?\d*|BERT|Transformer|ResNet|VGG|AlexNet|LSTM|GRU|CNN|RNN)\b',
            r'\b(LLaMA|ChatGPT|Claude|Gemini|PaLM|T5|CLIP|DALL-E)\b',
            r'\b(ViT|DeiT|Swin|EfficientNet|MobileNet|DenseNet)\b'
        ]
        
        # 算法和技术
        algorithm_patterns = [
            r'\b(Adam|SGD|RMSprop|Adagrad|LBFGS|Momentum)\b',
            r'\b(LoRA|QLoRA|AdaLoRA|DoRA|Adapter|Prefix)\b',
            r'\b(Attention|Self-attention|Cross-attention|Multi-head)\b'
        ]
        
        # 概念术语
        concept_patterns = [
            r'\b(机器学习|深度学习|神经网络|人工智能)\b',
            r'\b(监督学习|无监督学习|强化学习|迁移学习)\b',
            r'\b(自然语言处理|计算机视觉|语音识别)\b'
        ]
        
        pattern_types = [
            (model_patterns, "MODEL"),
            (algorithm_patterns, "ALGORITHM"), 
            (concept_patterns, "CONCEPT")
        ]
        
        for patterns, entity_type in pattern_types:
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_name = match.group()
                    entities.append(Entity(
                        name=entity_name,
                        entity_type=entity_type,
                        confidence=0.7,
                        source_chunks=[chunk_id]
                    ))
        
        return entities

class RelationExtractor:
    """关系抽取器"""
    
    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        logger.info(f"Loading Relation Extractor: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            token=token
        )
        
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 定义关系类型
        self.relation_types = {
            "IS_A": "是一种",
            "PART_OF": "是...的一部分",
            "USES": "使用",
            "IMPROVES": "改进",
            "BASED_ON": "基于",
            "EVALUATES": "评估",
            "PROPOSED_BY": "由...提出",
            "IMPLEMENTS": "实现",
            "OUTPERFORMS": "优于",
            "SIMILAR_TO": "类似于",
            "APPLIES_TO": "应用于",
            "REQUIRES": "需要"
        }
    
    def extract_relations(
        self, 
        text: str, 
        entities: List[Entity], 
        chunk_id: str
    ) -> List[Relation]:
        """从文本中抽取实体间关系"""
        if len(entities) < 2:
            return []
        
        # 检测语言
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        
        entity_names = [e.name for e in entities]
        
        if is_chinese:
            prompt = f"""请从以下AI技术文本中识别实体之间的关系。

已识别实体：{', '.join(entity_names)}

关系类型包括：
- IS_A: A是B的一种类型
- PART_OF: A是B的组成部分
- USES: A使用B
- IMPROVES: A改进了B
- BASED_ON: A基于B
- EVALUATES: A评估B
- PROPOSED_BY: A由B提出
- OUTPERFORMS: A优于B

文本：{text}

请按照以下JSON格式输出关系：
{{
    "relations": [
        {{"head": "实体1", "relation": "关系类型", "tail": "实体2", "confidence": 0.9}},
        ...
    ]
}}"""
        else:
            prompt = f"""Please identify relationships between entities in the following AI technology text.

Identified entities: {', '.join(entity_names)}

Relation types include:
- IS_A: A is a type of B
- PART_OF: A is part of B
- USES: A uses B
- IMPROVES: A improves B
- BASED_ON: A is based on B
- EVALUATES: A evaluates B
- PROPOSED_BY: A is proposed by B
- OUTPERFORMS: A outperforms B

Text: {text}

Please output relationships in the following JSON format:
{{
    "relations": [
        {{"head": "entity1", "relation": "relation_type", "tail": "entity2", "confidence": 0.9}},
        ...
    ]
}}"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取JSON部分
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                relations = []
                for rel_data in data.get('relations', []):
                    relation = Relation(
                        head_entity=rel_data['head'],
                        relation_type=rel_data.get('relation', 'RELATED_TO'),
                        tail_entity=rel_data['tail'],
                        confidence=rel_data.get('confidence', 0.8),
                        source_text=text[:200] + "..." if len(text) > 200 else text,
                        source_chunk_id=chunk_id
                    )
                    relations.append(relation)
                
                logger.info(f"Extracted {len(relations)} relations from chunk {chunk_id}")
                return relations
                
        except Exception as e:
            logger.error(f"Relation extraction failed for chunk {chunk_id}: {e}")
        
        # 后备：基于规则的关系抽取
        return self._rule_based_relation_extraction(text, entities, chunk_id)
    
    def _rule_based_relation_extraction(
        self, 
        text: str, 
        entities: List[Entity], 
        chunk_id: str
    ) -> List[Relation]:
        """基于规则的关系抽取"""
        relations = []
        entity_names = [e.name for e in entities]
        
        # 基于语言模式的关系识别
        relation_patterns = [
            (r'(\w+)\s+是\s+(\w+)', 'IS_A'),
            (r'(\w+)\s+使用\s+(\w+)', 'USES'),
            (r'(\w+)\s+基于\s+(\w+)', 'BASED_ON'),
            (r'(\w+)\s+改进\s+(\w+)', 'IMPROVES'),
            (r'(\w+)\s+is\s+a\s+(\w+)', 'IS_A'),
            (r'(\w+)\s+uses\s+(\w+)', 'USES'),
            (r'(\w+)\s+based\s+on\s+(\w+)', 'BASED_ON'),
            (r'(\w+)\s+improves\s+(\w+)', 'IMPROVES'),
        ]
        
        for pattern, relation_type in relation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                head, tail = match.groups()
                if head in entity_names and tail in entity_names:
                    relations.append(Relation(
                        head_entity=head,
                        relation_type=relation_type,
                        tail_entity=tail,
                        confidence=0.6,
                        source_text=match.group(),
                        source_chunk_id=chunk_id
                    ))
        
        return relations

class KnowledgeGraphDatabase:
    """知识图谱数据库"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        self._init_database()
        self.graph = nx.MultiDiGraph()  # 支持多重边的有向图
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            # 实体表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    entity_type TEXT,
                    aliases TEXT,
                    properties TEXT,
                    confidence REAL,
                    source_chunks TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            # 关系表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    head_entity TEXT,
                    relation_type TEXT,
                    tail_entity TEXT,
                    confidence REAL,
                    source_text TEXT,
                    source_chunk_id TEXT,
                    properties TEXT,
                    created_at TEXT,
                    FOREIGN KEY (head_entity) REFERENCES entities (name),
                    FOREIGN KEY (tail_entity) REFERENCES entities (name)
                )
            ''')
            
            # 创建索引
            conn.execute('CREATE INDEX IF NOT EXISTS idx_entities_name ON entities (name)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (entity_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_relations_head ON relations (head_entity)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_relations_tail ON relations (tail_entity)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_relations_type ON relations (relation_type)')
            
            conn.commit()
        
        logger.info(f"Knowledge Graph database initialized at {self.db_path}")
    
    def store_entity(self, entity: Entity):
        """存储实体"""
        with sqlite3.connect(self.db_path) as conn:
            now = datetime.now().isoformat()
            
            # 检查实体是否已存在
            cursor = conn.execute('SELECT * FROM entities WHERE name = ?', (entity.name,))
            existing = cursor.fetchone()
            
            if existing:
                # 更新现有实体
                conn.execute('''
                    UPDATE entities SET
                    entity_type = ?,
                    aliases = ?,
                    properties = ?,
                    confidence = ?,
                    source_chunks = ?,
                    updated_at = ?
                    WHERE name = ?
                ''', (
                    entity.entity_type,
                    json.dumps(list(entity.aliases)),
                    json.dumps(entity.properties),
                    entity.confidence,
                    json.dumps(entity.source_chunks),
                    now,
                    entity.name
                ))
            else:
                # 插入新实体
                conn.execute('''
                    INSERT INTO entities 
                    (name, entity_type, aliases, properties, confidence, source_chunks, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entity.name,
                    entity.entity_type,
                    json.dumps(list(entity.aliases)),
                    json.dumps(entity.properties),
                    entity.confidence,
                    json.dumps(entity.source_chunks),
                    now,
                    now
                ))
            
            conn.commit()
            
            # 添加到内存图
            self.graph.add_node(entity.name, **{
                'entity_type': entity.entity_type,
                'confidence': entity.confidence,
                'properties': entity.properties
            })
    
    def store_relation(self, relation: Relation):
        """存储关系"""
        with sqlite3.connect(self.db_path) as conn:
            now = datetime.now().isoformat()
            
            conn.execute('''
                INSERT INTO relations
                (head_entity, relation_type, tail_entity, confidence, source_text, source_chunk_id, properties, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                relation.head_entity,
                relation.relation_type,
                relation.tail_entity,
                relation.confidence,
                relation.source_text,
                relation.source_chunk_id,
                json.dumps(relation.properties),
                now
            ))
            
            conn.commit()
            
            # 添加到内存图
            self.graph.add_edge(
                relation.head_entity,
                relation.tail_entity,
                relation_type=relation.relation_type,
                confidence=relation.confidence,
                source=relation.source_chunk_id
            )
    
    def load_graph_from_db(self):
        """从数据库加载图到内存"""
        with sqlite3.connect(self.db_path) as conn:
            # 加载实体
            cursor = conn.execute('SELECT name, entity_type, confidence, properties FROM entities')
            for row in cursor.fetchall():
                name, entity_type, confidence, properties = row
                properties = json.loads(properties) if properties else {}
                self.graph.add_node(name, **{
                    'entity_type': entity_type,
                    'confidence': confidence,
                    'properties': properties
                })
            
            # 加载关系
            cursor = conn.execute('SELECT head_entity, tail_entity, relation_type, confidence, source_chunk_id FROM relations')
            for row in cursor.fetchall():
                head, tail, relation_type, confidence, source = row
                self.graph.add_edge(head, tail, **{
                    'relation_type': relation_type,
                    'confidence': confidence,
                    'source': source
                })
        
        logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def find_related_entities(
        self, 
        entity_name: str, 
        max_hops: int = 2,
        min_confidence: float = 0.5
    ) -> List[Dict]:
        """查找相关实体"""
        if entity_name not in self.graph:
            return []
        
        related = []
        
        # 1跳邻居
        for neighbor in self.graph.neighbors(entity_name):
            for edge_data in self.graph[entity_name][neighbor].values():
                if edge_data.get('confidence', 0) >= min_confidence:
                    related.append({
                        'entity': neighbor,
                        'relation': edge_data.get('relation_type'),
                        'confidence': edge_data.get('confidence'),
                        'hops': 1,
                        'path': [entity_name, neighbor]
                    })
        
        # 2跳邻居（如果需要）
        if max_hops >= 2:
            for neighbor in list(self.graph.neighbors(entity_name)):
                for second_neighbor in self.graph.neighbors(neighbor):
                    if second_neighbor != entity_name and second_neighbor not in [r['entity'] for r in related]:
                        for edge_data in self.graph[neighbor][second_neighbor].values():
                            if edge_data.get('confidence', 0) >= min_confidence:
                                related.append({
                                    'entity': second_neighbor,
                                    'relation': edge_data.get('relation_type'),
                                    'confidence': edge_data.get('confidence'),
                                    'hops': 2,
                                    'path': [entity_name, neighbor, second_neighbor]
                                })
        
        return related
    
    def get_entity_info(self, entity_name: str) -> Optional[Dict]:
        """获取实体信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT entity_type, aliases, properties, confidence, source_chunks FROM entities WHERE name = ?',
                (entity_name,)
            )
            row = cursor.fetchone()
            if row:
                entity_type, aliases, properties, confidence, source_chunks = row
                return {
                    'name': entity_name,
                    'entity_type': entity_type,
                    'aliases': json.loads(aliases) if aliases else [],
                    'properties': json.loads(properties) if properties else {},
                    'confidence': confidence,
                    'source_chunks': json.loads(source_chunks) if source_chunks else []
                }
        return None
    
    def search_entities_by_type(self, entity_type: str, limit: int = 50) -> List[str]:
        """按类型搜索实体"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT name FROM entities WHERE entity_type = ? ORDER BY confidence DESC LIMIT ?',
                (entity_type, limit)
            )
            return [row[0] for row in cursor.fetchall()]

class KnowledgeGraphIndexer:
    """知识图谱索引器"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 初始化数据库
        kg_db_path = config.get('knowledge_graph_db_path', 'data/knowledge_graph/kg.db')
        self.kg_db = KnowledgeGraphDatabase(Path(kg_db_path))
        
        # 初始化抽取器
        model_name = config.get('llm_model')
        token = config.get('HUGGING_FACE_TOKEN')
        device = config.get('device', 'auto')
        
        self.extractors_available = False
        if model_name:
            try:
                self.entity_extractor = EntityExtractor(
                    model_name=model_name, device=device, token=token
                )
                self.relation_extractor = RelationExtractor(
                    model_name=model_name, device=device, token=token
                )
                self.extractors_available = True
                logger.success("Knowledge Graph Indexer with LLM extractors initialized")
            except Exception as e:
                logger.error(f"Failed to initialize KG extractors: {e}")
        else:
            logger.warning("Knowledge Graph Indexer initialized without LLM extractors")
    
    def build_knowledge_graph(self, chunks: List[Dict]):
        """构建知识图谱"""
        if not self.extractors_available:
            logger.warning("KG extractors not available, skipping knowledge graph construction")
            return
        
        logger.info(f"Building knowledge graph from {len(chunks)} chunks...")
        
        total_entities = 0
        total_relations = 0
        
        for i, chunk in enumerate(chunks):
            if i % 50 == 0:
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            chunk_id = chunk['chunk_id']
            content = chunk['content']
            
            try:
                # 抽取实体
                entities = self.entity_extractor.extract_entities(content, chunk_id)
                
                # 存储实体
                for entity in entities:
                    self.kg_db.store_entity(entity)
                
                total_entities += len(entities)
                
                # 抽取关系
                if len(entities) >= 2:
                    relations = self.relation_extractor.extract_relations(content, entities, chunk_id)
                    
                    # 存储关系
                    for relation in relations:
                        self.kg_db.store_relation(relation)
                    
                    total_relations += len(relations)
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id}: {e}")
                continue
        
        # 加载图到内存
        self.kg_db.load_graph_from_db()
        
        logger.success(f"Knowledge graph constructed: {total_entities} entities, {total_relations} relations")
        logger.info(f"Graph stats: {self.kg_db.graph.number_of_nodes()} nodes, {self.kg_db.graph.number_of_edges()} edges")
    
    def query_knowledge_graph(
        self, 
        query: str, 
        max_entities: int = 10,
        max_hops: int = 2
    ) -> List[Dict]:
        """查询知识图谱"""
        # 从查询中识别实体
        mentioned_entities = []
        
        # 简单的实体识别（可以改进）
        with sqlite3.connect(self.kg_db.db_path) as conn:
            cursor = conn.execute('SELECT name FROM entities')
            all_entities = [row[0] for row in cursor.fetchall()]
        
        query_lower = query.lower()
        for entity in all_entities:
            if entity.lower() in query_lower:
                mentioned_entities.append(entity)
        
        if not mentioned_entities:
            return []
        
        # 收集相关信息
        kg_results = []
        
        for entity in mentioned_entities[:5]:  # 限制处理的实体数量
            # 获取实体信息
            entity_info = self.kg_db.get_entity_info(entity)
            if entity_info:
                kg_results.append({
                    'type': 'entity',
                    'entity': entity,
                    'info': entity_info
                })
            
            # 获取相关实体
            related_entities = self.kg_db.find_related_entities(
                entity, max_hops=max_hops, min_confidence=0.6
            )
            
            for related in related_entities[:max_entities]:
                kg_results.append({
                    'type': 'relation',
                    'source_entity': entity,
                    'target_entity': related['entity'],
                    'relation': related['relation'],
                    'confidence': related['confidence'],
                    'hops': related['hops'],
                    'path': related['path']
                })
        
        return kg_results
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        if not hasattr(self, 'kg_db') or not self.kg_db.graph:
            return {}
        
        # 确保图已加载
        if self.kg_db.graph.number_of_nodes() == 0:
            self.kg_db.load_graph_from_db()
        
        # 实体类型分布
        entity_types = {}
        for node, data in self.kg_db.graph.nodes(data=True):
            entity_type = data.get('entity_type', 'UNKNOWN')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # 关系类型分布
        relation_types = {}
        for u, v, data in self.kg_db.graph.edges(data=True):
            relation_type = data.get('relation_type', 'UNKNOWN')
            relation_types[relation_type] = relation_types.get(relation_type, 0) + 1
        
        return {
            'total_entities': self.kg_db.graph.number_of_nodes(),
            'total_relations': self.kg_db.graph.number_of_edges(),
            'entity_type_distribution': entity_types,
            'relation_type_distribution': relation_types,
            'is_connected': nx.is_connected(self.kg_db.graph.to_undirected()),
            'average_degree': sum(dict(self.kg_db.graph.degree()).values()) / max(self.kg_db.graph.number_of_nodes(), 1)
        }

# 使用示例
async def main():
    """测试知识图谱索引器"""
    config = {
        'llm_model': 'Qwen/Qwen2-7B-Instruct',
        'device': 'auto',
        'HUGGING_FACE_TOKEN': None,
        'knowledge_graph_db_path': 'data/knowledge_graph/test_kg.db'
    }
    
    indexer = KnowledgeGraphIndexer(config)
    
    # 测试数据
    test_chunks = [
        {
            'chunk_id': 'chunk_1',
            'content': 'Transformer模型由Vaswani等人在2017年提出，它使用注意力机制来处理序列数据。BERT模型基于Transformer架构。',
            'source_id': 'doc_1'
        },
        {
            'chunk_id': 'chunk_2', 
            'content': 'GPT模型是一种基于Transformer的生成式预训练语言模型。OpenAI开发了GPT系列模型。',
            'source_id': 'doc_2'
        }
    ]
    
    # 构建知识图谱
    indexer.build_knowledge_graph(test_chunks)
    
    # 查询知识图谱
    results = indexer.query_knowledge_graph("Transformer和BERT的关系是什么？")
    
    print("知识图谱查询结果:")
    for result in results:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 获取统计信息
    stats = indexer.get_graph_statistics()
    print("\n知识图谱统计:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())