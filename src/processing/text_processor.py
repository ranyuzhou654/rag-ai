# src/processing/text_processor.py
import re
from typing import List, Dict, Optional, Iterable
from dataclasses import dataclass, field
import numpy as np
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from pathlib import Path
from .multi_representation_indexer import MultiRepresentationIndexer
from src.optimization.model_registry import ModelRegistry
from numpy.linalg import norm


def _sentence_tokenize(text: str) -> Iterable[str]:
    """Simple sentence tokenizer that keeps punctuation."""
    pattern = re.compile(r'[^。！？!?\n]+(?:[。！？!?]\s*)?')
    return [seg.strip() for seg in pattern.findall(text) if seg.strip()]

@dataclass
class TextChunk:
    """优化的文本块结构"""
    content: str
    chunk_id: str
    source_id: str
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

class HierarchicalTextSplitter:
    """
    层次化文本分割器
    - 首先按逻辑章节分割 (Abstract, Introduction, etc.)
    - 然后在章节内进行递归字符分割
    """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.section_splitter = re.compile(r"\n(##?|Abstract|Introduction|Conclusion|Methodology|Discussion|Related Work)\n", re.IGNORECASE)
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split_document(self, doc_content: str, source_id: str, metadata: Dict) -> List[TextChunk]:
        """执行分层分割"""
        chunks = []
        
        # 1. 按章节分割
        sections = self.section_splitter.split(doc_content)
        
        # 处理分割后的列表，将分隔符与内容合并
        processed_sections = []
        i = 0
        while i < len(sections):
            if i + 1 < len(sections) and self.section_splitter.match("\n" + sections[i+1] + "\n"):
                # sections[i+1] is a delimiter
                header = sections[i+1].strip()
                content = sections[i+2]
                processed_sections.append((header, content))
                i += 3
            else:
                # No specific header, treat as 'content'
                processed_sections.append(("content", sections[i]))
                i += 1

        chunk_counter = 0
        for header, content in processed_sections:
            if not content.strip():
                continue
            
            # 2. 在章节内递归分割
            sub_chunks = self.recursive_splitter.split_text(content)
            
            for sub_chunk in sub_chunks:
                chunk_id = f"{source_id}_chunk_{chunk_counter}"
                chunk_metadata = {
                    **metadata,
                    'section': header.lower(),
                    'chunk_index': chunk_counter,
                }
                chunks.append(TextChunk(
                    content=sub_chunk,
                    chunk_id=chunk_id,
                    source_id=source_id,
                    metadata=chunk_metadata,
                ))
                chunk_counter += 1
                
        logger.info(f"Document {source_id} split into {len(chunks)} chunks.")
        return chunks


class SemanticGraphSplitter:
    """语义图驱动的分割器，结合句向量相似度形成更自然的片段"""

    def __init__(
        self,
        embedder: 'MultilingualEmbedder',
        chunk_size: int = 512,
        similarity_threshold: float = 0.7,
        max_sentences: int = 12
    ):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        self.max_sentences = max_sentences

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = norm(a) * norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def split_document(self, doc_content: str, source_id: str, metadata: Dict) -> List['TextChunk']:
        sentences = _sentence_tokenize(doc_content)
        if not sentences:
            return []

        embeddings = self.embedder.embed_sentences(sentences)

        chunks: List[TextChunk] = []
        buffer_sentences: List[str] = []
        buffer_embeddings: List[np.ndarray] = []
        chunk_counter = 0

        for sentence, embedding in zip(sentences, embeddings):
            if not buffer_sentences:
                buffer_sentences.append(sentence)
                buffer_embeddings.append(embedding)
                continue

            avg_embedding = np.mean(buffer_embeddings, axis=0)
            similarity = self._cosine(avg_embedding, embedding)
            projected_length = len(''.join(buffer_sentences)) + len(sentence)
            too_long = projected_length > self.chunk_size * 1.2
            too_many_sentences = len(buffer_sentences) >= self.max_sentences

            if similarity < self.similarity_threshold or too_long or too_many_sentences:
                chunk_text = '\n'.join(buffer_sentences)
                chunk_id = f"{source_id}_chunk_{chunk_counter}"
                chunk_metadata = {
                    **metadata,
                    'section': metadata.get('section', 'semantic'),
                    'chunk_index': chunk_counter,
                    'avg_similarity': similarity
                }
                chunks.append(TextChunk(
                    content=chunk_text,
                    chunk_id=chunk_id,
                    source_id=source_id,
                    metadata=chunk_metadata
                ))
                chunk_counter += 1
                buffer_sentences = [sentence]
                buffer_embeddings = [embedding]
            else:
                buffer_sentences.append(sentence)
                buffer_embeddings.append(embedding)

        if buffer_sentences:
            chunk_text = '\n'.join(buffer_sentences)
            chunk_id = f"{source_id}_chunk_{chunk_counter}"
            chunk_metadata = {
                **metadata,
                'section': metadata.get('section', 'semantic'),
                'chunk_index': chunk_counter
            }
            chunks.append(TextChunk(
                content=chunk_text,
                chunk_id=chunk_id,
                source_id=source_id,
                metadata=chunk_metadata
            ))

        logger.info(f"Semantic splitter produced {len(chunks)} chunks for {source_id}")
        return chunks

class MultilingualEmbedder:
    """
    多语言向量化器 (保持不变，但确认与BGE-M3模型一起使用)
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "auto"):
        logger.info(f"Loading multilingual embedding model: {model_name} (shared)")
        self.model = ModelRegistry.get_sentence_transformer(model_name, device=device)
        self.device = str(self.model.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.success(f"Embedder ready. Vector dimension: {self.embedding_dim}")

    def embed_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """批量向量化文本块"""
        if not chunks:
            return []
            
        logger.info(f"Starting to embed {len(chunks)} text chunks...")
        contents = [chunk.content for chunk in chunks]
        
        embeddings = self.model.encode(
            contents,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            
        return chunks
    
    def embed_query(self, query: str) -> np.ndarray:
        """查询向量化"""
        return self.model.encode(query, convert_to_numpy=True)

    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.array([])
        return self.model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )


class EnhancedTextProcessor:
    """增强文本处理管理器 - 支持多表示索引"""
    def __init__(self, config: Dict):
        self.config = config
        self.embedder = MultilingualEmbedder(
            model_name=config.get('embedding_model', 'BAAI/bge-m3'),
            device=config.get('device', 'auto')
        )
        splitter_type = config.get('splitter_type', 'hierarchical')
        if splitter_type == 'semantic':
            self.splitter = SemanticGraphSplitter(
                embedder=self.embedder,
                chunk_size=config.get('chunk_size', 512),
                similarity_threshold=config.get('semantic_similarity_threshold', 0.72),
                max_sentences=config.get('semantic_max_sentences', 10)
            )
        else:
            self.splitter = HierarchicalTextSplitter(
                chunk_size=config.get('chunk_size', 512),
                chunk_overlap=config.get('chunk_overlap', 50)
            )
        config['vector_size'] = self.embedder.embedding_dim # Dynamically set vector size
        
        # Initialize multi-representation indexer
        self.enable_multi_representation = config.get('enable_multi_representation', True)
        if self.enable_multi_representation:
            self.multi_rep_indexer = MultiRepresentationIndexer(config)
            logger.info("Enhanced Text Processor with Multi-Representation Indexing initialized")
        else:
            logger.info("Basic Text Processor initialized")

    async def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """完整的文档处理流程 - 支持多表示索引"""
        logger.info(f"Starting to process {len(documents)} documents...")
        all_chunks = []

        # Step 1: Split documents into chunks
        for doc in documents:
            chunks = self.splitter.split_document(
                doc_content=doc['content'],
                source_id=doc['id'],
                metadata={k: v for k, v in doc.items() if k not in ['content']}
            )
            all_chunks.extend(chunks)
        
        # Step 2: Embed chunks
        vectorized_chunks = self.embedder.embed_chunks(all_chunks)
        
        # Step 3: Multi-representation processing if enabled
        if self.enable_multi_representation and self.multi_rep_indexer:
            logger.info("Creating multi-representations...")
            # Convert TextChunk objects to dict format for multi-rep indexer
            chunks_dict = []
            for chunk in vectorized_chunks:
                chunk_dict = {
                    'content': chunk.content,
                    'chunk_id': chunk.chunk_id,
                    'source_id': chunk.source_id,
                    'metadata': chunk.metadata,
                    'embedding': chunk.embedding
                }
                chunks_dict.append(chunk_dict)
            
            # Create multi-representations asynchronously
            multi_rep_chunks = await self.multi_rep_indexer.create_multi_representations(chunks_dict)
            
            # Generate index entries for vector database
            index_entries = self.multi_rep_indexer.generate_index_entries(multi_rep_chunks)
            
            logger.success(f"Enhanced processing complete. Generated {len(index_entries)} total index entries from {len(vectorized_chunks)} original chunks.")
            return index_entries
        
        else:
            # Standard processing - convert to dict format
            chunks_dict = []
            for chunk in vectorized_chunks:
                chunk_dict = {
                    'content': chunk.content,
                    'chunk_id': chunk.chunk_id,
                    'source_id': chunk.source_id,
                    'metadata': chunk.metadata,
                    'embedding': chunk.embedding.tolist() if chunk.embedding is not None else None,
                    'semantic_type': 'content',
                    'representation_type': 'original'
                }
                chunks_dict.append(chunk_dict)
            
            logger.success(f"Standard processing complete. Generated {len(chunks_dict)} vectorized chunks.")
            return chunks_dict

    def save_processed_data(self, chunks: List[Dict], output_path: Path):
        """保存处理后的数据"""
        # chunks is already in dict format from process_documents
        serializable_chunks = []
        for chunk in chunks:
            chunk_data = chunk.copy()
            # Ensure embedding is in list format
            if 'embedding' in chunk_data and isinstance(chunk_data['embedding'], np.ndarray):
                chunk_data['embedding'] = chunk_data['embedding'].tolist()
            serializable_chunks.append(chunk_data)
        
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Processed data saved to: {output_path}")

# Keep compatibility with old TextProcessor name  
TextProcessor = EnhancedTextProcessor
