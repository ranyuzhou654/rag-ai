# src/processing/multi_representation_indexer.py
from typing import List, Dict, Optional, Tuple
import asyncio
from dataclasses import dataclass, field
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from loguru import logger
import re
import json
from pathlib import Path

@dataclass
class MultiRepresentationChunk:
    """多表示文本块"""
    content: str
    chunk_id: str
    source_id: str
    metadata: Dict = field(default_factory=dict)
    
    # Original embeddings
    content_embedding: Optional[np.ndarray] = None
    
    # Multi-representation content and embeddings
    summary: Optional[str] = None
    summary_embedding: Optional[np.ndarray] = None
    
    hypothetical_questions: List[str] = field(default_factory=list)
    questions_embeddings: List[np.ndarray] = field(default_factory=list)
    
    # Semantic type for better filtering
    semantic_type: str = 'content'  # 'content', 'summary', 'question'

class SummaryGenerator:
    """文档摘要生成器"""
    
    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        logger.info(f"Loading Summary Generator: {model_name}")
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
            max_new_tokens=200,  # Shorter for summaries
            temperature=0.3,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
    
    def generate_summary(self, text: str, max_length: int = 150) -> str:
        """生成文本摘要"""
        # Detect language
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        
        if is_chinese:
            prompt = f"""请为以下文本生成一个简洁准确的摘要，控制在{max_length}字以内：

原文：{text[:1000]}...

摘要："""
        else:
            prompt = f"""Generate a concise and accurate summary of the following text, keeping it under {max_length} words:

Original text: {text[:1000]}...

Summary:"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract summary
            if is_chinese:
                summary_part = response.split("摘要：")[-1].strip()
            else:
                summary_part = response.split("Summary:")[-1].strip()
            
            # Clean up and limit length
            summary = summary_part.split('\n')[0].strip()
            if len(summary) > max_length * 2:  # Rough character limit
                summary = summary[:max_length * 2] + "..."
            
            return summary if summary else text[:max_length]
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            # Fallback: return first part of text
            return text[:max_length] + "..." if len(text) > max_length else text

class QuestionGenerator:
    """假设性问题生成器"""
    
    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        logger.info(f"Loading Question Generator: {model_name}")
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
            max_new_tokens=300,
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
    
    def generate_questions(self, text: str, num_questions: int = 3) -> List[str]:
        """生成假设性问题"""
        # Detect language
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        
        if is_chinese:
            prompt = f"""基于以下文本内容，生成{num_questions}个相关的问题，这些问题应该能够通过该文本来回答：

文本内容：{text[:800]}

问题：
1."""
        else:
            prompt = f"""Based on the following text content, generate {num_questions} relevant questions that could be answered by this text:

Text content: {text[:800]}

Questions:
1."""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract questions
            if is_chinese:
                questions_part = response.split("问题：")[-1].strip()
            else:
                questions_part = response.split("Questions:")[-1].strip()
            
            # Parse numbered questions
            questions = []
            lines = questions_part.split('\n')
            
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    question = re.sub(r'^\d+\.\s*', '', line).strip()
                    if question and len(question) > 10 and question.endswith('?'):
                        questions.append(question)
                        if len(questions) >= num_questions:
                            break
            
            return questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Failed to generate questions: {e}")
            return []

class MultiRepresentationIndexer:
    """多表示索引器 - 为每个文本块创建多种表示形式"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize embedding model
        self.embedder = SentenceTransformer(
            config.get('embedding_model', 'BAAI/bge-m3'),
            device=config.get('device', 'auto')
        )
        
        # Initialize generators if LLM is available
        model_name = config.get('llm_model')
        token = config.get('HUGGING_FACE_TOKEN')
        device = config.get('device', 'auto')
        
        self.llm_available = False
        if model_name:
            try:
                self.summary_generator = SummaryGenerator(
                    model_name=model_name, device=device, token=token
                )
                self.question_generator = QuestionGenerator(
                    model_name=model_name, device=device, token=token
                )
                self.llm_available = True
                logger.success("Multi-Representation Indexer initialized with LLM support")
            except Exception as e:
                logger.error(f"Failed to initialize LLM generators: {e}")
        else:
            logger.warning("Multi-Representation Indexer initialized without LLM support")
    
    def create_multi_representations(
        self, 
        chunks: List[Dict]
    ) -> List[MultiRepresentationChunk]:
        """为文本块创建多表示索引"""
        logger.info(f"Creating multi-representations for {len(chunks)} chunks...")
        
        multi_rep_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i % 50 == 0:
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Create base multi-representation chunk
            multi_chunk = MultiRepresentationChunk(
                content=chunk['content'],
                chunk_id=chunk['chunk_id'],
                source_id=chunk['source_id'],
                metadata=chunk.get('metadata', {}),
                content_embedding=chunk.get('embedding')
            )
            
            # Generate additional representations if LLM is available
            if self.llm_available:
                try:
                    # Generate summary
                    multi_chunk.summary = self.summary_generator.generate_summary(
                        chunk['content'], max_length=150
                    )
                    
                    # Generate hypothetical questions
                    multi_chunk.hypothetical_questions = self.question_generator.generate_questions(
                        chunk['content'], num_questions=3
                    )
                    
                except Exception as e:
                    logger.error(f"Error generating representations for chunk {chunk['chunk_id']}: {e}")
            
            multi_rep_chunks.append(multi_chunk)
        
        # Batch embed all representations
        self._embed_representations(multi_rep_chunks)
        
        logger.success(f"Multi-representation indexing complete for {len(multi_rep_chunks)} chunks")
        return multi_rep_chunks
    
    def _embed_representations(self, chunks: List[MultiRepresentationChunk]):
        """批量嵌入所有表示形式"""
        logger.info("Embedding multi-representations...")
        
        # Collect all texts to embed
        texts_to_embed = []
        text_mappings = []  # Track which text belongs to which chunk and type
        
        for chunk in chunks:
            # Summary embedding
            if chunk.summary:
                texts_to_embed.append(chunk.summary)
                text_mappings.append((chunk, 'summary'))
            
            # Questions embeddings
            for question in chunk.hypothetical_questions:
                texts_to_embed.append(question)
                text_mappings.append((chunk, 'question'))
        
        if texts_to_embed:
            # Batch embedding
            logger.info(f"Embedding {len(texts_to_embed)} additional representations...")
            embeddings = self.embedder.encode(
                texts_to_embed,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Assign embeddings back to chunks
            for embedding, (chunk, rep_type) in zip(embeddings, text_mappings):
                if rep_type == 'summary':
                    chunk.summary_embedding = embedding
                elif rep_type == 'question':
                    chunk.questions_embeddings.append(embedding)
    
    def generate_index_entries(
        self, 
        chunks: List[MultiRepresentationChunk]
    ) -> List[Dict]:
        """生成多表示索引条目，用于向量数据库存储"""
        index_entries = []
        
        for chunk in chunks:
            # 1. Original content entry
            content_entry = {
                'chunk_id': chunk.chunk_id,
                'source_id': chunk.source_id,
                'content': chunk.content,
                'embedding': chunk.content_embedding.tolist() if chunk.content_embedding is not None else None,
                'semantic_type': 'content',
                'representation_type': 'original',
                'metadata': chunk.metadata
            }
            index_entries.append(content_entry)
            
            # 2. Summary entry
            if chunk.summary and chunk.summary_embedding is not None:
                summary_entry = {
                    'chunk_id': f"{chunk.chunk_id}_summary",
                    'source_id': chunk.source_id,
                    'content': chunk.summary,
                    'embedding': chunk.summary_embedding.tolist(),
                    'semantic_type': 'summary',
                    'representation_type': 'summary',
                    'original_chunk_id': chunk.chunk_id,
                    'metadata': {**chunk.metadata, 'is_summary': True}
                }
                index_entries.append(summary_entry)
            
            # 3. Question entries
            for j, (question, q_embedding) in enumerate(
                zip(chunk.hypothetical_questions, chunk.questions_embeddings)
            ):
                question_entry = {
                    'chunk_id': f"{chunk.chunk_id}_q{j}",
                    'source_id': chunk.source_id,
                    'content': question,
                    'embedding': q_embedding.tolist(),
                    'semantic_type': 'question',
                    'representation_type': 'hypothetical_question',
                    'original_chunk_id': chunk.chunk_id,
                    'original_content': chunk.content,
                    'metadata': {**chunk.metadata, 'is_question': True, 'question_index': j}
                }
                index_entries.append(question_entry)
        
        logger.info(f"Generated {len(index_entries)} total index entries from {len(chunks)} original chunks")
        return index_entries
    
    def save_multi_representations(
        self, 
        chunks: List[MultiRepresentationChunk], 
        output_path: Path
    ):
        """保存多表示索引数据"""
        # Convert to serializable format
        serializable_data = []
        
        for chunk in chunks:
            chunk_data = {
                'content': chunk.content,
                'chunk_id': chunk.chunk_id,
                'source_id': chunk.source_id,
                'metadata': chunk.metadata,
                'semantic_type': chunk.semantic_type,
                
                # Embeddings
                'content_embedding': chunk.content_embedding.tolist() if chunk.content_embedding is not None else None,
                'summary_embedding': chunk.summary_embedding.tolist() if chunk.summary_embedding is not None else None,
                'questions_embeddings': [emb.tolist() for emb in chunk.questions_embeddings],
                
                # Generated content
                'summary': chunk.summary,
                'hypothetical_questions': chunk.hypothetical_questions
            }
            serializable_data.append(chunk_data)
        
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Multi-representation data saved to: {output_path}")

# 使用示例
async def main():
    """测试多表示索引器"""
    config = {
        'embedding_model': 'BAAI/bge-m3',
        'llm_model': 'Qwen/Qwen2-7B-Instruct',
        'device': 'auto',
        'HUGGING_FACE_TOKEN': None
    }
    
    indexer = MultiRepresentationIndexer(config)
    
    # 测试文本块
    test_chunks = [
        {
            'content': 'Transformer模型是一种基于注意力机制的深度学习架构，由Vaswani等人在2017年提出。它摒弃了循环神经网络的序列处理方式，采用自注意力机制来捕获序列中的长距离依赖关系。',
            'chunk_id': 'test_chunk_1',
            'source_id': 'test_doc_1',
            'metadata': {'section': 'introduction'},
            'embedding': np.random.rand(1024).astype(np.float32)
        },
        {
            'content': 'The attention mechanism in Transformer models allows the model to focus on different parts of the input sequence when processing each element. This is achieved through the computation of attention weights that determine the relevance of each input token to the current processing step.',
            'chunk_id': 'test_chunk_2',
            'source_id': 'test_doc_2',
            'metadata': {'section': 'methodology'},
            'embedding': np.random.rand(1024).astype(np.float32)
        }
    ]
    
    # Create multi-representations
    multi_chunks = indexer.create_multi_representations(test_chunks)
    
    # Display results
    for chunk in multi_chunks:
        print(f"\n{'='*60}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Content: {chunk.content[:100]}...")
        
        if chunk.summary:
            print(f"\nSummary: {chunk.summary}")
        
        if chunk.hypothetical_questions:
            print(f"\nHypothetical Questions:")
            for i, q in enumerate(chunk.hypothetical_questions, 1):
                print(f"  {i}. {q}")
    
    # Generate index entries
    index_entries = indexer.generate_index_entries(multi_chunks)
    print(f"\nTotal index entries generated: {len(index_entries)}")

if __name__ == "__main__":
    asyncio.run(main())