# src/processing/multi_representation_indexer.py
from typing import List, Dict, Optional, Tuple
import asyncio
from dataclasses import dataclass, field
import numpy as np
from transformers import GenerationConfig
from loguru import logger
import re
import json
from pathlib import Path

from src.optimization.model_registry import ModelRegistry
from src.utils.progress_tracker import MultiStageProgressTracker, SimpleProgressBar

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


class _SharedLLMComponent:
    """Helper mixin to reuse cached LLM instances."""

    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None, component_name: str = "LLM component"):
        resource = ModelRegistry.get_llm(model_name, device=device, token=token)
        self.device = resource.device
        self.model_name = model_name
        self.tokenizer = resource.tokenizer
        self.model = resource.model
        logger.info(f"{component_name} using shared model: {model_name}")

class SummaryGenerator(_SharedLLMComponent):
    """文档摘要生成器"""

    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None):
        super().__init__(
            model_name=model_name,
            device=device,
            token=token,
            component_name="Summary Generator"
        )

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        self.generation_config = GenerationConfig(
            max_new_tokens=80,  # Shorter for summaries
            temperature=0.3,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=eos_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id
        )
    
    def generate_summaries(self, texts: List[str], max_length: int = 150) -> List[str]:
        """批量生成文本摘要"""
        if not texts:
            return []

        prompts: List[str] = []
        is_chinese_flags: List[bool] = []
        for text in texts:
            is_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
            is_chinese_flags.append(is_chinese)
            if is_chinese:
                prompt = (
                    f"请为以下文本生成一个简洁准确的摘要，控制在{max_length}字以内：\n\n"
                    f"原文：{text[:1000]}...\n\n摘要："
                )
            else:
                prompt = (
                    f"Generate a concise and accurate summary of the following text, keeping it under {max_length} words:\n\n"
                    f"Original text: {text[:1000]}...\n\nSummary:"
                )
            prompts.append(prompt)

        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )

            responses = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            summaries: List[str] = []
            for response, text, is_chinese in zip(responses, texts, is_chinese_flags):
                summary_part = (
                    response.split("摘要：")[-1].strip()
                    if is_chinese
                    else response.split("Summary:")[-1].strip()
                )
                summary = summary_part.split('\n')[0].strip()
                if len(summary) > max_length * 2:
                    summary = summary[:max_length * 2] + "..."
                if not summary:
                    summary = text[:max_length] if len(text) <= max_length else text[:max_length] + "..."
                summaries.append(summary)

            return summaries

        except Exception as e:
            logger.error(f"Failed to generate batch summaries: {e}")
            return [
                (text[:max_length] + "..." if len(text) > max_length else text)
                for text in texts
            ]

    def generate_summary(self, text: str, max_length: int = 150) -> str:
        """兼容单文本摘要生成"""
        return self.generate_summaries([text], max_length=max_length)[0]

class QuestionGenerator(_SharedLLMComponent):
    """假设性问题生成器"""

    def __init__(self, model_name: str, device: str = "auto", token: Optional[str] = None):
        super().__init__(
            model_name=model_name,
            device=device,
            token=token,
            component_name="Question Generator"
        )

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        self.generation_config = GenerationConfig(
            max_new_tokens=80,
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=eos_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id
        )
    
    def generate_questions_batch(
        self, texts: List[str], num_questions: int = 3
    ) -> List[List[str]]:
        """批量生成假设性问题"""
        if not texts:
            return []

        prompts: List[str] = []
        is_chinese_flags: List[bool] = []
        for text in texts:
            is_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
            is_chinese_flags.append(is_chinese)
            if is_chinese:
                prompt = (
                    f"基于以下文本内容，生成{num_questions}个相关的问题，这些问题应该能够通过该文本来回答：\n\n"
                    f"文本内容：{text[:800]}\n\n问题：\n1."
                )
            else:
                prompt = (
                    f"Based on the following text content, generate {num_questions} relevant questions that could be answered by this text:\n\n"
                    f"Text content: {text[:800]}\n\nQuestions:\n1."
                )
            prompts.append(prompt)

        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )

            responses = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            batched_questions: List[List[str]] = []
            for response, text, is_chinese in zip(responses, texts, is_chinese_flags):
                questions_part = (
                    response.split("问题：")[-1].strip()
                    if is_chinese
                    else response.split("Questions:")[-1].strip()
                )

                questions: List[str] = []
                lines = questions_part.split('\n')
                for line in lines:
                    line = line.strip()
                    if re.match(r'^\d+\.', line):
                        question = re.sub(r'^\d+\.\s*', '', line).strip()
                        if question and len(question) > 10 and question.endswith('?'):
                            questions.append(question)
                            if len(questions) >= num_questions:
                                break

                if not questions:
                    fallback = f"What is the main idea of: {text[:80]}?"
                    questions = [fallback]

                batched_questions.append(questions[:num_questions])

            return batched_questions

        except Exception as e:
            logger.error(f"Failed to generate batch questions: {e}")
            return [[] for _ in texts]

    def generate_questions(self, text: str, num_questions: int = 3) -> List[str]:
        """兼容单文本问题生成"""
        results = self.generate_questions_batch([text], num_questions=num_questions)
        return results[0] if results else []

class MultiRepresentationIndexer:
    """多表示索引器 - 为每个文本块创建多种表示形式"""
    
    def __init__(self, config: Dict):
        self.config = config

        # Initialize embedding model
        self.embedder = ModelRegistry.get_sentence_transformer(
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
    
    async def create_multi_representations(
        self,
        chunks: List[Dict],
        show_progress: bool = True
    ) -> List[MultiRepresentationChunk]:
        """为文本块创建多表示索引，使用异步生成提升吞吐"""
        logger.info(f"Creating multi-representations for {len(chunks)} chunks asynchronously...")

        # 计算各阶段的工作量
        total_chunks = len(chunks)
        if self.llm_available:
            # 每个chunk生成摘要和问题
            generation_items = total_chunks * 2  # summary + questions
            # 每个chunk有原文、摘要、3个问题的嵌入
            embedding_items = total_chunks * 5  # content + summary + 3 questions
        else:
            generation_items = 0
            embedding_items = total_chunks  # 只有原文嵌入

        # 初始化进度跟踪器
        progress_tracker = None
        if show_progress:
            progress_tracker = MultiStageProgressTracker("多表示处理")
            
            if self.llm_available:
                progress_tracker.add_stage("generation", "生成摘要和问题", generation_items, weight=3.0)
            progress_tracker.add_stage("embedding", "嵌入向量化", embedding_items, weight=2.0)
            progress_tracker.start_display()

        try:
            multi_rep_chunks = [
                MultiRepresentationChunk(
                    content=chunk['content'],
                    chunk_id=chunk['chunk_id'],
                    source_id=chunk['source_id'],
                    metadata=chunk.get('metadata', {}),
                    content_embedding=chunk.get('embedding')
                )
                for chunk in chunks
            ]

            if self.llm_available and multi_rep_chunks:
                if progress_tracker:
                    progress_tracker.start_stage("generation")

                batch_size = max(1, self.config.get('multi_rep_batch_size', 8))

                for start in range(0, len(multi_rep_chunks), batch_size):
                    batch = multi_rep_chunks[start:start + batch_size]
                    texts = [c.content for c in batch]

                    summaries = await asyncio.to_thread(
                        self.summary_generator.generate_summaries,
                        texts,
                        150
                    )

                    for chunk_obj, summary in zip(batch, summaries):
                        chunk_obj.summary = summary

                    if progress_tracker:
                        progress_tracker.increment_stage("generation", len(batch))

                    questions_batch = await asyncio.to_thread(
                        self.question_generator.generate_questions_batch,
                        texts,
                        3
                    )

                    for chunk_obj, questions in zip(batch, questions_batch):
                        chunk_obj.hypothetical_questions = questions

                    if progress_tracker:
                        progress_tracker.increment_stage("generation", len(batch))

                if progress_tracker:
                    progress_tracker.complete_stage("generation")
            else:
                logger.debug("LLM unavailable or no chunks; skipping generation stage")

            if progress_tracker:
                progress_tracker.start_stage("embedding")

            self._embed_representations(multi_rep_chunks, progress_tracker)

            if progress_tracker:
                progress_tracker.complete_stage("embedding")

        finally:
            if progress_tracker:
                progress_tracker.stop_display_thread()

        logger.success(f"Multi-representation indexing complete for {len(multi_rep_chunks)} chunks")
        return multi_rep_chunks
    
    def _embed_representations(self, chunks: List[MultiRepresentationChunk], progress_tracker: Optional[MultiStageProgressTracker] = None):
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
            
            # 自定义进度跟踪的批处理
            batch_size = 32
            
            all_embeddings = []
            for i in range(0, len(texts_to_embed), batch_size):
                batch_texts = texts_to_embed[i:i + batch_size]
                batch_embeddings = self.embedder.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,  # 我们自己管理进度
                    convert_to_numpy=True
                )
                all_embeddings.extend(batch_embeddings)
                
                # 更新进度
                if progress_tracker:
                    completed_items = min(i + batch_size, len(texts_to_embed))
                    progress_tracker.update_stage("embedding", completed_items)
            
            # Assign embeddings back to chunks
            for embedding, (chunk, rep_type) in zip(all_embeddings, text_mappings):
                if rep_type == 'summary':
                    chunk.summary_embedding = embedding
                elif rep_type == 'question':
                    chunk.questions_embeddings.append(embedding)
        else:
            # 如果没有额外的表示需要嵌入，直接完成
            if progress_tracker:
                progress_tracker.update_stage("embedding", len(chunks))
    
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
    multi_chunks = await indexer.create_multi_representations(test_chunks)
    
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
