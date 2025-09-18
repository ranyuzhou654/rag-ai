# src/generation/content_summarizer.py
"""
智能内容摘要生成器 - 基于LLM的高质量主题摘要
"""

import asyncio
import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

@dataclass
class SummaryRequest:
    """摘要请求"""
    topic_title: str
    documents: List[Dict]
    keywords: List[str]
    summary_type: str = "overview"  # overview, detailed, brief
    max_length: int = 200
    target_audience: str = "general"  # general, technical, beginner

@dataclass
class SummaryResult:
    """摘要结果"""
    summary: str
    key_points: List[str]
    confidence: float
    source_count: int
    generation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContentSummarizer:
    """智能内容摘要生成器"""
    
    def __init__(self, llm_generator=None):
        self.llm_generator = llm_generator
        
        # 摘要模板
        self.summary_templates = {
            "overview": {
                "prompt_template": """请为以下技术主题生成一个全面的概述摘要：

主题: {topic_title}
关键词: {keywords}

基于以下文档内容：
{document_excerpts}

请生成一个{max_length}字左右的摘要，包括：
1. 主题的核心概念和技术要点
2. 主要研究方向和应用场景
3. 重要的技术突破或进展

摘要应该{audience_style}，突出技术创新和实际应用价值。""",
                "max_length_default": 200
            },
            
            "detailed": {
                "prompt_template": """请为以下技术主题生成详细的技术分析摘要：

主题: {topic_title}
关键词: {keywords}
文档数量: {doc_count}

基于以下研究文档：
{document_excerpts}

请生成一个{max_length}字左右的详细摘要，包括：
1. 技术背景和发展脉络
2. 核心算法和方法论
3. 实验结果和性能表现
4. 未来发展趋势和挑战
5. 实际应用案例和影响

摘要应该{audience_style}，提供深入的技术洞察。""",
                "max_length_default": 400
            },
            
            "brief": {
                "prompt_template": """请为以下技术主题生成简洁的要点摘要：

主题: {topic_title}
关键词: {keywords}

基于{doc_count}篇相关文档：
{document_excerpts}

请生成一个{max_length}字以内的简洁摘要，突出：
1. 主要技术特点
2. 关键应用场景
3. 重要意义

摘要应该{audience_style}，简明扼要。""",
                "max_length_default": 100
            }
        }
        
        # 受众风格映射
        self.audience_styles = {
            "general": "通俗易懂，避免过于专业的术语",
            "technical": "使用专业术语，注重技术细节",
            "beginner": "适合初学者，包含基础概念解释"
        }
        
        logger.info("Content summarizer initialized")
    
    async def generate_summary(self, request: SummaryRequest) -> SummaryResult:
        """生成主题摘要"""
        
        start_time = datetime.now()
        
        try:
            # 1. 准备文档摘录
            document_excerpts = self._prepare_document_excerpts(
                request.documents, request.summary_type
            )
            
            # 2. 生成主摘要
            summary = await self._generate_main_summary(request, document_excerpts)
            
            # 3. 提取关键要点
            key_points = await self._extract_key_points(request, document_excerpts)
            
            # 4. 评估摘要质量
            confidence = self._assess_summary_quality(summary, request)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            result = SummaryResult(
                summary=summary,
                key_points=key_points,
                confidence=confidence,
                source_count=len(request.documents),
                generation_time=generation_time,
                metadata={
                    "summary_type": request.summary_type,
                    "target_audience": request.target_audience,
                    "keyword_count": len(request.keywords),
                    "template_used": request.summary_type
                }
            )
            
            logger.info(f"Summary generated for '{request.topic_title}' in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate summary for '{request.topic_title}': {e}")
            return self._create_fallback_summary(request, start_time)
    
    async def generate_batch_summaries(
        self, 
        requests: List[SummaryRequest]
    ) -> List[SummaryResult]:
        """批量生成摘要"""
        
        logger.info(f"Starting batch summary generation for {len(requests)} topics")
        
        # 并发生成摘要
        tasks = [self.generate_summary(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch summary failed for topic {i}: {result}")
                processed_results.append(self._create_fallback_summary(requests[i], datetime.now()))
            else:
                processed_results.append(result)
        
        logger.success(f"Batch summary generation completed: {len(processed_results)} summaries")
        return processed_results
    
    def _prepare_document_excerpts(self, documents: List[Dict], summary_type: str) -> str:
        """准备文档摘录"""
        
        excerpts = []
        max_docs = 5 if summary_type == "detailed" else 3  # 详细摘要使用更多文档
        excerpt_length = 300 if summary_type == "detailed" else 150
        
        for i, doc_info in enumerate(documents[:max_docs]):
            doc = doc_info.get('doc', doc_info)  # 兼容不同的数据结构
            
            title = doc['metadata'].get('title', f'Document {i+1}')
            content = doc['content']
            
            # 提取文档摘录
            excerpt = self._extract_meaningful_excerpt(content, excerpt_length)
            
            excerpts.append(f"文档{i+1}: {title}\n摘录: {excerpt}\n")
        
        return "\n".join(excerpts)
    
    def _extract_meaningful_excerpt(self, content: str, max_length: int) -> str:
        """提取有意义的文档摘录"""
        
        # 尝试找到包含关键信息的句子
        sentences = re.split(r'[.!?。！？]', content)
        
        # 优先选择包含技术关键词的句子
        tech_keywords = [
            'method', 'algorithm', 'model', 'approach', 'technique',
            'performance', 'result', 'experiment', 'evaluation',
            '方法', '算法', '模型', '技术', '性能', '结果', '实验'
        ]
        
        priority_sentences = []
        regular_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # 过滤太短的句子
                if any(keyword in sentence.lower() for keyword in tech_keywords):
                    priority_sentences.append(sentence)
                else:
                    regular_sentences.append(sentence)
        
        # 构建摘录
        excerpt_parts = []
        current_length = 0
        
        # 优先添加重要句子
        for sentence in priority_sentences:
            if current_length + len(sentence) <= max_length:
                excerpt_parts.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        # 如果还有空间，添加常规句子
        for sentence in regular_sentences:
            if current_length + len(sentence) <= max_length:
                excerpt_parts.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        if not excerpt_parts:
            # 如果没有合适的句子，直接截取内容的开头
            return content[:max_length] + "..."
        
        excerpt = ". ".join(excerpt_parts)
        if len(excerpt) > max_length:
            excerpt = excerpt[:max_length] + "..."
        
        return excerpt
    
    async def _generate_main_summary(
        self, 
        request: SummaryRequest, 
        document_excerpts: str
    ) -> str:
        """生成主摘要"""
        
        if not self.llm_generator:
            return self._generate_fallback_text_summary(request)
        
        try:
            # 获取模板
            template_info = self.summary_templates.get(
                request.summary_type, 
                self.summary_templates["overview"]
            )
            
            # 准备提示词
            audience_style = self.audience_styles.get(
                request.target_audience, 
                self.audience_styles["general"]
            )
            
            prompt = template_info["prompt_template"].format(
                topic_title=request.topic_title,
                keywords=", ".join(request.keywords[:8]),  # 限制关键词数量
                document_excerpts=document_excerpts,
                max_length=request.max_length,
                doc_count=len(request.documents),
                audience_style=audience_style
            )
            
            # 生成摘要
            summary = await self.llm_generator.generate_text(
                prompt,
                max_length=request.max_length * 2,  # 给LLM更多空间
                temperature=0.3  # 较低的温度以保持一致性
            )
            
            # 清理和优化摘要
            summary = self._clean_and_optimize_summary(summary, request.max_length)
            
            return summary
            
        except Exception as e:
            logger.error(f"LLM summary generation failed: {e}")
            return self._generate_fallback_text_summary(request)
    
    async def _extract_key_points(
        self, 
        request: SummaryRequest, 
        document_excerpts: str
    ) -> List[str]:
        """提取关键要点"""
        
        if not self.llm_generator:
            return self._extract_fallback_key_points(request)
        
        try:
            prompt = f"""基于以下技术主题和文档内容，提取3-5个最重要的技术要点：

主题: {request.topic_title}
关键词: {", ".join(request.keywords[:5])}

文档内容：
{document_excerpts[:1000]}  # 限制长度

请以简洁的要点形式列出最重要的技术特点、方法或发现，每个要点一行，以"-"开头："""

            key_points_text = await self.llm_generator.generate_text(
                prompt,
                max_length=300,
                temperature=0.2
            )
            
            # 解析要点
            key_points = self._parse_key_points(key_points_text)
            
            return key_points[:5]  # 最多5个要点
            
        except Exception as e:
            logger.error(f"Key points extraction failed: {e}")
            return self._extract_fallback_key_points(request)
    
    def _parse_key_points(self, text: str) -> List[str]:
        """解析关键要点文本"""
        
        key_points = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line:
                # 移除可能的编号和符号
                line = re.sub(r'^[-•\d\.]+\s*', '', line)
                if len(line) > 10:  # 过滤太短的要点
                    key_points.append(line)
        
        return key_points
    
    def _clean_and_optimize_summary(self, summary: str, max_length: int) -> str:
        """清理和优化摘要"""
        
        # 移除多余的空白
        summary = re.sub(r'\s+', ' ', summary)
        summary = summary.strip()
        
        # 确保摘要不超过指定长度
        if len(summary) > max_length:
            # 在句子边界截断
            sentences = re.split(r'[.!?。！？]', summary)
            truncated_summary = ""
            
            for sentence in sentences:
                if len(truncated_summary + sentence) <= max_length - 10:
                    truncated_summary += sentence + "。"
                else:
                    break
            
            summary = truncated_summary
        
        # 确保摘要以句号结尾
        if summary and not summary.endswith(('。', '.', '!', '?', '！', '？')):
            summary += "。"
        
        return summary
    
    def _assess_summary_quality(self, summary: str, request: SummaryRequest) -> float:
        """评估摘要质量"""
        
        confidence = 0.5  # 基础分数
        
        # 长度合适性
        target_length = request.max_length
        actual_length = len(summary)
        
        if 0.7 * target_length <= actual_length <= 1.2 * target_length:
            confidence += 0.2
        
        # 关键词覆盖率
        keywords_found = 0
        for keyword in request.keywords[:5]:
            if keyword.lower() in summary.lower():
                keywords_found += 1
        
        keyword_coverage = keywords_found / min(len(request.keywords), 5)
        confidence += keyword_coverage * 0.2
        
        # 句子结构合理性
        sentences = re.split(r'[.!?。！？]', summary)
        if 2 <= len(sentences) <= 8:  # 合理的句子数量
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_fallback_text_summary(self, request: SummaryRequest) -> str:
        """生成回退文本摘要"""
        
        keyword_text = "、".join(request.keywords[:5]) if request.keywords else "相关技术"
        
        fallback_templates = {
            "overview": f"{request.topic_title}是一个涉及{keyword_text}的重要技术领域。基于{len(request.documents)}篇相关文档的分析，该主题展现了显著的研究价值和应用潜力。",
            "detailed": f"{request.topic_title}技术主题的深入分析显示，{keyword_text}等关键技术在该领域发挥重要作用。通过对{len(request.documents)}篇研究文档的分析，我们发现了多个值得关注的技术发展方向和应用场景。",
            "brief": f"{request.topic_title}：{keyword_text}相关技术，{len(request.documents)}篇文档分析。"
        }
        
        return fallback_templates.get(request.summary_type, fallback_templates["overview"])
    
    def _extract_fallback_key_points(self, request: SummaryRequest) -> List[str]:
        """提取回退关键要点"""
        
        key_points = []
        
        if request.keywords:
            key_points.append(f"涉及{request.keywords[0]}技术")
            
            if len(request.keywords) > 1:
                key_points.append(f"应用{request.keywords[1]}方法")
        
        key_points.append(f"基于{len(request.documents)}篇研究文档")
        
        if len(request.documents) > 3:
            key_points.append("研究热度较高")
        
        return key_points
    
    def _create_fallback_summary(self, request: SummaryRequest, start_time: datetime) -> SummaryResult:
        """创建回退摘要结果"""
        
        fallback_summary = self._generate_fallback_text_summary(request)
        fallback_key_points = self._extract_fallback_key_points(request)
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return SummaryResult(
            summary=fallback_summary,
            key_points=fallback_key_points,
            confidence=0.3,  # 较低的置信度
            source_count=len(request.documents),
            generation_time=generation_time,
            metadata={
                "summary_type": request.summary_type,
                "fallback_used": True,
                "error_occurred": True
            }
        )
    
    async def generate_topic_comparison(
        self, 
        topic1: Dict, 
        topic2: Dict
    ) -> str:
        """生成主题对比摘要"""
        
        if not self.llm_generator:
            return self._generate_fallback_comparison(topic1, topic2)
        
        try:
            prompt = f"""请对比分析以下两个技术主题的异同点：

主题1: {topic1['title']}
关键词: {", ".join(topic1.get('keywords', [])[:5])}
文档数: {topic1.get('doc_count', 0)}

主题2: {topic2['title']}
关键词: {", ".join(topic2.get('keywords', [])[:5])}
文档数: {topic2.get('doc_count', 0)}

请从以下角度进行对比：
1. 技术方法和核心理念的异同
2. 应用场景的差异
3. 发展阶段和成熟度
4. 互补性和关联性

请生成150字左右的对比分析："""

            comparison = await self.llm_generator.generate_text(
                prompt,
                max_length=200,
                temperature=0.3
            )
            
            return comparison.strip()
            
        except Exception as e:
            logger.error(f"Topic comparison failed: {e}")
            return self._generate_fallback_comparison(topic1, topic2)
    
    def _generate_fallback_comparison(self, topic1: Dict, topic2: Dict) -> str:
        """生成回退对比分析"""
        
        return f"{topic1['title']}和{topic2['title']}是两个相关的技术领域。" \
               f"前者涉及{len(topic1.get('keywords', []))}个关键技术点，" \
               f"后者包含{len(topic2.get('keywords', []))}个技术要素。" \
               f"两个主题在研究方法和应用场景上既有相似性，也存在各自的特色和优势。"