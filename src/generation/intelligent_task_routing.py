# src/generation/intelligent_task_routing.py
"""
智能任务路由系统
基于查询复杂度、成本效益和性能需求，自动选择最优的模型和生成策略
实现60-80%的成本降低目标
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
import json
from datetime import datetime, timedelta
from loguru import logger

# LangChain组件
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# 本地组件
from .hybrid_retriever import EnhancedDocument


class TaskComplexity(Enum):
    """任务复杂度等级"""
    SIMPLE = "simple"         # 简单事实查询，关键词提取等
    MEDIUM = "medium"         # 需要一定推理的查询，摘要生成等  
    COMPLEX = "complex"       # 多步推理，比较分析，深度QA等
    CRITICAL = "critical"     # 创意写作，专业咨询，复杂规划等


class TaskType(Enum):
    """任务类型"""
    QUERY_REWRITING = "query_rewriting"
    SUMMARIZATION = "summarization"
    QUESTION_GENERATION = "question_generation"
    FINAL_GENERATION = "final_generation"
    EVALUATION = "evaluation"
    COMPRESSION = "compression"


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    model_type: str  # 'local', 'api'
    cost_per_token: float  # 每token成本
    max_tokens: int
    avg_latency: float  # 平均延迟(秒)
    quality_score: float  # 质量评分(0-1)
    supported_languages: List[str]
    model_instance: Optional[BaseLanguageModel] = None


@dataclass
class RoutingDecision:
    """路由决策结果"""
    selected_model: ModelConfig
    reasoning: str
    confidence: float
    estimated_cost: float
    estimated_latency: float
    fallback_models: List[ModelConfig]


@dataclass
class TaskRequest:
    """任务请求"""
    task_type: TaskType
    complexity: TaskComplexity
    content: str
    max_tokens: Optional[int] = None
    quality_requirement: float = 0.7  # 质量要求(0-1)
    latency_requirement: float = 30.0  # 延迟要求(秒)
    cost_sensitivity: float = 0.5  # 成本敏感度(0-1)
    language: str = 'zh'


@dataclass
class ExecutionResult:
    """执行结果"""
    output: str
    model_used: str
    actual_cost: float
    actual_latency: float
    token_count: int
    quality_estimation: float


class CostOptimizer:
    """成本优化器"""
    
    def __init__(self):
        self.usage_history = []
        self.cost_thresholds = {
            'daily': 100.0,    # 每日成本限制
            'hourly': 10.0,    # 每小时成本限制
            'per_query': 1.0   # 单次查询成本限制
        }
        
    def check_cost_constraints(self, estimated_cost: float) -> Tuple[bool, str]:
        """检查成本约束"""
        
        # 检查单次查询成本
        if estimated_cost > self.cost_thresholds['per_query']:
            return False, f"单次查询成本过高: ${estimated_cost:.4f} > ${self.cost_thresholds['per_query']}"
        
        # 检查每小时成本
        current_hour_cost = self._calculate_hourly_cost()
        if current_hour_cost + estimated_cost > self.cost_thresholds['hourly']:
            return False, f"小时成本限制: ${current_hour_cost + estimated_cost:.4f} > ${self.cost_thresholds['hourly']}"
        
        # 检查每日成本
        current_daily_cost = self._calculate_daily_cost()
        if current_daily_cost + estimated_cost > self.cost_thresholds['daily']:
            return False, f"每日成本限制: ${current_daily_cost + estimated_cost:.4f} > ${self.cost_thresholds['daily']}"
        
        return True, "成本约束满足"
    
    def _calculate_hourly_cost(self) -> float:
        """计算当前小时的成本"""
        now = datetime.now()
        current_hour_start = now.replace(minute=0, second=0, microsecond=0)
        
        hourly_cost = sum(
            record['cost'] for record in self.usage_history
            if record['timestamp'] >= current_hour_start
        )
        return hourly_cost
    
    def _calculate_daily_cost(self) -> float:
        """计算当日成本"""
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        daily_cost = sum(
            record['cost'] for record in self.usage_history
            if record['timestamp'] >= today_start
        )
        return daily_cost
    
    def record_usage(self, model_name: str, cost: float, tokens: int):
        """记录使用情况"""
        self.usage_history.append({
            'timestamp': datetime.now(),
            'model': model_name,
            'cost': cost,
            'tokens': tokens
        })
        
        # 保持历史记录在合理范围内（只保留7天）
        cutoff = datetime.now() - timedelta(days=7)
        self.usage_history = [
            record for record in self.usage_history
            if record['timestamp'] >= cutoff
        ]


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.model_stats = {}
        
    def record_performance(
        self,
        model_name: str,
        latency: float,
        quality_score: float,
        cost: float,
        success: bool
    ):
        """记录模型性能"""
        if model_name not in self.model_stats:
            self.model_stats[model_name] = {
                'latencies': [],
                'quality_scores': [],
                'costs': [],
                'success_count': 0,
                'total_count': 0
            }
        
        stats = self.model_stats[model_name]
        stats['latencies'].append(latency)
        stats['quality_scores'].append(quality_score)
        stats['costs'].append(cost)
        stats['total_count'] += 1
        
        if success:
            stats['success_count'] += 1
        
        # 保持统计数据在合理范围内
        max_records = 1000
        for key in ['latencies', 'quality_scores', 'costs']:
            if len(stats[key]) > max_records:
                stats[key] = stats[key][-max_records:]
    
    def get_model_performance(self, model_name: str) -> Dict[str, float]:
        """获取模型性能指标"""
        if model_name not in self.model_stats:
            return {
                'avg_latency': 0.0,
                'avg_quality': 0.0,
                'avg_cost': 0.0,
                'success_rate': 0.0
            }
        
        stats = self.model_stats[model_name]
        
        return {
            'avg_latency': sum(stats['latencies']) / len(stats['latencies']) if stats['latencies'] else 0.0,
            'avg_quality': sum(stats['quality_scores']) / len(stats['quality_scores']) if stats['quality_scores'] else 0.0,
            'avg_cost': sum(stats['costs']) / len(stats['costs']) if stats['costs'] else 0.0,
            'success_rate': stats['success_count'] / stats['total_count'] if stats['total_count'] > 0 else 0.0
        }


class IntelligentTaskRouter:
    """智能任务路由器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.cost_optimizer = CostOptimizer()
        self.performance_monitor = PerformanceMonitor()
        
        # 初始化模型配置
        self._initialize_models()
        
        # 路由策略配置
        self.routing_strategies = {
            TaskType.QUERY_REWRITING: self._route_query_rewriting,
            TaskType.SUMMARIZATION: self._route_summarization,
            TaskType.QUESTION_GENERATION: self._route_question_generation,
            TaskType.FINAL_GENERATION: self._route_final_generation,
            TaskType.EVALUATION: self._route_evaluation,
            TaskType.COMPRESSION: self._route_compression
        }
        
        logger.info(f"Intelligent Task Router initialized with {len(self.models)} models")
    
    def _initialize_models(self):
        """初始化模型配置"""
        
        # 1. 本地模型配置
        if self.config.get('local_model_enabled', False):
            self.models['qwen2_7b'] = ModelConfig(
                name='qwen2_7b',
                model_type='local',
                cost_per_token=0.0,  # 本地模型无API成本
                max_tokens=4096,
                avg_latency=2.0,  # 本地推理稍慢
                quality_score=0.75,
                supported_languages=['zh', 'en']
            )
        
        # 2. OpenAI模型配置
        if self.config.get('openai_api_key'):
            self.models['gpt-3.5-turbo'] = ModelConfig(
                name='gpt-3.5-turbo',
                model_type='api',
                cost_per_token=0.000002,  # $0.002 per 1K tokens
                max_tokens=4096,
                avg_latency=1.5,
                quality_score=0.85,
                supported_languages=['zh', 'en'],
                model_instance=ChatOpenAI(
                    model="gpt-3.5-turbo",
                    api_key=self.config['openai_api_key'],
                    temperature=0.1
                )
            )
            
            self.models['gpt-4'] = ModelConfig(
                name='gpt-4',
                model_type='api',
                cost_per_token=0.00003,  # $0.03 per 1K tokens
                max_tokens=8192,
                avg_latency=3.0,
                quality_score=0.95,
                supported_languages=['zh', 'en'],
                model_instance=ChatOpenAI(
                    model="gpt-4",
                    api_key=self.config['openai_api_key'],
                    temperature=0.1
                )
            )
        
        # 3. Claude模型配置
        if self.config.get('claude_api_key'):
            self.models['claude-3-sonnet'] = ModelConfig(
                name='claude-3-sonnet',
                model_type='api',
                cost_per_token=0.000015,  # $0.015 per 1K tokens
                max_tokens=4096,
                avg_latency=2.0,
                quality_score=0.92,
                supported_languages=['zh', 'en']
            )
    
    async def route_task(self, task_request: TaskRequest) -> RoutingDecision:
        """路由任务到最优模型"""
        
        # 1. 获取任务类型对应的路由策略
        routing_strategy = self.routing_strategies.get(
            task_request.task_type, 
            self._default_routing_strategy
        )
        
        # 2. 执行路由策略
        candidate_models = routing_strategy(task_request)
        
        # 3. 应用成本优化和性能约束
        optimal_decision = await self._optimize_selection(task_request, candidate_models)
        
        logger.debug(f"Task routed: {task_request.task_type.value} -> {optimal_decision.selected_model.name}")
        
        return optimal_decision
    
    def _route_query_rewriting(self, task_request: TaskRequest) -> List[ModelConfig]:
        """查询重写任务路由"""
        candidates = []
        
        if task_request.complexity in [TaskComplexity.SIMPLE, TaskComplexity.MEDIUM]:
            # 简单和中等复杂度优先使用本地模型
            if 'qwen2_7b' in self.models:
                candidates.append(self.models['qwen2_7b'])
            if 'gpt-3.5-turbo' in self.models:
                candidates.append(self.models['gpt-3.5-turbo'])
        else:
            # 复杂任务使用高质量模型
            if 'gpt-4' in self.models:
                candidates.append(self.models['gpt-4'])
            if 'claude-3-sonnet' in self.models:
                candidates.append(self.models['claude-3-sonnet'])
        
        return candidates
    
    def _route_summarization(self, task_request: TaskRequest) -> List[ModelConfig]:
        """摘要生成任务路由"""
        candidates = []
        
        if task_request.complexity == TaskComplexity.SIMPLE:
            # 简单摘要优先本地模型
            if 'qwen2_7b' in self.models:
                candidates.append(self.models['qwen2_7b'])
        
        # 中等和复杂摘要使用API模型
        if 'gpt-3.5-turbo' in self.models:
            candidates.append(self.models['gpt-3.5-turbo'])
        if 'claude-3-sonnet' in self.models:
            candidates.append(self.models['claude-3-sonnet'])
        
        return candidates
    
    def _route_question_generation(self, task_request: TaskRequest) -> List[ModelConfig]:
        """问题生成任务路由"""
        candidates = []
        
        # 问题生成通常需要创意，倾向于使用质量更高的模型
        if task_request.quality_requirement >= 0.8:
            if 'gpt-4' in self.models:
                candidates.append(self.models['gpt-4'])
        
        if 'gpt-3.5-turbo' in self.models:
            candidates.append(self.models['gpt-3.5-turbo'])
        if 'qwen2_7b' in self.models:
            candidates.append(self.models['qwen2_7b'])
        
        return candidates
    
    def _route_final_generation(self, task_request: TaskRequest) -> List[ModelConfig]:
        """最终生成任务路由"""
        candidates = []
        
        # 根据复杂度选择模型
        if task_request.complexity == TaskComplexity.SIMPLE:
            if 'qwen2_7b' in self.models:
                candidates.append(self.models['qwen2_7b'])
            if 'gpt-3.5-turbo' in self.models:
                candidates.append(self.models['gpt-3.5-turbo'])
        
        elif task_request.complexity == TaskComplexity.MEDIUM:
            if 'gpt-3.5-turbo' in self.models:
                candidates.append(self.models['gpt-3.5-turbo'])
            if 'claude-3-sonnet' in self.models:
                candidates.append(self.models['claude-3-sonnet'])
        
        elif task_request.complexity == TaskComplexity.COMPLEX:
            if 'gpt-4' in self.models:
                candidates.append(self.models['gpt-4'])
            if 'claude-3-sonnet' in self.models:
                candidates.append(self.models['claude-3-sonnet'])
        
        else:  # CRITICAL
            # 关键任务优先使用最高质量模型
            if 'claude-3-sonnet' in self.models:
                candidates.append(self.models['claude-3-sonnet'])
            if 'gpt-4' in self.models:
                candidates.append(self.models['gpt-4'])
        
        return candidates
    
    def _route_evaluation(self, task_request: TaskRequest) -> List[ModelConfig]:
        """评估任务路由"""
        candidates = []
        
        # 评估任务通常可以使用较便宜的模型
        if 'qwen2_7b' in self.models:
            candidates.append(self.models['qwen2_7b'])
        if 'gpt-3.5-turbo' in self.models:
            candidates.append(self.models['gpt-3.5-turbo'])
        
        return candidates
    
    def _route_compression(self, task_request: TaskRequest) -> List[ModelConfig]:
        """压缩任务路由"""
        candidates = []
        
        # 压缩任务对质量要求相对较低
        if 'qwen2_7b' in self.models:
            candidates.append(self.models['qwen2_7b'])
        if 'gpt-3.5-turbo' in self.models:
            candidates.append(self.models['gpt-3.5-turbo'])
        
        return candidates
    
    def _default_routing_strategy(self, task_request: TaskRequest) -> List[ModelConfig]:
        """默认路由策略"""
        # 返回所有可用模型，让优化器选择
        return list(self.models.values())
    
    async def _optimize_selection(
        self,
        task_request: TaskRequest,
        candidate_models: List[ModelConfig]
    ) -> RoutingDecision:
        """优化模型选择"""
        
        if not candidate_models:
            raise ValueError("No candidate models available")
        
        best_model = None
        best_score = -1
        reasoning_parts = []
        
        for model in candidate_models:
            # 1. 检查基本约束
            if task_request.language not in model.supported_languages:
                continue
            
            if task_request.max_tokens and task_request.max_tokens > model.max_tokens:
                continue
            
            # 2. 估算成本
            estimated_tokens = min(task_request.max_tokens or 1000, 1000)
            estimated_cost = model.cost_per_token * estimated_tokens
            
            # 3. 检查成本约束
            cost_ok, cost_reason = self.cost_optimizer.check_cost_constraints(estimated_cost)
            if not cost_ok:
                reasoning_parts.append(f"{model.name}: {cost_reason}")
                continue
            
            # 4. 获取实时性能数据
            perf_data = self.performance_monitor.get_model_performance(model.name)
            actual_latency = perf_data['avg_latency'] if perf_data['avg_latency'] > 0 else model.avg_latency
            actual_quality = perf_data['avg_quality'] if perf_data['avg_quality'] > 0 else model.quality_score
            
            # 5. 检查延迟约束
            if actual_latency > task_request.latency_requirement:
                reasoning_parts.append(f"{model.name}: 延迟过高 ({actual_latency:.1f}s > {task_request.latency_requirement:.1f}s)")
                continue
            
            # 6. 检查质量约束
            if actual_quality < task_request.quality_requirement:
                reasoning_parts.append(f"{model.name}: 质量不足 ({actual_quality:.2f} < {task_request.quality_requirement:.2f})")
                continue
            
            # 7. 计算综合评分
            # 质量权重 * 质量分 + 成本权重 * 成本分 + 延迟权重 * 延迟分
            quality_weight = 0.4
            cost_weight = task_request.cost_sensitivity
            latency_weight = 1.0 - quality_weight - cost_weight
            
            # 归一化分数 (0-1)
            quality_score = actual_quality
            cost_score = max(0, 1 - estimated_cost / 0.1)  # 假设0.1为高成本阈值
            latency_score = max(0, 1 - actual_latency / 10.0)  # 假设10s为高延迟阈值
            
            combined_score = (
                quality_weight * quality_score +
                cost_weight * cost_score +
                latency_weight * latency_score
            )
            
            reasoning_parts.append(f"{model.name}: 综合评分={combined_score:.3f} (质量={quality_score:.2f}, 成本={cost_score:.2f}, 延迟={latency_score:.2f})")
            
            if combined_score > best_score:
                best_score = combined_score
                best_model = model
        
        if not best_model:
            # 如果所有模型都不满足约束，选择成本最低的
            best_model = min(candidate_models, key=lambda m: m.cost_per_token)
            reasoning_parts.append(f"约束放宽：选择成本最低模型 {best_model.name}")
        
        # 准备备选模型（按评分排序）
        remaining_models = [m for m in candidate_models if m != best_model]
        fallback_models = sorted(remaining_models, key=lambda m: m.quality_score, reverse=True)[:2]
        
        return RoutingDecision(
            selected_model=best_model,
            reasoning=" | ".join(reasoning_parts),
            confidence=best_score if best_score > 0 else 0.5,
            estimated_cost=best_model.cost_per_token * (task_request.max_tokens or 1000),
            estimated_latency=self.performance_monitor.get_model_performance(best_model.name).get('avg_latency', best_model.avg_latency),
            fallback_models=fallback_models
        )
    
    async def execute_task(
        self,
        task_request: TaskRequest,
        routing_decision: RoutingDecision,
        context: str = ""
    ) -> ExecutionResult:
        """执行任务"""
        
        start_time = time.time()
        model = routing_decision.selected_model
        
        try:
            # 构建提示
            prompt = self._build_prompt(task_request, context)
            
            # 执行推理
            if model.model_instance:
                # API模型
                response = await model.model_instance.ainvoke(prompt)
                output = response.content if hasattr(response, 'content') else str(response)
            else:
                # 本地模型或其他实现
                output = await self._execute_local_model(model, prompt)
            
            # 计算实际指标
            actual_latency = time.time() - start_time
            token_count = len(output.split())  # 简化的token计算
            actual_cost = model.cost_per_token * token_count
            
            # 记录使用情况
            self.cost_optimizer.record_usage(model.name, actual_cost, token_count)
            
            # 质量评估（简化版）
            quality_estimation = self._estimate_quality(output, task_request)
            
            # 记录性能
            self.performance_monitor.record_performance(
                model.name,
                actual_latency,
                quality_estimation,
                actual_cost,
                True
            )
            
            return ExecutionResult(
                output=output,
                model_used=model.name,
                actual_cost=actual_cost,
                actual_latency=actual_latency,
                token_count=token_count,
                quality_estimation=quality_estimation
            )
            
        except Exception as e:
            logger.error(f"Task execution failed with {model.name}: {e}")
            
            # 记录失败
            self.performance_monitor.record_performance(
                model.name,
                time.time() - start_time,
                0.0,
                0.0,
                False
            )
            
            # 尝试备选模型
            if routing_decision.fallback_models:
                logger.info(f"Trying fallback model: {routing_decision.fallback_models[0].name}")
                fallback_decision = RoutingDecision(
                    selected_model=routing_decision.fallback_models[0],
                    reasoning="Fallback due to primary model failure",
                    confidence=0.5,
                    estimated_cost=0.0,
                    estimated_latency=0.0,
                    fallback_models=[]
                )
                return await self.execute_task(task_request, fallback_decision, context)
            
            raise
    
    def _build_prompt(self, task_request: TaskRequest, context: str) -> str:
        """构建提示"""
        
        if task_request.task_type == TaskType.FINAL_GENERATION:
            return f"""基于以下上下文信息回答问题：

上下文：
{context}

问题：{task_request.content}

请提供详细、准确的回答。"""
        
        elif task_request.task_type == TaskType.SUMMARIZATION:
            return f"""请对以下内容进行摘要：

{task_request.content}

请生成简洁、准确的摘要。"""
        
        elif task_request.task_type == TaskType.QUERY_REWRITING:
            return f"""请优化以下查询，使其更适合信息检索：

原始查询：{task_request.content}

请提供优化后的查询。"""
        
        else:
            return task_request.content
    
    async def _execute_local_model(self, model: ModelConfig, prompt: str) -> str:
        """执行本地模型（模拟实现）"""
        # 这里应该集成实际的本地模型推理
        # 目前返回模拟响应
        await asyncio.sleep(model.avg_latency)  # 模拟延迟
        return f"[{model.name}模拟响应] {prompt[:100]}..."
    
    def _estimate_quality(self, output: str, task_request: TaskRequest) -> float:
        """估算输出质量（简化版）"""
        # 基于输出长度和任务类型的简单质量评估
        base_quality = 0.7
        
        # 长度评估
        if len(output) < 50:
            base_quality -= 0.2
        elif len(output) > 200:
            base_quality += 0.1
        
        # 任务类型调整
        if task_request.task_type == TaskType.FINAL_GENERATION:
            base_quality += 0.1  # 最终生成通常要求更高
        
        return min(max(base_quality, 0.0), 1.0)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        return {
            'available_models': list(self.models.keys()),
            'cost_summary': {
                'daily_cost': self.cost_optimizer._calculate_daily_cost(),
                'hourly_cost': self.cost_optimizer._calculate_hourly_cost(),
                'cost_thresholds': self.cost_optimizer.cost_thresholds
            },
            'model_performance': {
                name: self.performance_monitor.get_model_performance(name)
                for name in self.models.keys()
            }
        }
    
    def update_cost_thresholds(self, thresholds: Dict[str, float]):
        """更新成本阈值"""
        self.cost_optimizer.cost_thresholds.update(thresholds)
        logger.info(f"Cost thresholds updated: {self.cost_optimizer.cost_thresholds}")