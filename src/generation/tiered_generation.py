# src/generation/tiered_generation.py
from typing import List, Dict, Optional, Tuple, Any, Union
import asyncio
import time
from dataclasses import dataclass
from enum import Enum
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from loguru import logger
import openai
from pathlib import Path
import json

class TaskComplexity(Enum):
    """任务复杂度"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    CRITICAL = "critical"

class TaskType(Enum):
    """任务类型"""
    QUERY_REWRITE = "query_rewrite"
    CONTEXT_COMPRESSION = "context_compression"
    QUALITY_EVALUATION = "quality_evaluation"
    FINAL_GENERATION = "final_generation"
    SUMMARIZATION = "summarization"
    FACT_CHECKING = "fact_checking"
    REASONING = "reasoning"

@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    model_type: str  # 'local', 'api'
    max_tokens: int
    temperature: float
    cost_per_token: float = 0.0
    speed_score: float = 1.0  # 相对速度评分
    quality_score: float = 1.0  # 质量评分
    capabilities: List[str] = None  # 支持的能力
    api_key: Optional[str] = None
    api_base: Optional[str] = None

@dataclass
class TaskRequest:
    """任务请求"""
    task_id: str
    task_type: TaskType
    complexity: TaskComplexity
    prompt: str
    context: Dict[str, Any]
    max_tokens: int = 512
    temperature: float = 0.1
    priority: int = 1  # 1-5, 5最高
    timeout: float = 30.0

@dataclass
class TaskResponse:
    """任务响应"""
    task_id: str
    result: str
    model_used: str
    execution_time: float
    token_count: int
    cost: float
    quality_score: float
    success: bool
    error_message: Optional[str] = None

class TaskRouter:
    """任务路由器"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 初始化模型配置
        self.models = self._initialize_models()
        
        # 任务-模型映射规则
        self.routing_rules = self._initialize_routing_rules()
        
        logger.info(f"Task Router initialized with {len(self.models)} models")
    
    def _initialize_models(self) -> Dict[str, ModelConfig]:
        """初始化模型配置"""
        models = {}
        
        # 本地模型（如果配置了）
        local_model = self.config.get('llm_model')
        if local_model:
            models['local_llm'] = ModelConfig(
                name=local_model,
                model_type='local',
                max_tokens=2048,
                temperature=0.1,
                cost_per_token=0.0,  # 本地模型无成本
                speed_score=0.6,  # 相对较慢
                quality_score=0.7,  # 中等质量
                capabilities=['chinese', 'reasoning', 'generation', 'summarization']
            )
        
        # 快速本地模型（7B参数）
        models['fast_local'] = ModelConfig(
            name=self.config.get('fast_model', local_model),
            model_type='local',
            max_tokens=1024,
            temperature=0.2,
            cost_per_token=0.0,
            speed_score=0.9,  # 快速
            quality_score=0.6,  # 质量稍低
            capabilities=['query_rewrite', 'compression', 'evaluation']
        )
        
        # API模型配置
        api_models_config = self.config.get('api_models', {})
        
        # GPT-4配置
        if api_models_config.get('gpt4_api_key'):
            models['gpt4'] = ModelConfig(
                name='gpt-4',
                model_type='api',
                max_tokens=4000,
                temperature=0.1,
                cost_per_token=0.00006,  # GPT-4大概成本
                speed_score=0.4,  # 较慢但质量高
                quality_score=0.95,
                capabilities=['reasoning', 'generation', 'analysis', 'chinese', 'english'],
                api_key=api_models_config['gpt4_api_key'],
                api_base=api_models_config.get('gpt4_api_base')
            )
        
        # GPT-3.5配置
        if api_models_config.get('gpt35_api_key'):
            models['gpt35'] = ModelConfig(
                name='gpt-3.5-turbo',
                model_type='api',
                max_tokens=2000,
                temperature=0.2,
                cost_per_token=0.000002,  # GPT-3.5成本
                speed_score=0.8,  # 较快
                quality_score=0.8,
                capabilities=['generation', 'rewrite', 'summarization', 'chinese', 'english'],
                api_key=api_models_config['gpt35_api_key'],
                api_base=api_models_config.get('gpt35_api_base')
            )
        
        # Claude配置
        if api_models_config.get('claude_api_key'):
            models['claude'] = ModelConfig(
                name='claude-3-sonnet-20240229',
                model_type='api',
                max_tokens=3000,
                temperature=0.1,
                cost_per_token=0.000015,
                speed_score=0.6,
                quality_score=0.9,
                capabilities=['reasoning', 'analysis', 'generation', 'chinese', 'english'],
                api_key=api_models_config['claude_api_key']
            )
        
        return models
    
    def _initialize_routing_rules(self) -> Dict[TaskType, Dict[TaskComplexity, str]]:
        """初始化路由规则"""
        rules = {
            TaskType.QUERY_REWRITE: {
                TaskComplexity.SIMPLE: 'fast_local',
                TaskComplexity.MEDIUM: 'fast_local',
                TaskComplexity.COMPLEX: 'gpt35',
                TaskComplexity.CRITICAL: 'gpt4'
            },
            TaskType.CONTEXT_COMPRESSION: {
                TaskComplexity.SIMPLE: 'fast_local',
                TaskComplexity.MEDIUM: 'local_llm',
                TaskComplexity.COMPLEX: 'gpt35',
                TaskComplexity.CRITICAL: 'gpt4'
            },
            TaskType.QUALITY_EVALUATION: {
                TaskComplexity.SIMPLE: 'fast_local',
                TaskComplexity.MEDIUM: 'local_llm',
                TaskComplexity.COMPLEX: 'gpt35',
                TaskComplexity.CRITICAL: 'claude'
            },
            TaskType.FINAL_GENERATION: {
                TaskComplexity.SIMPLE: 'local_llm',
                TaskComplexity.MEDIUM: 'gpt35',
                TaskComplexity.COMPLEX: 'gpt4',
                TaskComplexity.CRITICAL: 'claude'
            },
            TaskType.SUMMARIZATION: {
                TaskComplexity.SIMPLE: 'fast_local',
                TaskComplexity.MEDIUM: 'local_llm',
                TaskComplexity.COMPLEX: 'gpt35',
                TaskComplexity.CRITICAL: 'gpt4'
            },
            TaskType.FACT_CHECKING: {
                TaskComplexity.SIMPLE: 'local_llm',
                TaskComplexity.MEDIUM: 'gpt35',
                TaskComplexity.COMPLEX: 'claude',
                TaskComplexity.CRITICAL: 'gpt4'
            },
            TaskType.REASONING: {
                TaskComplexity.SIMPLE: 'local_llm',
                TaskComplexity.MEDIUM: 'gpt35',
                TaskComplexity.COMPLEX: 'claude',
                TaskComplexity.CRITICAL: 'gpt4'
            }
        }
        
        return rules
    
    def route_task(self, task: TaskRequest) -> str:
        """路由任务到合适的模型"""
        
        # 获取基础路由
        base_model = self.routing_rules.get(task.task_type, {}).get(
            task.complexity, 'local_llm'
        )
        
        # 检查模型可用性
        if base_model not in self.models:
            # 降级到可用模型
            available_models = list(self.models.keys())
            if 'local_llm' in available_models:
                logger.warning(f"Model {base_model} not available, using local_llm")
                return 'local_llm'
            elif available_models:
                fallback_model = available_models[0]
                logger.warning(f"Model {base_model} not available, using {fallback_model}")
                return fallback_model
            else:
                raise RuntimeError("No models available")
        
        # 检查优先级和成本约束
        model_config = self.models[base_model]
        
        # 高优先级任务可以使用更昂贵的模型
        if task.priority >= 4 and model_config.cost_per_token > 0:
            # 尝试升级到更好的模型
            better_models = [
                name for name, config in self.models.items()
                if (config.quality_score > model_config.quality_score and
                    config.cost_per_token <= model_config.cost_per_token * 2)
            ]
            if better_models:
                base_model = max(better_models, key=lambda x: self.models[x].quality_score)
        
        # 时间敏感任务优先考虑速度
        if task.timeout < 10.0:
            fast_models = [
                name for name, config in self.models.items()
                if config.speed_score >= 0.8
            ]
            if fast_models and base_model not in fast_models:
                base_model = max(fast_models, key=lambda x: self.models[x].quality_score)
        
        logger.debug(f"Routed {task.task_type.value} task to {base_model}")
        return base_model

class LocalModelExecutor:
    """本地模型执行器"""
    
    def __init__(self, model_config: ModelConfig, device: str = "auto", token: Optional[str] = None):
        self.config = model_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载模型
        logger.info(f"Loading local model: {model_config.name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.name, trust_remote_code=True, token=token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            token=token
        )
        
        logger.success(f"Local model {model_config.name} loaded")
    
    async def execute(self, task: TaskRequest) -> TaskResponse:
        """执行任务"""
        start_time = time.time()
        
        try:
            # 配置生成参数
            generation_config = GenerationConfig(
                max_new_tokens=min(task.max_tokens, self.config.max_tokens),
                temperature=task.temperature,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # 生成响应
            inputs = self.tokenizer(task.prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, generation_config=generation_config)
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的内容（移除原始prompt）
            if task.prompt in result:
                result = result.replace(task.prompt, "").strip()
            
            execution_time = time.time() - start_time
            token_count = len(outputs[0]) - len(inputs['input_ids'][0])
            
            return TaskResponse(
                task_id=task.task_id,
                result=result,
                model_used=self.config.name,
                execution_time=execution_time,
                token_count=token_count,
                cost=0.0,  # 本地模型无成本
                quality_score=self.config.quality_score,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Local model execution failed: {e}")
            
            return TaskResponse(
                task_id=task.task_id,
                result="",
                model_used=self.config.name,
                execution_time=execution_time,
                token_count=0,
                cost=0.0,
                quality_score=0.0,
                success=False,
                error_message=str(e)
            )

class APIModelExecutor:
    """API模型执行器"""
    
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        
        # 配置API客户端
        if 'gpt' in model_config.name.lower():
            openai.api_key = model_config.api_key
            if model_config.api_base:
                openai.api_base = model_config.api_base
        
        logger.info(f"API executor for {model_config.name} initialized")
    
    async def execute(self, task: TaskRequest) -> TaskResponse:
        """执行API任务"""
        start_time = time.time()
        
        try:
            if 'gpt' in self.config.name.lower():
                response = await self._execute_openai_task(task)
            elif 'claude' in self.config.name.lower():
                response = await self._execute_claude_task(task)
            else:
                raise NotImplementedError(f"API executor for {self.config.name} not implemented")
            
            execution_time = time.time() - start_time
            token_count = len(response.split())  # 简单估算
            cost = token_count * self.config.cost_per_token
            
            return TaskResponse(
                task_id=task.task_id,
                result=response,
                model_used=self.config.name,
                execution_time=execution_time,
                token_count=token_count,
                cost=cost,
                quality_score=self.config.quality_score,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"API model execution failed: {e}")
            
            return TaskResponse(
                task_id=task.task_id,
                result="",
                model_used=self.config.name,
                execution_time=execution_time,
                token_count=0,
                cost=0.0,
                quality_score=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_openai_task(self, task: TaskRequest) -> str:
        """执行OpenAI任务"""
        response = await openai.ChatCompletion.acreate(
            model=self.config.name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": task.prompt}
            ],
            max_tokens=min(task.max_tokens, self.config.max_tokens),
            temperature=task.temperature,
            timeout=task.timeout
        )
        
        return response.choices[0].message.content
    
    async def _execute_claude_task(self, task: TaskRequest) -> str:
        """执行Claude任务"""
        # 这里需要实际的Claude API调用
        # 暂时返回模拟响应
        logger.warning("Claude API execution not implemented, returning mock response")
        return f"Claude response for: {task.prompt[:50]}..."

class TieredGenerationSystem:
    """分层生成系统"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 初始化任务路由器
        self.task_router = TaskRouter(config)
        
        # 初始化模型执行器
        self.executors = {}
        self._initialize_executors()
        
        # 任务统计
        self.task_stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'total_cost': 0.0,
            'model_usage': {},
            'avg_response_time': 0.0
        }
        
        logger.success("Tiered Generation System initialized")
    
    def _initialize_executors(self):
        """初始化模型执行器"""
        for model_name, model_config in self.task_router.models.items():
            try:
                if model_config.model_type == 'local':
                    executor = LocalModelExecutor(
                        model_config, 
                        device=self.config.get('device', 'auto'),
                        token=self.config.get('HUGGING_FACE_TOKEN')
                    )
                elif model_config.model_type == 'api':
                    executor = APIModelExecutor(model_config)
                else:
                    continue
                
                self.executors[model_name] = executor
                logger.info(f"Initialized executor for {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize executor for {model_name}: {e}")
    
    async def execute_task(self, task: TaskRequest) -> TaskResponse:
        """执行单个任务"""
        # 路由任务
        selected_model = self.task_router.route_task(task)
        
        # 检查执行器可用性
        if selected_model not in self.executors:
            # fallback到第一个可用的执行器
            if self.executors:
                selected_model = list(self.executors.keys())[0]
                logger.warning(f"Fallback to {selected_model}")
            else:
                raise RuntimeError("No executors available")
        
        # 执行任务
        executor = self.executors[selected_model]
        response = await executor.execute(task)
        
        # 更新统计
        self._update_stats(response)
        
        logger.info(f"Task {task.task_id} completed with {selected_model} in {response.execution_time:.2f}s")
        return response
    
    def _update_stats(self, response: TaskResponse):
        """更新任务统计"""
        self.task_stats['total_tasks'] += 1
        if response.success:
            self.task_stats['successful_tasks'] += 1
        
        self.task_stats['total_cost'] += response.cost
        
        model_name = response.model_used
        if model_name not in self.task_stats['model_usage']:
            self.task_stats['model_usage'][model_name] = 0
        self.task_stats['model_usage'][model_name] += 1
        
        # 更新平均响应时间
        total_time = self.task_stats['avg_response_time'] * (self.task_stats['total_tasks'] - 1)
        self.task_stats['avg_response_time'] = (total_time + response.execution_time) / self.task_stats['total_tasks']
    
    async def execute_workflow(self, workflow_tasks: List[TaskRequest]) -> List[TaskResponse]:
        """执行任务工作流"""
        logger.info(f"Executing workflow with {len(workflow_tasks)} tasks")
        
        responses = []
        
        # 按优先级排序
        workflow_tasks.sort(key=lambda x: x.priority, reverse=True)
        
        # 并发执行（如果任务间无依赖）
        if all(task.priority <= 2 for task in workflow_tasks):
            # 低优先级任务可以并发
            tasks = [self.execute_task(task) for task in workflow_tasks]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # 高优先级任务串行执行
            for task in workflow_tasks:
                response = await self.execute_task(task)
                responses.append(response)
                
                # 如果关键任务失败，可能需要停止后续任务
                if task.priority >= 4 and not response.success:
                    logger.error(f"Critical task {task.task_id} failed, stopping workflow")
                    break
        
        logger.info(f"Workflow completed: {sum(1 for r in responses if isinstance(r, TaskResponse) and r.success)}/{len(responses)} tasks successful")
        return responses
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'available_models': list(self.executors.keys()),
            'task_stats': self.task_stats.copy(),
            'model_configs': {
                name: {
                    'type': config.model_type,
                    'quality_score': config.quality_score,
                    'speed_score': config.speed_score,
                    'cost_per_token': config.cost_per_token
                }
                for name, config in self.task_router.models.items()
            }
        }
    
    def estimate_task_cost(self, task: TaskRequest) -> Dict[str, float]:
        """估算任务成本"""
        selected_model = self.task_router.route_task(task)
        model_config = self.task_router.models[selected_model]
        
        estimated_tokens = len(task.prompt.split()) + task.max_tokens
        estimated_cost = estimated_tokens * model_config.cost_per_token
        estimated_time = estimated_tokens / (model_config.speed_score * 100)  # 简单估算
        
        return {
            'selected_model': selected_model,
            'estimated_cost': estimated_cost,
            'estimated_time': estimated_time,
            'estimated_tokens': estimated_tokens
        }

# 使用示例
async def main():
    """测试分层生成系统"""
    config = {
        'llm_model': 'Qwen/Qwen2-7B-Instruct',
        'fast_model': 'Qwen/Qwen2-7B-Instruct',
        'device': 'auto',
        'HUGGING_FACE_TOKEN': None,
        'api_models': {
            # 'gpt4_api_key': 'your-openai-key',
            # 'gpt35_api_key': 'your-openai-key',
        }
    }
    
    # 初始化系统
    tiered_system = TieredGenerationSystem(config)
    
    # 创建测试任务
    tasks = [
        TaskRequest(
            task_id="task_1",
            task_type=TaskType.QUERY_REWRITE,
            complexity=TaskComplexity.SIMPLE,
            prompt="重写查询：什么是机器学习？",
            context={},
            priority=1
        ),
        TaskRequest(
            task_id="task_2",
            task_type=TaskType.FINAL_GENERATION,
            complexity=TaskComplexity.COMPLEX,
            prompt="请详细解释Transformer模型的工作原理",
            context={},
            priority=4
        )
    ]
    
    # 执行任务
    for task in tasks:
        # 估算成本
        cost_info = tiered_system.estimate_task_cost(task)
        print(f"Task {task.task_id} cost estimation: {cost_info}")
        
        # 执行任务
        response = await tiered_system.execute_task(task)
        print(f"Task {task.task_id} response:")
        print(f"  Model: {response.model_used}")
        print(f"  Time: {response.execution_time:.2f}s")
        print(f"  Cost: ${response.cost:.4f}")
        print(f"  Success: {response.success}")
        print(f"  Result: {response.result[:100]}...")
        print("---")
    
    # 系统状态
    status = tiered_system.get_system_status()
    print("System Status:")
    print(json.dumps(status, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())