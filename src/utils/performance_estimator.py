# src/utils/performance_estimator.py
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from pathlib import Path


@dataclass
class GPUSpecs:
    """GPU规格参数"""
    name: str
    vram_gb: float
    cuda_cores: int
    tensor_cores: int
    memory_bandwidth_gbps: float
    base_clock_mhz: int
    boost_clock_mhz: int


@dataclass
class ProcessingEstimate:
    """处理时间估算结果"""
    total_chunks: int
    embedding_time_seconds: float
    llm_generation_time_seconds: float
    total_time_seconds: float
    memory_usage_gb: float
    chunks_per_second: float
    bottleneck: str  # "embedding", "llm", "memory", "io"


class PerformanceEstimator:
    """性能估算器 - 基于硬件配置估算处理时间"""
    
    # GPU性能数据库
    GPU_SPECS = {
        "RTX 3090": GPUSpecs(
            name="RTX 3090",
            vram_gb=24.0,
            cuda_cores=10496,
            tensor_cores=328,
            memory_bandwidth_gbps=936.2,
            base_clock_mhz=1395,
            boost_clock_mhz=1695
        ),
        "RTX 4090": GPUSpecs(
            name="RTX 4090",
            vram_gb=24.0,
            cuda_cores=16384,
            tensor_cores=512,
            memory_bandwidth_gbps=1008.0,
            base_clock_mhz=1230,
            boost_clock_mhz=2520
        ),
        "RTX 3080": GPUSpecs(
            name="RTX 3080",
            vram_gb=10.0,
            cuda_cores=8704,
            tensor_cores=272,
            memory_bandwidth_gbps=760.3,
            base_clock_mhz=1440,
            boost_clock_mhz=1710
        ),
        "A100": GPUSpecs(
            name="A100",
            vram_gb=40.0,
            cuda_cores=6912,
            tensor_cores=432,
            memory_bandwidth_gbps=1555.0,
            base_clock_mhz=765,
            boost_clock_mhz=1410
        )
    }
    
    def __init__(self):
        self.detected_gpu = self._detect_gpu()
        self.system_info = self._get_system_info()
        
    def _detect_gpu(self) -> Optional[GPUSpecs]:
        """检测当前GPU型号"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"检测到GPU: {gpu_name}")
                
                # 匹配已知GPU规格
                for known_gpu, specs in self.GPU_SPECS.items():
                    if known_gpu.replace(" ", "").lower() in gpu_name.replace(" ", "").lower():
                        logger.success(f"匹配GPU规格: {specs.name}")
                        return specs
                
                # 如果未匹配到，返回通用估算
                logger.warning(f"未找到GPU规格数据，使用RTX 3090作为默认值")
                return self.GPU_SPECS["RTX 3090"]
            else:
                logger.warning("未检测到CUDA GPU，将使用CPU处理（速度较慢）")
                return None
        except Exception as e:
            logger.error(f"GPU检测失败: {e}")
            return None
    
    def _get_system_info(self) -> Dict:
        """获取系统信息"""
        info = {
            "cpu_count": 8,  # 默认值
            "ram_gb": 32.0,  # 默认值
            "gpu_available": False,
            "gpu_count": 0
        }
        
        if PSUTIL_AVAILABLE:
            info["cpu_count"] = psutil.cpu_count()
            info["ram_gb"] = psutil.virtual_memory().total / (1024**3)
        
        if TORCH_AVAILABLE:
            info["gpu_available"] = torch.cuda.is_available()
            info["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
            
            if torch.cuda.is_available():
                info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
        return info
    
    def estimate_multi_representation_processing(
        self, 
        chunk_count: int,
        config: Dict,
        enable_llm: bool = True
    ) -> ProcessingEstimate:
        """估算多表示处理时间"""
        
        logger.info(f"估算 {chunk_count} 个chunks的多表示处理时间...")
        
        # 基础参数
        embedding_model = config.get('embedding_model', 'BAAI/bge-m3')
        llm_model = config.get('llm_model', 'Qwen/Qwen2-7B-Instruct')
        batch_size = config.get('batch_size', 32)
        
        # 嵌入处理估算
        embedding_time = self._estimate_embedding_time(
            chunk_count, embedding_model, batch_size
        )
        
        # LLM生成估算
        llm_time = 0.0
        if enable_llm:
            llm_time = self._estimate_llm_generation_time(
                chunk_count, llm_model
            )
        
        # 内存使用估算
        memory_usage = self._estimate_memory_usage(
            chunk_count, embedding_model, llm_model, enable_llm
        )
        
        # 确定瓶颈
        bottleneck = self._identify_bottleneck(embedding_time, llm_time, memory_usage)
        
        total_time = max(embedding_time, llm_time)  # 部分并行处理
        chunks_per_second = chunk_count / total_time if total_time > 0 else 0
        
        return ProcessingEstimate(
            total_chunks=chunk_count,
            embedding_time_seconds=embedding_time,
            llm_generation_time_seconds=llm_time,
            total_time_seconds=total_time,
            memory_usage_gb=memory_usage,
            chunks_per_second=chunks_per_second,
            bottleneck=bottleneck
        )
    
    def _estimate_embedding_time(
        self, 
        chunk_count: int, 
        model_name: str, 
        batch_size: int
    ) -> float:
        """估算嵌入处理时间"""
        
        # 模型复杂度系数（相对于bge-m3）
        model_complexity = {
            'BAAI/bge-m3': 1.0,
            'BAAI/bge-large-zh-v1.5': 1.2,
            'text-embedding-ada-002': 0.8,  # API调用
            'sentence-transformers/all-MiniLM-L6-v2': 0.4
        }
        
        complexity = model_complexity.get(model_name, 1.0)
        
        if self.detected_gpu:
            # GPU处理时间估算
            # RTX 3090基准: ~500 chunks/second (bge-m3, batch_size=32)
            base_throughput = {
                "RTX 3090": 500,
                "RTX 4090": 750,
                "RTX 3080": 350,
                "A100": 800
            }
            
            throughput = base_throughput.get(self.detected_gpu.name, 500)
            throughput = throughput / complexity  # 调整模型复杂度
            
            # 考虑批处理效率
            batch_efficiency = min(batch_size / 32, 1.5)
            throughput *= batch_efficiency
            
        else:
            # CPU处理时间估算 (much slower)
            throughput = 20 / complexity  # ~20 chunks/second on modern CPU
        
        # 包含模型加载时间
        model_load_time = 30 if self.detected_gpu else 60  # seconds
        processing_time = chunk_count / throughput
        
        return model_load_time + processing_time
    
    def _estimate_llm_generation_time(self, chunk_count: int, model_name: str) -> float:
        """估算LLM生成时间"""
        
        # 模型大小和速度映射
        model_speeds = {
            'Qwen/Qwen2-7B-Instruct': {'tokens_per_sec': 45, 'load_time': 45},
            'Qwen/Qwen2-1.5B-Instruct': {'tokens_per_sec': 80, 'load_time': 20},
            'Qwen/Qwen2-14B-Instruct': {'tokens_per_sec': 25, 'load_time': 60},
            'microsoft/DialoGPT-medium': {'tokens_per_sec': 60, 'load_time': 30}
        }
        
        model_speed = model_speeds.get(model_name, {'tokens_per_sec': 40, 'load_time': 45})
        
        if self.detected_gpu:
            tokens_per_sec = model_speed['tokens_per_sec']
            
            # RTX 3090性能调整
            if self.detected_gpu.name == "RTX 3090":
                tokens_per_sec *= 1.0  # 基准
            elif self.detected_gpu.name == "RTX 4090":
                tokens_per_sec *= 1.6  # 更快
            elif self.detected_gpu.name == "RTX 3080":
                tokens_per_sec *= 0.7  # 稍慢
            elif self.detected_gpu.name == "A100":
                tokens_per_sec *= 2.0  # 专业级
        else:
            tokens_per_sec = model_speed['tokens_per_sec'] * 0.1  # CPU much slower
        
        # 每个chunk需要生成的token数估算
        # 摘要: ~150 tokens, 问题: ~50 tokens * 3 = 150 tokens
        # 总计: ~300 tokens per chunk
        tokens_per_chunk = 300
        
        total_tokens = chunk_count * tokens_per_chunk
        generation_time = total_tokens / tokens_per_sec
        
        # 加载时间
        load_time = model_speed['load_time']
        
        return load_time + generation_time
    
    def _estimate_memory_usage(
        self, 
        chunk_count: int, 
        embedding_model: str, 
        llm_model: str, 
        enable_llm: bool
    ) -> float:
        """估算内存使用量(GB)"""
        
        # 模型内存使用估算
        embedding_memory = {
            'BAAI/bge-m3': 2.5,
            'BAAI/bge-large-zh-v1.5': 3.0,
            'sentence-transformers/all-MiniLM-L6-v2': 1.5
        }.get(embedding_model, 2.5)
        
        llm_memory = 0
        if enable_llm:
            llm_memory = {
                'Qwen/Qwen2-7B-Instruct': 14.0,  # ~14GB in fp16
                'Qwen/Qwen2-1.5B-Instruct': 3.5,
                'Qwen/Qwen2-14B-Instruct': 28.0
            }.get(llm_model, 14.0)
        
        # 数据内存使用
        # 假设每个chunk平均1KB原文 + 嵌入向量(1024*4bytes) + 生成内容
        chunk_memory_mb = 1 + 4 + 2  # ~7MB per chunk
        data_memory = (chunk_count * chunk_memory_mb) / 1024  # Convert to GB
        
        # 系统开销
        system_overhead = 2.0  # GB
        
        total_memory = embedding_memory + llm_memory + data_memory + system_overhead
        
        return total_memory
    
    def _identify_bottleneck(
        self, 
        embedding_time: float, 
        llm_time: float, 
        memory_usage: float
    ) -> str:
        """识别性能瓶颈"""
        
        # 检查内存瓶颈
        if self.detected_gpu and memory_usage > self.detected_gpu.vram_gb * 0.9:
            return "memory"
        elif memory_usage > self.system_info["ram_gb"] * 0.8:
            return "memory"
        
        # 检查计算瓶颈
        if llm_time > embedding_time * 2:
            return "llm"
        elif embedding_time > llm_time * 2:
            return "embedding"
        else:
            return "balanced"
    
    def print_estimate_report(self, estimate: ProcessingEstimate):
        """打印估算报告"""
        
        print("\n" + "="*60)
        print("🚀 多表示处理性能估算报告")
        print("="*60)
        
        print(f"📊 处理规模: {estimate.total_chunks:,} chunks")
        print(f"⚡ 处理速度: {estimate.chunks_per_second:.1f} chunks/秒")
        print()
        
        print("⏱️  时间估算:")
        print(f"  • 嵌入处理: {estimate.embedding_time_seconds/60:.1f} 分钟")
        print(f"  • LLM生成: {estimate.llm_generation_time_seconds/60:.1f} 分钟")
        print(f"  • 总处理时间: {estimate.total_time_seconds/60:.1f} 分钟")
        print()
        
        print(f"💾 内存使用: {estimate.memory_usage_gb:.1f} GB")
        
        if self.detected_gpu:
            print(f"🎮 GPU信息: {self.detected_gpu.name} ({self.detected_gpu.vram_gb}GB VRAM)")
            vram_usage_percent = (estimate.memory_usage_gb / self.detected_gpu.vram_gb) * 100
            print(f"  • VRAM使用率: {vram_usage_percent:.1f}%")
        
        print(f"🔍 性能瓶颈: {estimate.bottleneck}")
        
        # 优化建议
        print("\n💡 优化建议:")
        if estimate.bottleneck == "memory":
            print("  • 减少批处理大小")
            print("  • 考虑使用量化模型")
            print("  • 分批处理chunks")
        elif estimate.bottleneck == "llm":
            print("  • 考虑使用更小的LLM模型")
            print("  • 减少生成的问题数量")
            print("  • 使用并行处理")
        elif estimate.bottleneck == "embedding":
            print("  • 增加批处理大小")
            print("  • 考虑使用更快的嵌入模型")
        
        print("="*60)


def benchmark_system_performance() -> Dict:
    """系统性能基准测试"""
    logger.info("开始系统性能基准测试...")
    
    results = {}
    
    # GPU基准测试
    if TORCH_AVAILABLE and torch.cuda.is_available():
        device = torch.device("cuda")
        
        # 矩阵乘法测试
        start_time = time.time()
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        for _ in range(100):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        gpu_compute_time = time.time() - start_time
        
        results["gpu_compute_score"] = 100 / gpu_compute_time  # Higher is better
        
        # 内存带宽测试
        start_time = time.time()
        large_tensor = torch.randn(10000, 10000, device=device)
        for _ in range(10):
            copied = large_tensor.clone()
        torch.cuda.synchronize()
        memory_bandwidth_time = time.time() - start_time
        
        results["memory_bandwidth_score"] = 100 / memory_bandwidth_time
        
    else:
        results["gpu_compute_score"] = 0
        results["memory_bandwidth_score"] = 0
    
    # CPU基准测试
    try:
        start_time = time.time()
        import numpy as np
        a = np.random.randn(1000, 1000)
        b = np.random.randn(1000, 1000)
        for _ in range(100):
            c = np.dot(a, b)
        cpu_compute_time = time.time() - start_time
        
        results["cpu_compute_score"] = 100 / cpu_compute_time
    except ImportError:
        results["cpu_compute_score"] = 50  # 默认值
    
    logger.success("系统性能基准测试完成")
    return results


# 使用示例
if __name__ == "__main__":
    estimator = PerformanceEstimator()
    
    # 配置示例
    config = {
        'embedding_model': 'BAAI/bge-m3',
        'llm_model': 'Qwen/Qwen2-7B-Instruct',
        'batch_size': 32
    }
    
    # 估算731个chunks的处理时间
    estimate = estimator.estimate_multi_representation_processing(
        chunk_count=731,
        config=config,
        enable_llm=True
    )
    
    estimator.print_estimate_report(estimate)
    
    # 运行基准测试
    benchmark_results = benchmark_system_performance()
    print(f"\n🔬 基准测试结果: {benchmark_results}")