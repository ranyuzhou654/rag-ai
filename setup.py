# setup.py - Smart environment setup script for cloud environments like autodl
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

def install_requirements():
    """安装所有必需的依赖"""
    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        print(f"❌ 错误: 'requirements.txt' 文件未找到！")
        return False
    
    print("📦 正在安装依赖项...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
        print("✅ 所有依赖安装成功!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False

def configure_environment():
    """配置项目环境, 特别是数据和模型缓存路径"""
    print("\n... ⚙️ 开始配置项目环境 ...")
    
    # autodl 常见的数据盘路径
    autodl_paths = ["/root/autodl-tmp", "/root/autodl-fs"]
    detected_path = None
    
    for path in autodl_paths:
        if Path(path).exists():
            detected_path = path
            break
            
    if detected_path:
        print(f"✅ 检测到 autodl 数据盘路径: {detected_path}")
        use_path = input(f"是否使用此路径作为数据存储根目录? (Y/n): ").lower().strip()
        if use_path == 'n':
            data_disk_path_str = input("请输入你的数据盘路径: ")
        else:
            data_disk_path_str = detected_path
    else:
        print("⚠️ 未自动检测到 autodl 数据盘路径。")
        data_disk_path_str = input("请输入你的数据盘/大容量磁盘路径 (例如 /data/storage): ")

    data_disk_path = Path(data_disk_path_str).resolve()
    
    if not data_disk_path.exists() or not data_disk_path.is_dir():
        print(f"❌ 错误: 路径 '{data_disk_path}' 不存在或不是一个目录。")
        sys.exit(1)
        
    project_storage_path = data_disk_path / "rag_zh_project_data"
    model_cache_path = project_storage_path / "models"
    data_path = project_storage_path / "data"
    logs_path = project_storage_path / "logs"

    # Ask for Hugging Face Token
    print("\n... 🔑 配置Hugging Face访问 ...")
    print("为了下载部分模型（如Qwen2），需要配置Hugging Face访问令牌。")
    print("你可以从这里获取令牌: https://huggingface.co/settings/tokens")
    hf_token = input("请输入你的Hugging Face读取令牌 (留空则跳过): ").strip()
    
    # 创建 .env 文件，格式与仓库中的模板保持一致
    env_content = dedent(f"""
        # 环境配置文件
        # 请根据你的系统设置修改以下配置

        # === 存储路径配置 ===
        # 项目数据存储根目录（默认：当前目录下的project_data）
        STORAGE_ROOT={project_storage_path}

        # Hugging Face 模型缓存目录（可选，默认使用系统默认位置）
        HF_HOME={model_cache_path}
        TRANSFORMERS_CACHE={model_cache_path}
        SENTENCE_TRANSFORMERS_HOME={model_cache_path}

        # === Hugging Face 认证 ===
        # 获取token: https://huggingface.co/settings/tokens
        # 对于大多数公开模型，这是可选的，但建议设置以避免rate limit
        HUGGING_FACE_TOKEN={hf_token}

        # === 模型配置 ===
        # 嵌入模型（默认：BAAI/bge-m3）
        EMBEDDING_MODEL=BAAI/bge-m3

        # 本地LLM模型（默认：Qwen/Qwen2-7B-Instruct）
        LLM_MODEL=Qwen/Qwen2-7B-Instruct

        # 设备选择：auto, cpu, cuda, mps（Mac M1/M2）
        DEVICE=auto

        # === 向量数据库配置 ===
        QDRANT_HOST=localhost
        QDRANT_PORT=6333
        COLLECTION_NAME=ai_papers

        # === 处理参数 ===
        CHUNK_SIZE=512
        CHUNK_OVERLAP=50
        MAX_TOKENS=4096
        TEMPERATURE=0.1

        # === 数据源配置 ===
        # ArXiv相关配置
        MAX_PAPERS_PER_FETCH=50

        # 是否启用 Semantic Scholar 增强（需要网络访问）
        ENABLE_SEMANTIC_SCHOLAR=true
        # 可选：Semantic Scholar Graph API Key，用于更高的速率限制
        SEMANTIC_SCHOLAR_API_KEY=

        # === 功能开关 ===
        # 是否启用查询智能增强
        ENABLE_QUERY_INTELLIGENCE=true

        # 是否启用多表示索引
        ENABLE_MULTI_REPRESENTATION=true

        # 是否启用智能体RAG
        ENABLE_AGENTIC_RAG=true

        # 是否启用上下文压缩
        ENABLE_CONTEXTUAL_COMPRESSION=true

        # 是否启用知识图谱
        ENABLE_KNOWLEDGE_GRAPH=true

        # 是否启用分层生成
        ENABLE_TIERED_GENERATION=true

        # 是否启用反馈收集
        ENABLE_FEEDBACK_COLLECTION=true

        # === API模型配置（可选）===
        # 用于复杂任务的API模型，如果不设置则使用本地模型

        # OpenAI GPT-4
        # GPT4_API_KEY=your_gpt4_api_key
        # GPT4_API_BASE=https://api.openai.com/v1

        # OpenAI GPT-3.5
        # GPT35_API_KEY=your_gpt35_api_key
        # GPT35_API_BASE=https://api.openai.com/v1

        # Anthropic Claude
        # CLAUDE_API_KEY=your_claude_api_key
        # CLAUDE_API_BASE=https://api.anthropic.com

        # === 高级配置 ===
        # RAG模式：basic, enhanced, agentic, ultimate
        DEFAULT_RAG_MODE=ultimate

        # 智能体RAG最大迭代次数
        MAX_AGENTIC_ITERATIONS=3

        # 最小chunks阈值
        MIN_CHUNKS_THRESHOLD=2

        # 压缩方法：sentence_extraction, llm_compression, hybrid
        COMPRESSION_METHOD=hybrid

        # 是否使用智能重排序
        USE_SMART_RERANKING=true
        """)

    with open(".env", "w") as f:
        f.write(env_content.strip() + "\n")
    print("✅ 成功创建 .env 配置文件！")

    # 创建必要的目录结构
    for p in [project_storage_path, model_cache_path, data_path, logs_path, data_path / "raw", data_path / "processed"]:
        p.mkdir(exist_ok=True)
    print("✅ 目录结构创建成功。")


if __name__ == "__main__":
    print("🚀 开始设置RAG项目环境...")
    configure_environment()
    install_requirements()
    print("\n🎉 环境设置完成!")
    print("请运行 `run_rag_system.py` 来采集数据。")
