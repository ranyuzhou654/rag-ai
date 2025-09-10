import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
# 这应该在所有其他导入之前执行
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# --- 设置Hugging Face缓存目录 ---
# 必须在导入transformers或sentence_transformers之前设置
# 这样可以确保模型从一开始就被下载到正确的位置
hf_home = os.getenv("HF_HOME")
if hf_home:
    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = hf_home
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = hf_home
    print(f"✅ Hugging Face cache directory set to: {hf_home}")


class Config:
    # 项目代码的根目录
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # 从.env文件读取存储根目录
    # 如果.env文件不存在或未设置，则默认为项目代码目录下的 'project_data'
    STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", PROJECT_ROOT / "project_data"))
    
    # 基于存储根目录定义其他路径
    DATA_DIR = STORAGE_ROOT / "data"
    MODEL_DIR = STORAGE_ROOT / "models" # 现在由HF_HOME环境变量控制
    LOG_DIR = STORAGE_ROOT / "logs"
    
    # Hugging Face Token
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
    
    # 模型配置 (从.env或默认值)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2-7B-Instruct")
    DEVICE = os.getenv("DEVICE", "auto")
    
    # 向量数据库
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ai_papers")
    
    # 处理参数
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4096))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
    
    # 数据源
    ARXIV_BASE_URL = os.getenv("ARXIV_BASE_URL")
    MAX_PAPERS_PER_FETCH = int(os.getenv("MAX_PAPERS_PER_FETCH", 50))
    
    # 增强功能开关
    ENABLE_QUERY_INTELLIGENCE = bool(os.getenv("ENABLE_QUERY_INTELLIGENCE", "true").lower() == "true")
    ENABLE_MULTI_REPRESENTATION = bool(os.getenv("ENABLE_MULTI_REPRESENTATION", "true").lower() == "true")
    ENABLE_AGENTIC_RAG = bool(os.getenv("ENABLE_AGENTIC_RAG", "true").lower() == "true")
    ENABLE_CONTEXTUAL_COMPRESSION = bool(os.getenv("ENABLE_CONTEXTUAL_COMPRESSION", "true").lower() == "true")
    
    # 智能体RAG参数
    MAX_AGENTIC_ITERATIONS = int(os.getenv("MAX_AGENTIC_ITERATIONS", 3))
    MIN_CHUNKS_THRESHOLD = int(os.getenv("MIN_CHUNKS_THRESHOLD", 2))
    
    # 压缩和重排序参数
    COMPRESSION_METHOD = os.getenv("COMPRESSION_METHOD", "hybrid")  # "sentence_extraction", "llm_compression", "hybrid"
    USE_SMART_RERANKING = bool(os.getenv("USE_SMART_RERANKING", "true").lower() == "true")
    
    # 反馈系统
    FEEDBACK_DB_PATH = STORAGE_ROOT / "feedback" / "feedback.db"
    ENABLE_FEEDBACK_COLLECTION = bool(os.getenv("ENABLE_FEEDBACK_COLLECTION", "true").lower() == "true")
    
    # 评估系统
    EVALUATION_OUTPUT_DIR = STORAGE_ROOT / "evaluation"
    GOLDEN_TEST_SET_PATH = EVALUATION_OUTPUT_DIR / "golden_test_set.json"
    
    # 知识图谱
    ENABLE_KNOWLEDGE_GRAPH = bool(os.getenv("ENABLE_KNOWLEDGE_GRAPH", "true").lower() == "true")
    KNOWLEDGE_GRAPH_DB_PATH = STORAGE_ROOT / "knowledge_graph" / "kg.db"
    
    # 分层生成系统
    ENABLE_TIERED_GENERATION = bool(os.getenv("ENABLE_TIERED_GENERATION", "true").lower() == "true")
    
    # API模型配置
    API_MODELS = {
        'gpt4_api_key': os.getenv("GPT4_API_KEY"),
        'gpt4_api_base': os.getenv("GPT4_API_BASE"),
        'gpt35_api_key': os.getenv("GPT35_API_KEY"),
        'gpt35_api_base': os.getenv("GPT35_API_BASE"),
        'claude_api_key': os.getenv("CLAUDE_API_KEY"),
        'claude_api_base': os.getenv("CLAUDE_API_BASE")
    }
    
    # 快速模型配置（用于简单任务）
    FAST_MODEL = os.getenv("FAST_MODEL", LLM_MODEL)
    
    # 系统模式
    DEFAULT_RAG_MODE = os.getenv("DEFAULT_RAG_MODE", "ultimate")  # basic, enhanced, agentic, ultimate

# 创建一个全局可用的配置实例
config = Config()

