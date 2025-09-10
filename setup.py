# setup.py - Smart environment setup script for cloud environments like autodl
import subprocess
import sys
from pathlib import Path
import os

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
    
    # 创建 .env 文件
    env_content = f"""
# RAG项目环境变量配置文件
STORAGE_ROOT={project_storage_path}
HF_HOME={model_cache_path}
TRANSFORMERS_CACHE={model_cache_path}
SENTENCE_TRANSFORMERS_HOME={model_cache_path}
HUGGING_FACE_TOKEN={hf_token}
QDRANT_STORAGE_PATH={project_storage_path / "qdrant_storage"}
"""
    
    with open(".env", "w") as f:
        f.write(env_content.strip())
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

