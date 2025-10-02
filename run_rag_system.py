# run_rag_system.py - 项目一键运行脚本
import asyncio
import argparse
import sys
from pathlib import Path
import json
import time
import subprocess
from datetime import datetime

# 在所有其他导入之前加载配置，以设置环境变量
try:
    from configs.config import config
except ImportError:
    print("无法导入配置，请确保configs/config.py存在")
    sys.exit(1)
    
from loguru import logger
from huggingface_hub import login

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 设置日志
log_file = config.LOG_DIR / f"rag_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file.parent.mkdir(exist_ok=True)
logger.add(log_file, rotation="1 day", retention="30 days")

def hf_login():
    """使用配置中的令牌登录Hugging Face"""
    token = config.HUGGING_FACE_TOKEN
    if token:
        try:
            login(token=token, add_to_git_credential=False)
            logger.success("✅ Hugging Face 登录成功")
        except Exception as e:
            logger.warning(f"🤔 HF登录失败: {e}")
    else:
        logger.info("💡 未配置HF令牌，部分模型可能受限。")

class RAGSystemRunner:
    """RAG系统运行器 - 管理整个项目的生命周期"""
    
    def __init__(self):
        self.project_root = project_root
        self.data_dir = config.DATA_DIR
        self.config_file = self.project_root / "configs" / "config.py"
        
        # 检查项目结构
        self._check_project_structure()
    
    def _check_project_structure(self):
        """检查项目目录结构"""
        required_dirs = [
            "data/raw",
            "data/processed", 
            "src/data_ingestion",
            "src/processing",
            "src/retrieval",
            "src/generation",
            "logs",
            "configs"
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                logger.warning(f"创建缺失目录: {dir_path}")
                full_path.mkdir(parents=True, exist_ok=True)
    
    def check_environment(self) -> bool:
        """检查运行环境"""
        logger.info("🔍 检查运行环境...")
        
        success = True
        
        # 1. 检查Python版本
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            logger.error(f"❌ Python版本过低: {python_version.major}.{python_version.minor}")
            success = False
        else:
            logger.info(f"✅ Python版本: {python_version.major}.{python_version.minor}")
        
        # 2. 检查关键依赖
        required_packages = [
            'torch',
            'transformers', 
            'sentence_transformers',
            'qdrant_client',
            'streamlit',
            'pymupdf4llm'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} 已安装")
            except ImportError:
                logger.error(f"❌ {package} 未安装")
                success = False
        
        # 3. 检查Qdrant服务
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host="localhost", port=6333)
            collections = client.get_collections()
            logger.info("✅ Qdrant服务连接正常")
        except Exception as e:
            logger.warning(f"⚠️ Qdrant服务连接失败: {e}")
            logger.info("提示: 请确保Docker中的Qdrant服务正在运行")
            success = False
        
        return success

    async def collect_data(self, max_papers: int = 50, days_back: int = 7) -> bool:
        """数据收集步骤"""
        logger.info("📥 开始数据收集...")
        
        try:
            from src.data_ingestion.multi_source_collector import MultiSourceCollector
            
            # 初始化收集器
            collector = MultiSourceCollector(self.data_dir / "raw")
            
            # 收集数据
            all_data = await collector.collect_all(days_back=days_back)
            
            if not all_data:
                logger.error("❌ 没有收集到任何数据")
                return False
            
            logger.success(f"✅ 数据收集完成: {len(all_data)} 条数据已由收集器保存。")
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据收集失败: {e}")
            return False
    
    async def process_data(self) -> bool:
        """数据处理步骤"""
        logger.info("⚙️ 开始数据处理...")
        
        try:
            from src.processing.text_processor import TextProcessor
            
            # 检查输入数据
            input_file = self.data_dir / "raw" / "raw_collected_data.json"
            if not input_file.exists():
                logger.error(f"❌ 找不到输入数据文件: {input_file}")
                return False
            
            # 加载数据
            with open(input_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            logger.info(f"📄 加载到 {len(documents)} 个文档")
            
            # 初始化处理器
            processor_config = {
                'chunk_size': config.CHUNK_SIZE,
                'chunk_overlap': config.CHUNK_OVERLAP,
                'embedding_model': config.EMBEDDING_MODEL,
                'device': config.DEVICE,
                'enable_multi_representation': config.ENABLE_MULTI_REPRESENTATION,
                'llm_model': config.LLM_MODEL,
                'HUGGING_FACE_TOKEN': config.HUGGING_FACE_TOKEN
            }

            processor = TextProcessor(processor_config)

            # 处理文档
            chunks = await processor.process_documents(documents)
            
            if not chunks:
                logger.error("❌ 文档处理失败，没有生成有效的文本块")
                return False
            
            # 保存结果
            output_file = self.data_dir / "processed" / "processed_chunks.json"
            processor.save_processed_data(chunks, output_file)
            
            logger.info(f"✅ 数据处理完成: {len(chunks)} 个文本块已保存至 {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据处理失败: {e}")
            return False
    
    def build_knowledge_base(self) -> bool:
        """构建向量知识库"""
        logger.info("🗄️ 开始构建向量知识库...")
        
        try:
            from src.retrieval.vector_database import VectorDatabaseManager
            
            # 检查处理后的数据
            processed_file = self.data_dir / "processed" / "processed_chunks.json"
            if not processed_file.exists():
                logger.error(f"❌ 找不到处理后的数据文件: {processed_file}")
                return False
            
            with open(processed_file, 'r', encoding='utf-8') as f:
                processed_chunks = json.load(f)

            sample_vector = next((chunk.get('embedding') for chunk in processed_chunks if chunk.get('embedding')), None)
            vector_size = len(sample_vector) if sample_vector else 0
            if not vector_size:
                vector_size = 1024  # 合理的默认值

            db_config = {
                'qdrant_host': config.QDRANT_HOST,
                'qdrant_port': config.QDRANT_PORT,
                'collection_name': config.COLLECTION_NAME,
                'vector_size': vector_size,
                'qdrant_timeout': getattr(config, 'QDRANT_TIMEOUT', 120)
            }

            db_manager = VectorDatabaseManager(db_config)

            # 构建知识库
            success = db_manager.build_knowledge_base(processed_file, chunks=processed_chunks)
            
            if success:
                # 显示统计信息
                stats = db_manager.db.get_collection_stats()
                logger.info(f"✅ 知识库构建成功!")
                logger.info(f"   总向量数: {stats.get('total_points', 0)}")
                logger.info(f"   向量维度: {stats.get('vector_size', 0)}")
                return True
            else:
                logger.error("❌ 知识库构建失败")
                return False
                
        except Exception as e:
            logger.error(f"❌ 知识库构建失败: {e}")
            return False
    
    def test_system(self) -> bool:
        """测试系统功能"""
        logger.info("🧪 开始系统测试...")
        
        try:
            from src.generation.rag_generator import RAGSystem
            from src.retrieval.vector_database import VectorDatabaseManager
            
            rag_settings = {
                'embedding_model': config.EMBEDDING_MODEL,
                'llm_model': config.LLM_MODEL,
                'device': config.DEVICE,
                'qdrant_host': config.QDRANT_HOST,
                'qdrant_port': config.QDRANT_PORT,
                'collection_name': config.COLLECTION_NAME,
                'HUGGING_FACE_TOKEN': config.HUGGING_FACE_TOKEN
            }

            rag_system = RAGSystem(rag_settings)
            db_manager = VectorDatabaseManager(rag_settings)
            rag_system.set_database(db_manager)
            
            # 测试查询
            test_queries = [
                "什么是Transformer架构？",
                "How do attention mechanisms work?"
            ]
            
            success_count = 0
            
            for query in test_queries:
                try:
                    logger.info(f"测试查询: {query}")

                    result = asyncio.run(
                        rag_system.generate_answer(query, top_k=3)
                    )

                    if result and result.answer:
                        logger.info(f"✅ 查询成功 - 置信度: {result.confidence:.2f}")
                        success_count += 1
                    else:
                        logger.warning(f"⚠️ 查询返回空结果")
                        
                except Exception as e:
                    logger.error(f"❌ 查询失败: {e}")
            
            success_rate = success_count / len(test_queries)
            if success_rate >= 0.5:
                logger.info(f"✅ 系统测试通过 - 成功率: {success_rate:.1%}")
                return True
            else:
                logger.error(f"❌ 系统测试失败 - 成功率: {success_rate:.1%}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 系统测试失败: {e}")
            return False
    
    def launch_frontend(self, port: int = 8501):
        """启动Streamlit前端"""
        logger.info(f"🚀 启动前端界面 (端口: {port})...")
        
        try:
            candidate_files = [
                self.project_root / "app.py",
                self.project_root / "enhanced_app.py",
                self.project_root / "frontend" / "app.py",
                self.project_root / "frontend" / "streamlit_app.py"
            ]

            app_file = None
            for candidate in candidate_files:
                if candidate.exists():
                    app_file = candidate
                    break

            if not app_file:
                logger.error("❌ 找不到前端应用文件，请确认 app.py 或 enhanced_app.py 是否存在")
                return False

            # 启动Streamlit
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                str(app_file),
                "--server.port", str(port),
                "--server.address", "localhost"
            ]
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            # 在新进程中启动
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待一段时间检查是否启动成功
            time.sleep(3)
            
            if process.poll() is None:  # 进程仍在运行
                logger.info(f"✅ 前端启动成功!")
                logger.info(f"🌐 访问地址: http://localhost:{port}")
                
                try:
                    process.wait()  # 等待进程结束
                except KeyboardInterrupt:
                    logger.info("收到中断信号，正在关闭...")
                    process.terminate()
                    process.wait()
                
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"❌ 前端启动失败")
                logger.error(f"stdout: {stdout}")
                logger.error(f"stderr: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 前端启动失败: {e}")
            return False
    
    async def run_full_pipeline(self, args):
        """运行完整的RAG系统流程"""
        logger.info("🚀 开始运行完整的RAG系统流程...")
        
        start_time = time.time()
        
        # 1. 环境检查
        if not args.skip_check and not self.check_environment():
            logger.error("❌ 环境检查失败，请修复问题后重试")
            return False
        
        # 2. 数据收集（可选）
        if not args.skip_collect:
            if not await self.collect_data(args.max_papers, args.days_back):
                logger.error("❌ 数据收集失败")
                return False
        
        # 3. 数据处理
        if not args.skip_process:
            if not await self.process_data():
                logger.error("❌ 数据处理失败") 
                return False
        
        # 4. 构建知识库
        if not args.skip_build:
            if not self.build_knowledge_base():
                logger.error("❌ 知识库构建失败")
                return False
        
        # 5. 系统测试（可选）
        if args.test:
            if not self.test_system():
                logger.warning("⚠️ 系统测试失败，但仍可继续启动前端")
        
        total_time = time.time() - start_time
        logger.info(f"✅ RAG系统准备完成! 总耗时: {total_time:.1f}秒")
        
        # 6. 启动前端
        if not args.no_frontend:
            self.launch_frontend(args.port)
        
        return True

def main():
    """主函数"""

    hf_login()
    
    parser = argparse.ArgumentParser(
        description="RAG AI技术资讯分析师系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_rag_system.py                    # 运行完整流程
  python run_rag_system.py --quick           # 快速启动(跳过数据收集)
  python run_rag_system.py --frontend-only   # 仅启动前端
  python run_rag_system.py --test            # 运行系统测试
        """
    )
    
    # 基本参数
    parser.add_argument('--port', type=int, default=8501, help='前端端口号')
    parser.add_argument('--max-papers', type=int, default=50, help='最大论文收集数量')
    parser.add_argument('--days-back', type=int, default=7, help='收集最近几天的数据')
    
    # 流程控制
    parser.add_argument('--skip-check', action='store_true', help='跳过环境检查')
    parser.add_argument('--skip-collect', action='store_true', help='跳过数据收集')
    parser.add_argument('--skip-process', action='store_true', help='跳过数据处理')
    parser.add_argument('--skip-build', action='store_true', help='跳过知识库构建')
    parser.add_argument('--no-frontend', action='store_true', help='不启动前端')
    parser.add_argument('--test', action='store_true', help='运行系统测试')
    
    # 便捷选项
    parser.add_argument('--quick', action='store_true', 
                       help='快速启动(跳过数据收集，直接使用现有数据)')
    parser.add_argument('--frontend-only', action='store_true',
                       help='仅启动前端界面')
    
    args = parser.parse_args()
    
    # 处理便捷选项
    if args.quick:
        args.skip_collect = True
        
    if args.frontend_only:
        args.skip_check = True
        args.skip_collect = True
        args.skip_process = True
        args.skip_build = True
    
    # 初始化运行器
    runner = RAGSystemRunner()
    
    print("""
🤖 ===================================================
   RAG AI技术资讯分析师系统
   Real-time AI Tech Info Analyst with RAG
🤖 ===================================================
    """)
    
    try:
        if args.frontend_only:
            # 仅启动前端
            runner.launch_frontend(args.port)
        else:
            # 运行完整流程
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(runner.run_full_pipeline(args))
            loop.close()
            
            if not success:
                print("❌ 系统启动失败，请检查日志文件获取详细信息")
                sys.exit(1)
            
    except KeyboardInterrupt:
        print("\\n👋 用户中断，系统退出")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"系统运行异常: {e}")
        print(f"❌ 系统异常: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
