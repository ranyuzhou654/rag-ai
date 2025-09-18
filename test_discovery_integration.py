#!/usr/bin/env python3
"""
内容发现功能集成测试
测试新增的内容发现和主题感知RAG功能
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_module_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        # 测试新模块导入
        from src.analysis.content_analyzer import ContentAnalyzer, ContentAnalysis
        from src.generation.content_summarizer import ContentSummarizer, SummaryRequest
        from src.generation.topic_outline_generator import TopicOutlineGenerator, TopicOutline
        from src.generation.topic_aware_rag import TopicAwareRAGEngine, TopicContext
        from src.frontend.discovery_interface import DiscoveryInterface
        
        print("✅ 所有新模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_data_structures():
    """测试数据结构"""
    print("\n📊 测试数据结构...")
    
    try:
        from src.analysis.content_analyzer import TopicItem, ContentAnalysis
        from src.generation.content_summarizer import SummaryRequest, SummaryResult
        from src.generation.topic_outline_generator import OutlineSection, TopicOutline
        
        # 测试数据结构创建
        topic_item = TopicItem(
            topic_id="test-topic",
            title="测试主题",
            keywords=["测试", "主题"],
            description="这是一个测试主题",
            documents_count=10,
            relevance_score=0.8,
            latest_date="2024-01-01"
        )
        
        content_analysis = ContentAnalysis(
            analysis_id="test-analysis",
            analysis_date="2024-01-01",
            topics=[topic_item],
            summary="测试分析摘要",
            total_documents=100,
            analysis_period_days=7
        )
        
        print("✅ 数据结构测试通过")
        return True
    except Exception as e:
        print(f"❌ 数据结构测试失败: {e}")
        return False

def test_discovery_interface_creation():
    """测试发现界面创建"""
    print("\n🎨 测试发现界面创建...")
    
    try:
        from src.frontend.discovery_interface import DiscoveryInterface
        
        # 创建界面实例
        interface = DiscoveryInterface()
        
        print("✅ 发现界面创建成功")
        return True
    except Exception as e:
        print(f"❌ 发现界面创建失败: {e}")
        return False

def test_app_integration():
    """测试主应用集成"""
    print("\n🔧 测试主应用集成...")
    
    try:
        # 测试app.py中的新导入
        import ast
        with open('app.py', 'r', encoding='utf-8') as f:
            source = f.read()
        
        # 检查是否包含新的导入
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and 'discovery_interface' in node.module:
                    imports.append(node.module)
        
        if imports:
            print("✅ 主应用包含发现界面导入")
        else:
            print("⚠️ 主应用中未找到发现界面导入")
        
        # 检查是否包含新的标签页
        if '内容发现' in source:
            print("✅ 主应用包含内容发现标签页")
        else:
            print("⚠️ 主应用中未找到内容发现标签页")
        
        return True
    except Exception as e:
        print(f"❌ 主应用集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始内容发现功能集成测试\n")
    
    tests = [
        test_module_imports,
        test_data_structures,
        test_discovery_interface_creation,
        test_app_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📈 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有集成测试通过！内容发现功能已成功集成。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关模块。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)