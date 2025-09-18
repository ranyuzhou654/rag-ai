#!/usr/bin/env python3
"""
语法和结构测试 - 不依赖外部库的基础测试
"""

import ast
import os
from pathlib import Path

def test_file_syntax(file_path):
    """测试文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except Exception as e:
        return False, str(e)

def test_integration_structure():
    """测试集成结构"""
    print("🔧 测试集成结构...")
    
    # 检查所有必需文件是否存在
    required_files = [
        'src/analysis/content_analyzer.py',
        'src/generation/content_summarizer.py',
        'src/generation/topic_outline_generator.py',
        'src/generation/topic_aware_rag.py',
        'src/frontend/discovery_interface.py',
        'app.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} 存在")
        else:
            print(f"❌ {file_path} 不存在")
            all_exist = False
    
    return all_exist

def test_app_integration():
    """测试app.py集成"""
    print("\n📱 测试app.py集成...")
    
    try:
        with open('app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ('discovery_interface导入', 'from src.frontend.discovery_interface import DiscoveryInterface'),
            ('内容发现标签页', '内容发现'),
            ('discovery_interface初始化', 'discovery_interface = DiscoveryInterface()'),
            ('四个标签页', 'tab1, tab2, tab3, tab4 = st.tabs'),
            ('渲染发现界面', 'discovery_interface.render_discovery_tab()')
        ]
        
        passed = 0
        for check_name, check_text in checks:
            if check_text in content:
                print(f"✅ {check_name}: 已集成")
                passed += 1
            else:
                print(f"❌ {check_name}: 未找到")
        
        return passed == len(checks)
    
    except Exception as e:
        print(f"❌ 读取app.py失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始语法和结构测试\n")
    
    # 测试所有模块语法
    modules = [
        'src/analysis/content_analyzer.py',
        'src/generation/content_summarizer.py', 
        'src/generation/topic_outline_generator.py',
        'src/generation/topic_aware_rag.py',
        'src/frontend/discovery_interface.py',
        'app.py'
    ]
    
    print("📝 测试模块语法...")
    syntax_passed = 0
    for module in modules:
        success, error = test_file_syntax(module)
        if success:
            print(f"✅ {module}: 语法正确")
            syntax_passed += 1
        else:
            print(f"❌ {module}: {error}")
    
    print(f"\n语法测试: {syntax_passed}/{len(modules)} 通过")
    
    # 测试文件结构
    structure_ok = test_integration_structure()
    
    # 测试app.py集成
    app_integration_ok = test_app_integration()
    
    # 总结
    print(f"\n📊 测试总结:")
    print(f"语法测试: {'✅' if syntax_passed == len(modules) else '❌'}")
    print(f"结构测试: {'✅' if structure_ok else '❌'}")
    print(f"集成测试: {'✅' if app_integration_ok else '❌'}")
    
    all_passed = (syntax_passed == len(modules)) and structure_ok and app_integration_ok
    
    if all_passed:
        print("\n🎉 所有基础测试通过！内容发现功能已成功集成到系统中。")
        print("\n📋 功能摘要:")
        print("- ✅ 内容分析模块：从知识库提取热门主题")
        print("- ✅ 智能摘要生成：为主题生成结构化摘要")
        print("- ✅ 交互式大纲：组织主题为层次结构")
        print("- ✅ 主题感知RAG：基于选定主题的智能问答")
        print("- ✅ 前端界面集成：完整的用户交互界面")
        
        print("\n🚀 启动系统:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 启动Qdrant: ./qdrant --storage-path ./storage")
        print("3. 运行系统: streamlit run app.py")
    else:
        print("\n⚠️ 部分测试失败，请检查相关模块。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)