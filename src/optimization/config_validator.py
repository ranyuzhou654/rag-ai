# src/optimization/config_validator.py
import os
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
import socket
import requests
from urllib.parse import urlparse

@dataclass
class ValidationIssue:
    """验证问题"""
    level: str  # error, warning, info
    category: str  # config, dependency, network, security
    message: str
    fix_suggestion: Optional[str] = None
    auto_fixable: bool = False

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)
    
    def add_issue(self, level: str, category: str, message: str, 
                  fix_suggestion: Optional[str] = None, auto_fixable: bool = False):
        """添加验证问题"""
        issue = ValidationIssue(level, category, message, fix_suggestion, auto_fixable)
        
        if level == "error":
            self.issues.append(issue)
            self.is_valid = False
        elif level == "warning":
            self.warnings.append(issue)
        else:
            self.info.append(issue)
    
    def get_summary(self) -> Dict[str, int]:
        """获取验证摘要"""
        return {
            "errors": len(self.issues),
            "warnings": len(self.warnings),
            "info": len(self.info),
            "total_issues": len(self.issues) + len(self.warnings) + len(self.info)
        }

class ConfigValidator:
    """配置验证器"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.required_files = [
            "requirements.txt",
            "configs/config.py",
            ".env.example"
        ]
        
        self.required_env_vars = [
            "STORAGE_ROOT",
            "EMBEDDING_MODEL", 
            "LLM_MODEL",
            "QDRANT_HOST",
            "QDRANT_PORT",
            "COLLECTION_NAME"
        ]
        
        self.recommended_env_vars = [
            "HUGGING_FACE_TOKEN",
            "DEVICE",
            "CHUNK_SIZE",
            "CHUNK_OVERLAP"
        ]
        
        self.security_sensitive_vars = [
            "HUGGING_FACE_TOKEN",
            "GPT4_API_KEY",
            "CLAUDE_API_KEY",
            "DATABASE_PASSWORD",
            "SECRET_KEY"
        ]
    
    def validate_project_structure(self) -> ValidationResult:
        """验证项目结构"""
        result = ValidationResult(is_valid=True)
        
        # 检查必需文件
        for file_path in self.required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                result.add_issue(
                    "error", "config",
                    f"Missing required file: {file_path}",
                    f"Create the file at {full_path}",
                    auto_fixable=True
                )
        
        # 检查关键目录
        required_dirs = [
            "src",
            "configs", 
            "api",
            "frontend",
            "data",
            "logs"
        ]
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                result.add_issue(
                    "warning", "config",
                    f"Missing directory: {dir_name}",
                    f"Create directory: mkdir {dir_path}",
                    auto_fixable=True
                )
        
        return result
    
    def validate_environment_config(self) -> ValidationResult:
        """验证环境配置"""
        result = ValidationResult(is_valid=True)
        
        # 检查 .env 文件
        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"
        
        if not env_file.exists():
            if env_example.exists():
                result.add_issue(
                    "warning", "config",
                    ".env file not found",
                    "Copy .env.example to .env and configure it",
                    auto_fixable=True
                )
            else:
                result.add_issue(
                    "error", "config",
                    "Neither .env nor .env.example found",
                    "Create .env file with required variables"
                )
                return result
        
        # 加载环境变量
        env_vars = self._load_env_file(env_file)
        
        # 检查必需的环境变量
        for var in self.required_env_vars:
            if var not in env_vars:
                result.add_issue(
                    "error", "config",
                    f"Required environment variable missing: {var}",
                    f"Add {var}=<value> to .env file"
                )
            elif not env_vars[var]:
                result.add_issue(
                    "error", "config", 
                    f"Environment variable is empty: {var}",
                    f"Set a value for {var} in .env file"
                )
        
        # 检查推荐的环境变量
        for var in self.recommended_env_vars:
            if var not in env_vars:
                result.add_issue(
                    "info", "config",
                    f"Recommended environment variable missing: {var}",
                    f"Consider adding {var}=<value> to .env file"
                )
        
        # 验证具体的配置值
        self._validate_config_values(env_vars, result)
        
        # 安全检查
        self._validate_security_config(env_vars, result)
        
        return result
    
    def _load_env_file(self, env_file: Path) -> Dict[str, str]:
        """加载环境变量文件"""
        env_vars = {}
        
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip().strip('"\'')
        except Exception as e:
            logger.error(f"Error reading .env file: {e}")
        
        return env_vars
    
    def _validate_config_values(self, env_vars: Dict[str, str], result: ValidationResult):
        """验证配置值"""
        
        # 验证路径
        if "STORAGE_ROOT" in env_vars:
            storage_path = Path(env_vars["STORAGE_ROOT"])
            if not storage_path.parent.exists():
                result.add_issue(
                    "warning", "config",
                    f"Storage root parent directory doesn't exist: {storage_path.parent}",
                    f"Create directory: mkdir -p {storage_path.parent}"
                )
        
        # 验证模型名称
        if "EMBEDDING_MODEL" in env_vars:
            model_name = env_vars["EMBEDDING_MODEL"]
            if not self._is_valid_model_name(model_name):
                result.add_issue(
                    "warning", "config",
                    f"Invalid embedding model name format: {model_name}",
                    "Use format like 'BAAI/bge-m3' or 'sentence-transformers/all-MiniLM-L6-v2'"
                )
        
        # 验证端口
        if "QDRANT_PORT" in env_vars:
            try:
                port = int(env_vars["QDRANT_PORT"])
                if not (1024 <= port <= 65535):
                    result.add_issue(
                        "warning", "config",
                        f"Qdrant port {port} outside recommended range",
                        "Use port between 1024-65535"
                    )
            except ValueError:
                result.add_issue(
                    "error", "config",
                    f"Invalid Qdrant port: {env_vars['QDRANT_PORT']}",
                    "Port must be a number"
                )
        
        # 验证设备设置
        if "DEVICE" in env_vars:
            device = env_vars["DEVICE"].lower()
            if device not in ["auto", "cpu", "cuda", "mps"]:
                result.add_issue(
                    "warning", "config",
                    f"Unknown device setting: {device}",
                    "Use 'auto', 'cpu', 'cuda', or 'mps'"
                )
    
    def _is_valid_model_name(self, model_name: str) -> bool:
        """验证模型名称格式"""
        # 检查是否符合 HuggingFace 模型名称格式
        pattern = r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$|^[a-zA-Z0-9_.-]+$'
        return re.match(pattern, model_name) is not None
    
    def _validate_security_config(self, env_vars: Dict[str, str], result: ValidationResult):
        """验证安全配置"""
        
        # 检查敏感信息是否被意外提交
        for var in self.security_sensitive_vars:
            if var in env_vars and env_vars[var]:
                # 检查是否是默认或示例值
                value = env_vars[var].lower()
                if value in ["your_key_here", "example", "changeme", "default", "test"]:
                    result.add_issue(
                        "warning", "security",
                        f"Security variable {var} contains default/example value",
                        f"Set a real value for {var}"
                    )
                
                # 检查值长度（API keys通常较长）
                if len(env_vars[var]) < 10:
                    result.add_issue(
                        "warning", "security", 
                        f"Security variable {var} value seems too short",
                        "Verify the key is complete and correct"
                    )
        
        # 检查是否有硬编码的密钥
        sensitive_patterns = [
            r'api[_-]?key\s*=\s*["\']?[a-zA-Z0-9]{20,}',
            r'secret[_-]?key\s*=\s*["\']?[a-zA-Z0-9]{20,}',
            r'password\s*=\s*["\']?[a-zA-Z0-9]{8,}'
        ]
        
        # 检查配置文件
        config_files = [
            self.project_root / "configs" / "config.py",
            self.project_root / "app.py",
            self.project_root / "api" / "main.py"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for pattern in sensitive_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            result.add_issue(
                                "warning", "security",
                                f"Potential hardcoded sensitive data in {config_file.name}",
                                "Move sensitive data to environment variables"
                            )
                except Exception as e:
                    logger.warning(f"Could not read {config_file}: {e}")
    
    def validate_dependencies(self) -> ValidationResult:
        """验证依赖项"""
        result = ValidationResult(is_valid=True)
        
        # 检查 requirements.txt
        req_file = self.project_root / "requirements.txt"
        if not req_file.exists():
            result.add_issue(
                "error", "dependency",
                "requirements.txt not found",
                "Create requirements.txt with project dependencies"
            )
            return result
        
        # 读取依赖项
        try:
            with open(req_file, 'r', encoding='utf-8') as f:
                requirements = f.read().splitlines()
            
            # 检查关键依赖
            critical_deps = [
                "torch", "transformers", "sentence-transformers",
                "qdrant-client", "streamlit", "fastapi", "loguru"
            ]
            
            found_deps = set()
            for req in requirements:
                if req.strip() and not req.strip().startswith('#'):
                    dep_name = req.split('>=')[0].split('==')[0].split('[')[0].strip()
                    found_deps.add(dep_name.lower())
            
            for dep in critical_deps:
                if dep.lower() not in found_deps:
                    result.add_issue(
                        "warning", "dependency",
                        f"Critical dependency missing: {dep}",
                        f"Add '{dep}' to requirements.txt"
                    )
            
            # 检查版本冲突的常见情况
            self._check_version_conflicts(requirements, result)
            
        except Exception as e:
            result.add_issue(
                "error", "dependency",
                f"Error reading requirements.txt: {e}",
                "Fix requirements.txt format"
            )
        
        return result
    
    def _check_version_conflicts(self, requirements: List[str], result: ValidationResult):
        """检查版本冲突"""
        
        # 常见的版本冲突检查
        version_checks = {
            "transformers": {"min": "4.35.0", "reason": "Required for latest model support"},
            "torch": {"min": "2.0.0", "reason": "Performance improvements"},
            "streamlit": {"min": "1.28.0", "reason": "Latest UI features"}
        }
        
        for req in requirements:
            if req.strip() and not req.strip().startswith('#') and '>=' in req:
                parts = req.split('>=')
                if len(parts) == 2:
                    dep_name = parts[0].strip().lower()
                    version = parts[1].strip()
                    
                    if dep_name in version_checks:
                        expected = version_checks[dep_name]
                        if self._compare_versions(version, expected["min"]) < 0:
                            result.add_issue(
                                "warning", "dependency",
                                f"{dep_name} version {version} is below recommended {expected['min']}",
                                f"Update to {dep_name}>={expected['min']} - {expected['reason']}"
                            )
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """比较版本号"""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # 补齐长度
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            for i in range(max_len):
                if v1_parts[i] < v2_parts[i]:
                    return -1
                elif v1_parts[i] > v2_parts[i]:
                    return 1
            
            return 0
        except:
            return 0
    
    def validate_network_config(self) -> ValidationResult:
        """验证网络配置"""
        result = ValidationResult(is_valid=True)
        
        # 检查 Qdrant 连接
        env_file = self.project_root / ".env"
        if env_file.exists():
            env_vars = self._load_env_file(env_file)
            
            host = env_vars.get("QDRANT_HOST", "localhost")
            port = env_vars.get("QDRANT_PORT", "6333")
            
            try:
                port_int = int(port)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                connection_result = sock.connect_ex((host, port_int))
                sock.close()
                
                if connection_result != 0:
                    result.add_issue(
                        "warning", "network",
                        f"Cannot connect to Qdrant at {host}:{port}",
                        "Start Qdrant service or check connection settings"
                    )
                else:
                    result.add_issue(
                        "info", "network",
                        f"Qdrant connection successful at {host}:{port}",
                        None
                    )
            except Exception as e:
                result.add_issue(
                    "warning", "network",
                    f"Network check failed for Qdrant: {e}",
                    "Verify Qdrant service is running"
                )
        
        # 检查HuggingFace连接
        try:
            response = requests.get("https://huggingface.co", timeout=10)
            if response.status_code == 200:
                result.add_issue(
                    "info", "network",
                    "HuggingFace Hub connection successful",
                    None
                )
            else:
                result.add_issue(
                    "warning", "network",
                    "HuggingFace Hub connection issue",
                    "Check internet connection"
                )
        except:
            result.add_issue(
                "warning", "network",
                "Cannot connect to HuggingFace Hub",
                "Check internet connection or proxy settings"
            )
        
        return result
    
    def validate_all(self) -> ValidationResult:
        """执行所有验证"""
        logger.info("Starting comprehensive project validation...")
        
        all_results = [
            self.validate_project_structure(),
            self.validate_environment_config(),
            self.validate_dependencies(),
            self.validate_network_config()
        ]
        
        # 合并所有结果
        combined_result = ValidationResult(is_valid=True)
        
        for result in all_results:
            if not result.is_valid:
                combined_result.is_valid = False
            
            combined_result.issues.extend(result.issues)
            combined_result.warnings.extend(result.warnings)
            combined_result.info.extend(result.info)
        
        logger.info(f"Validation completed: {combined_result.get_summary()}")
        return combined_result
    
    def auto_fix_issues(self, result: ValidationResult) -> int:
        """自动修复可修复的问题"""
        fixed_count = 0
        
        for issue in result.issues + result.warnings:
            if issue.auto_fixable:
                try:
                    if self._attempt_auto_fix(issue):
                        fixed_count += 1
                        logger.info(f"Auto-fixed: {issue.message}")
                except Exception as e:
                    logger.error(f"Auto-fix failed for '{issue.message}': {e}")
        
        return fixed_count
    
    def _attempt_auto_fix(self, issue: ValidationIssue) -> bool:
        """尝试自动修复单个问题"""
        
        if "Missing required file" in issue.message and "requirements.txt" in issue.message:
            # 创建基本的 requirements.txt
            req_content = """# Core ML and NLP
torch>=2.0.0
transformers>=4.35.0
sentence-transformers>=2.3.0
huggingface_hub
numpy>=1.24.0

# Vector Database
qdrant-client>=1.7.0

# Web Framework
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn>=0.24.0

# Utilities
loguru>=0.7.0
python-dotenv>=1.0.0
pandas>=2.0.0
"""
            req_file = self.project_root / "requirements.txt"
            with open(req_file, 'w', encoding='utf-8') as f:
                f.write(req_content)
            return True
        
        elif "Missing directory" in issue.message:
            # 创建缺失的目录
            dir_name = issue.message.split(": ")[-1]
            dir_path = self.project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            return True
        
        elif ".env file not found" in issue.message:
            # 从 .env.example 复制到 .env
            env_example = self.project_root / ".env.example"
            env_file = self.project_root / ".env"
            if env_example.exists():
                import shutil
                shutil.copy2(env_example, env_file)
                return True
        
        return False
    
    def generate_report(self, result: ValidationResult) -> str:
        """生成验证报告"""
        report = ["# 项目配置验证报告", ""]
        
        summary = result.get_summary()
        report.append(f"## 总体状态: {'✅ 通过' if result.is_valid else '❌ 需要修复'}")
        report.append("")
        report.append(f"- 错误: {summary['errors']}")
        report.append(f"- 警告: {summary['warnings']}")
        report.append(f"- 信息: {summary['info']}")
        report.append("")
        
        if result.issues:
            report.append("## ❌ 错误项")
            for issue in result.issues:
                report.append(f"- **{issue.category}**: {issue.message}")
                if issue.fix_suggestion:
                    report.append(f"  - 建议: {issue.fix_suggestion}")
            report.append("")
        
        if result.warnings:
            report.append("## ⚠️ 警告项")
            for warning in result.warnings:
                report.append(f"- **{warning.category}**: {warning.message}")
                if warning.fix_suggestion:
                    report.append(f"  - 建议: {warning.fix_suggestion}")
            report.append("")
        
        if result.info:
            report.append("## ℹ️ 信息项")
            for info in result.info:
                report.append(f"- **{info.category}**: {info.message}")
            report.append("")
        
        return "\n".join(report)

# 使用示例
def main():
    """测试配置验证器"""
    project_root = Path(".")
    validator = ConfigValidator(project_root)
    
    # 执行验证
    result = validator.validate_all()
    
    # 尝试自动修复
    fixed_count = validator.auto_fix_issues(result)
    if fixed_count > 0:
        print(f"自动修复了 {fixed_count} 个问题")
        # 重新验证
        result = validator.validate_all()
    
    # 生成报告
    report = validator.generate_report(result)
    print(report)
    
    # 保存报告
    report_file = project_root / "validation_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n报告已保存到: {report_file}")

if __name__ == "__main__":
    main()