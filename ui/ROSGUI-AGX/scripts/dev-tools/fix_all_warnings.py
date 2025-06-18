#!/usr/bin/env python3
"""
Comprehensive script to fix all documentation warnings.
"""

import os
import re
from pathlib import Path

def fix_all_title_underlines():
    """Fix all title underline issues in RST files."""
    
    # Define all the problematic titles and their correct underlines
    fixes = [
        # File: docs/development/architecture.rst
        ("Result<T> 模板类 - 已完全实现", "Result<T> 模板类 - 已完全实现\n--------------------------------"),
        
        # File: docs/development/building.rst  
        ("构建指南", "构建指南\n================"),
        
        # File: docs/development/environment-management.rst
        ("rosgui-docs 环境 📚", "rosgui-docs 环境 📚\n----------------------"),
        ("路径前缀配置", "路径前缀配置\n---------------"),
        ("📋 支持的配置变量", "📋 支持的配置变量\n===================="),
        ("核心路径变量", "核心路径变量\n---------------"),
        ("系统配置变量", "系统配置变量\n---------------"),
        ("命令路径解析", "命令路径解析\n---------------"),
        ("Conda环境使用", "Conda环境使用\n----------------"),
        ("🚀 实际应用效果", "🚀 实际应用效果\n=================="),
        ("命令路径转换", "命令路径转换\n---------------"),
        ("灵活部署支持", "灵活部署支持\n---------------"),
        ("环境变量查找流程", "环境变量查找流程\n=================="),
        ("配置文件示例", "配置文件示例\n==============="),
        ("完整的 environment.env 配置", "完整的 environment.env 配置\n------------------------------"),
        ("配置文件管理", "配置文件管理\n---------------"),
        ("路径配置原则", "路径配置原则\n---------------"),
        
        # File: docs/development/modern-cpp.rst
        ("Builder模式实现", "Builder模式实现\n------------------"),
        ("实际性能改进（已验证）", "实际性能改进（已验证）\n------------------------"),
        ("综合测试框架（已实现）", "综合测试框架（已实现）\n------------------------"),
        ("静态分析工具", "静态分析工具\n--------------"),
        
        # File: docs/development/parameter-system.rst
        ("参数对象结构", "参数对象结构\n---------------"),
        ("字符串输入 (StringWidget)", "字符串输入 (StringWidget)\n----------------------------"),
        ("整数输入 (IntegerWidget)", "整数输入 (IntegerWidget)\n---------------------------"),
        ("浮点数输入 (FloatWidget)", "浮点数输入 (FloatWidget)\n---------------------------"),
        ("布尔输入 (BooleanWidget)", "布尔输入 (BooleanWidget)\n---------------------------"),
        ("文件/目录选择 (FileWidget/DirectoryWidget)", "文件/目录选择 (FileWidget/DirectoryWidget)\n---------------------------------------------"),
        ("选择列表 (ChoiceWidget)", "选择列表 (ChoiceWidget)\n--------------------------"),
        ("🔧 参数验证系统", "🔧 参数验证系统\n=================="),
        ("自定义验证器", "自定义验证器\n---------------"),
    ]
    
    docs_dir = Path.cwd() / 'docs'
    
    for rst_file in docs_dir.rglob('*.rst'):
        try:
            content = rst_file.read_text(encoding='utf-8')
            original_content = content
            
            # Apply all fixes
            for title, replacement in fixes:
                if title in content:
                    # Find the title and fix its underline
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip() == title:
                            # Check if next line is an underline
                            if i + 1 < len(lines):
                                next_line = lines[i + 1]
                                if next_line.strip() and all(c in '=-~^"\'`#*+<>' for c in next_line.strip()):
                                    # Replace the title and underline
                                    lines[i] = title
                                    lines[i + 1] = replacement.split('\n')[1]
                                    content = '\n'.join(lines)
                                    print(f"Fixed '{title}' in {rst_file}")
                                    break
            
            if content != original_content:
                rst_file.write_text(content, encoding='utf-8')
                
        except Exception as e:
            print(f"Error processing {rst_file}: {e}")

def fix_toctree_issues():
    """Fix toctree issues."""
    docs_dir = Path.cwd() / 'docs'
    
    # Add missing documents to main index.rst
    index_file = docs_dir / 'index.rst'
    if index_file.exists():
        content = index_file.read_text(encoding='utf-8')
        
        # Add the missing markdown files to toctree if not already present
        markdown_files = ['DOCUMENTATION_COMPLETE', 'MODERNIZATION_COMPLETE', 'MODERN_CPP_GUIDE']
        
        for md_file in markdown_files:
            if md_file not in content:
                # Find the main toctree and add the file
                toctree_pattern = r'(\.\. toctree::\s*\n(?:\s+:[^:]+:[^\n]*\n)*)((?:\s+[^\n]+\n)*)'
                match = re.search(toctree_pattern, content)
                if match:
                    toctree_header = match.group(1)
                    toctree_content = match.group(2)
                    toctree_content += f"   {md_file}\n"
                    new_content = content.replace(match.group(0), toctree_header + toctree_content)
                    content = new_content
                    print(f"Added {md_file} to main toctree")
        
        index_file.write_text(content, encoding='utf-8')
    
    # Remove duplicate references from api/index.rst
    api_index = docs_dir / 'api' / 'index.rst'
    if api_index.exists():
        content = api_index.read_text(encoding='utf-8')
        
        # Remove duplicate toctree entries
        duplicates = [
            'application-lifecycle',
            'commands', 
            'file-descriptor',
            'process-manager',
            'result-system',
            'ui-manager',
            'utilities'
        ]
        
        lines = content.split('\n')
        filtered_lines = []
        
        for line in lines:
            # Skip lines that reference duplicate files in toctree
            if any(dup in line.strip() for dup in duplicates) and line.strip().startswith('   ') and not line.strip().startswith('   :'):
                print(f"Removed duplicate toctree reference: {line.strip()}")
                continue
            filtered_lines.append(line)
        
        api_index.write_text('\n'.join(filtered_lines), encoding='utf-8')

def main():
    """Main function."""
    print("🔧 Fixing all documentation warnings...")
    
    # Change to docs directory if we're in project root
    if Path.cwd().name != 'docs' and (Path.cwd() / 'docs').exists():
        os.chdir(Path.cwd() / 'docs')
    
    print("📝 Fixing title underlines...")
    fix_all_title_underlines()
    
    print("📚 Fixing toctree issues...")
    fix_toctree_issues()
    
    print("✅ All documentation warnings should be fixed!")
    print("Run 'sphinx-build -b html . _build/html' to verify.")

if __name__ == '__main__':
    main()
