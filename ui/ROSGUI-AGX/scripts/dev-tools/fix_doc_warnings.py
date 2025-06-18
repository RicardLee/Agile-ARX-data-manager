#!/usr/bin/env python3
"""
Script to fix documentation warnings by correcting title underlines
for Chinese characters and other formatting issues.
"""

import os
import re
import sys
from pathlib import Path

def fix_title_underlines(content):
    """Fix title underlines that are too short for Chinese characters."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a title line followed by underline
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            
            # Check for underline patterns (=, -, ~, ^, ", ', `, #, *, +, <, >)
            underline_chars = ['=', '-', '~', '^', '"', "'", '`', '#', '*', '+', '<', '>']
            
            for char in underline_chars:
                if next_line.strip() and all(c == char for c in next_line.strip()):
                    # This is an underline, check if it's long enough
                    title_length = len(line.strip())
                    underline_length = len(next_line.strip())
                    
                    if underline_length < title_length:
                        # Fix the underline length
                        new_underline = char * title_length
                        lines[i + 1] = new_underline
                        print(f"Fixed underline for: '{line.strip()}'")
                    break
        
        # Check for overline + title + underline pattern
        if i + 2 < len(lines) and i > 0:
            prev_line = lines[i - 1]
            next_line = lines[i + 1]
            
            for char in underline_chars:
                if (prev_line.strip() and all(c == char for c in prev_line.strip()) and
                    next_line.strip() and all(c == char for c in next_line.strip())):
                    # This is an overline + underline pattern
                    title_length = len(line.strip())
                    overline_length = len(prev_line.strip())
                    underline_length = len(next_line.strip())
                    
                    if overline_length < title_length:
                        lines[i - 1] = char * title_length
                        print(f"Fixed overline for: '{line.strip()}'")
                    if underline_length < title_length:
                        lines[i + 1] = char * title_length
                        print(f"Fixed underline for: '{line.strip()}'")
                    break
        
        i += 1
    
    return '\n'.join(lines)

def fix_toctree_issues(docs_dir):
    """Fix toctree issues by adding missing documents to index.rst."""
    index_file = docs_dir / 'index.rst'
    
    if not index_file.exists():
        print(f"Warning: {index_file} not found")
        return
    
    content = index_file.read_text(encoding='utf-8')
    
    # Check if the markdown files are included
    markdown_files = [
        'DOCUMENTATION_COMPLETE',
        'MODERNIZATION_COMPLETE', 
        'MODERN_CPP_GUIDE'
    ]
    
    # Find the toctree section
    toctree_pattern = r'(\.\. toctree::\s*\n(?:\s+:[^:]+:[^\n]*\n)*)((?:\s+[^\n]+\n)*)'
    match = re.search(toctree_pattern, content)
    
    if match:
        toctree_header = match.group(1)
        toctree_content = match.group(2)
        
        # Add missing files to toctree
        for md_file in markdown_files:
            if md_file not in toctree_content:
                toctree_content += f"   {md_file}\n"
                print(f"Added {md_file} to main toctree")
        
        # Replace the toctree section
        new_content = content.replace(match.group(0), toctree_header + toctree_content)
        index_file.write_text(new_content, encoding='utf-8')

def fix_duplicate_toctree_references(docs_dir):
    """Fix duplicate toctree references by removing from api/index.rst."""
    api_index = docs_dir / 'api' / 'index.rst'
    
    if not api_index.exists():
        print(f"Warning: {api_index} not found")
        return
    
    content = api_index.read_text(encoding='utf-8')
    
    # Remove the duplicate references that are already in main index.rst
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
        # Skip lines that reference duplicate files
        if any(dup in line.strip() for dup in duplicates) and line.strip().startswith('   '):
            print(f"Removed duplicate toctree reference: {line.strip()}")
            continue
        filtered_lines.append(line)
    
    api_index.write_text('\n'.join(filtered_lines), encoding='utf-8')

def process_rst_file(file_path):
    """Process a single RST file to fix warnings."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        content = fix_title_underlines(content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed warnings in: {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    """Main function to fix all documentation warnings."""
    # Check if we're in the docs directory or project root
    current_dir = Path.cwd()
    if current_dir.name == 'docs':
        docs_dir = current_dir
    else:
        docs_dir = current_dir / 'docs'

    if not docs_dir.exists():
        print(f"Error: docs directory not found at {docs_dir}")
        sys.exit(1)

    print("🔧 Fixing documentation warnings...")
    print(f"Working directory: {docs_dir}")

    # Fix title underlines in all RST files
    rst_files = list(docs_dir.rglob('*.rst'))
    print(f"Found {len(rst_files)} RST files")

    for rst_file in rst_files:
        process_rst_file(rst_file)

    # Fix toctree issues
    print("\n🔧 Fixing toctree issues...")
    fix_toctree_issues(docs_dir)
    fix_duplicate_toctree_references(docs_dir)
    
    print("\n✅ Documentation warning fixes completed!")
    print("Run 'sphinx-build -b html . _build/html' to verify fixes.")

if __name__ == '__main__':
    main()
