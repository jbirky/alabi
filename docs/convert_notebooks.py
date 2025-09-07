#!/usr/bin/env python3
"""
Convert Jupyter notebooks to Python scripts for download links
"""

import json
import os
import re
from pathlib import Path

def notebook_to_python(notebook_path):
    """Convert a Jupyter notebook to a Python script"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse VS Code notebook format
    if content.strip().startswith('<VSCode.Cell'):
        return vscode_notebook_to_python(content)
    else:
        # Parse standard Jupyter notebook format
        try:
            nb = json.loads(content)
            return jupyter_notebook_to_python(nb)
        except json.JSONDecodeError:
            print(f"Error parsing notebook: {notebook_path}")
            return None

def vscode_notebook_to_python(content):
    """Convert VS Code notebook format to Python"""
    lines = []
    
    # Split by cell markers
    cells = re.split(r'<VSCode\.Cell[^>]*>', content)
    
    for cell in cells[1:]:  # Skip first empty part
        if '</VSCode.Cell>' not in cell:
            continue
            
        # Extract cell content
        cell_content = cell.split('</VSCode.Cell>')[0].strip()
        
        # Skip empty cells
        if not cell_content:
            continue
            
        # Add cell separator comment
        lines.append("# %%")
        
        # Add cell content
        lines.extend(cell_content.split('\n'))
        lines.append("")  # Empty line between cells
    
    return '\n'.join(lines)

def jupyter_notebook_to_python(nb):
    """Convert standard Jupyter notebook to Python"""
    lines = []
    
    for cell in nb.get('cells', []):
        cell_type = cell.get('cell_type', '')
        source = cell.get('source', [])
        
        if cell_type == 'code' and source:
            lines.append("# %%")
            if isinstance(source, list):
                lines.extend(source)
            else:
                lines.append(source)
            lines.append("")
        elif cell_type == 'markdown' and source:
            lines.append("# %% [markdown]")
            if isinstance(source, list):
                for line in source:
                    lines.append(f"# {line.rstrip()}")
            else:
                lines.append(f"# {source.rstrip()}")
            lines.append("")
    
    return '\n'.join(lines)

def main():
    """Convert all notebooks in the docs/source directory"""
    source_dir = Path(__file__).parent
    
    for notebook_path in source_dir.glob("*.ipynb"):
        if notebook_path.name.startswith('.'):
            continue
            
        print(f"Converting {notebook_path.name}...")
        
        python_content = notebook_to_python(notebook_path)
        if python_content:
            python_path = notebook_path.with_suffix('.py')
            
            # Add header
            header = f'''"""
{notebook_path.stem}
{'=' * len(notebook_path.stem)}

This Python script was automatically generated from the Jupyter notebook
{notebook_path.name}.

You can run this script directly or copy sections into your own code.
"""

'''
            
            with open(python_path, 'w', encoding='utf-8') as f:
                f.write(header + python_content)
            
            print(f"Created {python_path.name}")

if __name__ == "__main__":
    main()
