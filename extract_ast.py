import ast
import os
import json
import logging

logging.basicConfig(level=logging.INFO)

def is_ignore_path(path):
    """Checks if the path should be ignored."""
    ignored = [".git", ".docker", "venv", ".gitignore", "Dockerfile", "Lib", "Scripts", "share", "etc"]
    return any(ignore in path for ignore in ignored)

def extract_comments_and_docstrings(tree):
    """Extracts docstrings and comments from an AST."""
    docstrings = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
            docstring = ast.get_docstring(node)
            if docstring:
                docstrings.append({"type": type(node).__name__, "name": node.name, "docstring": docstring})
    
    return docstrings

def process_file(filepath):
    """Processes a Python file to extract AST-related information."""
    encodings = ['utf-8', 'latin1']  # Add more encodings as needed
    
    for encoding in encodings:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        logging.error(f"Failed to decode {filepath}. Skipping.")
        return None
    
    try:
        tree = ast.parse(content)
    except Exception as e:
        logging.error(f"Failed to parse {filepath}: {e}")
        return None
    
    docstrings = extract_comments_and_docstrings(tree)
    return {"filepath": filepath, "docstrings": docstrings}

def process_directory(dirpath):
    """Processes a directory to look for Python files."""
    output = []
    
    for root, _, files in os.walk(dirpath):
        if is_ignore_path(root):
            continue
        
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                logging.info(f"Processing {filepath} ...")
                file_data = process_file(filepath)
                if file_data:
                    output.append(file_data)
    
    return output

def main():
    dirpath = "."  # Start from current directory. Modify this path as needed.
    output = process_directory(dirpath)
    
    with open("ast_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    main()
