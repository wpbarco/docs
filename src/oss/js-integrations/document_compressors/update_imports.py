import os
import sys
from pathlib import Path
import re

def to_pascal_case(filename: str) -> str:
    # Remove extension and split on underscores
    parts = filename.split('.')[0].split('_')
    # Capitalize each part and join
    return ''.join(part.capitalize() for part in parts)

def validate_new_path(original_path: str, file_path: Path) -> bool:
    # Convert to absolute path if it isn't already
    file_path = file_path.resolve()
    
    # Navigate up to src directory
    src_dir = file_path.parent
    while src_dir.name != 'src' and src_dir.parent != src_dir:
        src_dir = src_dir.parent
    
    if src_dir.name != 'src':
        print(f"Error: Could not find src directory from {file_path}")
        return False
    
    # Construct the full path to the target file
    new_path = src_dir / 'snippets' / 'javascript-integrations' / 'examples' / Path(original_path).with_suffix('.mdx')
    exists = new_path.exists()
    if not exists:
        print(f"Warning: Target file does not exist: {new_path}")
        print(f"Absolute path: {new_path.absolute()}")
    return exists

def update_imports(content: str, file_path: Path) -> str:
    # Find import statements and validate paths before updating
    import_pattern = r'import \w+ from "@examples/(.*?)\.ts";'
    
    def replace_import(match):
        path = match.group(1)
        # Get the filename from the path and convert to PascalCase
        filename = os.path.basename(path)
        component_name = to_pascal_case(filename)
        
        if validate_new_path(path, file_path):
            return f'import {component_name} from "/snippets/javascript-integrations/examples/{path}.mdx";'
        else:
            # Keep the original if target doesn't exist
            return match.group(0)
    
    # Store original content to check for changes
    original_content = content
    
    # Update imports
    content = re.sub(import_pattern, replace_import, content)
    
    # Find and update CodeBlock usage if content changed
    if content != original_content:
        # Find all old import statements to get the mapping of old to new names
        old_imports = re.finditer(r'import (\w+) from "@examples/(.*?)\.ts";', original_content)
        for old_import in old_imports:
            old_name = old_import.group(1)
            old_path = old_import.group(2)
            new_name = to_pascal_case(os.path.basename(old_path))
            
            # Update CodeBlock usage with new component name
            codeblock_pattern = f'<CodeBlock language="typescript">{{{old_name}}}</CodeBlock>'
            new_codeblock = f'<{new_name} />'
            content = content.replace(codeblock_pattern, new_codeblock)
    
    return content

def process_file(file_path: Path) -> None:
    # Convert to absolute path if it isn't already
    file_path = file_path.resolve()
    
    if file_path.suffix != '.mdx':
        print(f"Skipping non-MDX file: {file_path}")
        return
        
    try:
        # Read the file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Look for the patterns we want to update
        if '@examples/' not in content and '<CodeBlock language="typescript">' not in content:
            print(f"No updates needed for: {file_path}")
            return
            
        # Update the content
        new_content = update_imports(content, file_path)
        
        if new_content != content:
            # Write the updated content
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"Updated: {file_path}")
        else:
            print(f"No changes made to: {file_path}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def process_path(path: Path) -> None:
    # Convert to absolute path if it isn't already
    path = path.resolve()
    
    if path.is_file():
        process_file(path)
    elif path.is_dir():
        print(f"\nProcessing directory: {path}")
        for root, _, files in os.walk(path):
            root_path = Path(root)
            for file in files:
                process_file(root_path / file)
    else:
        print(f"Error: {path} does not exist")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <path1> [path2] [path3] ...")
        print("Paths can be files or directories")
        sys.exit(1)
    
    print("This script will:")
    print("1. Find @examples/... imports and convert them to /snippets/javascript-integrations/examples/...")
    print("2. Convert filenames to PascalCase for component names")
    print("3. Replace CodeBlock components with direct component usage")
    print("4. Validate that target MDX files exist before updating")
    print("\nProcessing will begin in 3 seconds...")
    
    import time
    time.sleep(3)
    
    for path_str in sys.argv[1:]:
        process_path(Path(path_str))
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()