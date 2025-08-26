import os
import sys
from pathlib import Path

def process_file(file_path: Path) -> None:
    # Read the original content
    with open(file_path, 'r') as f:
        content = f.read().rstrip()  # rstrip() removes trailing whitespace and newlines
    
    # Create the new content with typescript code block
    new_content = f"```typescript\n{content}\n```"
    
    # Write the wrapped content to a new .mdx file
    new_path = file_path.with_suffix('.mdx')
    with open(new_path, 'w') as f:
        f.write(new_content)
    
    # Remove the original .ts file
    os.remove(file_path)
    print(f"Processed: {file_path} -> {new_path}")

def process_directory(directory: Path) -> None:
    # Walk through all files and directories
    for root, _, files in os.walk(directory):
        root_path = Path(root)
        for file in files:
            file_path = root_path / file
            # Only process .ts files
            if file_path.suffix == '.ts':
                process_file(file_path)
            else:
                print(f"Skipping non-typescript file: {file_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)
    
    print(f"\nStarting conversion in directory: {directory}")
    print("Will only process .ts files and convert them to .mdx\n")
    
    process_directory(directory)
    print("\nConversion complete!")

if __name__ == "__main__":
    main()