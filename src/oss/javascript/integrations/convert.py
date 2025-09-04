import re
from pathlib import Path

def extract_packages(content: str) -> str | None:
    """Extract package names from Npm2Yarn component."""
    pattern = r'<Npm2Yarn>\s*(.*?)\s*</Npm2Yarn>'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def create_codegroup(packages: str) -> str:
    """Create CodeGroup component with npm, yarn, and pnpm instructions."""
    return f'''<CodeGroup>
```bash npm
npm install {packages}
```

```bash yarn
yarn add {packages}
```

```bash pnpm
pnpm add {packages}
```
</CodeGroup>'''

def convert_file(file_path: Path) -> bool:
    """Convert a single file's installation blocks."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the mdx block
    mdx_pattern = r'```\{=mdx\}.*?```'
    match = re.search(mdx_pattern, content, re.DOTALL)  # Fixed: using mdx_pattern instead of pattern

    if not match:
        print(f"No mdx block found in {file_path}")
        return False

    mdx_block = match.group(0)
    packages = extract_packages(mdx_block)

    if not packages:
        print(f"No Npm2Yarn component found in {file_path}")
        return False

    # Create new CodeGroup
    new_content = content.replace(mdx_block, create_codegroup(packages))

    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"Successfully converted {file_path}")
    return True

def main():
    import sys

    # If no argument provided, process all files in current directory
    if len(sys.argv) == 1:
        base_dir = Path(__file__).parent
        files = []
        # Walk through all subdirectories
        for ext in ['.md', '.mdx']:
            files.extend(list(base_dir.rglob(f'*{ext}')))

        print(f"Found {len(files)} files to process")
        for file_path in files:
            print(f"\nProcessing {file_path}...")
            convert_file(file_path)

    # If file path provided, process single file
    elif len(sys.argv) == 2:
        file_path = Path(sys.argv[1])
        if not file_path.exists():
            print(f"File not found: {file_path}")
            sys.exit(1)
        convert_file(file_path)

    else:
        print("Usage: python script.py [file_path]")
        print("If no file_path is provided, will process all .md/.mdx files in current directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
