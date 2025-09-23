"""Code snippet linting for markdown/MDX files.

This module provides functionality to lint code snippets within markdown files,
specifically targeting Python and JavaScript code blocks.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple, Union


class CodeBlock(NamedTuple):
    """Represents a code block found in markdown."""

    language: Union[str, None]
    content: str
    start_line: int
    end_line: int


class LintResult(NamedTuple):
    """Result of linting a code block."""

    original_content: str
    linted_content: str
    has_changes: bool
    errors: list[str]


def extract_code_blocks(markdown_content: str) -> list[CodeBlock]:
    """Extract all code blocks from markdown content.

    Args:
        markdown_content: The markdown file content as a string

    Returns:
        List of CodeBlock objects containing the language, content, and line numbers
    """
    code_blocks = []
    lines = markdown_content.split('\n')
    in_code_block = False
    current_block_language = None
    current_block_content = []
    current_block_start = 0

    for line_num, line in enumerate(lines, 1):
        # Check for opening code fence
        if line.strip().startswith('```'):
            if not in_code_block:
                # Starting a new code block
                in_code_block = True
                current_block_start = line_num
                # Extract language from the opening fence
                fence_content = line.strip()[3:].strip()
                current_block_language = fence_content.split()[0] if fence_content else None
                current_block_content = []
            else:
                # Ending current code block
                in_code_block = False
                code_blocks.append(CodeBlock(
                    language=current_block_language,
                    content='\n'.join(current_block_content),
                    start_line=current_block_start + 1,  # Content starts after opening fence
                    end_line=line_num - 1  # Content ends before closing fence
                ))
                current_block_language = None
                current_block_content = []
        elif in_code_block:
            # We're inside a code block, collect the content
            current_block_content.append(line)

    return code_blocks


def lint_python_code(code: str) -> LintResult:
    """Lint Python code using ruff (placeholder implementation).

    Args:
        code: Python code to lint

    Returns:
        LintResult with linting results
    """
    # For now, this is a no-op placeholder
    # TODO: Implement actual ruff integration
    return LintResult(
        original_content=code,
        linted_content=code,  # No changes for now
        has_changes=False,
        errors=[]
    )


def lint_javascript_code(code: str) -> LintResult:
    """Lint JavaScript code using appropriate linter (placeholder implementation).

    Args:
        code: JavaScript code to lint

    Returns:
        LintResult with linting results
    """
    # For now, this is a no-op placeholder
    # TODO: Implement actual JS linting integration
    return LintResult(
        original_content=code,
        linted_content=code,  # No changes for now
        has_changes=False,
        errors=[]
    )


def lint_code_block(block: CodeBlock) -> LintResult:
    """Lint a single code block based on its language.

    Args:
        block: CodeBlock to lint

    Returns:
        LintResult with linting results
    """
    if not block.language:
        # No language specified, skip linting
        return LintResult(
            original_content=block.content,
            linted_content=block.content,
            has_changes=False,
            errors=[]
        )

    language = block.language.lower()

    if language == 'python' or language == 'py':
        return lint_python_code(block.content)
    elif language in ('javascript', 'js', 'jsx', 'typescript', 'ts', 'tsx'):
        return lint_javascript_code(block.content)
    else:
        # Unsupported language, skip linting
        return LintResult(
            original_content=block.content,
            linted_content=block.content,
            has_changes=False,
            errors=[]
        )


def apply_lint_changes(markdown_content: str, code_blocks: list[CodeBlock], lint_results: list[LintResult]) -> str:
    """Apply linting changes back to the original markdown content.

    Args:
        markdown_content: Original markdown content
        code_blocks: List of extracted code blocks
        lint_results: List of corresponding lint results

    Returns:
        Updated markdown content with linting changes applied
    """
    lines = markdown_content.split('\n')

    # Apply changes in reverse order to maintain line numbers
    for block, result in reversed(list(zip(code_blocks, lint_results))):
        if result.has_changes:
            # Replace the content lines of this code block
            new_content_lines = result.linted_content.split('\n')
            # Replace lines from start_line to end_line (inclusive, 1-based)
            lines[block.start_line - 1:block.end_line] = new_content_lines

    return '\n'.join(lines)


def lint_markdown_file(file_path: Path, *, dry_run: bool = False) -> tuple[str, list[str]]:
    """Lint all code snippets in a markdown file.

    Args:
        file_path: Path to the markdown file
        dry_run: If True, return the changes without writing to file

    Returns:
        Tuple of (updated_content, list_of_errors)
    """
    if not file_path.exists():
        return "", [f"File not found: {file_path}"]

    try:
        original_content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        return "", [f"Error reading file: {e}"]

    # Extract all code blocks
    code_blocks = extract_code_blocks(original_content)

    # Lint each code block
    lint_results = []
    all_errors = []

    for block in code_blocks:
        result = lint_code_block(block)
        lint_results.append(result)
        all_errors.extend(result.errors)

    # Apply changes
    updated_content = apply_lint_changes(original_content, code_blocks, lint_results)

    # Write back if not dry run and there are changes
    if not dry_run and updated_content != original_content:
        try:
            file_path.write_text(updated_content, encoding='utf-8')
        except Exception as e:
            all_errors.append(f"Error writing file: {e}")

    return updated_content, all_errors