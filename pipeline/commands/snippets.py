"""Lint command implementation."""

import logging
from pathlib import Path
from typing import Any

from pipeline.core.snippets import DocumentationSnippetsParser

logger = logging.getLogger(__name__)


def clean_snippets_command(args: Any) -> int:  # noqa: ANN401
    """Cleans the snippets directory."""
    src_dir_path = Path(getattr(args, "src_dir", "src"))
    output_dir_path = Path(getattr(args, "output_dir", "snippets"))
    snippets_parser = DocumentationSnippetsParser(src_dir_path, output_dir_path)
    snippets_parser.clean_all()
    return 0


def snippets_command(args: Any) -> int:  # noqa: ANN401
    """Exports code snippets from doc files.

    This function serves as the entry point for the snippets command, handling
    the process of exporting code snippets from doc files in the source directory
    to the lint output directory to be linted by external tools.

    Returns:
        Exit code: 0 for success, 1 for failure (e.g., source directory
        not found).
    """
    src_dir_path = Path(getattr(args, "src_dir", "src"))
    output_dir_path = Path(getattr(args, "output_dir", "snippets"))
    export_only = getattr(args, "export_only", False)
    lint_only = getattr(args, "lint_only", False)

    if not src_dir_path.exists():
        logger.error("Error: src directory not found")
        return 1

    # Create lint output directory
    output_dir_path.mkdir(exist_ok=True)

    # Initialize snippets parser
    snippets_parser = DocumentationSnippetsParser(src_dir_path, output_dir_path)
    if lint_only is False:
        snippets_parser.export_all()
    if export_only is False:
        snippets_parser.lint_all()
    return 0
