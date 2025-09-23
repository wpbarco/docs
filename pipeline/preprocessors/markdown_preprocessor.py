"""Markdown preprocessor for handling cross-references and conditional blocks.

This module provides functionality to preprocess markdown files by:
1. Transforming @[link_name] references to proper markdown links
2. Processing conditional language blocks (:::python, :::js)
"""

import logging
import os
import re
from pathlib import Path

from pipeline import constants
from pipeline.preprocessors.handle_auto_links import replace_autolinks

logger = logging.getLogger(__name__)


def _apply_conditional_rendering(md_text: str, target_language: str) -> str:
    r"""Apply conditional rendering to markdown content.

    Processes conditional blocks like:
    :::python
    This content is only shown for Python
    :::

    Escaped blocks (with backslash) are preserved as literal text:
    \:::python
    This will appear as :::python in the output
    \:::

    Args:
        md_text: The markdown content to process.
        target_language: The target language ("python" or "js").

    Returns:
        Processed markdown with conditional blocks resolved and escaped tags unescaped.

    Raises:
        ValueError: If target_language is not "python" or "js".
    """
    if target_language not in {"python", "js"}:
        msg = "target_language must be 'python' or 'js'"
        raise ValueError(msg)

    def replace_conditional_blocks(match: re.Match) -> str:
        """Keep active conditionals."""
        language = match.group("language")
        content = match.group("content")

        if language not in {"python", "js"}:
            # If the language is not supported, return the original block
            return match.group(0)

        if language == target_language:
            return content

        # If the language does not match, return an empty string
        return ""

    def replace_ignored_code_blocks(match: re.Match) -> str:
        """Replace ignored code blocks with an empty string."""
        attributes = match.group("attributes").strip()
        if "ignore" in attributes:
            return ""
        return match.group(0)

    # Process conditional blocks first
    result = constants.CONDITIONAL_BLOCK_PATTERN.sub(
        replace_conditional_blocks, md_text
    )

    # Then unescape escaped tags by removing the backslash
    result = re.sub(r"\\(:::)", r"\1", result)

    # Then replace ignored code blocks
    return constants.CODE_BLOCK_PATTERN.sub(replace_ignored_code_blocks, result)


def preprocess_markdown(
    content: str,
    file_path: Path,
    *,
    target_language: str | None = None,
    default_scope: str | None = None,
) -> str:
    """Preprocess markdown content with cross-references and conditional blocks.

    Args:
        content: The markdown content to process.
        file_path: Path to the file being processed (for error reporting).
        target_language: Target language for conditional blocks ("python" or "js").
                        If None, uses TARGET_LANGUAGE environment variable.
        default_scope: Default scope for cross-references. If None,
            uses target_language.

    Returns:
        Processed markdown content.
    """
    # Determine target language
    if target_language is None:
        target_language = os.environ.get("TARGET_LANGUAGE", "python")

    # Determine default scope for cross-references
    if default_scope is None:
        default_scope = target_language

    # Apply cross-reference preprocessing
    content = replace_autolinks(content, str(file_path), default_scope=default_scope)

    # Apply conditional rendering for code blocks
    return _apply_conditional_rendering(content, target_language)
