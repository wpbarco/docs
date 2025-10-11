"""Helpers for substituting documentation constants in markdown content."""

from __future__ import annotations

import logging
import re

from pipeline.preprocessors.constants_map import CONSTANTS_MAP

logger = logging.getLogger(__name__)

CONSTANT_PATTERN = re.compile(r"(?<!\\)\$\[(?P<constant>[^\]]+)\]")


def replace_constants(markdown: str, file_path: str) -> str:
    """Replace $[constant] tokens with strings from CONSTANTS_MAP.

    Args:
        markdown: Raw markdown/MDX content.
        file_path: Source filename for logging context.

    Returns:
        Markdown string with any matching constants substituted. Unknown
        constants are left unchanged.
    """

    def _replace(match: re.Match[str]) -> str:
        constant_name = match.group("constant")
        replacement = CONSTANTS_MAP.get(constant_name)

        if replacement is None:
            logger.info(
                "%s: Constant '%s' not found in constants map.",
                file_path,
                constant_name,
            )
            return match.group(0)

        return replacement

    substituted = CONSTANT_PATTERN.sub(_replace, markdown)
    # Unescape literal \$[constant] sequences that were intentionally left untouched.
    return re.sub(r"\\(\$\[)", r"\1", substituted)

