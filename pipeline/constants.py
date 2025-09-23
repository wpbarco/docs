"""Constants used for docs building."""

import re

CONDITIONAL_FENCE_PATTERN = re.compile(
    r"""
    ^                       # Start of line
    (?P<indent>[ \t]*)      # Optional indentation (spaces or tabs)
    :::                     # Literal fence marker
    (?P<language>\w+)?      # Optional language identifier (named group: language)
    \s*                     # Optional trailing whitespace
    $                       # End of line
    """,
    re.VERBOSE,
)

CONDITIONAL_BLOCK_PATTERN = re.compile(
    r"(?P<indent>[ \t]*)(?<!\\):::(?P<language>\w+)\s*\n"
    r"(?P<content>((?:.*\n)*?))"  # Capture content inside the block
    r"(?P=indent)[ \t]*(?<!\\):::"  # Match closing, same indentation, not escaped
)

CROSS_REFERENCE_PATTERN = re.compile(
    r"""
    (?:                     # Non-capturing group for two possible formats:
        @\[                 # @ symbol followed by opening bracket for title
        (?P<title>[^\]]+)   # Custom title - one or more non-bracket characters
        \]                  # Closing bracket for title
        \[                  # Opening bracket for link name
        (?P<link_name_with_title>[^\]]+)  # Link name - non-bracket chars
        \]                  # Closing bracket for link name
        |                   # OR
        @\[                 # @ symbol followed by opening bracket
        (?P<link_name>[^\]]+)   # Link name - one or more non-bracket characters
        \]                  # Closing bracket
    )
    """,
    re.VERBOSE,
)

# Pattern to find code blocks with highlight comments, supporting optional indentation
CODE_BLOCK_PATTERN = re.compile(
    r"(?P<indent>[ \t]*)```(?P<language>\w+)[ ]*(?P<attributes>[^\n]*)\n"
    r"(?P<code>((?:.*\n)*?))"  # Capture the code inside the block using named group
    r"(?P=indent)```"  # Match closing backticks with the same indentation
)

LANGUAGE_COMMENT_SYNTAX = {
    "python": "# ",
    "typescript": "// ",
}
