"""Central mapping for reusable documentation constants.

This module stores string constants that can be reused across the
documentation set. Use the ``$[constant-name]`` syntax in markdown/MDX
files to reference any entry defined here.
"""

from typing import Final

CONSTANTS_MAP: Final[dict[str, str]] = {}
"""Global registry of reusable documentation constants.

Add entries here with keys representing the constant token and values
containing the string that should replace the token during preprocessing.
"""

