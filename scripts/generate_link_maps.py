"""Generate link maps from Sphinx objects.inv files.

This script fetches and parses Sphinx inventory files (objects.inv) from
various documentation sources and generates Python code for link mappings
that can be used in the documentation preprocessor.

Usage:
    python scripts/generate_link_maps.py

The generated link maps will be written to stdout and can be redirected
to update the link_map.py file.
"""

import json
import sys
import zlib
from collections import defaultdict
from collections.abc import Callable
from typing import Any
from urllib.request import urlopen


class SphinxInventory:
    """Parser for Sphinx objects.inv files."""

    def __init__(self, url: str) -> None:
        """Initialize the inventory parser.

        Args:
            url: URL to the objects.inv file.
        """
        self.url = url
        self.objects: list[dict[str, Any]] = []

    def fetch_and_parse(self) -> None:
        """Fetch and parse the objects.inv file from the URL."""
        with urlopen(self.url) as response:  # noqa: S310
            data = response.read()

        # Parse the header (first 4 lines)
        lines = data.split(b"\n", 4)
        if len(lines) < 5:  # noqa: PLR2004
            msg = "Invalid objects.inv format"
            raise ValueError(msg)

        # The compressed data starts after the header
        compressed_data = lines[4]

        # Decompress the inventory data
        decompressed = zlib.decompress(compressed_data).decode("utf-8")

        # Parse each line of the inventory
        for line in decompressed.strip().split("\n"):
            if not line or line.startswith("#"):
                continue

            parts = line.split(None, 4)
            if len(parts) < 5:  # noqa: PLR2004
                continue

            name, domain_role, priority, location, display_name = parts

            # Split domain:role
            if ":" in domain_role:
                domain, role = domain_role.split(":", 1)
            else:
                domain = domain_role
                role = ""

            # Handle location anchors
            if location.endswith("$"):
                # $ means the name is appended to the location
                location = location[:-1] + name

            self.objects.append({
                "name": name,
                "domain": domain,
                "role": role,
                "priority": priority,
                "location": location,
                "display_name": display_name if display_name != "-" else name,
            })

    def get_objects_by_role(self, role: str | None = None) -> list[dict[str, Any]]:
        """Get all objects, optionally filtered by role.

        Args:
            role: Optional role to filter by (e.g., 'class', 'function', 'module').

        Returns:
            List of matching objects.
        """
        if role is None:
            return self.objects
        return [obj for obj in self.objects if obj["role"] == role]


def generate_link_map_from_inventory(
    host: str,
    scope: str,
    inventory_url: str,
    *,
    include_roles: list[str] | None = None,
    name_transform: Callable[[dict[str, Any]], str | None] | None = None,
    url_transform: Callable[[dict[str, Any], str], str] | None = None,
) -> dict[str, Any]:
    """Generate a link map from a Sphinx inventory.

    Args:
        host: Base URL for the documentation site.
        scope: Scope identifier (e.g., 'python', 'js').
        inventory_url: URL to the objects.inv file.
        include_roles: List of roles to include (None = all).
        name_transform: Optional function to transform object names.
        url_transform: Optional function to transform URLs.

    Returns:
        Dictionary containing the link map in the format expected by link_map.py.
    """
    inv = SphinxInventory(inventory_url)
    inv.fetch_and_parse()

    links: dict[str, str] = {}

    for obj in inv.objects:
        # Filter by role if specified
        if include_roles and obj["role"] not in include_roles:
            continue

        # Get the name and location
        name = obj["name"]
        location = obj["location"]

        # Apply name transformation if provided
        if name_transform:
            name = name_transform(obj)
            if name is None:
                continue

        # Apply URL transformation if provided
        if url_transform:
            location = url_transform(obj, location)

        links[name] = location

    return {
        "host": host,
        "scope": scope,
        "links": links,
    }


def python_langchain_name_transform(obj: dict[str, Any]) -> str | None:
    """Transform object names for Python LangChain reference.

    Args:
        obj: Object dictionary from the inventory.

    Returns:
        Transformed name or None to skip this object.
    """
    name = obj["name"]
    role = obj["role"]

    # For classes and functions, use the short name
    if role in ("class", "function", "method", "attribute"):
        # Get the last component (e.g., AIMessage from langchain_core.messages.AIMessage)
        return name.split(".")[-1]

    # For modules, keep the full path
    if role == "module":
        return name

    # Skip other types
    return None


# Configuration for different documentation sources
INVENTORY_CONFIGS = [
    {
        "host": "https://reference.langchain.com/python/",
        "scope": "python",
        "inventory_url": "https://reference.langchain.com/python/objects.inv",
        "include_roles": ["class", "function", "method", "module"],
        "name_transform": python_langchain_name_transform,
    },
    {
        "host": "https://langchain-ai.github.io/langgraph/",
        "scope": "python",
        "inventory_url": "https://langchain-ai.github.io/langgraph/objects.inv",
        "include_roles": ["class", "function", "method"],
    },
    # Add more configs as needed
    # {
    #     "host": "https://langchain-ai.github.io/langgraphjs/",
    #     "scope": "js",
    #     "inventory_url": "https://langchain-ai.github.io/langgraphjs/objects.inv",
    # },
]


def format_python_dict(data: dict[str, str], indent: int = 12) -> str:
    """Format a dictionary as Python code.

    Args:
        data: Dictionary to format.
        indent: Number of spaces for indentation.

    Returns:
        Formatted Python dictionary string.
    """
    if not data:
        return "{}"

    lines = ["{"]
    indent_str = " " * indent
    for key, value in sorted(data.items()):
        # Escape quotes in strings
        key_escaped = key.replace('"', '\\"')
        value_escaped = value.replace('"', '\\"')
        lines.append(f'{indent_str}"{key_escaped}": "{value_escaped}",')
    lines.append(" " * (indent - 4) + "}")
    return "\n".join(lines)


def main() -> None:
    """Generate link maps from configured inventory sources."""
    link_maps = []

    for config in INVENTORY_CONFIGS:
        try:
            print(f"Processing {config['inventory_url']}...", file=sys.stderr)
            link_map = generate_link_map_from_inventory(**config)
            link_maps.append(link_map)
            print(
                f"  -> Generated {len(link_map['links'])} links",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"  -> Error processing {config['inventory_url']}: {e}",
                file=sys.stderr,
            )

    # Output as Python code
    print('"""Auto-generated link mappings from Sphinx objects.inv files.')
    print()
    print("This file is generated by scripts/generate_link_maps.py.")
    print("Do not edit manually - regenerate using:")
    print()
    print("    python scripts/generate_link_maps.py > pipeline/preprocessors/link_map_generated.py")  # noqa: E501
    print('"""')
    print()
    print("from collections.abc import Mapping")
    print("from typing import TypedDict")
    print()
    print()
    print('class LinkMap(TypedDict):')
    print('    """Typed mapping describing each link map entry."""')
    print()
    print("    host: str")
    print("    scope: str")
    print("    links: Mapping[str, str]")
    print()
    print()
    print("AUTO_GENERATED_LINK_MAPS: list[LinkMap] = [")

    for link_map in link_maps:
        print("    {")
        print(f'        "host": "{link_map["host"]}",')
        print(f'        "scope": "{link_map["scope"]}",')
        print('        "links": ' + format_python_dict(link_map["links"], 12))
        print("    },")

    print("]")

    # Also show some statistics
    print("\n# Statistics:", file=sys.stderr)
    scope_counts: dict[str, int] = defaultdict(int)
    for link_map in link_maps:
        scope = link_map["scope"]
        count = len(link_map["links"])
        scope_counts[scope] += count
        print(f"  {link_map['host']}: {count} links", file=sys.stderr)

    print("\nTotal by scope:", file=sys.stderr)
    for scope, count in scope_counts.items():
        print(f"  {scope}: {count} links", file=sys.stderr)


if __name__ == "__main__":
    main()
