# Middleware Hooks Visualizer

Minimal interactive visualizer for LangChain middleware hooks and agent graphs.

## Usage

The `index.html` file can be embedded directly into Mintlify documentation.

**Binary naming scheme (5 bits):**
```
Bit 0: tools
Bit 1: before_agent
Bit 2: before_model
Bit 3: after_model
Bit 4: after_agent
```

Example: `10110` = tools + before_model + after_model

## Files

- `index.html` - Minimal embeddable widget (checkboxes + diagram)
- `diagrams.json` - 32 pre-generated Mermaid diagrams (for reference)
- `diagrams_inline.js` - Inline JavaScript version of diagrams (for Mintlify serving)
- `generate_middleware_diagrams.py` - Diagram generator

## Regenerating Diagrams

To regenerate the diagrams after making changes to middleware hooks:

```bash
cd langchain  # Navigate to the langchain repo root
uv run python src/plugins/middleware_visualization/generate_middleware_diagrams.py
```

This will generate:
- `diagrams.json` - Pretty-printed JSON for reference/debugging
- `diagrams_inline.js` - Minified JavaScript constant for serving

**Note**: Mintlify's dev server doesn't serve JSON files, so we convert the data to a JavaScript file that can be loaded via `<script src>`.
