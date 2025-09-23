.PHONY: lint

lint:
	@command -v uv >/dev/null 2>&1 || { echo >&2 "uv is not installed. Please install uv to proceed (https://docs.astral.sh/uv/getting-started/installation/)"; exit 1; }
	uv pip install --quiet .
	uv run ruff format . --diff
	uv run ruff check . --diff
	uv run mypy .
