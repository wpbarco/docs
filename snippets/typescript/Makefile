.PHONY: lint

lint:
	@command -v pnpm >/dev/null 2>&1 || { echo >&2 "pnpm is not installed. Please install pnpm to proceed (https://pnpm.io/installation)"; exit 1; }
	pnpm install
	pnpm lint
