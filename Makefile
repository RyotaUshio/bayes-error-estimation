all: format lint type test

.PHONY: format
format:
	uv run ruff format

.PHONY: lint
lint:
	uv run ruff check

.PHONY: lint-fix
lint-fix:
	uv run ruff check --fix

.PHONY: test
test:
	uv run pytest
