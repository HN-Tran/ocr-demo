# Contributing to docread

## Development setup

```bash
uv sync --all-groups
uv run uvicorn app.main:app --reload
```

## Quality checks (same as CI)

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy app eval tests
uv run pytest -q
```

## Commit messages

Use clear [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, …).

## Pull requests

1. Branch from `master`.
2. Open a PR; wait for **CI**.
3. Merge when ready.

Use PRs instead of pushing directly to `master` whenever possible.
