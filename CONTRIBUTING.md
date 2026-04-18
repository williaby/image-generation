# Contributing

Thank you for your interest in contributing to image-generation!

## Getting Started

1. Fork the repository and clone your fork.
2. Install dependencies: `uv sync --all-extras`.
3. Install pre-commit hooks: `pre-commit install`.

## Development Workflow

- Create a feature branch from `main` using the naming convention
  `feat/<short-description>` or `fix/<short-description>`.
- Write or update tests for any changed behaviour.
- Run `pre-commit run --all-files` before committing.
- Commits must follow [Conventional Commits](https://www.conventionalcommits.org/)
  and be GPG-signed.
- Open a pull request against `main`. The PR description should reference any
  related issues.

## Code Standards

- Python: Ruff format and lint (88-char line length), BasedPyright strict mode.
- No em-dashes (`-`) in any text (code, comments, docs, commit messages).
- Security: run `uv run pip-audit` and address any findings before merging.
- Coverage targets: 80% line, 70% branch, 90% for critical paths.

## Reporting Issues

Open a GitHub issue with a clear description and steps to reproduce. For
security vulnerabilities see [SECURITY.md](SECURITY.md).
