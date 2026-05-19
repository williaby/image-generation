# scripts/ Folder Conventions

> Folder scope. Inherits from the project root `CLAUDE.md` and
> `~/.claude/CLAUDE.md`. Only what differs from those scopes is documented here.

## Directory contents

This directory contains a single production entry point:
`generate_image.py` (1700+ lines). It is the sole CLI surface for image
generation and Topaz enhancement. New files added here should be treated as
production code and held to the same standards.

## Ruff per-file-ignore policy

`pyproject.toml` encodes a set of per-file suppressions for
`scripts/generate_image.py` that reflect pre-existing complexity and a
tracked cleanup backlog (GCA-001 follow-on). These suppressions are
**specific to that file**. Any new file added to `scripts/` starts with the
full rule set enabled. Add suppression entries deliberately, with a comment
explaining the reason and, where applicable, a tracking reference.

Key suppressions on `generate_image.py` and their rationale:

| Code | Reason |
| --- | --- |
| `T20` | `print()` is user-facing CLI output, not debug logging (see below) |
| `FBT` | Legacy boolean flag API; refactor tracked separately |
| `ANN` | Full type annotation deferred; tracked separately |
| `PLR` | Complexity refactor deferred; tracked separately |
| `C901` | Same complexity backlog as `PLR` |
| `D` | Docstring style cleanup tracked separately |
| `N806` | Variable naming cleanup tracked separately |
| `G004` | structlog renders f-strings eagerly; the form is idiomatic here |

## Print vs. structlog: when to use each

`T20` (flake8-print) is suppressed for this file because `print()` is the
correct tool for user-facing CLI output written to stdout. This is intentional
and should be preserved for any future code that writes informational output
directly to the user.

Use `structlog` only for error and warning paths that go to stderr. Do not
replace deliberate `print()` calls with `structlog.info()`.

## Error hierarchy

The `AppError` hierarchy defined in `generate_image.py` is the single source
of structured error signaling for this CLI:

```
AppError                  # base; main() catches this for clean stderr + exit 1
  ConfigError             # missing or malformed configuration (e.g. unset env var)
  GeminiAPIError          # Gemini API returned an error or unparseable response
  TopazAPIError           # Topaz Labs API returned an error or unparseable response
  FileIOError             # expected file read or write failed
```

New error paths must raise the appropriate subclass, not bare `Exception` or
`RuntimeError`. The `except Exception` block in `main()` is the safety net for
programmer bugs only; it is not a substitute for typed error signaling.

## HTTP client

The canonical HTTP client for this directory is `httpx` (sync). It is already
declared in `pyproject.toml` dependencies. Do not introduce `requests` or any
other HTTP library in new code under `scripts/`. If you need async HTTP, open a
discussion first; the current architecture is sync throughout.

## Settings loader

Environment configuration is loaded via `pydantic-settings`:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings): ...
```

Do not introduce manual `.env` parsing (`dotenv.load_dotenv`, `open(".env")`,
`os.environ` string splitting, etc.) in new code. The `Settings(BaseSettings)`
pattern handles `.env` loading, type coercion, and validation.

## Cross-references

- Full Python conventions (path-scoped): `~/.claude/.claude/rules/python.md`
- Project-wide RAD tagging and verification: project root `CLAUDE.md`
- Ruff and basedpyright configuration: `pyproject.toml`
