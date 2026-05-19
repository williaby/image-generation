# tests/ Folder Conventions

> Folder scope. Inherits from the project root `CLAUDE.md` and
> `~/.claude/CLAUDE.md`. Only what differs from those scopes is documented here.

## Per-file-ignore entries active for tests/

`pyproject.toml` suppresses several rule groups for `tests/**`. The suppressions
are intentional; do not remove them without understanding their rationale:

| Code(s) | Reason |
| --- | --- |
| `S101` | Assertions are the standard assertion mechanism in pytest |
| `D` | Docstrings on test classes and methods add little value; pytest uses function names for reporting |
| `PLR2004` | Magic numbers in test assertions are acceptable and readable |
| `PLC0415` | Deferred imports inside test methods are the structural pattern (see below) |
| `ANN` | Type annotations are not required in tests |
| `SLF` | Tests legitimately access private members to verify internal state |
| `FBT` | Boolean values in parametrize arguments are expected |
| `TRY` | Exception-handling style is relaxed; tests verify exception types, not message formatting |
| `ARG001`, `ARG002` | pytest fixture parameters appear unused to static analysis |
| `EM` | Error message construction style is relaxed in tests |
| `ERA` | Commented-out reference code is allowed in tests |
| `FURB` | Refurb modernizations are deferred for the test suite |
| `TCH003` | Typing imports under `TYPE_CHECKING` are not needed in tests |

## Deferred imports inside test methods

Test methods import from `scripts.generate_image` inside the method body rather
than at module level:

```python
def test_something(self, tmp_path: Path) -> None:
    from scripts import generate_image as mod
    ...
```

This pattern avoids triggering `generate_image.py` module-level side-effects
(environment variable reads, `Settings` instantiation, structlog setup) during
pytest collection. Importing at collection time would cause failures in
environments where `GEMINI_API_KEY` is not set.

`PLC0415` (import not at top of file) is suppressed repo-wide for `tests/**`
specifically to permit this pattern. Do not move these imports to module level
without first verifying that `generate_image.py` has no collection-time
side-effects.

## basedpyright scope

The pre-commit `basedpyright` hook covers both `tests/` and
`scripts/generate_image.py` (added in GCA-002). The `pyproject.toml`
`[tool.basedpyright]` section disables the `reportUnknown*` family and
`reportPrivateUsage` to accommodate pytest fixture patterns and test access to
private helpers. These relaxations apply to the whole project, including tests.

## Test file layout

| File | Coverage focus |
| --- | --- |
| `test_generate_image.py` | Main CLI flow: pure functions, file I/O helpers, API key helpers, and `generate_image()` with a mocked google-genai SDK client |
| `test_security_hardening.py` | Security properties: `MAX_INPUT_IMAGE_BYTES` enforcement, Markdown sanitization in `document_image_prompt`, random-token filename regression |
| `test_topaz_enhance.py` | Topaz integration path: all branches of `topaz_enhance_image()`, including SSRF guards, polling retry, timeout, and disk-write failure |

## Fixture conventions

Tests use `unittest.mock` exclusively (no `pytest-mock`). The fixture conventions
in use across the test suite are:

- `tmp_path` (built-in pytest): file I/O tests write to this temporary directory.
  Use it for any test that creates, reads, or writes files.
- `monkeypatch` (built-in pytest): environment variable overrides and module
  attribute patching. Use `monkeypatch.setattr(mod, "ATTR", value)` to override
  module-level constants without leaving state behind.
- `capsys` (built-in pytest): capture stdout/stderr for assertions on CLI output.
- Module-level helper functions prefixed with `_` (e.g. `_make_input_png`,
  `_mock_response`, `_happy_path_mocks`) are build helpers, not fixtures. They
  are plain functions called directly in test methods.
- Test classes are named `Test<Subject>` and group related cases. Each method
  tests one behaviour; use `@pytest.mark.parametrize` for input variations rather
  than loops inside a single test body.

## Cross-references

- Full testing conventions (path-scoped): `~/.claude/.claude/rules/testing.md`
- Coverage thresholds and branch requirements: `pyproject.toml` `[tool.coverage.report]`
- RAD tagging for test verification steps: project root `CLAUDE.md`
