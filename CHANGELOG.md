# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Topaz Labs API integration as an optional post-processing pipeline for
  generated images (`--enhance`, `--topaz-model`, `--topaz-sharpen`,
  `--topaz-denoise`, `--topaz-face`, `--topaz-face-strength` flags).
- SSRF protection for Topaz image downloads: hostname validated against an
  allowlist (`api.topazlabs.com`, `cdn.topazlabs.com`) before fetching.
- RAD annotations (`#CRITICAL`, `#ASSUME`) marking external-API assumptions
  in `topaz_enhance_image` for future verification.
- `renovate.json` configuration to re-enable Renovate dependency updates
  after the onboarding PR was closed without merging. Covers `pep621`,
  `pip_requirements`, `github-actions`, and `pre-commit` managers, with
  weekly `lockFileMaintenance` to keep `uv.lock` in sync, auto-merge for
  GitHub Actions minor/patch bumps, SHA pinning for actions, and manual
  review gating for Python version updates.

### Changed

- `renovate.json` `packageRules` extended with a cross-manager grouping entry
  for `pep621` and `pip_requirements`. Single-package Python bumps now
  produce one PR that updates both `pyproject.toml` and `requirements.txt`
  in lockstep (`groupName: "Python dep {{depName}}"` templates per-package
  so different dependencies still get separate PRs). Prevents the
  version-skew pattern that surfaced in PR #26, where `google-genai` was
  bumped in `requirements.txt` only and `pyproject.toml` retained the prior
  constraint, causing `uv sync` (used by every CI workflow) to silently
  resolve the old version.
- `.pre-commit-config.yaml` `astral-sh/ruff-pre-commit` rev bumped from
  `v0.7.4` (SHA `cafecb2f...`) to `v0.15.11` (SHA `d1b83317...`) to match
  the project's pinned ruff in `uv.lock` (`ruff==0.15.11`). The older pin's
  ruff formatter disagreed with the project's modern ruff, so
  `pre-commit run --all-files` locally would reformat files that current
  ruff considered already-formatted. With the pins aligned, pre-commit and
  CI agree on canonical formatting. SHA-pin format preserved per PR #18
  PC-* compliance work. Two source files surfaced minor adjustments under
  the new hook: `scripts/generate_image.py` (two SIM222 simplifications,
  `x if x else y` -> `x or y`) and `tests/test_generate_image.py` (two
  assertion-message line-wrap adjustments). Discovered while triaging
  local pre-commit drift during PR #26 review.
- `google-genai` dependency upgraded from `>=0.4.0,<2.0.0` (resolved at
  `0.8.0`) to `>=2.2.0,<2.3.0`. Pin updated in `pyproject.toml`,
  `requirements.txt`, and `uv.lock` so all three manifests agree. Upstream
  v2.0 release notes scope breaking changes to the Interactions API, which
  this project does not use. The `client.models.generate_content` codepath
  in `scripts/generate_image.py` was verified against v2.2.0's
  `types.GenerateContentResponse`, `types.GenerateContentConfig`, and
  `types.Part` schemas; all attributes the script reads (`candidates`,
  `prompt_feedback`, `usage_metadata`, `text`, `inline_data`,
  `thought_signature`) remain present unchanged. Satisfies the `#VERIFY`
  marker at `scripts/generate_image.py:823` for google-genai version bumps.
- `python-compatibility.yml` `paths:` filters and `source-directory:` aligned
  from `src/` to `scripts/` to match this repo's actual layout. The previous
  `src/` references were a copy-paste leftover from the org template; no
  `src/` directory exists in this repo, so the workflow's path-trigger never
  fired on Python changes.
- `python-compatibility.yml` matrix trimmed from
  `["3.10", "3.11", "3.12", "3.13"]` to `["3.12", "3.13"]` to match
  `requires-python = ">=3.12"` in `pyproject.toml`. The earlier 3.10/3.11
  cells could not produce meaningful test results under the declared
  support range.

### Removed

- `.github/workflows/fips-compatibility.yml` removed. This repo has no
  FIPS components: no federal, HIPAA, or financial services deployment
  target; no custom cryptography in code (HTTPS via `requests` ->
  `urllib3` -> OpenSSL is the only crypto-adjacent surface, and FIPS
  posture there is governed by the deployment host's OpenSSL build,
  not by application code). The workflow referenced
  `scripts/check_fips_compatibility.py`, which this repo never
  contained, so every run failed with `FileNotFoundError`. Part of the
  WF-16 10-repo cleanup sweep; canonical template hardened separately
  so future deployments self-skip when the script is absent.

### Fixed

- Python compatibility matrix `test-command` in
  `.github/workflows/python-compatibility.yml` simplified from
  `pytest tests/ -v --tb=short -x --ignore=tests/integration --ignore=tests/load -m "not slow and not integration"`
  to `pytest tests/ -v --tb=short -x`. The org reusable workflow
  (`ByronWilliamsCPA/.github` `python-compatibility.yml@de201234`,
  introduced by PR #25) re-executes `test-command` through a shell layer
  that strips inner double quotes, so pytest received `-m not` and
  treated `slow` as a positional path argument, failing all six matrix
  cells (Python 3.12/3.13 x ubuntu/macos/windows) with exit code 4 since
  commit `d11c726`. The removed filter and ignores were dead boilerplate
  for this repo (no `slow` or `integration` markers exist; no
  `tests/integration/` or `tests/load/` directories). Same workflow file
  now also triggers on its own path so workflow-only edits get validated
  by CI. Restore the filter only after the org workflow properly quotes
  `inputs.test-command` AND this repo introduces those markers. (PR #33)
- Coverage gate raised from 60% to 80% to match CLAUDE.md graduated coverage
  requirement.
- Pre-commit em-dash hook `types_or` filter removed; the filter was ANDed with
  the `files` regex, silently excluding TOML, YAML, JSON, and RST files from
  the check.
- Ruff SIM102 and SIM108 violations resolved in `generate_image.py`: nested
  `if` statements collapsed to compound conditions, ternary expressions used
  where appropriate.
- `getattr(part, "thought_signature", None)` replaces `hasattr` + direct
  attribute access for `thought_signature` in `generate_image.py` to avoid a
  potential `AttributeError` on absent attributes.
- `datetime.UTC` import corrected in `generate_image.py`.
- RAD markers (`#CRITICAL`, `#ASSUME`, `#EDGE`, `#VERIFY`) added to
  `generate_image()` covering API key logging, response schema, and
  `candidates[0].content is None` edge case.
- `del mock.attribute` pattern replaced with `mock.attribute = None` in test
  helpers; `del` on a fresh `MagicMock` raises `AttributeError` at runtime.
- Dead unreachable `with patch` blocks removed from
  `test_reads_key_from_dot_env_file`.
- Unicode characters in argparse help strings and section comments replaced
  with ASCII equivalents to satisfy Ruff RUF001.
- Bare `dict[]` key accesses on Topaz API JSON responses replaced with
  `.get()` calls and explicit `None` checks to avoid misleading `KeyError`
  exceptions swallowed by broad `except Exception` blocks.
- Disk write after a billed Topaz API call now wrapped in `try/except OSError`
  so I/O failures are reported rather than silently lost.
- Output file extension mismatch between user-supplied path and detected image
  format now emits a warning instead of silently writing the wrong extension.
- All diagnostic `print()` calls in `get_topaz_api_key` redirected to
  `sys.stderr`.
- `.env` file read in `_load_api_key` wrapped in `try/except OSError`.
- `topaz_enhance_image` validation `print()` calls (unknown-model,
  invalid `output_format`, strength out-of-range) routed to `sys.stderr`
  so wrapper scripts capturing only stdout are not confused by error
  text mixed into data output.
- Malformed-JSON responses from the Topaz submit and download-URL
  endpoints (200 status, non-JSON body) are now caught: `resp.json()`
  and `dl_resp.json()` moved inside the existing `try/except
  (RequestException, ValueError)` blocks, and the parsed payloads are
  guarded with `isinstance(payload, dict)` so a non-object top-level
  JSON value (array, scalar, `null`) raises a clean error instead of an
  uncaught `AttributeError` from `.get(...)`. The status-poll path
  received the same `isinstance` guard for consistency.
- Malformed-JSON error messages now include the HTTP status code and a
  200-character body snippet so operators can distinguish a transport
  failure from a non-JSON 200 response without reading the live wire.
- `KeyError` removed from four Topaz `requests` `except` tuples
  (submit, status-poll, download URL, image download). No protected
  `try` body did a dict subscript that could raise `KeyError`; the
  catch was dead defensive code with no test exercising it.
- CRLF and other consecutive newline runs in prompts collapsed to a
  single space via `re.sub(r"[\r\n]+", " ", ...)` in
  `document_image_prompt`. Replaces the chained
  `.replace("\r", " ").replace("\n", " ")` that turned a single CRLF
  into two spaces in the generated `PROMPTS.md` table cell.

### Security

- Pinned `urllib3>=2.7.0,<3.0.0` (transitive via `requests`) to address
  CVE-2026-44431 (sensitive headers forwarded across origins in proxied
  low-level redirects, affects 1.23..<2.7.0) and CVE-2026-44432
  (decompression-bomb safeguards bypassed in parts of the streaming API,
  affects 2.6.0..<2.7.0). Resolver moves urllib3 from 2.6.3 to 2.7.0.

## [0.1.0] - 2026-03-01

### Added

- Initial release: Gemini image generation tools.
- `generate_image.py` script with model selection, aspect ratio, and output
  path configuration.
- Diagram-specialist agent with Claude Code frontmatter.
