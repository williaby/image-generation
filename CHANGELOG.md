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
