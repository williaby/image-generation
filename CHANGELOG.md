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

### Fixed

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

## [0.1.0] - 2026-03-01

### Added

- Initial release: Gemini image generation tools.
- `generate_image.py` script with model selection, aspect ratio, and output
  path configuration.
- Diagram-specialist agent with Claude Code frontmatter.
