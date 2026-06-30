# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| main    | Yes       |

## Security Surface

This repository is a Python command-line application that invokes external AI
image-generation APIs (Google Gemini via `google-genai`, optional Topaz Labs
via `requests`). It does **not** run a network server, accept untrusted input
over the wire, or hold persistent state for other users. Primary security
concerns:

- **Credential exposure**: `GEMINI_API_KEY` and `TOPAZ_API_KEY` are read from
  the environment or a local `.env` file. They must never be committed.
- **Prompt injection**: user-supplied prompts are passed verbatim to Gemini;
  downstream consumers of generated images or `PROMPTS.md` files should treat
  prompt content as untrusted.
- **Supply chain**: GitHub Actions are pinned to 40-character SHAs; Python
  dependencies are tracked in `uv.lock` and audited by `pip-audit`.
- **Output integrity**: generated images and signature files are written under
  `output/`; that directory should not be served as static content without
  separate review.

Mitigations in place: GitHub Actions SHA pinning, `pip-audit` in CI, signed
commits, required-status-check branch protection, REUSE license compliance,
secret-scanning pre-commit hooks (TruffleHog), and the `no-em-dash` pre-commit
hook for writing-style enforcement.

## Accepted Static-Analysis Findings

Some static-analysis findings are accepted as not applicable to this project's
threat model rather than fixed in code. They are documented here for an audit
trail.

### Snyk Code `python/PT` (Path Traversal, CWE-23) in `scripts/generate_image.py`

- **Status**: Accepted (not a vulnerability under this threat model).
- **What it flags**: command-line path arguments (`--output`, `--finalize`,
  `--enhance`, `--reference`) flow into `open()` calls.
- **Why accepted**: this is a single-user, local CLI with no network server,
  no upload surface, and no multi-tenant boundary (see Security Surface above).
  The path inputs are supplied by the same user who runs the process and are
  used under that user's own filesystem privileges, so directing the tool to
  read or write a chosen path is intended behavior, not a privilege escalation.
  All findings are Low severity.
- **Why not gated in code**: the only sanitizer Snyk credits for this rule is a
  `resolve()`-then-reject containment check against a fixed root. Applying that
  would reject reading or writing images outside the allowed root (a core
  workflow, e.g. enhancing `~/Pictures/foo.png`) and break the test suite, which
  drives `main()` with paths under the system temp directory. The trade is not
  justified for Low-severity, no-trust-boundary findings.
- **Defense in depth retained**: CLI path arguments are checked for NUL bytes at
  the entry boundary (`_reject_nul_byte_in_path_args`), and generated output is
  re-anchored under `output/` via `_reanchor_output_path`.
- **Suppression mechanism**: `.snyk` policy ignores are not supported for Snyk
  Code, so these are dismissed in the Snyk platform (Ignore -> "Won't fix /
  not vulnerable") rather than in-repo. Re-review if a network, server, or
  upload surface, or automation/CI context (scheduled runners, containerized
  pipelines) where path arguments arrive from untrusted sources is ever added.
  Reviewed 2026-06-29.

## Reporting a Vulnerability

**Preferred reporting channel: GitHub Private Vulnerability Reporting.**
Open a new private advisory at
[https://github.com/williaby/image-generation/security/advisories/new](https://github.com/williaby/image-generation/security/advisories/new).
Do **not** open a public GitHub issue for security reports.

If GitHub PVR is unavailable, email
[byronawilliams@gmail.com](mailto:byronawilliams@gmail.com) with the subject
line `[SECURITY] image-generation vulnerability report`.

Include as much detail as possible:

- A description of the vulnerability and its potential impact.
- Steps to reproduce or a minimal proof-of-concept.
- Any suggested mitigation or fix.

### Response SLA

- **Initial acknowledgment within 14 days** of submission (target: 72 hours).
- Once the issue is confirmed and a fix is available, a security advisory will
  be published and the fix will be released.

## Known Vulnerabilities

See [docs/known-vulnerabilities.md](docs/known-vulnerabilities.md) for any
open CVEs that cannot be immediately resolved.
