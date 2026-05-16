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
