# Known Vulnerabilities

Last reviewed: 2026-05-15

This file documents vulnerabilities identified by `uv run pip-audit` that
cannot be immediately resolved. Each entry must be reassessed within 60 days.
No entry may age past 60 days without reassessment (OpenSSF release gate).

## Open vulnerabilities

None at this time. Last `pip-audit` run returned clean on 2026-05-15
after the urllib3 update below.

## Resolved vulnerabilities

- 2026-05-15: bumped urllib3 2.6.3 -> 2.7.0 to clear CVE-2026-44431
  (cross-origin sensitive-header forwarding in proxied low-level
  redirects) and CVE-2026-44432 (decompression-bomb safeguards bypassed
  on parts of the streaming API). Transitive dependency via `requests`.
  Surfaced by the SBOM workflow during PR #17; pin
  (`urllib3>=2.7.0,<3.0.0`) and `uv.lock` regeneration applied in
  PR #24, commit `8a9d79c`.

## Review schedule

Next review due: 2026-07-14

To re-audit: `uv run pip-audit`
