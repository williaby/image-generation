# ADR 001: Single-Script Architecture for Image Generation

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: @williaby
- **Supersedes**: none
- **Superseded by**: none

## Context

This repository provides image-generation tooling that calls Google's Gemini
API (and optionally Topaz Labs) from the command line. Early in the project we
considered three packaging shapes:

1. A single script under `scripts/`, invoked directly via `uv run`.
2. A Python package under `src/image_generation/` with an installable CLI
   entry point (`pyproject.toml [project.scripts]`).
3. A long-running HTTP service that exposes generation as an internal API.

## Decision

We adopted option 1: a single script (`scripts/generate_image.py`) plus a
small set of agent definitions under `agents/`. There is no installable
package, no HTTP server, and no daemon.

## Consequences

### Positive

- **Minimal blast radius**: no network listener, no multi-tenant boundary, no
  request handler. The security surface is restricted to the local invoker
  and the outbound API calls.
- **Low operational overhead**: no deployment, no service-account management,
  no uptime SLO. Contributors run the script the same way the maintainer does.
- **Tight feedback loop**: changes ship as one PR touching one script; CI
  validates the script and a single coverage target.

### Negative

- **No PyPI distribution**: external users must clone the repo or copy the
  script. We accept this because the script is a personal utility, not a
  reusable library.
- **Coverage scope is narrow**: `pytest --cov=scripts` covers the only code
  path; adding new modules will require updating `[tool.coverage.run]`.

## Security Considerations

This decision shapes the threat model documented in [SECURITY.md](../../SECURITY.md):

- **Credential surface**: API keys live in the invoker's environment or a
  local `.env` file. No service-account, no token broker, no key rotation
  workflow is required. The mitigation is to never commit `.env` (enforced
  by `.gitignore` and pre-commit secret scanning via TruffleHog) and to
  prefer ephemeral session-scoped keys when feasible.
- **Input handling**: prompts and file paths are CLI-supplied by the script
  invoker. There is no remote attacker model for prompt injection at the
  script boundary, but downstream consumers of generated images or
  `PROMPTS.md` files should treat all output as untrusted markdown/binary.
- **Outbound trust**: the script trusts Google Gemini and Topaz Labs to
  return well-formed responses. Topaz responses are downloaded as image
  files; the script does not execute or unmarshal them beyond writing to
  disk.
- **Supply chain**: dependencies are pinned via `uv.lock` and audited by
  `pip-audit` in CI; GitHub Actions are SHA-pinned and Renovate (via PR #11)
  will maintain pin freshness.
- **No persistence boundary**: there is no database, no session store, and
  no PII surface. The script does not run inside CI; CI only validates code
  quality and runs unit tests against mocked clients.

## Alternatives considered

- **Option 2 (installable package)** was rejected because the single-user,
  single-script use case does not justify the maintenance cost of a
  versioned package with semantic-versioned releases.
- **Option 3 (HTTP service)** was rejected because exposing image generation
  as a network service would introduce authentication, rate-limiting, abuse
  prevention, and key-broker requirements that are out of scope for a
  personal utility.

## References

- [README.md](../../README.md) for the user-facing description.
- [SECURITY.md](../../SECURITY.md) for the full security posture.
- [docs/IMAGE_GENERATION_GUIDE.md](../IMAGE_GENERATION_GUIDE.md) for
  prompt-engineering workflows.
