# Project Claude Code Instructions

> Project scope. Inherits from `~/.claude/CLAUDE.md` (global) and any folder-level
> `CLAUDE.md` files. Only overrides or project-specific guidance lives here.

## Project context

This repository provides image-generation tooling built on Google's Gemini API
with optional Topaz Labs enhancement. The primary surface area is a single
script (`scripts/generate_image.py`) plus a small set of agent definitions
under `agents/`. There is no HTTP server, no multi-tenant boundary, and no
upload surface.

When asked about project goals or design decisions, search this file,
[README.md](README.md), [AGENTS.md](AGENTS.md), [docs/](docs/), and
[SECURITY.md](SECURITY.md) before answering. If the topic is not covered,
state what was searched and answer from training knowledge, prefixed with
`[Not in project docs, answer from training knowledge only]`.

## Model Selection

Use the right model for the task to balance quality and cost:

| Task type | Model | When |
| --- | --- | --- |
| Complex reasoning, planning, architecture | Opus 4.7 | Multi-step decisions, ADRs, deep code review |
| Standard development work | Sonnet 4.6 (default) | Most coding, editing, PR descriptions |
| Read-only exploration | Haiku 4.5 | File scanning, structure mapping, quick lookups |
| Image-generation runtime (not Claude) | Gemini 2.5 Flash / Gemini 3 Pro | See [GEMINI.md](GEMINI.md) |

In subagent configuration, set `model: haiku` for the built-in `Explore` agent
when used in this repo, and default agents that write code to `sonnet` unless
the task requires deep reasoning.

## Response-Aware Development (RAD)

Tag assumptions that could cause production failures using `#CRITICAL`,
`#ASSUME`, and `#EDGE` comment markers paired with `#VERIFY` instructions.
Mandatory categories for this repo:

- **External resources**: Gemini API, Topaz API, network egress.
- **Data integrity**: image file writes to `output/`, prompt files.
- **Security**: API key handling, prompt-injection inputs, environment
  variable loading order.

Example:

```python
# CRITICAL: GEMINI_API_KEY must be present before client construction.
# VERIFY: tests/test_env.py asserts KeyError raised when var absent.
api_key = os.environ["GEMINI_API_KEY"]
```

See the global guide at `~/.claude/docs/response-aware-development.md`
for full tagging syntax and verification workflow.

## Essential commands

- Run the script: `uv run scripts/generate_image.py --prompt "..."`
- Run tests with coverage: `uv run pytest`
- Format and lint: `uv run ruff format scripts tests && uv run ruff check scripts tests --fix`
- Type check: `uv run basedpyright`
- Pre-commit: `pre-commit run --all-files`
- Dependency audit: `uv run pip-audit`

## Cross-references

- Global standards: `~/.claude/CLAUDE.md`
- Python conventions (path-scoped): `~/.claude/.claude/rules/python.md`
- Testing conventions (path-scoped): `~/.claude/.claude/rules/testing.md`
- Git workflow: `~/.claude/.claude/rules/git-workflow.md`
- Pre-commit checklist: `~/.claude/.claude/rules/pre-commit.md`
- Writing rules: `~/.claude/.claude/rules/writing.md`
