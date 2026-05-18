# Gemini Notes

This file is the project-level companion to [AGENTS.md](AGENTS.md) and
[CLAUDE.md](CLAUDE.md) for the Gemini family of tools.

## Scope

This repository uses Gemini in two distinct ways:

1. **As a runtime image-generation backend** via `scripts/generate_image.py`,
   which calls the `google-genai` SDK against Gemini 2.5 Flash, Gemini 3.1
   Flash Image (Nano Banana 2), and Gemini 3 Pro Image for image synthesis.
2. **As an optional reviewer model** when contributors use the Gemini CLI or
   the Gemini API for code review and analysis tasks. This is opt-in; no
   workflow depends on it.

## Configuration

Gemini API access requires `GEMINI_API_KEY` in the environment. See
[.env.example](.env.example) for the expected variable. Never commit a real
`.env` file; secrets-management is enforced by the no-em-dash and (planned)
detect-secrets pre-commit hooks.

## Model selection for Gemini

The script exposes three Gemini image models. Defaults and recommendations
were updated 2026-05 when Google released Nano Banana 2 and made it the
ecosystem default.

| Task | Model (script key) | API model ID | Why |
| ---- | ------------------ | ------------ | --- |
| **Default image generation** | `flash-2` | `gemini-3.1-flash-image-preview` | Pro-quality reasoning at Flash speed/cost; new default since May 2026 |
| Highest-fidelity text in technical diagrams | `pro` | `gemini-3-pro-image-preview` | Best fidelity for L1-L4 network diagrams; slower |
| Legacy fast iteration (no aspect/size control) | `flash` | `gemini-2.5-flash-image` | Lower latency, lower cost; predates the 3.x line |
| Reviewing PR diffs (opt-in) | n/a | `gemini-3-pro` | Long-context reasoning; not image-generation |

Override the default with `--model pro` when generating text-heavy technical
diagrams. The pre-2026-05 default was `pro`; scripts that relied on the
implicit default may want to pin `--model pro` explicitly.

## Operational notes

- Image generations write to `output/` (gitignored, except `.gitkeep`).
- Rate limits and quotas are enforced by the API; the script does not implement
  retry/backoff, so contributors should treat transient 429 responses as a
  signal to slow down rather than retry blindly.
- Topaz Labs is invoked optionally for post-processing via the `--enhance`
  flag; see [docs/IMAGE_GENERATION_GUIDE.md](docs/IMAGE_GENERATION_GUIDE.md)
  for the enhancement workflow.
