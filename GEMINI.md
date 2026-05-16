# Gemini Notes

This file is the project-level companion to [AGENTS.md](AGENTS.md) and
[CLAUDE.md](CLAUDE.md) for the Gemini family of tools.

## Scope

This repository uses Gemini in two distinct ways:

1. **As a runtime image-generation backend** via `scripts/generate_image.py`,
   which calls the `google-genai` SDK against Gemini 2.5 Flash and Gemini 3 Pro
   for image synthesis.
2. **As an optional reviewer model** when contributors use the Gemini CLI or
   the Gemini API for code review and analysis tasks. This is opt-in; no
   workflow depends on it.

## Configuration

Gemini API access requires `GEMINI_API_KEY` in the environment. See
[.env.example](.env.example) for the expected variable. Never commit a real
`.env` file; secrets-management is enforced by the no-em-dash and (planned)
detect-secrets pre-commit hooks.

## Model selection for Gemini

| Task | Model | Why |
| ---- | ----- | --- |
| High-quality image generation | `gemini-3-pro` | Best fidelity, slower |
| Fast iteration on prompts | `gemini-2.5-flash` | Lower latency, lower cost |
| Reviewing PR diffs (opt-in) | `gemini-3-pro` | Long-context reasoning |

## Operational notes

- Image generations write to `output/` (gitignored, except `.gitkeep`).
- Rate limits and quotas are enforced by the API; the script does not implement
  retry/backoff, so contributors should treat transient 429 responses as a
  signal to slow down rather than retry blindly.
- Topaz Labs is invoked optionally for post-processing via the `--enhance`
  flag; see [docs/IMAGE_GENERATION_GUIDE.md](docs/IMAGE_GENERATION_GUIDE.md)
  for the enhancement workflow.
