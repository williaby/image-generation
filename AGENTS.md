# Agents

This repository hosts a small set of Claude/Gemini agent definitions used
alongside the image-generation scripts. Each agent is a focused, single-purpose
configuration intended to be invoked from a parent Claude Code session or via
the Claude Agent SDK.

## Active agents

| Agent | File | Purpose |
| ----- | ---- | ------- |
| diagram-specialist | [agents/diagram-specialist.md](agents/diagram-specialist.md) | Validates AI-generated technical diagrams for clarity, accuracy, and visual quality |

## Conventions

- Agent definition files live under `agents/` as Markdown with YAML frontmatter.
- Each agent file describes its purpose, allowed tools, model selection, and
  invocation conventions.
- Agents are intended to be invoked, not edited at runtime. Pull requests are
  the only path to update an agent.

## Adding a new agent

1. Create `agents/<agent-name>.md` with YAML frontmatter (`name`, `description`,
   `model`, `tools`).
2. Document the agent's purpose, the inputs it expects, and the outputs it
   produces.
3. Add a row to the table above.
4. Open a PR; the standard CI gates will validate the file.

## Related documentation

- [CLAUDE.md](CLAUDE.md) for project-level Claude Code instructions.
- [docs/IMAGE_GENERATION_GUIDE.md](docs/IMAGE_GENERATION_GUIDE.md) for the
  primary script's prompt-engineering guidance.
