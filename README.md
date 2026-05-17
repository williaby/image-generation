# Image Generation Tools

A collection of scripts and agents for generating images using Google's Gemini API, with Claude-based validation for technical diagrams.

## Overview

This repository contains:

- **`scripts/generate_image.py`** - Full-featured Python script for generating images using Gemini 2.5 Flash, Gemini 3.1 Flash Image (Nano Banana 2), and Gemini 3 Pro
- **`docs/IMAGE_GENERATION_GUIDE.md`** - Best practices guide for prompt engineering and workflows
- **`agents/diagram-specialist.md`** - Claude agent configuration for validating AI-generated technical diagrams

## Features

### Gemini Image Generation Script

The `generate_image.py` script supports:

- **Three Gemini Models**:
  - `flash`: Gemini 2.5 Flash (legacy fast generation)
  - `flash-2`: **Nano Banana 2** / Gemini 3.1 Flash Image - *default* (Pro-quality at Flash speed, 14 aspect ratios, 512/1K/2K/4K, configurable thinking, Search grounding)
  - `pro`: Nano Banana Pro / Gemini 3 Pro (highest quality, best text rendering, Google Search grounding)

- **Draft-Then-Finalize Workflow**: `--draft-mode` generates at 1K for fast iteration, then `--finalize` upscales to 2K/4K for production
- **Aspect Ratio Control**: flash-2 supports 14 ratios (1:1, 1:4, 1:8, 2:3, 3:2, 3:4, 4:1, 4:3, 4:5, 5:4, 8:1, 9:16, 16:9, 21:9); pro supports 1:1, 3:4, 4:3, 9:16, 16:9
- **Resolution Options**: 512 (flash-2 only), 1K, 2K, 4K
- **Thinking Control** (flash-2): `--thinking minimal|high` to trade latency for quality
- **Reference Image Support**: Edit existing images or transfer styles
- **Multi-Part Story Generation**: Automatic continuity for visual sequences
- **Thinking Mode**: View intermediate reasoning steps
- **Automatic Format Detection**: Fixes JPEG/PNG mismatches from API

### Claude Diagram Specialist Agent

A specialized Claude agent for:

- Creating technical network diagrams (L1-L4 hierarchy)
- Validating AI-generated images against PlantUML source
- Ensuring text accuracy and visual consistency
- Detecting file format issues that cause session crashes

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/williaby/image-generation.git
cd image-generation

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependency
pip install google-genai
```

### Set API Key

```bash
export GEMINI_API_KEY='your-api-key'
```

Or create a `.env` file in the repository root:

```env
GEMINI_API_KEY=your-api-key
```

### Generate an Image

```bash
# Basic generation
python scripts/generate_image.py "A futuristic city at sunset"

# With options
python scripts/generate_image.py \
  "Professional network architecture diagram" \
  --model pro \
  --aspect 16:9 \
  --size 2K \
  -o diagram.png
```

### Draft-Then-Finalize Workflow

```bash
# Step 1: Generate draft (1K, fast, cheap)
python scripts/generate_image.py \
  "Network diagram showing VLAN segmentation" \
  --draft-mode \
  -o network_draft.png

# Step 2: Iterate on draft
python scripts/generate_image.py \
  "Make the firewall larger and add security icons" \
  -r output/drafts/network_draft.png \
  --draft-mode \
  -o network_draft_v2.png

# Step 3: Finalize at higher resolution
python scripts/generate_image.py \
  --finalize output/drafts/network_draft_v2.png \
  --size 2K \
  -o network_final.png
```

## Command Reference

```bash
python scripts/generate_image.py --help

# Key options:
--model flash-2          # Nano Banana 2 / Gemini 3.1 Flash Image (default)
--model pro              # Nano Banana Pro / Gemini 3 Pro (highest quality)
--model flash            # Gemini 2.5 Flash (legacy)
--aspect 16:9            # Aspect ratio (flash-2 supports 14; pro supports 5)
--size 2K                # Resolution (512 flash-2 only; 1K, 2K, 4K)
--thinking high          # Thinking level for flash-2 (minimal|high)
-r image.png             # Reference image (can use multiple)
--draft-mode             # Generate at 1K for iteration
--finalize draft.png     # Upscale draft to final resolution
--verbose, -v            # Show thinking process
--save-thoughts          # Save intermediate thought images
--search                 # Enable Google Search grounding
--story-parts N          # Generate N-part sequence
-o output.png            # Output filename
--list-models            # Show available models
```

## Project Structure

```
image-generation/
├── README.md                           # This file
├── scripts/
│   └── generate_image.py               # Main image generation script
├── docs/
│   └── IMAGE_GENERATION_GUIDE.md       # Best practices and examples
├── agents/
│   └── diagram-specialist.md           # Claude agent for validation
├── output/                             # Generated images (gitignored)
│   ├── drafts/                         # Draft images (1K)
│   └── finals/                         # Final images (2K/4K)
├── examples/                           # Example prompts and outputs
│   └── PROMPTS.md                      # Registry of generation prompts
├── requirements.txt                    # Python dependencies
└── .gitignore
```

## Resolution Guidelines

| Use Case | Resolution | Aspect Ratio |
|----------|------------|--------------|
| Draft/iteration | 1K | 16:9 |
| Documentation | 2K | 16:9 |
| Large prints/posters | 4K | 16:9 |
| Square icons/badges | 2K | 1:1 |
| Vertical infographics | 2K | 9:16 |

### File Size Expectations

| Configuration | Dimensions | File Size |
|---------------|-----------|-----------|
| 16:9, 1K | 1408 x 768 | ~1 MB |
| 16:9, 2K | 2752 x 1536 | ~3 MB |
| 16:9, 4K | 5504 x 3072 | ~7 MB |

## Prompt Engineering Tips

### Core Principle: Brief Like a Human Artist

**DON'T**: Use keyword spam ("4k, trending on artstation, masterpiece")

**DO**: Write natural creative briefs ("Create a technical network diagram showing...")

### Recommended Prompt Structure

```text
[Action Verb] + [Subject] + [Composition/Layout] + [Style] + [Technical Specs] + [Constraints]
```

### Example Prompt

```text
Create a professional network architecture diagram for a homelab.
Show an OPNsense firewall at the top connected to a MikroTik switch.
The switch connects to 7 color-coded VLANs arranged in a clean grid.
Use a modern flat design with subtle gradients.
Include icons for firewall, switch, server, and WiFi access points.
Use a professional color scheme with blue for trusted, orange for IoT, red for isolated.
```

## Claude Agent Integration

The `agents/diagram-specialist.md` can be used with Claude Code for:

1. **Creating Diagrams**: Generate PlantUML source and Gemini visuals
2. **Validating Images**: Ensure AI-generated images match source diagrams
3. **Format Checking**: Detect JPEG/PNG mismatches that cause issues

### Usage with Claude Code

```text
/diagram create l1 "Homelab external connections with Cloudflare and VPN"
/diagram validate l1-context-gemini.jpg
/diagram check
```

## Common Issues

### File Extension Mismatch (CRITICAL)

**Problem**: Gemini API may return JPEG data even when PNG is expected. This can crash tools that read files by extension.

**Solution**: The script automatically detects actual format from magic bytes and corrects extensions.

**Manual Check**:
```bash
file output/*.png  # Verify format matches extension
```

### Text is Blurry or Misspelled

- Use 2K or higher resolution
- Explicitly request "clear, legible text"
- Spell out critical text exactly in prompt

### Inconsistent Style Across Images

- Generate first image with detailed style specs
- Use that image as reference for subsequent images
- Maintain consistent aspect ratio and resolution

## License

MIT License - See [LICENSE](LICENSE) for details.

## Credits

Originally developed for homelab infrastructure documentation at [homelab-infra](https://github.com/ByronWilliamsCPA/homelab-infra).
