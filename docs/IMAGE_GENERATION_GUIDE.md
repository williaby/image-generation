# Gemini Image Generation Best Practices Guide

> **Purpose**: Generate high-quality, consistent images for documentation and presentations
> **Models**: Gemini 3.1 Flash Image (Nano Banana 2, default) and Gemini 3 Pro Image Preview (Nano Banana Pro)
> **Last Updated**: 2026-05-15

## Overview

This repo supports three Gemini image models. **Nano Banana 2** (`flash-2`,
`gemini-3.1-flash-image-preview`) is the default: it delivers Pro-tier reasoning
and fidelity at Flash latency, supports 14 aspect ratios (including ultra-wide
21:9 and ultra-tall 1:8), and adds a 512px resolution tier for rapid iteration.
**Nano Banana Pro** (`pro`, `gemini-3-pro-image-preview`) remains the choice when
you need the highest-fidelity text rendering on detailed technical diagrams.
Both use a "thinking process" to reason through complex prompts; on flash-2 you
can tune that with `--thinking minimal|high`.

### When to pick which model

| Use case | Model | Notes |
|----------|-------|-------|
| Default / general images | `flash-2` | Pro-quality at Flash speed and cost |
| High-volume iteration | `flash-2 --size 512` | New 0.5K tier minimizes latency |
| Ultra-wide / ultra-tall layouts | `flash-2` | Only flash-2 supports 21:9, 8:1, 4:1, 1:8, 1:4 |
| Highest-fidelity text in diagrams | `pro` | Best for L1-L4 network diagrams |
| Lowest cost, no aspect/size control | `flash` | Legacy 2.5 model |

## Quick Start

```bash
# Install dependency
pip install google-genai

# Set API key
export GEMINI_API_KEY='your-api-key'

# Generate a network diagram
python scripts/generate_image.py \
  "Professional network architecture diagram showing a homelab with firewall, switch, and VLANs" \
  --aspect 16:9 \
  --size 2K \
  -o network_diagram.png
```

## Script Capabilities

The `scripts/generate_image.py` script supports:

- **Draft-Then-Finalize Workflow**: `--draft-mode` picks the smallest tier the model supports (512 on flash-2, 1K elsewhere) for fast iteration; `--finalize` upscales to high-res
- **Thinking Process Visibility**: `--verbose` flag shows reasoning steps
- **Thought Image Saving**: `--save-thoughts` captures intermediate refinement images
- **Aspect Ratio Control**: flash-2 supports 14 ratios (1:1, 1:4, 1:8, 2:3,
  3:2, 3:4, 4:1, 4:3, 4:5, 5:4, 8:1, 9:16, 16:9, 21:9); pro supports 5
  (1:1, 3:4, 4:3, 9:16, 16:9)
- **Resolution Control**: 1K (draft), 2K (standard), 4K (premium)
- **Reference Images**: Up to 14 reference images for style/composition consistency
- **Google Search Grounding**: `--search` for factually accurate real-world content
- **Multi-Part Story Generation**: `--story-parts N` for visual sequences
- **Image Editing**: Use `-r reference.png` to edit existing images

## Prompt Engineering

### Core Principle: Brief Like a Human Artist

**DON'T**: Use keyword spam ("4k, trending on artstation, masterpiece, ultra detailed")
**DO**: Write natural creative briefs ("Create a technical network diagram showing...")

### Recommended Prompt Structure

```text
[Action Verb] + [Subject] + [Composition/Layout] + [Style] + [Technical Specs] + [Constraints]
```

**Example for Network Diagram**:

```text
Create a professional network architecture diagram for a homelab.
Show an OPNsense firewall at the top connected to a MikroTik switch.
The switch connects to 7 color-coded VLANs arranged in a clean grid.
Use a modern flat design with subtle gradients.
Include icons for firewall, switch, server, and WiFi access points.
Use a professional color scheme with blue for trusted, orange for IoT, red for isolated.
```

## Network Diagram Best Practices

### Essential Elements for Network Diagrams

1. **Layout Type**: Specify "hierarchical," "radial," "grid," or "flow-based"
2. **Drawing Style**: "technical diagram," "flat design," "blueprint," "whiteboard sketch"
3. **Icon Style**: "simple icons," "detailed icons," "geometric shapes"
4. **Color Coding**: Explicitly define what colors mean (security zones, VLANs, etc.)
5. **Labels**: "clean labels," "minimal text," "detailed annotations"
6. **Legend**: "include legend," "color key in corner"

### Example: Homelab Network Diagram

```bash
python scripts/generate_image.py \
  "Create a professional network architecture diagram for Byron's homelab.

   Layout: Hierarchical with Internet at top, firewall below, then switch, then VLANs.

   Components:
   - Cloud icon for Internet (top)
   - OPNsense firewall box with security badge
   - MikroTik switch in center
   - 7 VLAN segments below, color-coded:
     * Blue: LAN (Infrastructure) - servers, APs
     * Indigo: VLAN 10 (Control) - management
     * Green: VLAN 20 (Secure) - workstations
     * Purple: VLAN 30 (Home) - entertainment
     * Red: VLAN 40 (Guest) - isolated
     * Orange: VLAN 50 (IoT) - smart devices
     * Pink: VLAN 60 (Cameras) - no internet

   Style: Modern flat design with clean lines, subtle shadows, rounded rectangles.
   Include small icons for each device type.
   Add a color-coded legend in the bottom right." \
  --model pro \
  --aspect 16:9 \
  --size 2K \
  -o homelab_architecture.png
```

## Draft-Then-Finalize Workflow

For most images, use this cost-effective workflow:

### Step 1: Generate Draft (1K, fast, cheap)

```bash
python scripts/generate_image.py \
  "Network diagram showing VLAN segmentation" \
  --draft-mode \
  -o network_draft.png
```

### Step 2: Iterate on Draft

```bash
python scripts/generate_image.py \
  "Make the firewall larger and add security icons" \
  -r output/drafts/network_draft.png \
  --draft-mode \
  -o network_draft_v2.png
```

### Step 3: Finalize at Higher Resolution

```bash
python scripts/generate_image.py \
  --finalize output/drafts/network_draft_v2.png \
  --size 2K \
  -o network_final.png
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

## Multi-Part Sequences

Generate consistent visual sequences:

```bash
python scripts/generate_image.py \
  "A 4-part visual evolution of network security: from basic to enterprise" \
  --story-parts 4 \
  --aspect 16:9 \
  --size 2K \
  -o security_evolution
```

Output:

- `security_evolution_part1.png` - Opening scene
- `security_evolution_part2.png` - Development
- `security_evolution_part3.png` - Development
- `security_evolution_part4.png` - Final state

## Using Reference Images

### Style Transfer

```bash
python scripts/generate_image.py \
  "Create a VLAN segmentation diagram in the exact same visual style" \
  -r existing_diagram.png \
  --model pro \
  -o new_diagram.png
```

### Iterative Refinement (80% Rule)

When an image is 80% correct, use targeted edits:

```bash
python scripts/generate_image.py \
  "Change the color scheme to navy blue and add a title: 'HOMELAB NETWORK'" \
  -r diagram_v1.png \
  --model pro \
  -o diagram_v2.png
```

## Common Pitfalls and Solutions

### Issue: Text is Blurry or Misspelled

- Use 2K or higher resolution
- Explicitly request "clear, legible text"
- Spell out critical text exactly

### Issue: Inconsistent Style Across Images

- Generate first image with detailed style specs
- Use that image as reference for all subsequent images
- Maintain consistent aspect ratio and resolution

### Issue: Diagram Looks Cluttered

- Request "clean, organized layout with clear visual separation"
- Break complex diagrams into multiple simpler ones
- Specify information hierarchy

### Issue: File Extension Mismatch (CRITICAL)

**Problem**: Gemini API may return JPEG data even when PNG is expected. If the file is saved with `.png` extension but contains JPEG data, tools that read the file by extension (including Claude Code) will fail with MIME type mismatch errors. This can crash Claude Code sessions.

**Solution** (Implemented in generate_image.py):

- The script now detects actual image format from magic bytes
- Automatically corrects file extension to match actual format
- Warns when API MIME type doesn't match detected format

**If you encounter a mismatch with existing files**:

```bash
# Check actual file format
file output/*.png

# If it shows "JPEG image data" for a .png file, rename it:
mv image.png image.jpg

# Update any references in PROMPTS.md
```

**Prevention**:

- Always use the updated `generate_image.py` which auto-detects format
- If using Gemini API directly, check magic bytes before saving

## Directory Structure

Generated images are automatically organized:

```text
output/
├── drafts/             # Draft images (1K, temporary)
│   └── draft_*.png
├── finals/             # Final images (2K/4K, production)
│   └── *_final.png
└── *.signature.bin     # Thought signatures for multi-turn

examples/
└── PROMPTS.md          # Registry of all images with prompts
```

## Command Reference

```bash
# Full help
python scripts/generate_image.py --help

# Quick reference
--model flash-2          # Nano Banana 2 / Gemini 3.1 Flash Image (default)
--model pro              # Nano Banana Pro / Gemini 3 Pro (highest quality)
--model flash            # Gemini 2.5 Flash (legacy)
--aspect 16:9            # Aspect ratio (flash-2: 14 options; pro: 5 options)
--size 2K                # Resolution (flash-2: 512/1K/2K/4K; pro: 1K/2K/4K)
--thinking high          # flash-2 thinking level: minimal | high
-r image.png             # Reference image (can use multiple)
--draft-mode             # Generate at smallest model-supported tier (512 on flash-2, else 1K)
--finalize draft.png     # Upscale draft to final resolution
--verbose, -v            # Show thinking process
--save-thoughts          # Save intermediate thought images
--search                 # Enable Google Search grounding
--story-parts N          # Generate N-part sequence
-o output.png            # Output filename
--list-models            # Show available models
```

## Integration with PlantUML

For technical documentation, use **both** approaches:

1. **PlantUML** for version-controlled source of truth
2. **Gemini** for polished visuals for presentations/documentation

Example workflow:

```bash
# 1. Create PlantUML diagram for technical reference
# (stored in your docs repository)

# 2. Generate polished visual using PlantUML as reference
python scripts/generate_image.py \
  "Create a professional, polished version of this network diagram" \
  -r path/to/plantuml-output.png \
  --model pro \
  --size 2K \
  -o network_visual.png
```

---

*Based on the Investment Data Operations Gemini image generation system.*
