#!/usr/bin/env python3
"""
Generate images using Google Gemini's image generation models, with optional
Topaz Labs post-processing for professional-grade enhancement and upscaling.

Supports:
- Nano Banana (gemini-2.5-flash-preview-image-generation) - Fast generation
- Nano Banana Pro (gemini-3-pro-image-preview) - 4K, better text rendering, Google Search grounding
- Topaz Labs API - Post-generation enhancement: upscaling, denoising, sharpening, face enhancement

Environment Variables:
    GEMINI_API_KEY: Required. Your Google AI API key.
    TOPAZ_API_KEY:  Required for Topaz features. Get from developer.topazlabs.com.

Usage:
    # Basic text-to-image generation
    python generate_image.py "A futuristic city at sunset"

    # With output filename
    python generate_image.py "A futuristic city at sunset" -o city.png

    # Using Nano Banana Pro model (Gemini 3)
    python generate_image.py "A futuristic city at sunset" --model pro

    # With reference image for editing/style
    python generate_image.py "Make this building taller" -r reference.png

    # With aspect ratio and resolution (pro model only)
    python generate_image.py "A landscape" --model pro --aspect 16:9 --size 4K

    # Show thinking process and save thought images (pro model only)
    python generate_image.py "Complex blueprint" --save-thoughts --verbose

    # Multi-part story generation
    python generate_image.py "A 3-part story about data governance" --story-parts 3 -o story

    # List available models
    python generate_image.py --list-models

    # Enhance an existing image with Topaz (standalone post-processing)
    python generate_image.py --enhance photo.png
    python generate_image.py --enhance photo.png --topaz-model "High Fidelity V2"
    python generate_image.py --enhance photo.png --topaz-denoise 0.5 --topaz-sharpen 0.3
    python generate_image.py --enhance photo.png --topaz-face-enhance -o portrait_enhanced.png

    # Generate then immediately enhance with Topaz
    python generate_image.py "A network diagram" --topaz
    python generate_image.py "A network diagram" --topaz --topaz-model "Text Refine"

    # Use Topaz to finalize a draft (instead of re-generating via Gemini)
    python generate_image.py --finalize draft.png --topaz -o final.png
"""

import argparse
import base64
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

GENAI_AVAILABLE = True
try:
    from google import genai
    from google.genai import types
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None

REQUESTS_AVAILABLE = True
try:
    import requests
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None


# Model configurations
# Note: Actual API model IDs are gemini-2.5-flash-image and gemini-3-pro-image-preview
MODELS = {
    "flash": {
        "id": "gemini-2.5-flash-image",
        "name": "Nano Banana (Gemini 2.5 Flash)",
        "description": "Fast image generation model",
        "supports_image_config": False,
    },
    "pro": {
        "id": "gemini-3-pro-image-preview",
        "name": "Nano Banana Pro (Gemini 3 Pro)",
        "description": "4K resolution, better text rendering, Google Search grounding, thinking mode",
        "supports_image_config": True,
    },
}

DEFAULT_MODEL = "pro"  # Default to Gemini 3 Pro for best quality

# Valid aspect ratios for pro model
ASPECT_RATIOS = ["1:1", "3:4", "4:3", "9:16", "16:9"]

# Valid image sizes for pro model
IMAGE_SIZES = ["1K", "2K", "4K"]

# Topaz Labs API base URL
TOPAZ_BASE_URL = "https://api.topazlabs.com/image/v1"

# Topaz model registry: "enhance" -> /enhance/async; "enhance-gen" -> /enhance-gen/async
# Generative models (Wonder, Bloom) cost ~6-12x more credits than precision models.
TOPAZ_MODELS = {
    # Gigapixel precision upscaling (24 MP per credit)
    "Standard V2": {
        "endpoint": "enhance/async",
        "description": "Precision upscaling, best for most images",
    },
    "High Fidelity V2": {
        "endpoint": "enhance/async",
        "description": "Highest quality, preserves fine detail",
    },
    "Low Resolution V2": {
        "endpoint": "enhance/async",
        "description": "Optimized for very low-resolution sources",
    },
    "CGI": {
        "endpoint": "enhance/async",
        "description": "Optimized for CGI and rendered imagery",
    },
    "Text Refine": {
        "endpoint": "enhance/async",
        "description": "Preserves and sharpens text in diagrams",
    },
    "Detail Faces": {
        "endpoint": "enhance/async",
        "description": "Enhances facial clarity",
    },
    "Recover Faces": {
        "endpoint": "enhance/async",
        "description": "Restores damaged or degraded faces",
    },
    "Transparency Upscale": {
        "endpoint": "enhance/async",
        "description": "Upscales images with alpha transparency",
    },
    # Generative upscaling (4 MP per credit; significantly more expensive)
    "Wonder": {
        "endpoint": "enhance-gen/async",
        "description": "Generative upscaling, adds intelligent detail",
    },
    "Wonder 2": {
        "endpoint": "enhance-gen/async",
        "description": "Improved generative upscaling",
    },
    "Standard Max": {
        "endpoint": "enhance-gen/async",
        "description": "Maximum quality generative upscaling",
    },
    "Recover 3": {
        "endpoint": "enhance-gen/async",
        "description": "Advanced recovery with generation",
    },
    "Redefine": {
        "endpoint": "enhance-gen/async",
        "description": "Creative reinterpretation with upscaling",
    },
    "Bloom": {
        "endpoint": "enhance-gen/async",
        "description": "Creative upscaling for AI-generated art",
    },
}

DEFAULT_TOPAZ_MODEL = "Standard V2"


def _load_api_key(env_var: str) -> str | None:
    """Load an API key from the environment or the repo-root .env file."""
    api_key = os.environ.get(env_var)
    if not api_key:
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            try:
                with open(env_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(f"{env_var}="):
                            api_key = (
                                line.split("=", 1)[1].strip().strip('"').strip("'")
                            )
                            break
            except OSError:
                pass
    return api_key or None


def get_api_key() -> str:
    """Get the Gemini API key from environment."""
    api_key = _load_api_key("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Set it with: export GEMINI_API_KEY='your-api-key'")
        print("Or create a .env file in the repository root.")
        sys.exit(1)
    return api_key


def get_topaz_api_key() -> str | None:
    """Get the Topaz Labs API key from environment or .env file."""
    api_key = _load_api_key("TOPAZ_API_KEY")
    if not api_key:
        print("Error: TOPAZ_API_KEY not set.", file=sys.stderr)
        print("Set it with: export TOPAZ_API_KEY='your-api-key'", file=sys.stderr)
        print("Or add it to the .env file in the repository root.", file=sys.stderr)
        print("Get a key at: https://developer.topazlabs.com", file=sys.stderr)
    return api_key


def topaz_enhance_image(
    input_path: Path,
    output_path: Path | None = None,
    model: str = DEFAULT_TOPAZ_MODEL,
    output_format: str = "png",
    sharpen: float | None = None,
    denoise: float | None = None,
    face_enhancement: bool = False,
    face_enhancement_strength: float | None = None,
    verbose: bool = False,
) -> Path | None:
    """Enhance an image using the Topaz Labs API (async job with polling).

    Precision models (Gigapixel family): 24 MP per credit.
    Generative models (Wonder, Bloom): 2-4 MP per credit, significantly more expensive.
    """
    if not REQUESTS_AVAILABLE:
        print("Error: 'requests' package is required for Topaz enhancement.")
        print("Install with: pip install requests")
        return None

    api_key = get_topaz_api_key()
    if not api_key:
        return None

    if not input_path.exists():
        print(f"Error: Input image not found: {input_path}")
        return None

    model_config = TOPAZ_MODELS.get(model)
    if model_config is None:
        print(f"Error: Unknown Topaz model '{model}'.")
        print(f"Available: {', '.join(TOPAZ_MODELS)}")
        return None

    valid_formats = ("png", "jpg", "jpeg", "webp")
    if output_format not in valid_formats:
        print(
            f"Error: output_format must be one of {valid_formats}, got '{output_format}'"
        )
        return None

    def _check_strength(name: str, value: float | None) -> bool:
        if value is not None and not (0.0 <= value <= 1.0):
            print(f"Error: {name} must be between 0.0 and 1.0, got {value}")
            return False
        return True

    if not all(
        [
            _check_strength("sharpen", sharpen),
            _check_strength("denoise", denoise),
            _check_strength("face_enhancement_strength", face_enhancement_strength),
        ]
    ):
        return None

    if face_enhancement_strength is not None and not face_enhancement:
        print(
            "Error: --topaz-face-strength requires --topaz-face-enhance to be set.",
            file=sys.stderr,
        )
        return None

    # #CRITICAL: API key sent as X-API-KEY header; never log headers containing this value.
    # #ASSUME: Topaz API at TOPAZ_BASE_URL accepts multipart/form-data for all model endpoints.
    _TOPAZ_DOWNLOAD_HOSTS = frozenset({"api.topazlabs.com", "cdn.topazlabs.com"})
    headers = {"X-API-KEY": api_key}
    endpoint_url = f"{TOPAZ_BASE_URL}/{model_config['endpoint']}"

    if verbose:
        print(f"Topaz enhance: {input_path.name} [{model}]")

    data: dict = {"model": model, "output_format": output_format}
    if sharpen is not None:
        data["sharpen"] = sharpen
    if denoise is not None:
        data["denoise"] = denoise
    if face_enhancement:
        data["face_enhancement"] = "true"
        if face_enhancement_strength is not None:
            data["face_enhancement_strength"] = face_enhancement_strength

    # Submit async job
    try:
        with open(input_path, "rb") as f:
            resp = requests.post(
                endpoint_url,
                headers=headers,
                data=data,
                files={"image": f},
                timeout=30,
            )
        resp.raise_for_status()
    except Exception as e:
        print(f"Error submitting Topaz job: {e}", file=sys.stderr)
        return None

    process_id = resp.json().get("process_id")
    if not process_id:
        print(
            f"Error: Topaz API returned unexpected response (missing process_id): {resp.text[:200]}",
            file=sys.stderr,
        )
        return None
    if verbose:
        print(f"  Job submitted: {process_id}")

    # #ASSUME: job completes within 25 poll iterations; max wall time ~5 minutes.
    # #EDGE: sustained 429 responses exhaust iterations and are reported as "did not complete".
    wait = 2.0
    for _ in range(25):
        time.sleep(wait)
        try:
            status_resp = requests.get(
                f"{TOPAZ_BASE_URL}/status/{process_id}",
                headers=headers,
                timeout=15,
            )
            if status_resp.status_code == 429:
                wait = min(wait * 2, 30)
                continue
            status_resp.raise_for_status()
            status = status_resp.json().get("status", "")
            if verbose:
                print(f"  Status: {status}")
            if status == "Completed":
                break
            if status in ("Failed", "Error"):
                print(
                    f"Error: Topaz job {process_id} failed (status: {status})",
                    file=sys.stderr,
                )
                return None
        except Exception as e:
            print(
                f"Error polling Topaz status for job {process_id}: {e}", file=sys.stderr
            )
            return None
        wait = min(wait * 1.5, 15)
    else:
        print(
            f"Error: Topaz job {process_id} did not complete within the polling limit.",
            file=sys.stderr,
        )
        return None

    # Get download URL
    try:
        dl_resp = requests.get(
            f"{TOPAZ_BASE_URL}/download/{process_id}",
            headers=headers,
            timeout=15,
        )
        dl_resp.raise_for_status()
    except Exception as e:
        print(
            f"Error getting Topaz download URL for job {process_id}: {e}",
            file=sys.stderr,
        )
        return None

    download_url = dl_resp.json().get("url")
    if not download_url:
        print(
            f"Error: Topaz download response missing URL for job {process_id}",
            file=sys.stderr,
        )
        return None

    # #ASSUME: download_url is a valid HTTPS URL served from api.topazlabs.com or its CDN.
    _parsed = urlparse(download_url)
    if _parsed.scheme != "https" or _parsed.hostname not in _TOPAZ_DOWNLOAD_HOSTS:
        print(
            f"Error: Topaz returned unexpected download URL: {download_url!r}",
            file=sys.stderr,
        )
        return None

    # Download the enhanced image
    try:
        img_resp = requests.get(download_url, timeout=120, allow_redirects=False)
        img_resp.raise_for_status()
        image_data = img_resp.content
    except Exception as e:
        print(
            f"Error downloading Topaz result for job {process_id}: {e}", file=sys.stderr
        )
        return None

    if not image_data:
        print(
            f"Error: Topaz download returned empty response for job {process_id}",
            file=sys.stderr,
        )
        return None

    # Resolve output path
    if output_path is None:
        detected_ext = detect_image_format(image_data)
        output_path = input_path.parent / f"{input_path.stem}_topaz{detected_ext}"
    else:
        # Correct extension if needed
        detected_ext = detect_image_format(image_data)
        user_ext = output_path.suffix.lower()
        if user_ext not in (".png", ".jpg", ".jpeg", ".webp"):
            output_path = output_path.with_suffix(detected_ext)
        elif user_ext != detected_ext:
            print(
                f"Warning: specified extension {user_ext!r} does not match Topaz result"
                f" {detected_ext!r}; correcting.",
                file=sys.stderr,
            )
            output_path = output_path.with_suffix(detected_ext)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(image_data)
    except OSError as e:
        print(f"Error writing output file {output_path}: {e}", file=sys.stderr)
        return None

    print(f"Topaz result saved to: {output_path}")
    return output_path


def list_topaz_models() -> None:
    """Print available Topaz enhancement models."""
    print("Available Topaz models:\n")
    print("  Precision upscaling (24 MP/credit):")
    for name, cfg in TOPAZ_MODELS.items():
        if cfg["endpoint"] == "enhance/async":
            print(f"    {name:<22}  {cfg['description']}")
    print("\n  Generative upscaling (2-4 MP/credit, more expensive):")
    for name, cfg in TOPAZ_MODELS.items():
        if cfg["endpoint"] == "enhance-gen/async":
            print(f"    {name:<22}  {cfg['description']}")
    print()


def detect_image_format(data: bytes) -> str:
    """Detect actual image format from magic bytes.

    Returns file extension (with dot) based on file signature.
    """
    # Check magic bytes (file signatures)
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    # Default to PNG if unknown
    return ".png"


def get_extension_for_mime(mime_type: str) -> str:
    """Get file extension for a MIME type."""
    mime_to_ext = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }
    return mime_to_ext.get(mime_type, ".png")


def load_image_as_base64(image_path: Path) -> tuple[str, str]:
    """Load an image file and return base64 data and mime type.

    Detects actual image format from file contents (magic bytes),
    not from file extension, to avoid MIME type mismatches.
    """
    with open(image_path, "rb") as f:
        raw_data = f.read()

    # Detect actual format from magic bytes
    detected_ext = detect_image_format(raw_data)
    ext_to_mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = ext_to_mime.get(detected_ext, "image/png")

    # Warn if extension doesn't match actual format
    suffix = image_path.suffix.lower()
    if suffix != detected_ext and suffix in ext_to_mime:
        print(
            f"Warning: File {image_path.name} has extension {suffix} but contains {detected_ext} data"
        )
        print(f"  Using detected MIME type: {mime_type}")

    data = base64.standard_b64encode(raw_data).decode("utf-8")
    return data, mime_type


def document_image_prompt(
    image_path: Path,
    prompt: str,
    model_key: str,
    aspect_ratio: str | None,
    image_size: str | None,
    reference_images: list[Path] | None = None,
    is_draft: bool = False,
    is_final: bool = False,
    purpose: str | None = None,
) -> None:
    """
    Document image generation in PROMPTS.md registry.
    Matches existing format from web UI documentation.
    """
    # Prompts file is in examples directory
    prompts_file = Path(__file__).parent.parent / "examples" / "PROMPTS.md"

    # Read existing content
    if prompts_file.exists():
        with open(prompts_file, encoding="utf-8") as f:
            content = f.read()
    else:
        content = """# AI-Generated Image Registry

This folder contains images generated by AI models. Each entry documents the model, prompt, and generation parameters.

## Image Registry

| Filename | Model | Date | Prompt | Parameters |
|----------|-------|------|--------|------------|

## Detailed Prompts

"""

    # Skip if already documented
    if f"### {image_path.name}" in content:
        return

    # Prepare data
    timestamp = datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")
    model_name = MODELS[model_key]["name"]
    params_str = f"--aspect {aspect_ratio or '16:9'} --size {image_size or '2K'}"

    # Relative path from output root
    try:
        rel_path = image_path.relative_to(Path(__file__).parent.parent / "output")
    except ValueError:
        rel_path = image_path.name

    # Short description for table
    prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt

    # Add to registry table
    table_end = content.find("\n## Detailed Prompts")
    if table_end > 0:
        new_row = f"| {rel_path} | {model_name} | {timestamp} | {prompt_short} | {params_str} |\n"
        content = content[:table_end] + new_row + content[table_end:]

    # Build detailed entry matching web UI format
    detailed_entry = f"""
### {image_path.name}

- **Model**: {model_name} ({MODELS[model_key]["id"]})
- **Date Generated**: {timestamp}
"""

    if reference_images:
        detailed_entry += "- **Attachments**:\n"
        for ref in reference_images:
            detailed_entry += f"  - {ref.name}\n"

    detailed_entry += f"""- **Prompt**:
```
{prompt}
```
- **Parameters**: {params_str}
"""

    if purpose:
        detailed_entry += f"- **Purpose**: {purpose}\n"

    # Only document final images location (drafts are temporary)
    if is_final:
        detailed_entry += f"- **Location**: `{rel_path}`\n"

    if is_draft:
        detailed_entry += (
            "- **Type**: Draft (temporary, will be removed after finalization)\n"
        )
    elif is_final:
        detailed_entry += "- **Type**: Final (production ready)\n"

    detailed_entry += "\n---\n"

    # Append
    content += detailed_entry

    # Write
    with open(prompts_file, "w", encoding="utf-8") as f:
        f.write(content)

    if is_final:
        print("✓ Documented in PROMPTS.md")


def generate_image(
    prompt: str,
    model_key: str = DEFAULT_MODEL,
    reference_images: list[Path] | None = None,
    output_path: Path | None = None,
    aspect_ratio: str | None = None,
    image_size: str | None = None,
    use_search: bool = False,
    save_thoughts: bool = False,
    verbose: bool = False,
    is_draft: bool = False,
    document_prompt: bool = True,
) -> Path | None:
    """
    Generate an image using Gemini.

    Args:
        prompt: Text description of the image to generate
        model_key: Model to use ('flash' or 'pro')
        reference_images: Optional list of reference images for editing/style
        output_path: Optional output file path
        aspect_ratio: Aspect ratio for pro model (e.g., "16:9", "1:1")
        image_size: Image size for pro model ("1K", "2K", "4K")
        use_search: Enable Google Search grounding (pro model only)
        save_thoughts: Save intermediate thought images (pro model only)
        verbose: Show detailed thinking process and thought signatures

    Returns:
        Path to the generated image, or None on failure
    """
    api_key = get_api_key()

    if model_key not in MODELS:
        print(f"Error: Unknown model '{model_key}'. Use --list-models to see options.")
        return None

    model_config = MODELS[model_key]
    model_id = model_config["id"]

    print(f"Using model: {model_config['name']}")
    print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

    # Initialize client
    client = genai.Client(api_key=api_key)

    # Build the content parts
    contents: list = []

    # Add reference images if provided
    if reference_images:
        for img_path in reference_images:
            if not img_path.exists():
                print(f"Warning: Reference image not found: {img_path}")
                continue

            print(f"Including reference image: {img_path}")
            img_data, mime_type = load_image_as_base64(img_path)
            contents.append(
                types.Part.from_bytes(
                    data=base64.standard_b64decode(img_data),
                    mime_type=mime_type,
                )
            )

    # Add the text prompt
    contents.append(prompt)

    # Build config kwargs
    config_kwargs = {
        "response_modalities": ["IMAGE", "TEXT"],
    }

    # Add image config for pro model
    if model_config.get("supports_image_config"):
        image_config_kwargs = {}
        if aspect_ratio:
            if aspect_ratio not in ASPECT_RATIOS:
                print(
                    f"Warning: Invalid aspect ratio '{aspect_ratio}'. Valid: {ASPECT_RATIOS}"
                )
            else:
                image_config_kwargs["aspect_ratio"] = aspect_ratio
                print(f"Aspect ratio: {aspect_ratio}")
        if image_size:
            if image_size not in IMAGE_SIZES:
                print(
                    f"Warning: Invalid image size '{image_size}'. Valid: {IMAGE_SIZES}"
                )
            else:
                image_config_kwargs["image_size"] = image_size
                print(f"Image size: {image_size}")

        if image_config_kwargs:
            config_kwargs["image_config"] = types.ImageConfig(**image_config_kwargs)

        # Add Google Search grounding if requested
        if use_search:
            config_kwargs["tools"] = [{"google_search": {}}]
            print("Google Search grounding: enabled")

    # Configure generation
    generate_config = types.GenerateContentConfig(**config_kwargs)

    try:
        print("Generating image...")
        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=generate_config,
        )

        # Process response
        if not response.candidates:
            print("Error: No response candidates returned.")
            if hasattr(response, "prompt_feedback"):
                print(f"Feedback: {response.prompt_feedback}")
            return None

        # Track thoughts and final images
        thought_count = 0
        final_image_data = None
        final_mime_type = None
        final_signature = None

        # Process all parts in response
        for part in response.candidates[0].content.parts:
            # Check if this is a thought (intermediate reasoning step)
            is_thought = hasattr(part, "thought") and part.thought

            if is_thought:
                thought_count += 1
                if verbose:
                    print(f"\n[Thought {thought_count}]")

                # Handle thought text
                if part.text is not None:
                    if verbose:
                        print(f"Reasoning: {part.text}")

                # Handle thought image
                if part.inline_data is not None and save_thoughts:
                    thought_data = part.inline_data.data
                    thought_mime = part.inline_data.mime_type
                    thought_ext = ".png" if "png" in thought_mime else ".jpg"

                    # Save thought image
                    if output_path:
                        thought_path = (
                            output_path.parent
                            / f"{output_path.stem}_thought{thought_count}{thought_ext}"
                        )
                    else:
                        timestamp = datetime.now(tz=datetime.UTC).strftime(
                            "%Y%m%d_%H%M%S"
                        )
                        thought_path = Path(
                            f"thought{thought_count}_{timestamp}{thought_ext}"
                        )

                    thought_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(thought_path, "wb") as f:
                        f.write(thought_data)

                    print(f"Thought image {thought_count} saved to: {thought_path}")

            # Non-thought content (final output)
            elif part.inline_data is not None:
                # Final image
                final_image_data = part.inline_data.data
                final_mime_type = part.inline_data.mime_type

                # Extract thought signature if available
                if hasattr(part, "thought_signature") and part.thought_signature:
                    final_signature = part.thought_signature
                    if verbose:
                        print(f"\n[Thought Signature]: {final_signature[:100]}...")

            elif part.text is not None:
                # Final text response
                print(f"\nModel response: {part.text}")

                # Extract thought signature from text part if available
                if hasattr(part, "thought_signature") and part.thought_signature:
                    final_signature = part.thought_signature
                    if verbose:
                        print(f"[Thought Signature]: {final_signature[:100]}...")

        # Save final image
        if final_image_data is not None:
            # Detect actual image format from magic bytes (more reliable than MIME type)
            detected_ext = detect_image_format(final_image_data)

            # Also check MIME type for comparison
            mime_ext = (
                get_extension_for_mime(final_mime_type) if final_mime_type else ".png"
            )
            if detected_ext != mime_ext:
                print(
                    f"Note: API returned MIME type for {mime_ext}, but data is {detected_ext}"
                )
                print(f"  Using detected format: {detected_ext}")

            # Get script directory for output paths
            script_dir = Path(__file__).parent.parent

            # Determine output filename and auto-organize into drafts/finals
            if output_path is None:
                timestamp = datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
                ext = detected_ext  # Use detected format, not MIME type

                # Auto-organize: drafts go to drafts/, finals would go to root
                if is_draft:
                    output_path = script_dir / f"output/drafts/draft_{timestamp}{ext}"
                else:
                    output_path = script_dir / f"output/generated_{timestamp}{ext}"
            else:
                # If user specified path, check if extension matches actual format
                user_ext = output_path.suffix.lower()
                if user_ext and user_ext != detected_ext:
                    # User specified different extension - correct it
                    print(
                        f"Warning: Specified extension {user_ext} doesn't match actual format {detected_ext}"
                    )
                    output_path = output_path.with_suffix(detected_ext)
                    print(f"  Saving as: {output_path.name}")
                elif not user_ext:
                    # No extension specified - add the detected one
                    output_path = output_path.with_suffix(detected_ext)

                # If path doesn't start with output/, prepend it
                if not str(output_path).startswith("output"):
                    if is_draft:
                        output_path = script_dir / "output/drafts" / output_path.name
                    # Check if this looks like a final (contains "final" in name)
                    elif "final" in output_path.stem.lower():
                        output_path = script_dir / "output/finals" / output_path.name
                    else:
                        output_path = script_dir / "output" / output_path.name

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write image
            with open(output_path, "wb") as f:
                f.write(final_image_data)

            if thought_count > 0:
                print(f"\nProcessed {thought_count} thought step(s)")
            print(f"Final image saved to: {output_path}")

            # Optionally save thought signature to sidecar file
            if final_signature and verbose:
                sig_path = output_path.with_suffix(".signature.bin")
                with open(sig_path, "wb") as f:
                    # Signature is binary data
                    if isinstance(final_signature, bytes):
                        f.write(final_signature)
                    else:
                        f.write(str(final_signature).encode())
                print(f"Thought signature saved to: {sig_path}")

            # Document prompt if requested
            if document_prompt:
                is_final_image = (
                    "final" in output_path.stem.lower()
                    or output_path.parent.name == "finals"
                )
                document_image_prompt(
                    image_path=output_path,
                    prompt=prompt,
                    model_key=model_key,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    reference_images=reference_images,
                    is_draft=is_draft,
                    is_final=is_final_image,
                )

            return output_path

        print("Error: No image data in response.")
        return None

    except Exception as e:
        print(f"Error generating image: {e}")
        if "API_KEY" in str(e).upper():
            print("Check that your GEMINI_API_KEY is valid.")
        return None


def generate_story_sequence(
    base_prompt: str,
    num_parts: int,
    model_key: str = DEFAULT_MODEL,
    output_prefix: Path | None = None,
    aspect_ratio: str | None = None,
    image_size: str | None = None,
    verbose: bool = False,
) -> list[Path]:
    """
    Generate a multi-part story sequence using conversational refinement.

    Args:
        base_prompt: Base story description
        num_parts: Number of story parts to generate
        model_key: Model to use
        output_prefix: Prefix for output files (e.g., "story" -> story_part1.png, story_part2.png)
        aspect_ratio: Aspect ratio for all images
        image_size: Image size for all images
        verbose: Show detailed process

    Returns:
        List of paths to generated images
    """
    if num_parts < 1:
        print("Error: Number of story parts must be at least 1")
        return []

    if output_prefix is None:
        timestamp = datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
        output_prefix = Path(f"story_{timestamp}")

    get_api_key()  # Validate API key exists
    if model_key not in MODELS:
        print(f"Error: Unknown model '{model_key}'. Use --list-models to see options.")
        return []

    generated_images = []
    previous_image_path = None

    print(f"Generating {num_parts}-part story sequence...")
    print(f"Base prompt: {base_prompt}\n")

    for part_num in range(1, num_parts + 1):
        print(f"\n{'=' * 60}")
        print(f"PART {part_num}/{num_parts}")
        print(f"{'=' * 60}")

        # Build prompt for this part
        if part_num == 1:
            prompt = f"{base_prompt}\n\nThis is part 1 of {num_parts}. Create the opening scene that establishes the context and visual style for the entire sequence."
        elif part_num == num_parts:
            prompt = f"This is part {part_num} of {num_parts}, the final scene. Building on the previous image, create a concluding scene that resolves the narrative. Maintain visual consistency with the established style."
        else:
            prompt = f"This is part {part_num} of {num_parts}. Building on the previous image, advance the narrative while maintaining visual consistency with the established style."

        # Build output path
        output_path = output_prefix.parent / f"{output_prefix.stem}_part{part_num}.png"

        # Build reference images list
        reference_images = [previous_image_path] if previous_image_path else None

        print(f"Prompt: {prompt[:100]}...")

        # Generate this part
        result = generate_image(
            prompt=prompt,
            model_key=model_key,
            reference_images=reference_images,
            output_path=output_path,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            use_search=False,
            save_thoughts=False,
            verbose=verbose,
        )

        if result:
            generated_images.append(result)
            previous_image_path = result
            print(f"✓ Part {part_num} complete: {result}")
        else:
            print(f"✗ Failed to generate part {part_num}")
            break

    print(f"\n{'=' * 60}")
    print(
        f"Story sequence complete: {len(generated_images)}/{num_parts} parts generated"
    )
    print(f"{'=' * 60}\n")

    for i, path in enumerate(generated_images, 1):
        print(f"  Part {i}: {path}")

    return generated_images


def list_models():
    """Print available models."""
    print("Available models:\n")
    for key, config in MODELS.items():
        print(f"  {key}:")
        print(f"    Name: {config['name']}")
        print(f"    ID: {config['id']}")
        print(f"    Description: {config['description']}")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate images using Google Gemini (Nano Banana / Nano Banana Pro)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image generation
  %(prog)s "A serene mountain landscape at dawn"
  %(prog)s "A data governance diagram" -o governance.png

  # Draft-then-finalize workflow (cost-effective iteration)
  %(prog)s "A technical blueprint" --draft-mode -o draft.png
  %(prog)s "Adjust colors" -r draft.png --draft-mode -o draft_v2.png
  %(prog)s --finalize draft_v2.png --size 2K -o final.png

  # Image editing with reference
  %(prog)s "Make the building taller" -r blueprint.png
  %(prog)s "Refine this architectural drawing" -r img1.png -r img2.png

  # Advanced options
  %(prog)s "A landscape" --aspect 16:9 --size 4K
  %(prog)s "Current weather in Tokyo" --search
  %(prog)s "Complex blueprint design" --save-thoughts --verbose

  # Multi-part story generation (automatic continuity)
  %(prog)s "A 3-part journey through data governance" --story-parts 3 -o journey
  %(prog)s "Evolution of a data platform" --story-parts 4 --aspect 16:9 --size 2K -o evolution
        """,
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="Text prompt describing the image to generate",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path (default: generated_TIMESTAMP.png)",
    )

    parser.add_argument(
        "-m",
        "--model",
        choices=list(MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "-r",
        "--reference",
        type=Path,
        action="append",
        dest="references",
        help="Reference image(s) for editing or style (can be used multiple times)",
    )

    parser.add_argument(
        "--aspect",
        choices=ASPECT_RATIOS,
        help="Aspect ratio (pro model only): 1:1, 3:4, 4:3, 9:16, 16:9",
    )

    parser.add_argument(
        "--size",
        choices=IMAGE_SIZES,
        help="Image size (pro model only): 1K, 2K, 4K",
    )

    parser.add_argument(
        "--search",
        action="store_true",
        help="Enable Google Search grounding for real-time data (pro model only)",
    )

    parser.add_argument(
        "--save-thoughts",
        action="store_true",
        help="Save intermediate thought images (pro model only)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed thinking process and save thought signatures",
    )

    parser.add_argument(
        "--story-parts",
        type=int,
        metavar="N",
        help="Generate a multi-part story with N parts (uses previous image as reference for continuity)",
    )

    parser.add_argument(
        "--draft-mode",
        action="store_true",
        help="Generate at 1K resolution for faster, lower-cost iteration (upgrades to 2K/4K when finalized)",
    )

    parser.add_argument(
        "--finalize",
        type=Path,
        metavar="DRAFT_IMAGE",
        help="Finalize a draft image by regenerating at higher resolution (2K by default, or use --size)",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Gemini models and exit",
    )

    parser.add_argument(
        "--list-topaz-models",
        action="store_true",
        help="List available Topaz enhancement models and exit",
    )

    # --- Topaz post-processing ---
    topaz_group = parser.add_argument_group(
        "Topaz Labs post-processing (requires TOPAZ_API_KEY)"
    )

    topaz_group.add_argument(
        "--enhance",
        type=Path,
        metavar="IMAGE",
        help="Enhance an existing image with Topaz (standalone mode, no generation)",
    )

    topaz_group.add_argument(
        "--topaz",
        action="store_true",
        help="Pipe generated (or finalized) image through Topaz after generation",
    )

    topaz_group.add_argument(
        "--topaz-model",
        default=DEFAULT_TOPAZ_MODEL,
        metavar="MODEL",
        help=f"Topaz model to use (default: {DEFAULT_TOPAZ_MODEL}). See --list-topaz-models.",
    )

    topaz_group.add_argument(
        "--topaz-sharpen",
        type=float,
        metavar="STRENGTH",
        help="Sharpening strength 0.0-1.0 applied during Topaz enhancement",
    )

    topaz_group.add_argument(
        "--topaz-denoise",
        type=float,
        metavar="STRENGTH",
        help="Denoising strength 0.0-1.0 applied during Topaz enhancement",
    )

    topaz_group.add_argument(
        "--topaz-face-enhance",
        action="store_true",
        help="Enable Topaz face enhancement during post-processing",
    )

    topaz_group.add_argument(
        "--topaz-face-strength",
        type=float,
        metavar="STRENGTH",
        help="Face enhancement strength 0.0-1.0 (used with --topaz-face-enhance)",
    )

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if args.list_topaz_models:
        list_topaz_models()
        return

    if not args.prompt and not args.finalize and not args.enhance:
        parser.print_help()
        sys.exit(1)

    # --- Standalone Topaz enhancement mode ---
    if args.enhance:
        result = topaz_enhance_image(
            input_path=args.enhance,
            output_path=args.output,
            model=args.topaz_model,
            sharpen=args.topaz_sharpen,
            denoise=args.topaz_denoise,
            face_enhancement=args.topaz_face_enhance,
            face_enhancement_strength=args.topaz_face_strength,
            verbose=args.verbose,
        )
        sys.exit(0 if result else 1)

    # Check for google-genai package
    if not GENAI_AVAILABLE:
        print("Error: google-genai package not installed.")
        print()
        print("To install, create a virtual environment:")
        print("  python3 -m venv .venv")
        print("  source .venv/bin/activate")
        print("  pip install google-genai")
        print()
        print("Or use uv:")
        print("  uv venv && source .venv/bin/activate && uv pip install google-genai")
        sys.exit(1)

    # Finalize mode - upscale a draft to final resolution
    if args.finalize:
        if not args.finalize.exists():
            print(f"Error: Draft image not found: {args.finalize}")
            sys.exit(1)

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = args.finalize.parent / f"{args.finalize.stem}_final.png"

        # --- Topaz finalization path ---
        if args.topaz:
            print(f"Finalizing draft image via Topaz: {args.finalize}")
            print(f"Topaz model: {args.topaz_model}")
            result = topaz_enhance_image(
                input_path=args.finalize,
                output_path=output_path,
                model=args.topaz_model,
                sharpen=args.topaz_sharpen,
                denoise=args.topaz_denoise,
                face_enhancement=args.topaz_face_enhance,
                face_enhancement_strength=args.topaz_face_strength,
                verbose=args.verbose,
            )

            if result:
                print(f"\n{'=' * 60}")
                print("Topaz finalization complete!")
                print(f"Draft: {args.finalize}")
                print(f"Final: {result}")
                print(f"{'=' * 60}")

            sys.exit(0 if result else 1)

        # --- Gemini finalization path ---
        final_size = args.size if args.size else "2K"
        final_aspect = args.aspect if args.aspect else "16:9"

        print(f"Finalizing draft image: {args.finalize}")
        print(f"Target resolution: {final_size} ({final_aspect})")

        prompts_file = Path(__file__).parent.parent / "examples" / "PROMPTS.md"
        if prompts_file.exists():
            with open(prompts_file, encoding="utf-8") as f:
                content = f.read()
                image_name = args.finalize.stem
                if image_name in content:
                    print("Found original prompt in PROMPTS.md")
                    if not args.prompt:
                        print(
                            "\nNote: Use the same prompt as the draft, or provide a refinement prompt."
                        )
                        print(
                            'Example: python scripts/generate_image.py --finalize draft.png "Same as draft"'
                        )
                        print("\nProceeding with reference-based upscaling...")

        prompt = (
            args.prompt
            if args.prompt
            else "Recreate this image at higher resolution with the same composition, style, and details"
        )

        result = generate_image(
            prompt=prompt,
            model_key=args.model,
            reference_images=[args.finalize],
            output_path=output_path,
            aspect_ratio=final_aspect,
            image_size=final_size,
            use_search=False,
            save_thoughts=args.save_thoughts,
            verbose=args.verbose,
        )

        if result:
            print(f"\n{'=' * 60}")
            print("Finalization complete!")
            print(f"Draft (1K): {args.finalize}")
            print(f"Final ({final_size}): {result}")
            print(f"{'=' * 60}")

        sys.exit(0 if result else 1)

    # Story sequence mode
    if args.story_parts:
        if args.story_parts < 2:
            print("Error: Story must have at least 2 parts")
            sys.exit(1)

        if args.output:
            output_prefix = args.output
        else:
            output_prefix = Path("story")

        results = generate_story_sequence(
            base_prompt=args.prompt,
            num_parts=args.story_parts,
            model_key=args.model,
            output_prefix=output_prefix,
            aspect_ratio=args.aspect,
            image_size=args.size,
            verbose=args.verbose,
        )

        if args.topaz and results:
            print(f"\nApplying Topaz enhancement to {len(results)} image(s)...")
            enhanced = []
            for path in results:
                r = topaz_enhance_image(
                    input_path=path,
                    model=args.topaz_model,
                    sharpen=args.topaz_sharpen,
                    denoise=args.topaz_denoise,
                    face_enhancement=args.topaz_face_enhance,
                    face_enhancement_strength=args.topaz_face_strength,
                    verbose=args.verbose,
                )
                if r:
                    enhanced.append(r)
            results = enhanced

        sys.exit(0 if len(results) == args.story_parts else 1)

    # Single image mode
    else:
        effective_size = "1K" if args.draft_mode else args.size

        if args.draft_mode:
            print("Draft mode: Generating at 1K resolution for fast iteration")
            print("Drafts are stored in output/drafts/")
            print("Use --finalize <draft_image.png> to upscale to final resolution\n")

        result = generate_image(
            prompt=args.prompt,
            model_key=args.model,
            reference_images=args.references,
            output_path=args.output,
            aspect_ratio=args.aspect,
            image_size=effective_size,
            use_search=args.search,
            save_thoughts=args.save_thoughts,
            verbose=args.verbose,
            is_draft=args.draft_mode,
            document_prompt=True,
        )

        if result and args.topaz:
            result = topaz_enhance_image(
                input_path=result,
                model=args.topaz_model,
                sharpen=args.topaz_sharpen,
                denoise=args.topaz_denoise,
                face_enhancement=args.topaz_face_enhance,
                face_enhancement_strength=args.topaz_face_strength,
                verbose=args.verbose,
            )

        if result and args.draft_mode:
            print(f"\n{'=' * 60}")
            print("Draft complete! To finalize at higher resolution:")
            print(f"  python scripts/generate_image.py --finalize {result} --size 2K")
            print(f"  # Or with Topaz: --finalize {result} --topaz")
            print(f"{'=' * 60}")

        sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
