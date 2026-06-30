"""Static configuration and the typed error hierarchy for the image CLI.

This module holds the pure *data* and *exception* layer for
``generate_image``: model registries, validation constants, and the
:class:`AppError` hierarchy. It deliberately contains no behavior, no I/O, and
no third-party imports beyond ``typing`` so it can be imported cheaply from
anywhere (and patched/inspected in tests) without side effects.

``scripts.generate_image`` re-imports every public name defined here, so both
``from scripts.generate_image import ConfigError`` and direct
``from scripts._config import ConfigError`` resolve to the same objects.
"""

from typing import Any

# Hard cap on reference / enhancement input image size. 4K PNGs are ~10 MiB;
# 100 MiB is generous and protects against pathological local inputs that would
# otherwise be fully loaded into memory. Peak memory is roughly 1.33x the input
# size for the base64 path; the raw-bytes path used by generation holds a single
# copy.
MAX_INPUT_IMAGE_BYTES = 100 * 1024 * 1024

# Topaz Labs download host allowlist. SSRF guard for the image fetch step in
# topaz_enhance_image: Topaz returns a signed download URL; we restrict the
# host to a known allowlist before issuing the GET. Kept at module scope so
# the security boundary is visible at audit time, not buried mid-function.
TOPAZ_DOWNLOAD_HOSTS: frozenset[str] = frozenset(
    {"api.topazlabs.com", "cdn.topazlabs.com"}
)

# Model configurations
# Each entry carries its own aspect_ratios / image_sizes because the three
# Gemini image models accept different sets. Heterogeneous values are typed
# as ``Any`` so the static checker does not treat ``MODELS[key]["foo"]`` as a
# concrete union; per-call site checks (e.g. ``if not supports_image_config``)
# operate on the values directly.
MODELS: dict[str, dict[str, Any]] = {
    "flash": {
        "id": "gemini-2.5-flash-image",
        "name": "Nano Banana (Gemini 2.5 Flash)",
        "description": "Legacy fast image generation model (no aspect/size control)",
        "supports_image_config": False,
        "supports_thinking_config": False,
        "aspect_ratios": [],
        "image_sizes": [],
    },
    "flash-2": {
        "id": "gemini-3.1-flash-image-preview",
        "name": "Nano Banana 2 (Gemini 3.1 Flash Image)",
        "description": "Pro-quality reasoning at Flash speed; 512/1K/2K/4K, 14 aspect ratios, configurable thinking, Search grounding",
        "supports_image_config": True,
        "supports_thinking_config": True,
        "aspect_ratios": [
            "1:1",
            "1:4",
            "1:8",
            "2:3",
            "3:2",
            "3:4",
            "4:1",
            "4:3",
            "4:5",
            "5:4",
            "8:1",
            "9:16",
            "16:9",
            "21:9",
        ],
        "image_sizes": ["512", "1K", "2K", "4K"],
    },
    "pro": {
        "id": "gemini-3-pro-image-preview",
        "name": "Nano Banana Pro (Gemini 3 Pro)",
        "description": "Highest quality, best text rendering, Google Search grounding, thinking mode",
        "supports_image_config": True,
        "supports_thinking_config": False,
        "aspect_ratios": ["1:1", "3:4", "4:3", "9:16", "16:9"],
        "image_sizes": ["1K", "2K", "4K"],
    },
}

DEFAULT_MODEL = "flash-2"  # Google's new default; Pro-quality at Flash speed.

# Union of valid values across all models (used for argparse choices). Derived
# from MODELS so a future model edit cannot silently desync the CLI from the
# actual capabilities. Per-model validation still happens inside
# generate_image() against MODELS[key] for accurate error messages.
# dict.fromkeys preserves first-seen order while deduplicating.
ASPECT_RATIOS = list(
    dict.fromkeys(r for m in MODELS.values() for r in m.get("aspect_ratios", []))
)
IMAGE_SIZES = list(
    dict.fromkeys(s for m in MODELS.values() for s in m.get("image_sizes", []))
)

# Valid thinking levels for models that support thinking_config.
THINKING_LEVELS = ["minimal", "high"]

# Topaz Labs API base URL
TOPAZ_BASE_URL = "https://api.topazlabs.com/image/v1"

# Topaz endpoint paths, named once so the registry below shares a single literal.
TOPAZ_ENDPOINT_ENHANCE = "enhance/async"
TOPAZ_ENDPOINT_ENHANCE_GEN = "enhance-gen/async"

# Topaz model registry: "enhance" -> /enhance/async; "enhance-gen" -> /enhance-gen/async
# Generative models (Wonder, Bloom) cost ~6-12x more credits than precision models.
TOPAZ_MODELS = {
    # Gigapixel precision upscaling (24 MP per credit)
    "Standard V2": {
        "endpoint": TOPAZ_ENDPOINT_ENHANCE,
        "description": "Precision upscaling, best for most images",
    },
    "High Fidelity V2": {
        "endpoint": TOPAZ_ENDPOINT_ENHANCE,
        "description": "Highest quality, preserves fine detail",
    },
    "Low Resolution V2": {
        "endpoint": TOPAZ_ENDPOINT_ENHANCE,
        "description": "Optimized for very low-resolution sources",
    },
    "CGI": {
        "endpoint": TOPAZ_ENDPOINT_ENHANCE,
        "description": "Optimized for CGI and rendered imagery",
    },
    "Text Refine": {
        "endpoint": TOPAZ_ENDPOINT_ENHANCE,
        "description": "Preserves and sharpens text in diagrams",
    },
    "Detail Faces": {
        "endpoint": TOPAZ_ENDPOINT_ENHANCE,
        "description": "Enhances facial clarity",
    },
    "Recover Faces": {
        "endpoint": TOPAZ_ENDPOINT_ENHANCE,
        "description": "Restores damaged or degraded faces",
    },
    "Transparency Upscale": {
        "endpoint": TOPAZ_ENDPOINT_ENHANCE,
        "description": "Upscales images with alpha transparency",
    },
    # Generative upscaling (4 MP per credit; significantly more expensive)
    "Wonder": {
        "endpoint": TOPAZ_ENDPOINT_ENHANCE_GEN,
        "description": "Generative upscaling, adds intelligent detail",
    },
    "Wonder 2": {
        "endpoint": TOPAZ_ENDPOINT_ENHANCE_GEN,
        "description": "Improved generative upscaling",
    },
    "Standard Max": {
        "endpoint": TOPAZ_ENDPOINT_ENHANCE_GEN,
        "description": "Maximum quality generative upscaling",
    },
    "Recover 3": {
        "endpoint": TOPAZ_ENDPOINT_ENHANCE_GEN,
        "description": "Advanced recovery with generation",
    },
    "Redefine": {
        "endpoint": TOPAZ_ENDPOINT_ENHANCE_GEN,
        "description": "Creative reinterpretation with upscaling",
    },
    "Bloom": {
        "endpoint": TOPAZ_ENDPOINT_ENHANCE_GEN,
        "description": "Creative upscaling for AI-generated art",
    },
}

DEFAULT_TOPAZ_MODEL = "Standard V2"


# ---------------------------------------------------------------------------
# Typed exception hierarchy
# ---------------------------------------------------------------------------


class AppError(Exception):
    """Base for all expected, user-visible failures in this CLI.

    ``main()`` catches ``AppError`` and produces a clean stderr message + an
    exit code of 1, instead of a Python traceback. Any subclass implies the
    failure is part of normal operation (missing config, upstream API error,
    bad file I/O); programmer bugs should raise ordinary exceptions and
    surface as tracebacks via the outermost ``except Exception`` safety net.
    """


class ConfigError(AppError):
    """Configuration is missing or malformed (e.g. unset GEMINI_API_KEY)."""


class GeminiAPIError(AppError):
    """The Gemini API returned an error or unparseable response."""


class TopazAPIError(AppError):
    """The Topaz Labs API returned an error or unparseable response."""


class FileIOError(AppError):
    """An expected file write or read failed (e.g. output directory unwritable)."""
