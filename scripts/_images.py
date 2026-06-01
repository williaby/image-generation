"""Pure image-format helpers (no I/O, no module-global state).

These functions inspect raw bytes / MIME strings and return file extensions.
They are kept side-effect free and dependency free so they are trivially unit
testable. File-reading loaders that enforce ``MAX_INPUT_IMAGE_BYTES`` live in
``scripts.generate_image`` because their size cap is a patch seam exercised by
the test suite via ``monkeypatch.setattr(mod, "MAX_INPUT_IMAGE_BYTES", ...)``.
"""

# Canonical extension <-> MIME map. ``MIME_TO_EXT`` is derived as the inverse so
# the two directions can never drift. Unknown values fall back to PNG, matching
# ``detect_image_format``'s default.
EXT_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
MIME_TO_EXT = {mime: ext for ext, mime in EXT_TO_MIME.items()}


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
    return MIME_TO_EXT.get(mime_type, ".png")


def get_mime_for_extension(ext: str) -> str:
    """Get the MIME type for a dot-prefixed file extension."""
    return EXT_TO_MIME.get(ext, "image/png")
