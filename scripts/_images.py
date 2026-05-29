"""Pure image-format helpers (no I/O, no module-global state).

These functions inspect raw bytes / MIME strings and return file extensions.
They are kept side-effect free and dependency free so they are trivially unit
testable. File-reading loaders that enforce ``MAX_INPUT_IMAGE_BYTES`` live in
``scripts.generate_image`` because their size cap is a patch seam exercised by
the test suite via ``monkeypatch.setattr(mod, "MAX_INPUT_IMAGE_BYTES", ...)``.
"""


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
