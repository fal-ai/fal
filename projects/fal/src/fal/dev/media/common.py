"""What preview, term_media and term_video share; imports no Pillow."""

from __future__ import annotations

import re

OFF = "off"
ENV_PROTOCOL = "FAL_DEV_PREVIEW"
ENV_MAX_ROWS = "FAL_DEV_PREVIEW_ROWS"
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}
ANIMATED_SUFFIXES = {".gif", ".webp", ".png"}
# Formats Pillow may open: it would otherwise sniff anything (EPS runs Ghostscript).
IMAGE_FORMATS = ["PNG", "JPEG", "GIF", "WEBP", "BMP", "TIFF"]
CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def clean(text: str) -> str:
    """Text safe to write to a terminal: control characters become "?".

    Captions carry JSON keys, file names and error messages from the app's
    response, which must not be able to send escape sequences.
    """
    return CONTROL_CHARS.sub("?", text)
