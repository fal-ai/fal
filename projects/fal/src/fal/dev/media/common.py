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


def sniff_image(path) -> tuple[str, tuple[int, int] | None] | None:
    """(content type, (width, height) or None) from a file's header, or None
    when it is not a still image format `view` knows. Reads a few KB, no Pillow."""
    import struct

    try:
        with open(path, "rb") as f:
            head = f.read(32)
            if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24:
                width, height = struct.unpack(">II", head[16:24])
                return "image/png", (width, height)
            if head[:6] in (b"GIF87a", b"GIF89a") and len(head) >= 10:
                width, height = struct.unpack("<HH", head[6:10])
                return "image/gif", (width, height)
            if head.startswith(b"BM") and len(head) >= 26:
                width, height = struct.unpack("<ii", head[18:26])
                return "image/bmp", (width, abs(height))
            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                return "image/webp", _webp_size(head)
            if head[:4] in (b"II*\x00", b"MM\x00*"):
                return "image/tiff", None
            if head.startswith(b"\xff\xd8"):
                return "image/jpeg", _jpeg_size(f)
    except OSError:
        return None
    return None


def _webp_size(head: bytes) -> tuple[int, int] | None:
    import struct

    kind = head[12:16]
    try:
        if kind == b"VP8X":
            width = int.from_bytes(head[24:27], "little") + 1
            height = int.from_bytes(head[27:30], "little") + 1
            return width, height
        if kind == b"VP8L" and head[20] == 0x2F:
            bits = struct.unpack("<I", head[21:25])[0]
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if kind == b"VP8 " and head[23:26] == b"\x9d\x01\x2a":
            width, height = struct.unpack("<HH", head[26:30])
            return width & 0x3FFF, height & 0x3FFF
    except (IndexError, struct.error):
        pass
    return None


def _jpeg_size(f) -> tuple[int, int] | None:
    """Walk the marker segments to the first frame header."""
    import struct

    f.seek(2)
    while True:
        byte = f.read(1)
        while byte and byte != b"\xff":
            byte = f.read(1)
        marker = f.read(1)
        while marker == b"\xff":
            marker = f.read(1)
        if not marker or marker == b"\xda":  # end of file, or start of scan
            return None
        code = marker[0]
        if code in (0x01, 0xD8) or 0xD0 <= code <= 0xD7:  # no length field
            continue
        raw = f.read(2)
        if len(raw) < 2:
            return None
        length = struct.unpack(">H", raw)[0]
        if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
            frame = f.read(5)
            if len(frame) < 5:
                return None
            height, width = struct.unpack(">xHH", frame)
            return width, height
        f.seek(length - 2, 1)
