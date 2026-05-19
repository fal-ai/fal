from __future__ import annotations

from typing import TextIO


def make_terminal_safe(text: str, stream: TextIO) -> str:
    """Escape characters that cannot be encoded by the target stream."""
    encoding = stream.encoding or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)
