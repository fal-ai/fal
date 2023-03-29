from __future__ import annotations


class KoldstartException(Exception):
    """Base exception type for koldstart related flows and APIs."""

    message: str

    hint: str | None

    def __init__(self, message: str, hint: str | None = None) -> None:
        self.message = message
        self.hint = hint
        super().__init__(message)

    def __str__(self) -> str:
        return self.message
