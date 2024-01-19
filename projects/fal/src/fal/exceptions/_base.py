from __future__ import annotations


class FalServerlessException(Exception):
    """Base exception type for fal Serverless related flows and APIs."""

    message: str

    hint: str | None

    def __init__(self, message: str, hint: str | None = None) -> None:
        self.message = message
        self.hint = hint
        super().__init__(message)

    def __str__(self) -> str:
        return self.message + (f"\nHint: {self.hint}" if self.hint else "")
