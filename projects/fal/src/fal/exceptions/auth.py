from __future__ import annotations

from ._base import FalServerlessException


class UnauthenticatedException(FalServerlessException):
    def __init__(self) -> None:
        super().__init__(
            "You must be authenticated. "
            "Login via `fal auth login` or make sure to setup fal keys correctly."
        )
