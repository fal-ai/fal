from __future__ import annotations

from ._base import FalServerlessException


class UnauthenticatedException(FalServerlessException):
    """Exception that indicates that"""

    def __init__(self) -> None:
        super().__init__(
            message="You must be authenticated.",
            hint="Login via `fal auth login` or make sure to setup fal keys correctly.",
        )
