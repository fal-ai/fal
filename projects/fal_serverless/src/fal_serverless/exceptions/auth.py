from __future__ import annotations

from ._base import KoldstartException


class UnauthenticatedException(KoldstartException):
    """Exception that indicates that"""

    def __init__(self) -> None:
        super().__init__(
            message="You must be authenticated.",
            hint="Login via `koldstart auth login`",
        )
