from __future__ import annotations

from ._base import FalServerlessException


class UnauthenticatedException(FalServerlessException):
    def __init__(self) -> None:
        super().__init__(
            "You must be authenticated. "
            "Use [bold]fal auth login[/] or [bold]fal profile key[/] to set your fal key."  # noqa: E501
        )
