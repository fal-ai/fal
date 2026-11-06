"""The error type `fal dev` raises for failures it can explain to the user."""

from __future__ import annotations

from typing import Sequence


class DevError(Exception):
    """A `fal dev` failure; the message is shown to the user as is.

    `title` names the failure in a few words, and `fixes` are (command, why)
    pairs to try next; the CLI shows both in an error panel.
    """

    def __init__(
        self,
        message: str,
        *,
        title: str | None = None,
        fixes: Sequence[tuple[str, str]] = (),
    ) -> None:
        super().__init__(message)
        self.title = title
        self.fixes = list(fixes)
