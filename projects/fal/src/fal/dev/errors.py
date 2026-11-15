"""The error type `fal dev` raises for failures it can explain to the user."""

from __future__ import annotations

from typing import Sequence

# Exit codes: 0 ok, 1 general error, 2 usage, 3 box not found or stopped, 4 auth
# or team, 5 timeout, 130 Ctrl-C. `exec` passes the remote command's code through.
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_NO_BOX = 3
EXIT_AUTH = 4
EXIT_TIMEOUT = 5
EXIT_INTERRUPTED = 130


class DevError(Exception):
    """A `fal dev` failure; the message is shown to the user as is.

    `title` names the failure in a few words (the command's name when it has
    none), and `fixes` are (command, why) pairs to try next; the CLI shows both
    in an error panel. `code` is the exit status, `usage` a usage line to show
    first.
    """

    def __init__(
        self,
        message: str,
        *,
        title: str | None = None,
        fixes: Sequence[tuple[str, str]] = (),
        code: int = EXIT_ERROR,
        usage: str | None = None,
    ) -> None:
        super().__init__(message)
        self.title = title
        self.fixes = list(fixes)
        self.code = code
        self.usage = usage
