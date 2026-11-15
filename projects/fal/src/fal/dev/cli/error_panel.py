"""`fal dev` failures as a panel: what broke, why, and the command to run next.

Off a terminal the same content is plain lines: `error: <title>: <message>` and
one `try: <command>  (<why>)` line per fix.
"""

from __future__ import annotations

import json
from typing import Sequence

from rich.console import Console, Group, RenderableType
from rich.table import Table
from rich.text import Text

from fal.dev.errors import DevError
from fal.dev.loading import panel

ERROR = "red"
WARNING = "yellow"
MUTED = "dim"


def _fix_table(fixes: Sequence[tuple[str, str]]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style=MUTED, no_wrap=True)
    table.add_column(style="bold", no_wrap=True)
    table.add_column(style=MUTED)
    for i, (command, why) in enumerate(fixes):
        table.add_row("try" if i == 0 else "or", command, why)
    return table


def _panel(
    label: str,
    style: str,
    title: str,
    lines: Sequence[RenderableType | str],
    fixes: Sequence[tuple[str, str]],
) -> RenderableType:
    body: list[RenderableType] = [
        Text(line) if isinstance(line, str) else line for line in lines
    ]
    if fixes:
        body += [Text(""), _fix_table(fixes)]
    heading = Text.styled(label, style=f"bold {style}") + Text(
        f"  {title}", style="bold default"
    )
    return panel(Group(*body), heading)


def error_panel(
    title: str,
    lines: Sequence[RenderableType | str],
    fixes: Sequence[tuple[str, str]] = (),
) -> RenderableType:
    """A rounded panel titled "error - <title>", the lines, then the fixes."""
    return _panel("error", ERROR, title, lines, fixes)


def plain_lines(
    label: str, title: str | None, message: str, fixes: Sequence[tuple[str, str]]
) -> list[Text]:
    """`label: <title>: <message>` then one `try:` line per fix, unstyled."""
    head = f"{label}: {title}: {message}" if title else f"{label}: {message}"
    return [Text(head)] + [Text(f"try: {cmd}  ({why})") for cmd, why in fixes]


def _print_plain(console: Console, lines: Sequence[Text]) -> None:
    for line in lines:
        console.print(line, highlight=False, soft_wrap=True)


def title_of(exc: DevError, command: str | None = None) -> str:
    return exc.title or command or "fal dev"


def show(exc: DevError, console: Console, command: str | None = None) -> None:
    """Print a DevError: as a panel on a terminal, as plain lines otherwise.

    The title is the error's own, else `command` (the failing command's name).
    """
    title = title_of(exc, command)
    if exc.usage:
        console.print(Text(exc.usage.rstrip()), highlight=False, soft_wrap=True)
    if not console.is_terminal:
        _print_plain(console, plain_lines("error", title, str(exc), exc.fixes))
        return
    console.print(error_panel(title, [str(exc)], exc.fixes))


def warn(
    message: str,
    console: Console,
    fixes: Sequence[tuple[str, str]] = (),
    title: str = "warning",
) -> None:
    """A warning in the error panel's style: yellow panel, or `warning:` lines."""
    if not console.is_terminal:
        _print_plain(console, plain_lines("warning", None, message, fixes))
        return
    console.print(_panel("warning", WARNING, title, [message], fixes))


def as_json(exc: DevError, command: str | None = None) -> str:
    """The error as one JSON object: {"error": {title, message, fixes}}."""
    fixes = [{"command": cmd, "why": why} for cmd, why in exc.fixes]
    error = {"title": title_of(exc, command), "message": str(exc), "fixes": fixes}
    return json.dumps({"error": error})
