"""`fal dev` failures as a panel: what broke, why, and the command to run next."""

from __future__ import annotations

from typing import Sequence

from rich.console import Console, Group, RenderableType
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from fal.dev.errors import DevError
from fal.dev.loading import panel

ERROR = "red"
MUTED = "dim"


def error_panel(
    title: str,
    lines: Sequence[RenderableType | str],
    fixes: Sequence[tuple[str, str]] = (),
) -> RenderableType:
    """A rounded panel titled "error - <title>", the lines, then the fixes."""
    body: list[RenderableType] = [
        Text(line) if isinstance(line, str) else line for line in lines
    ]
    if fixes:
        table = Table.grid(padding=(0, 2))
        table.add_column(style=MUTED, no_wrap=True)
        table.add_column(style="bold", no_wrap=True)
        table.add_column(style=MUTED)
        for i, (command, why) in enumerate(fixes):
            table.add_row("try" if i == 0 else "or", command, why)
        body += [Text(""), table]
    heading = Text.styled("error", style=f"bold {ERROR}") + Text(
        f"  {title}", style="bold default"
    )
    return panel(Group(*body), heading)


def show(exc: DevError, console: Console) -> None:
    """Print a DevError: as a panel on a terminal, as plain lines otherwise."""
    message = str(exc)
    if not console.is_terminal:
        console.print(f"Error: {escape(message)}", highlight=False)
        for command, why in exc.fixes:
            console.print(f"  try: {escape(command)}  ({escape(why)})", highlight=False)
        return
    console.print(error_panel(exc.title or "fal dev", [message], exc.fixes))
