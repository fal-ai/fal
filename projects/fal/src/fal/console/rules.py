from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console


def print_rule(console: Console, title: Any = "", *, style: str = "") -> None:
    if console.options.ascii_only is not True:
        from rich.rule import Rule

        console.print(Rule(title, style=style))
        return

    width = max(1, min(console.width, 120))
    title_text = getattr(title, "plain", str(title))
    if not title_text:
        console.print("-" * width, style=style)
        return

    padded_title = f" {title_text} "
    if len(padded_title) >= width:
        console.print(title_text, style=style)
        return

    left = (width - len(padded_title)) // 2
    right = width - len(padded_title) - left
    console.print(f"{'-' * left}{padded_title}{'-' * right}", style=style)
