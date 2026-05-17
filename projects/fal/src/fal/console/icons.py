from __future__ import annotations

from fal.console import console


def _select_icon(
    unicode_icon: str,
    ascii_icon: str,
    *,
    ascii_only: bool | None = None,
) -> str:
    if ascii_only is None:
        ascii_only = console.options.ascii_only

    return ascii_icon if ascii_only else unicode_icon


CHECK_ICON = _select_icon("[bold green]\u2713[/]", "[bold green]+[/]")
CROSS_ICON = _select_icon("[bold red]\u2718[/]", "[bold red]x[/]")
SECTION_ICON = _select_icon("\u25b8", ">")
BULLET_ICON = _select_icon("\u2022", "-")
STATUS_QUEUED_ICON = _select_icon("\u23f3", "...")
STATUS_PROGRESS_ICON = _select_icon("\U0001f504", "~")
STATUS_DONE_ICON = _select_icon("\u2705", "+")
WORKFLOW_LOADED_ICON = _select_icon("\U0001f927", "*")
WORKFLOW_COMPLETE_ICON = _select_icon("\U0001f389", "+")
