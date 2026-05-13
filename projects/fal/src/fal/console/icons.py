from __future__ import annotations

from typing import TYPE_CHECKING

from fal.console import console as default_console

if TYPE_CHECKING:
    from rich.console import Console


def _select_icon(
    unicode_icon: str,
    ascii_icon: str,
    *,
    target_console: Console | None = None,
    ascii_only: bool | None = None,
) -> str:
    if ascii_only is None:
        target_console = target_console or default_console
        ascii_only = target_console.options.ascii_only

    return ascii_icon if ascii_only else unicode_icon


def get_check_icon(target_console: Console | None = None) -> str:
    return _select_icon(
        "[bold green]\u2713[/]", "[bold green]+[/]", target_console=target_console
    )


def get_cross_icon(target_console: Console | None = None) -> str:
    return _select_icon(
        "[bold red]\u2718[/]", "[bold red]x[/]", target_console=target_console
    )


def get_section_icon(target_console: Console | None = None) -> str:
    return _select_icon("\u25b8", ">", target_console=target_console)


def get_bullet_icon(target_console: Console | None = None) -> str:
    return _select_icon("\u2022", "-", target_console=target_console)


def get_status_queued_icon(target_console: Console | None = None) -> str:
    return _select_icon("\u23f3", "...", target_console=target_console)


def get_status_progress_icon(target_console: Console | None = None) -> str:
    return _select_icon("\U0001f504", "~", target_console=target_console)


def get_status_done_icon(target_console: Console | None = None) -> str:
    return _select_icon("\u2705", "+", target_console=target_console)


def get_workflow_loaded_icon(target_console: Console | None = None) -> str:
    return _select_icon("\U0001f927", "*", target_console=target_console)


def get_workflow_complete_icon(target_console: Console | None = None) -> str:
    return _select_icon("\U0001f389", "+", target_console=target_console)


_ICON_GETTERS = {
    "CHECK_ICON": get_check_icon,
    "CROSS_ICON": get_cross_icon,
    "SECTION_ICON": get_section_icon,
    "BULLET_ICON": get_bullet_icon,
    "STATUS_QUEUED_ICON": get_status_queued_icon,
    "STATUS_PROGRESS_ICON": get_status_progress_icon,
    "STATUS_DONE_ICON": get_status_done_icon,
    "WORKFLOW_LOADED_ICON": get_workflow_loaded_icon,
    "WORKFLOW_COMPLETE_ICON": get_workflow_complete_icon,
}


def __getattr__(name: str) -> str:
    try:
        return _ICON_GETTERS[name]()
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
