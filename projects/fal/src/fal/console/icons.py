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


def get_column_separator_icon(target_console: Console | None = None) -> str:
    return _select_icon("\u2502", "|", target_console=target_console)


def get_inline_separator_icon(target_console: Console | None = None) -> str:
    return _select_icon("\u00b7", "-", target_console=target_console)


def get_bar_full_icon(target_console: Console | None = None) -> str:
    return _select_icon("\u2588", "#", target_console=target_console)


def get_bar_empty_icon(target_console: Console | None = None) -> str:
    return _select_icon("\u2591", ".", target_console=target_console)


def get_sparkline_icons(target_console: Console | None = None) -> str:
    """Eight levels, lowest first."""
    return _select_icon(
        "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588",
        "_.-~=+*#",
        target_console=target_console,
    )


def get_state_dot_icon(target_console: Console | None = None) -> str:
    return _select_icon("\u25cf", "*", target_console=target_console)


def get_degrees_celsius_icon(target_console: Console | None = None) -> str:
    return _select_icon("\u00b0C", " C", target_console=target_console)


def get_ellipsis_icon(target_console: Console | None = None) -> str:
    return _select_icon("\u2026", "...", target_console=target_console)


# Drawn only on UTF-8 streams (terminal image previews, `fal dev` loading states).
UPPER_HALF_BLOCK = "\u2580"
BRAILLE_SPINNER = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
# Dust squares, smallest to largest: middle dot, small square, square.
DUST_SQUARES = "\u00b7\u25aa\u25a0"


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
