from __future__ import annotations

import colorama

LEVEL_STYLES = {
    "critical": colorama.Fore.RED,
    "exception": colorama.Fore.RED,
    "error": colorama.Fore.RED,
    "warning": colorama.Fore.YELLOW,
    "info": colorama.Fore.BLUE,
    "debug": colorama.Style.DIM,
    "trace": colorama.Style.DIM,
    "stdout": colorama.Fore.LIGHTCYAN_EX,
    "notset": colorama.Style.DIM,
}
