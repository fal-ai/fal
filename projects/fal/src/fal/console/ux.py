from __future__ import annotations

import os
import webbrowser


def is_headless() -> bool:
    """Detect environments where opening a browser is unlikely to work."""
    # WSL (Windows Subsystem for Linux)
    if os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop") or os.path.exists(
        "/proc/version"
    ):
        try:
            with open("/proc/version") as f:
                if "microsoft" in f.read().lower():
                    return True
        except OSError:
            pass

    # SSH session
    if os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"):
        return True

    # Linux without a display server
    if (
        os.name == "posix"
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
        and os.uname().sysname == "Linux"
    ):
        return True

    return False


def maybe_open_browser_tab(url) -> None:
    if is_headless():
        return

    try:
        # Avoids unwanted output in the console from the standard `webbrowser.open()`.
        # See https://stackoverflow.com/a/19199794
        browser = webbrowser.get()

        browser.open_new_tab(url)
    except webbrowser.Error:
        pass
