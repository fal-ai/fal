from __future__ import annotations

import webbrowser


def get_browser() -> webbrowser.BaseBrowser | None:
    """Gets a reference to the default browser, if available.

    This allows us to decide on a flow before showing the actual browser window.
    It also avoids unwanted output in the console from the standard `webbrowser.open()`.

    See https://stackoverflow.com/a/19199794
    """
    try:
        return webbrowser.get()
    except webbrowser.Error:
        return None
