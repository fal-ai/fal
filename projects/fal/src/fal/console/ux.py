from __future__ import annotations

import webbrowser


def maybe_open_browser_tab(url) -> None:
    try:
        # Avoids unwanted output in the console from the standard `webbrowser.open()`.
        # See https://stackoverflow.com/a/19199794
        browser = webbrowser.get()

        browser.open_new_tab(url)
    except webbrowser.Error:
        pass
