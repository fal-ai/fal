"""`fal dev app dash` before a box exists: one frame, refreshed in place.

The dash can be started first (in a split pane) and attach by itself when the
agent or user starts a box. Until then it shows what is known: nothing yet, or
the phase of a box being started. An error (several boxes, a broken session
file) is a line in the same frame, not a new box on every retry.
"""

from __future__ import annotations

import sys
import time
from typing import Callable

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.text import Text

from fal.dev import loading
from fal.dev.app import live
from fal.dev.app.session import NoSession, Session, find_session
from fal.dev.errors import DevError

PHASE_TEXT = {
    live.DEPLOYING: "deploying the app",
    live.PROVISIONING: "provisioning: waiting for a runner and uploading the source",
    live.SETUP: "running setup() on the box",
}
WAITING_TEXT = "waiting for a dev box (start one with `fal dev app start`)"


def frame(starting: list[dict], now: float, error: str | None = None) -> RenderableType:
    """The waiting frame: a title, one status line, the error if any."""
    title = Text.styled("fal dev app", style="bold") + Text(
        "  dash", style=loading.MUTED
    )
    if starting:
        box = starting[0]
        elapsed = loading.format_elapsed(now - box["since"])
        status = Text.assemble(
            ("starting ", "bold"),
            (box["name"], "bold"),
            (f": {PHASE_TEXT[box['phase']]} ", ""),
            (f"({elapsed})", loading.MUTED),
        )
    else:
        status = Text(WAITING_TEXT, style=loading.MUTED)
    rows: list[RenderableType] = [status]
    if error:
        rows.append(Text(error, style=loading.ERROR))
    return loading.panel(Group(*rows), title)


def _starting_line(box: dict) -> str:
    return f"starting {box['name']}: {PHASE_TEXT[box['phase']]}"


def find(
    name: str | None, *, settled: bool = False
) -> tuple[Session | None, str | None]:
    """(session, None) when one exists; (None, error) when it cannot be picked
    yet, such as several boxes; (None, None) when there is none. With
    ``settled``, a box still being started is not there yet: a watcher must
    not boot it while `start` does."""
    try:
        session = find_session(name)
        if settled and any(s["name"] == session.name for s in live.starting()):
            return None, None
        return session, None
    except NoSession:
        return None, None
    except DevError as exc:
        return None, str(exc)


def wait_for_session(
    name: str | None,
    console: Console,
    *,
    interval: float = 1.0,
    settled: bool = False,
    find_session_fn: Callable[[str | None], tuple[Session | None, str | None]]
    | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
) -> Session:
    """Block until a box can be attached; draw one frame in place meanwhile.

    On a terminal the frame refreshes in place. Otherwise a line prints only
    when the status changes, so a log never fills with repeats.
    """
    if find_session_fn is None:

        def find_session_fn(n):
            return find(n, settled=settled)

    session, error = find_session_fn(name)
    if session:
        return session
    tty = console.is_terminal and sys.stdout.isatty()
    last_line = None
    live_view = None
    try:
        while True:
            starting = live.starting()
            if tty:
                rendered = frame(starting, clock(), error)
                if live_view is None:
                    live_view = Live(
                        rendered, console=console, auto_refresh=False, transient=True
                    )
                    live_view.start()
                live_view.update(rendered, refresh=True)
            else:
                line = error or (
                    _starting_line(starting[0]) if starting else WAITING_TEXT
                )
                if line != last_line:
                    console.print(line, markup=False, highlight=False)
                    last_line = line
            sleep(interval)
            session, error = find_session_fn(name)
            if session:
                return session
    finally:
        if live_view is not None:
            live_view.stop()
