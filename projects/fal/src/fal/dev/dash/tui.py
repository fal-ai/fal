"""The interactive loop of `fal dev dash`: a rich Live screen plus single-key commands.

rich has no input handling, so keys are read from stdin in cbreak mode with
select(); that is all the dashboard needs, without a full TUI framework.
"""

from __future__ import annotations

import os
import select
import sys
import threading
import time
from contextlib import contextmanager

from rich.console import Console
from rich.live import Live
from rich.text import Text

from fal.dev.box import Box
from fal.dev.dash import model
from fal.dev.dash.view import render

CONFIRM_SECONDS = 5.0


@contextmanager
def _cbreak(fd: int):
    """Keys arrive one at a time, unechoed; Ctrl-C still interrupts."""
    import termios
    import tty

    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _read_key(fd: int, timeout: float) -> str | None:
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        return None
    data = os.read(fd, 32)
    if not data:
        return "q"  # the terminal went away
    # Arrow keys and other escape sequences are not commands.
    if data.startswith(b"\x1b"):
        return None
    return data[:1].decode(errors="ignore").lower()


def print_once(box: Box, console: Console | None = None) -> None:
    """One poll, one frame: for scripts, captures, and terminals without a TTY."""
    console = console or Console()
    poller = model.Poller(box)
    poller.prime()
    console.print(render(poller.snapshot, box))


def run_dashboard(
    box: Box, interval: float = 3.0, console: Console | None = None
) -> None:
    """Show the dashboard until q or Ctrl-C."""
    console = console or Console()
    # No termios on Windows; no screen to draw on without a terminal.
    if os.name == "nt" or not (sys.stdin.isatty() and sys.stdout.isatty()):
        print_once(box, console)
        return
    poller = model.Poller(box, interval=interval)
    poller.start()
    fd = sys.stdin.fileno()
    pending_down = 0.0
    pending_quit = False
    busy = threading.Lock()

    def act(label: str, func, *args) -> None:
        """Run a slow action without freezing the screen."""

        def run() -> None:
            try:
                poller.say(f"{label} ...", seconds=3600)
                try:
                    poller.say(func(*args))
                except Exception as exc:
                    poller.say(f"{label} failed: {exc}")
                poller.refresh_state()
            finally:
                busy.release()

        # One action at a time; a key pressed meanwhile is dropped, not queued.
        if not busy.acquire(blocking=False):
            poller.say("busy, try again in a moment")
            return
        threading.Thread(target=run, daemon=True).start()

    try:
        with _cbreak(fd):
            live = Live(
                render(poller.snapshot, box, screen=True),
                console=console,
                screen=True,
                auto_refresh=False,
            )
            live.start()
            try:
                while True:
                    key = _read_key(fd, 1.0)
                    runner_id = poller.snapshot.runner_id
                    if key == "q":
                        if not busy.locked() or pending_quit:
                            break
                        pending_quit = True
                        poller.say(
                            "an action is still running; press q again to quit "
                            "and abandon it (a stop not yet done leaves the box up)"
                        )
                        continue
                    if key:
                        pending_quit = False
                    if key and key != "d" and pending_down:
                        # Any other key cancels a pending down.
                        pending_down = 0.0
                        poller.say("", seconds=0)
                    if key == "d":
                        if time.time() - pending_down < CONFIRM_SECONDS:
                            pending_down = 0.0
                            act("saving /work and stopping the box", model.down, box)
                        else:
                            pending_down = time.time()
                            poller.say(
                                "press d again to save /work and stop the box",
                                seconds=CONFIRM_SECONDS,
                            )
                    elif key in ("s", "h", "r") and not runner_id:
                        poller.say("the box is not running")
                    elif key == "s" and runner_id:
                        live.stop()
                        with _restored(fd):
                            console.print(
                                Text(
                                    f"opening a shell on {runner_id}; "
                                    "exit it to return",
                                    style="dim",
                                )
                            )
                            try:
                                model.open_shell(box, runner_id)
                            except Exception as exc:
                                poller.say(f"shell failed: {exc}")
                        live.start()
                    elif key == "h" and runner_id:
                        act("holding", model.hold, box)
                    elif key == "r" and runner_id:
                        act(
                            "restarting the hot server",
                            model.hot_restart,
                            box,
                            runner_id,
                        )
                    live.update(render(poller.snapshot, box, screen=True), refresh=True)
            finally:
                live.stop()
    except KeyboardInterrupt:
        pass
    finally:
        poller.stop()


@contextmanager
def _restored(fd: int):
    """Hand the terminal back in its normal mode for an interactive shell."""
    import termios

    current = termios.tcgetattr(fd)
    # Back to canonical, echoing input; the shell sets raw mode itself.
    normal = list(current)
    normal[3] |= termios.ICANON | termios.ECHO
    termios.tcsetattr(fd, termios.TCSADRAIN, normal)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, current)
