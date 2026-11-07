"""`fal dev app split`: the dash and the sync watcher next to your current pane.

tmux when ``$TMUX`` is set, iTerm via AppleScript when running in iTerm,
otherwise the two commands are printed to run in two panes yourself.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from typing import Mapping, Sequence

from fal.dev.errors import DevError


def commands(name: str | None) -> list[list[str]]:
    """The dash and the watcher; both can start before a box exists."""
    extra = ["--name", name] if name else []
    # The interpreter running this command, so a venv's fal is the one used.
    fal = [sys.executable, "-m", "fal"]
    return [
        [*fal, "dev", "app", "dash", *extra],
        [*fal, "dev", "app", "sync", "--watch", *extra],
    ]


def _shell(command: Sequence[str], root: str) -> str:
    return f"cd {shlex.quote(root)} && {shlex.join(command)}"


def tmux_dash(dash: Sequence[str], root: str) -> list[str]:
    """Dash on the right of this pane; prints the new pane's id. Focus stays."""
    return [
        "tmux", "split-window", "-h", "-d", "-P", "-F", "#{pane_id}",
        "-c", root, _shell(dash, root),
    ]  # fmt: skip


def tmux_sync(pane: str, sync: Sequence[str], root: str) -> list[str]:
    """The watcher under the dash pane."""
    return [
        "tmux",
        "split-window",
        "-v",
        "-d",
        "-t",
        pane,
        "-c",
        root,
        _shell(sync, root),
    ]


def iterm_script(cmds: Sequence[Sequence[str]], root: str) -> str:
    dash, sync = (_shell(c, root) for c in cmds)

    def q(text: str) -> str:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'

    return "\n".join(
        [
            'tell application "iTerm2"',
            "  tell current session of current window",
            "    set dashSession to (split vertically with default profile)",
            "  end tell",
            "  tell dashSession",
            f"    write text {q(dash)}",
            "    set syncSession to (split horizontally with default profile)",
            "  end tell",
            "  tell syncSession",
            f"    write text {q(sync)}",
            "  end tell",
            "  tell current session of current window to select",
            "end tell",
        ]
    )


def run(
    name: str | None,
    root: str,
    env: Mapping[str, str] | None = None,
    runner=subprocess.run,
) -> str:
    """Open the panes if this terminal allows it. Returns "tmux", "iterm" or
    "print" (the caller prints the commands)."""
    env = os.environ if env is None else env
    cmds = commands(name)
    try:
        if env.get("TMUX"):
            made = runner(
                tmux_dash(cmds[0], root), check=True, capture_output=True, text=True
            )
            pane = (made.stdout or "").strip() or "{right}"
            runner(tmux_sync(pane, cmds[1], root), check=True)
            return "tmux"
        if env.get("TERM_PROGRAM") == "iTerm.app":
            runner(["osascript", "-e", iterm_script(cmds, root)], check=True)
            return "iterm"
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DevError(f"Could not open the panes: {exc}") from exc
    return "print"
