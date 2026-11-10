"""The edit loop of `fal dev hot sync --watch --run`, as you save.

A card says what is being watched and which code version the model runs. Each
save then prints one line per step (sync, run, error), with the image result
drawn under its line; a failed run shows an error panel that says whether the
model is still loaded, and the next good save says the error is fixed.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from fal.dev.cli.error_panel import error_panel
from fal.dev.hot import RunResult
from fal.dev.loading import panel

OK, ERROR, WARNING, MUTED = "green", "red", "yellow", "dim"
# Longer exceptions go in the error panel's body, not its title.
TITLE_CHARS = 60
FRAME = re.compile(r'^\s*File "(?P<path>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)')


def failing_line(traceback: str | None) -> tuple[str, str] | None:
    """("infer.py:14", "image = sepia(image)") for the last frame of your code."""
    lines = (traceback or "").splitlines()
    found = None
    for i, line in enumerate(lines):
        match = FRAME.match(line)
        if match and "site-packages" not in match["path"]:
            path = Path(match["path"])
            if path.name == "hot_server.py":
                continue  # the server calling infer(), not your code
            code = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if FRAME.match(code) or code.startswith("^"):
                code = ""
            found = (f"{path.name}:{match['line']}", code)
    return found


def exception_line(traceback: str | None) -> str | None:
    """The last line of a traceback: `NameError: name 'x' is not defined`."""
    lines = [line for line in (traceback or "").splitlines() if line.strip()]
    return lines[-1].strip() if lines else None


class LoopView:
    """Reports each save of a watched hot project; see fal.dev.hot.WatchView."""

    def __init__(
        self,
        console: Console,
        status: dict | None = None,
        on_image: Callable[[Path, dict], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.console = console
        self.status = status  # None: the hot server did not answer
        self.on_image = on_image
        self.clock = clock
        self.failing = False  # the last run failed; the next good one says fixed

    def _event(self, word: str, style: str, *parts: Text | str) -> None:
        row = Text(no_wrap=True, overflow="ellipsis")
        row.append(f" {time.strftime('%H:%M:%S', time.localtime(self.clock()))}  ")
        row.append(f"{word:<7}", style=f"bold {style}")
        for i, part in enumerate(parts):
            if i:
                row.append("  ")
            row.append(part if isinstance(part, Text) else Text(part))
        self.console.print(row, no_wrap=True, overflow="ellipsis", crop=True)

    def watching(self, root: Path, run: bool) -> None:
        status = self.status or {}
        version = status.get("code_version")
        loaded = status.get("load_seconds")
        if self.status is None:
            model = Text("unknown (the hot server did not answer)", style=WARNING)
        elif not status.get("ready"):
            model = Text("not loaded: start it with fal dev hot start", style=WARNING)
        else:
            model = Text(f"loaded in {loaded}s" if loaded else "loaded")
        facts = Table.grid(padding=(0, 3))
        facts.add_column(no_wrap=True)
        facts.add_column(no_wrap=True)
        facts.add_row(
            Text.assemble(
                ("code ", MUTED), f"v{version}" if version is not None else "-"
            ),
            Text.assemble(("model ", MUTED), model),
        )
        then = "sync it and run a request" if run else "sync it"
        title = Text.styled("fal dev hot", style="bold cyan")
        subtitle = Text("the box keeps running", style=MUTED)
        self.console.print(
            panel(
                _stack(
                    Text.assemble(("watching ", MUTED), _short(root)),
                    facts,
                    Text(f"save a file to {then}; Ctrl-C stops watching", style=MUTED),
                ),
                title,
                subtitle=subtitle,
                subtitle_align="right",
            )
        )

    def synced(self, files: list[str], after_save: float) -> None:
        names = ", ".join(files)
        self._event(
            "sync", "cyan", Text(names), Text(f"{after_save:.2f}s after save", MUTED)
        )

    def _detail(self, text: str, hint: str) -> None:
        """An error's full text and what happens next, under its event line."""
        self.console.print(Padding(Text(text), (0, 0, 0, 18)))
        self.console.print(Text(f"{' ' * 18}{hint}", style=MUTED), crop=True)

    def sync_failed(self, files: list[str], error: str) -> None:
        self._event("sync", ERROR, Text("failed", style="bold"), Text(", ".join(files)))
        self._detail(error, "sent again with the next save")

    def restarting(self, reason: str) -> None:
        self._event(
            "reload",
            WARNING,
            Text(f"FULL RESTART ({reason})"),
            Text("the weights reload", MUTED),
        )

    def lost(self, error: str) -> None:
        self._event("error", ERROR, Text("no result", style="bold"))
        self._detail(error, "the next save retries")

    def ran(self, result: RunResult, edited_at: float) -> None:
        if not result.ok:
            self._failed(result)
            return
        headers = result.headers
        parts: list[Text | str] = [
            Text(f"v{headers.get('x-hot-code-version', '?')}", "bold")
        ]
        parts.append(Text(f"infer {headers.get('x-hot-infer-ms', '?')} ms"))
        if headers.get("x-hot-reload-ms"):
            parts.append(Text(f"reload {headers['x-hot-reload-ms']} ms", MUTED))
        parts.append(Text(f"edit to result {self.clock() - edited_at:.2f}s", MUTED))
        if result.restarted:
            parts.append(Text("weights reloaded", style=WARNING))
        if self.failing:
            parts.append(Text("fixed", style=f"bold {OK}"))
            self.failing = False
        self._event("ok", OK, *parts)
        if result.path:
            self.console.print(Text(f"{' ' * 18}{result.path}", style=MUTED), crop=True)
        if result.path and self.on_image:
            self.on_image(result.path, headers)  # HotPreview never raises
        elif not result.path:
            body = result.body.decode(errors="replace")[:2000]
            self.console.print(Padding(Text(body), (0, 0, 0, 18)))

    def _failed(self, result: RunResult) -> None:
        self.failing = True
        error = result.error
        summary = exception_line(error["traceback"]) or error["message"] or ""
        what = error["message"] or "the request failed"
        self._event("error", ERROR, Text(f"{result.status}"), Text(what))
        # A long exception does not fit a title: the type there, the rest below.
        title = summary
        if len(summary) > TITLE_CHARS:
            head = summary.split(":", 1)[0]
            title = head if len(head) <= TITLE_CHARS else "the request failed"
        lines: list[Text | str] = [Text(summary)] if title != summary else []
        if where := failing_line(error["traceback"]):
            location, code = where
            lines.append(
                Text.assemble((location, "cyan"), ("  " + code if code else ""))
            )
        if result.model_loaded:
            lines.append(
                Text(
                    "the model is still loaded: fix it and save, the next save retries",
                    MUTED,
                )
            )
        else:
            lines.append(Text("the next save retries", MUTED))
        self.console.print(error_panel(title or what, lines))


def _short(root: Path) -> str:
    """The watched folder relative to here when it is under it."""
    try:
        return str(root.resolve().relative_to(Path.cwd().resolve())) or "."
    except ValueError:
        return str(root)


def _stack(*rows) -> Table:
    table = Table.grid()
    for row in rows:
        table.add_row(row)
    return table
