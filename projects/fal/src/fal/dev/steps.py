"""A framed step checklist for `fal dev up`: what will run, what is done, how long.

Each step goes pending, active, then done or failed, and shows its elapsed time.
The active step carries the glitch dust of `fal.dev.loading`, with the typical
wait where one is known. A header names the team, the box and the machine type.
Deploy logs are captured rather than streamed: the last lines are kept for a
failure (all of them with `verbose`).

Without a TTY, with ``NO_COLOR``, ``TERM=dumb``, a non-UTF-8 stream or
``FAL_DEV_NO_ANIMATION=1`` (the switches of `loading.animation_enabled`), the
same steps print as plain lines.

rich is imported only when something is drawn.
"""

from __future__ import annotations

import collections
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

from fal.dev.loading import (
    ACCENT,
    BELL_SECONDS,
    DUST_ROWS,
    ERROR,
    FPS,
    HEARTBEAT_SECONDS,
    MIN_PANEL_WIDTH,
    MUTED,
    animation_enabled,
    format_elapsed,
    typically,
)

if TYPE_CHECKING:
    from rich.console import Console, RenderableType
    from rich.live import Live
    from rich.text import Text

PENDING, ACTIVE, DONE, FAILED = "pending", "active", "done", "failed"
# Deploy log lines kept for a failure, unless verbose.
LOG_TAIL = 15
MAX_PANEL_WIDTH = 100
# The Ctrl-C promise only holds once a runner is being started.
HINT_STEPS = ("runner", "ready")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


@dataclass
class Step:
    key: str
    label: str
    expected: float | None = None  # typical seconds, when known
    state: str = PENDING
    detail: str = ""
    started: float | None = None
    ended: float | None = None


class Steps:
    """The checklist of one command. Unknown step keys are ignored, so callers
    can report every step without knowing which ones this run includes."""

    def __init__(
        self,
        specs: Sequence[tuple[str, str] | tuple[str, str, float | None]],
        *,
        title: str,
        team: str,
        machine: str,
        hint: str | None = None,
        verbose: bool = False,
        stream=None,
        animate: bool | None = None,
        environ: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.steps = [Step(*spec) for spec in specs]
        self.title, self.team, self.machine = title, team, machine
        self.hint = hint
        self.verbose = verbose
        self._stream = sys.stderr if stream is None else stream
        self._clock = clock
        self._animate = (
            animation_enabled(self._stream, environ) if animate is None else animate
        )
        self._t0 = clock()
        self._live: Live | None = None
        self._console: Console | None = None
        self._stop = threading.Event()
        self._opened = False
        self._closed = False
        self._lock = threading.RLock()
        self._log: collections.deque[str] | list[str] = (
            [] if verbose else collections.deque(maxlen=LOG_TAIL)
        )

    # --- state ------------------------------------------------------------

    def _get(self, key: str) -> Step | None:
        return next((s for s in self.steps if s.key == key), None)

    def state(self, key: str) -> str | None:
        step = self._get(key)
        return step.state if step else None

    def started(self, key: str) -> bool:
        step = self._get(key)
        return bool(step and step.started is not None)

    def elapsed(self, key: str) -> float | None:
        """Seconds the step took (so far, while active); None if it never began."""
        step = self._get(key)
        if step is None or step.started is None:
            return None
        return (step.ended if step.ended is not None else self._clock()) - step.started

    @property
    def active(self) -> Step | None:
        return next((s for s in self.steps if s.state == ACTIVE), None)

    def begin(self, key: str, detail: str = "") -> None:
        with self._lock:
            step = self._get(key)
            if step is None or step.state != PENDING:
                return
            self.open()
            step.state, step.started, step.detail = ACTIVE, self._clock(), detail
            self._plain(step, "...")
            if not self._animate and self.hint and key in HINT_STEPS:
                self.say(self.hint)

    def finish(self, key: str, detail: str | None = None) -> None:
        with self._lock:
            step = self._get(key)
            if step is None or step.state in (DONE, FAILED):
                return
            self.open()
            now = self._clock()
            if step.started is None:
                step.started = now
            step.state, step.ended = DONE, now
            step.detail = detail or ""
            self._plain(step, "done")

    def update(self, key: str, detail: str) -> None:
        """Change what an active step says it is doing."""
        with self._lock:
            step = self._get(key)
            if step is None or step.detail == detail:
                return
            step.detail = detail
            if step.state == ACTIVE and not self._animate and self._opened:
                self.say(f"{self._index(step)} {step.label}: {detail} ...")

    def fail(self, detail: str | None = None) -> None:
        """Mark the active step failed (else the first pending one), and stop."""
        with self._lock:
            # An error before any step began leaves the steps as they are.
            step = self.active
            if step is not None:
                now = self._clock()
                step.state, step.ended = FAILED, now
                if detail is not None:
                    step.detail = detail
                self._plain(step, "failed")
            self.close()

    # --- logs -------------------------------------------------------------

    def capture(self, line: str) -> None:
        line = _ANSI.sub("", line).rstrip()
        if line:
            self._log.append(line)

    @property
    def log_lines(self) -> list[str]:
        return list(self._log)

    def say(self, line: str) -> None:
        """A message above the panel (or a plain line)."""
        with self._lock:
            if self._console is not None:
                self._console.print(line, markup=False, highlight=False)
            else:
                self._stream.write(line + "\n")
                self._stream.flush()

    def print_log(self, console=None, *, title: str = "deploy log") -> None:
        """The captured log under the panel: the tail, or all of it if verbose."""
        lines = self.log_lines
        if not lines:
            return
        header = f"{title} ({'all' if self.verbose else f'last {len(lines)}'} lines):"
        if not self.verbose and len(lines) == LOG_TAIL:
            header += " rerun with --verbose for the whole log"
        if console is not None:
            console.print(header, style="dim", markup=False)
            for line in lines:
                console.print(f"  {line}", markup=False, highlight=False)
        else:
            self.say(header)
            for line in lines:
                self.say(f"  {line}")

    # --- drawing ----------------------------------------------------------

    def _index(self, step: Step) -> str:
        return f"[{self.steps.index(step) + 1}/{len(self.steps)}]"

    def _plain(self, step: Step, what: str) -> None:
        if self._animate:
            return
        if what == "...":
            hint = typically(step.expected)
            self.say(
                f"{self._index(step)} {step.label} ..." + (f" ({hint})" if hint else "")
            )
            return
        took = format_elapsed(self.elapsed(step.key) or 0)
        extra = f" - {step.detail}" if step.detail else ""
        self.say(f"{self._index(step)} {step.label}: {what} ({took}){extra}")

    def header_line(self) -> str:
        return f"{self.title}: team {self.team}, {self.machine}"

    def open(self) -> None:
        if self._opened:
            return
        self._opened = True
        self._t0 = self._clock()
        if self._animate:
            from rich.console import Console
            from rich.live import Live

            self._console = Console(file=self._stream, highlight=False)
            self._live = Live(
                get_renderable=self._renderable,
                console=self._console,
                refresh_per_second=FPS,
                transient=False,
            )
            self._live.start()
        else:
            self.say(self.header_line())
            threading.Thread(target=self._beat, daemon=True).start()

    def _beat(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            step = self.active
            if step is not None:
                took = format_elapsed(self.elapsed(step.key) or 0)
                self.say(f"still {step.label} ({took})")

    def _renderable(self):
        assert self._console is not None
        width, height = self._console.size
        return self.renderable(width, height)

    def _row(self, step: Step, label_width: int, t: float) -> Text:
        from rich.text import Text

        from fal.console.icons import (
            BRAILLE_SPINNER,
            STEP_DONE,
            STEP_FAILED,
            STEP_PENDING,
        )

        row = Text(no_wrap=True, overflow="ellipsis")
        took = self.elapsed(step.key)
        if step.state == DONE:
            row.append(f"{STEP_DONE} ", style="green")
            row.append(step.label.ljust(label_width))
            row.append(f"  {format_elapsed(took or 0):>6}", style="bold")
            if step.detail:
                row.append(f"  {step.detail}", style=f"default {MUTED}")
        elif step.state == FAILED:
            row.append(f"{STEP_FAILED} ", style=ERROR)
            row.append(step.label.ljust(label_width), style=ERROR)
            row.append(f"  {format_elapsed(took or 0):>6}", style="bold")
            if step.detail:
                row.append(f"  {step.detail}", style=ERROR)
        elif step.state == ACTIVE:
            frame = BRAILLE_SPINNER[int(t * 10) % len(BRAILLE_SPINNER)]
            row.append(f"{frame} ", style=ACCENT)
            row.append(step.label.ljust(label_width), style="bold")
            row.append(f"  {format_elapsed(took or 0):>6}", style="bold")
            note = step.detail or ""
            hint = typically(step.expected)
            if hint:
                note = f"{note}  {hint}" if note else hint
            if note:
                row.append(f"  {note}", style=f"default {MUTED}")
        else:
            row.append(f"{STEP_PENDING} ", style=f"default {MUTED}")
            row.append(step.label.ljust(label_width), style=f"default {MUTED}")
        return row

    def renderable(self, width: int, height: int) -> RenderableType:
        from rich.console import Group
        from rich.padding import Padding
        from rich.text import Text

        from fal.dev.loading import Dust, estimate_progress, panel

        now = self._clock()
        label_width = max(len(s.label) for s in self.steps)
        head = Text(no_wrap=True, overflow="ellipsis")
        head.append(f"team {self.team}", style="bold")
        head.append(f"   {self.machine}", style=f"default {MUTED}")
        parts: list = [head, Text("")]
        room = height >= len(self.steps) + DUST_ROWS + 8
        for step in self.steps:
            parts.append(self._row(step, label_width, now - self._t0))
            if step.state == ACTIVE and room:
                t = self.elapsed(step.key) or 0
                dust = Dust(t, DUST_ROWS, estimate_progress(t, step.expected))
                parts.append(Padding(dust, (0, 0, 0, 2)))
        active = self.active
        if self.hint and active is not None and active.key in HINT_STEPS:
            parts.append(Text(self.hint, style=f"default {MUTED}"))
        body = Group(*parts)
        if width < MIN_PANEL_WIDTH or height < len(self.steps) + 6:
            return body
        return panel(body, self.title, width=min(width, MAX_PANEL_WIDTH))

    def close(self) -> None:
        """Stop drawing; the panel stays on screen as it is."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            if self._live is not None:
                self._live.stop()
                self._live = None
                self._console = None

    def done(self) -> None:
        """Close after success; ring the bell if the whole run was a long one."""
        total = self._clock() - self._t0
        self.close()
        if self._opened and total >= BELL_SECONDS:
            try:
                if self._stream.isatty():
                    self._stream.write("\a")
                    self._stream.flush()
            except (AttributeError, ValueError, OSError):
                pass
