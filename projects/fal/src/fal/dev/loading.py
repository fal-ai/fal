"""Glitch-dust loading states for `fal dev` waits and the dash.

Chunky clusters of square pixels grow, flicker and fade in a panel while
something slow happens, with the real status, the elapsed time and the typical
duration under it. More clusters burn as the progress (a time-based guess
unless the caller knows better) moves on; now and then a cluster's row jumps
sideways. The status text itself is never scrambled.

`Loading` runs a wait in a terminal (`fal dev up`, `hot start`, `hot restart`);
the dash calls `dust_panel_body` from its own render loop. Without a TTY, with
``NO_COLOR``, ``TERM=dumb``, a non-UTF-8 stream or ``FAL_DEV_NO_ANIMATION=1``,
`Loading` prints plain status lines instead.

Frames are pure functions of the time, so they are deterministic and testable.
rich is imported only when something is drawn: importing this module is free.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from typing import TYPE_CHECKING, Callable, Mapping

if TYPE_CHECKING:
    from rich.console import Console, RenderableType
    from rich.live import Live
    from rich.text import Text

FPS = 12
HEARTBEAT_SECONDS = 30
NO_ANIMATION_ENV = "FAL_DEV_NO_ANIMATION"
# Narrower or shorter terminals get one line instead of the panel.
MIN_PANEL_WIDTH = 60
MIN_PANEL_HEIGHT = 12
DUST_ROWS = 4

# Waits of `fal dev`: label and typical seconds, from real runs.
STATES: dict[str, tuple[str, int]] = {
    "gpu-runner": ("waiting for a runner", 100),
    "weights": ("loading the model", 35),
    "hot-restart": ("restarting the hot server: loading weights", 35),
}

# Runner states, as the platform names them, in words.
RUNNER_PHASES = {
    "pending": "queued",
    "docker_pull": "pulling the image",
    "setup": "starting the box",
}
# A wait at least this long ends with the terminal bell, for whoever looked away.
BELL_SECONDS = 20


def runner_phase(states: str) -> str:
    """'docker_pull, setup' -> 'pulling the image, starting the box'."""
    return ", ".join(RUNNER_PHASES.get(s.strip(), s.strip()) for s in states.split(","))


# Named ANSI colors, as in the dash: the terminal's palette keeps them readable.
BORDER = "bright_black"
ACCENT = "cyan"
ERROR = "red"
MUTED = "dim"
TITLE = "bold default"

# One square per two columns, so the grid is about square in a terminal cell.
CELL_COLUMNS = 2
# Flickers per second of the squares inside a cluster.
FLICKER = 8
# Square colors and their odds, as in the fal glitch dust: blue, light blue,
# ink and a few grey.
DUST_COLORS = (
    ("blue", 0.42),
    ("bright_blue", 0.34),
    ("bold default", 0.12),
    ("bright_black", 0.12),
)


def _rand(*keys: int) -> float:
    """A stable pseudo-random number in [0, 1) for integer keys."""
    h = 2166136261
    for key in keys:
        h = ((h ^ (key & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0x5BD1E995) & 0xFFFFFFFF
    h ^= h >> 15
    return h / 2**32


def format_elapsed(seconds: float) -> str:
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    return f"{total // 60}m{total % 60:02d}s"


def estimate_progress(t: float, expected: float | None) -> float:
    """Time-based guess in [0, 0.985): fast at first, never reaches 1."""
    if not expected or expected <= 0:
        expected = 60
    return min(0.985, 1 - math.exp(-3 * t / expected))


def _burst(t: float, period: float, length: float, chance: float, salt: int):
    """(active, index) for a glitch burst of `length` seconds every `period`."""
    index = int(t / period)
    active = (t % period) < length and _rand(index, salt) < chance
    return active, index


def _clusters(
    t: float, cols: int, rows: int, progress: float
) -> list[tuple[float, float, float, float]]:
    """(x, y, radius, strength) of each cluster burning at `t`, in grid cells.

    Each slot grows a cluster at a random spot, holds it and fades it out, then
    starts over somewhere else. More slots burn as `progress` (0..1) grows.
    """
    slots = max(1, cols // 8)
    burning = max(1, round(slots * (0.25 + 0.75 * progress)))
    out = []
    for i in range(burning):
        period = 2.4 + 1.6 * _rand(i, 41)
        cycle = t / period + _rand(i, 42)
        n = int(cycle)
        # Grow fast, hold, fade: a flattened sine.
        strength = min(1.0, 1.6 * math.sin(math.pi * (cycle - n)))
        radius = (2.5 + 3.5 * _rand(i, n, 45)) * (0.4 + 0.6 * strength)
        out.append((_rand(i, n, 43) * cols, _rand(i, n, 44) * rows, radius, strength))
    return out


def _square(x: int, y: int, t: float, clusters, squares: str) -> tuple[str, str] | None:
    """The square at grid cell (x, y): (glyph, style), or None for empty."""
    tick = int(t * FLICKER)
    heat = 0.0
    for cx, cy, radius, strength in clusters:
        if radius > 0:
            heat = max(heat, strength * (1 - math.hypot(x - cx, y - cy) / radius))
    # Ragged edges and a flicker inside, rather than round blobs.
    heat *= 0.7 + 0.6 * _rand(x, y, tick, 6)
    if heat > 0.1 and _rand(x, y, tick, 7) < 0.45 + heat:
        size = 2 if heat > 0.45 else 1 if heat > 0.2 else 0
        pick, style = _rand(x, y, int(t * 3), 8), DUST_COLORS[-1][0]
        for color, odds in DUST_COLORS:
            if pick < odds:
                style = color
                break
            pick -= odds
        return squares[size], style
    if _rand(x, y, tick // 4, 9) < 0.006:
        return squares[0], "bright_black"
    return None


def dust_rows(t: float, width: int, rows: int, progress: float) -> list[Text]:
    """Clusters of square pixels, more of them as `progress` (0..1) grows."""
    from rich.text import Text

    from fal.console.icons import DUST_SQUARES

    cols = width // CELL_COLUMNS
    clusters = _clusters(t, cols, rows, progress)
    jump, g = _burst(t, 3.1, 0.2, 0.7, 31)
    jump_row = int(_rand(g, 32) * rows)
    out = []
    for y in range(rows):
        cells = [_square(x, y, t, clusters, DUST_SQUARES) for x in range(cols)]
        if jump and y == jump_row and cols > 1:
            dx = (int((_rand(g, 33) - 0.5) * 6) or 2) % cols
            cells = cells[-dx:] + cells[:-dx]
        row = Text(no_wrap=True, overflow="crop")
        for cell in cells:
            if cell is None:
                row.append(" " * CELL_COLUMNS)
            else:
                row.append(cell[0], style=cell[1])
                row.append(" " * (CELL_COLUMNS - 1))
        row.append(" " * (width - cols * CELL_COLUMNS))
        out.append(row)
    return out


def status_line(
    t: float, text: str, elapsed: float, hint: str = "", *, spinner: bool = True
) -> Text:
    """Spinner, status, elapsed time and a dim hint such as 'typically ~35s'."""
    from rich.text import Text

    from fal.console.icons import BRAILLE_SPINNER

    line = Text(no_wrap=True, overflow="ellipsis")
    if spinner:
        frame = BRAILLE_SPINNER[int(t * 10) % len(BRAILLE_SPINNER)]
        line.append(f"{frame} ", ACCENT)
    line.append(text)
    line.append(f"  {format_elapsed(elapsed)}", style="bold")
    if hint:
        line.append(f"  {hint}", style=f"default {MUTED}")
    return line


def sweep(t: float, text: str) -> Text:
    """Bold text with a bright scanline passing over it; the text is unchanged."""
    from rich.text import Text

    out = Text(no_wrap=True, overflow="ellipsis")
    pos = (t * 18) % (len(text) + 14) - 7
    for i, char in enumerate(text):
        out.append(char, style=f"bold {ACCENT}" if abs(i - pos) < 3 else "bold")
    return out


def typically(expected: float | None) -> str:
    return f"typically ~{format_elapsed(expected)}" if expected else ""


class Dust:
    """Dust rows as wide as the space they are drawn in."""

    def __init__(self, t: float, rows: int, progress: float):
        self.t, self.rows, self.progress = t, rows, progress

    def __rich_console__(self, console, options):
        yield from dust_rows(
            self.t, max(10, options.max_width), self.rows, self.progress
        )


class StatusLine:
    """status_line that drops its hint rather than cut it when space is short."""

    def __init__(self, t: float, text: str, hint: str, spinner: bool = True):
        self.t, self.text, self.hint, self.spinner = t, text, hint, spinner

    def __rich_console__(self, console, options):
        t, text, spinner = self.t, self.text, self.spinner
        line = status_line(t, text, t, self.hint, spinner=spinner)
        if line.cell_len > options.max_width:
            line = status_line(t, text, t, spinner=spinner)
        yield line


def dust_panel_body(
    t: float,
    status: str,
    *,
    expected: float | None = None,
    headline: str | None = None,
    extra: list | None = None,
    animate: bool = True,
) -> RenderableType:
    """What goes inside a panel while waiting: [headline], dust, status, [extra].

    `t` is the seconds since the wait began. Without `animate` (the switches
    in animation_enabled), only the text: no dust, spinner or scanline.
    """
    from rich.console import Group
    from rich.text import Text

    parts: list = []
    if headline:
        parts.append(sweep(t, headline) if animate else Text(headline, style="bold"))
    if animate:
        parts.append(Dust(t, DUST_ROWS, estimate_progress(t, expected)))
    parts.append(StatusLine(t, status, typically(expected), spinner=animate))
    return Group(*parts, *(extra or []))


def one_line(t: float, width: int, status: str, expected: float | None) -> Text:
    """Six dust cells and the status, for narrow terminals."""
    p = estimate_progress(t, expected)
    dust = dust_rows(t, 6, 1, 0.3 + 0.6 * p)[0]
    dust.append(" ")
    line = status_line(t, status, t, typically(expected))
    if len(dust.plain) + len(line.plain) > width:
        line = status_line(t, status, t)  # drop the hint rather than cut it
    dust.append_text(line)
    dust.truncate(width, overflow="ellipsis")
    return dust


def panel(content: RenderableType, title: str | Text, **kwargs):
    """A frame in the dash style: rounded, muted, bold title on the left."""
    from rich.box import ROUNDED
    from rich.panel import Panel
    from rich.text import Text

    if isinstance(title, str):
        title = Text(title, style=TITLE)
    return Panel(
        content,
        title=title,
        title_align="left",
        box=ROUNDED,
        border_style=BORDER,
        **kwargs,
    )


# --- switches and the runner ---------------------------------------------


def animation_enabled(stream=None, environ: Mapping[str, str] | None = None) -> bool:
    """False for non-TTYs, NO_COLOR, TERM=dumb, non-UTF-8 and the off switch."""
    env = os.environ if environ is None else environ
    stream = sys.stderr if stream is None else stream
    if env.get(NO_ANIMATION_ENV, "").strip().lower() not in ("", "0", "false", "no"):
        return False
    if env.get("NO_COLOR") or env.get("TERM", "").lower() == "dumb":
        return False
    try:
        if not stream.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
    return encoding.startswith("utf")


class Loading:
    """Context manager for a wait: a dust panel when animating, else plain lines.

    Messages written with `say` while the panel is up print above it.
    """

    def __init__(
        self,
        status: str,
        *,
        title: str = "",
        expected: float | None = None,
        headline: str | None = None,
        hint: str | None = None,
        subtitle: str | None = None,
        stream=None,
        animate: bool | None = None,
        environ: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.status = status
        self.title = title
        self.expected = expected
        self.headline = headline
        self.hint = hint
        self.subtitle = subtitle
        self._stream = sys.stderr if stream is None else stream
        self._clock = clock
        self._animate = (
            animation_enabled(self._stream, environ) if animate is None else animate
        )
        self._start = clock()
        self._live: Live | None = None
        self._console: Console | None = None
        self._stop = threading.Event()
        self._started = False
        self._closed = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def elapsed(self) -> float:
        return self._clock() - self._start

    def update(self, status: str) -> None:
        if status != self.status:
            self.status = status
            if self._started and not self._animate:
                self.say(f"{status} ...")

    def say(self, line: str) -> None:
        if self._console is not None:
            self._console.print(line, markup=False, highlight=False)
        else:
            # One write, so a heartbeat from its thread cannot split a log line.
            self._stream.write(line + "\n")
            self._stream.flush()

    @property
    def summary(self) -> str:
        """The wait in one line: the headline when there is one."""
        return self.headline or self.status

    def renderable(self, width: int, height: int) -> RenderableType:
        from rich.text import Text

        t = self.elapsed
        if width < MIN_PANEL_WIDTH or height < MIN_PANEL_HEIGHT:
            return one_line(t, width, self.summary, self.expected)
        extra = [Text(self.hint, style=f"default {MUTED}")] if self.hint else None
        body = dust_panel_body(
            t,
            self.status,
            expected=self.expected,
            headline=self.headline,
            extra=extra,
        )
        subtitle = (
            Text(self.subtitle, style=f"default {MUTED}") if self.subtitle else None
        )
        return panel(
            body,
            self.title,
            subtitle=subtitle,
            subtitle_align="right",
            width=min(width, 100),
        )

    def _renderable(self):
        assert self._console is not None
        width, height = self._console.size
        return self.renderable(width, height)

    def start(self) -> Loading:
        """Begin showing the wait (once); the elapsed time counts from here."""
        if self._started:
            return self
        self._started = True
        self._start = self._clock()
        if self._animate:
            from rich.console import Console
            from rich.live import Live

            self._console = Console(file=self._stream, highlight=False)
            self._live = Live(
                get_renderable=self._renderable,
                console=self._console,
                refresh_per_second=FPS,
                transient=True,
            )
            self._live.start()
        else:
            hint = typically(self.expected)
            self.say(f"{self.summary} ..." + (f" ({hint})" if hint else ""))
            threading.Thread(target=self._beat, daemon=True).start()
        return self

    def __enter__(self) -> Loading:
        return self.start()

    def _beat(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            self.say(f"still {self.summary} ({format_elapsed(self.elapsed)})")

    def close(self) -> None:
        """Clear the animation without a final line."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._live is not None:
            self._live.stop()
            self._live = None
            self._console = None

    def done(self) -> None:
        """Clear the wait after success; ring the bell if it was a long one."""
        self.close()
        if self._started and self.elapsed >= BELL_SECONDS:
            try:
                if self._stream.isatty():
                    self._stream.write("\a")
                    self._stream.flush()
            except (AttributeError, ValueError, OSError):
                pass

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
