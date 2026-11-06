"""Rendering for `fal dev dash`: a Snapshot in, a rich renderable out."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable

from rich.box import ROUNDED
from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.padding import Padding
from rich.panel import Panel
from rich.segment import Segment
from rich.table import Table
from rich.text import Text

from fal.console.icons import (
    get_bar_empty_icon,
    get_bar_full_icon,
    get_column_separator_icon,
    get_degrees_celsius_icon,
    get_inline_separator_icon,
    get_state_dot_icon,
)
from fal.dev.dash.model import (
    _CONTROL,
    HOLD_HOURS,
    IdleView,
    Snapshot,
    idle_view,
    price_per_hour,
)

# One style per role, after oh-my-pi's theme tokens: frames are all the same
# muted color, and color goes on titles, badges and values, never on borders.
# Named ANSI colors, so the terminal's own palette keeps them readable on
# light and dark backgrounds.
BORDER = "bright_black"
ACCENT = "cyan"
SUCCESS = "green"
WARNING = "yellow"
ERROR = "red"
MUTED = "dim"
TITLE = "bold default"


def _separator() -> Text:
    return Text(f" {get_column_separator_icon()} ", style=BORDER)


def _dot() -> str:
    return get_inline_separator_icon()


STATE_STYLES = {
    "running": SUCCESS,
    "starting": WARNING,
    "stopping": WARNING,
    "stopped": ERROR,
    "failed": ERROR,
    "unknown": MUTED,
}
# (key, label, needs a live runner)
KEY_HINTS = [
    ("s", "shell", True),
    ("h", f"hold {HOLD_HOURS:g}h", True),
    ("d", "down", False),
    ("r", "hot restart", True),
    ("q", "quit", False),
]
FOOTER_LINES = 3
# Narrower than this, the GPU row no longer fits half the width.
TWO_COLUMN_WIDTH = 100
PRINT_LOG_LINES = 8
# The logs panel needs a frame plus at least one line.
MIN_FILL = 3
# Lines the platform logs for every runner exec, including the dashboard's own probes.
NOISE_PREFIXES = ("Skipping authentication for /Isolate/",)


def safe(value) -> str:
    """Box-originated text, flattened to one line without control characters."""
    return _CONTROL.sub("", " ".join(str(value).split()))


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h {minutes:02d}m"
    return f"{hours // 24}d {hours % 24}h"


def size(num_bytes: float | None) -> str:
    if num_bytes is None:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return (
                f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
            )
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def ago(then: float | None, now: float) -> str:
    return "never" if not then else f"{duration(now - then)} ago"


def _bar(pct: float | None, width: int = 8) -> Text:
    if pct is None:
        return Text("-" * width, style="dim")
    filled = round(min(max(pct, 0), 100) / 100 * width)
    style = ERROR if pct >= 90 else WARNING if pct >= 60 else SUCCESS
    return Text.styled(get_bar_full_icon() * filled, style=style) + Text(
        get_bar_empty_icon() * (width - filled), style=BORDER
    )


def _kv(rows: list[tuple[str, RenderableType | str]]) -> Table:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    for key, value in rows:
        # Plain strings become Text so that box text is never parsed as markup.
        table.add_row(key, Text(value) if isinstance(value, str) else value)
    return table


def _panel(content: RenderableType, title: str | Text, **kwargs) -> Panel:
    """Every frame looks the same: rounded, muted, bold title on the left."""
    # Titles take the border's style as a base; reset the color to the default.
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


def _waiting(title: str) -> Panel:
    return _panel(Text("waiting for the box", style=MUTED), title)


class _StatusLine:
    """Segments joined by thin separators, like a shell status line.

    A narrow terminal wraps between segments, never inside one.
    """

    def __init__(self, *segments: Text) -> None:
        self.segments = segments

    def __rich_console__(self, console, options):
        width = options.max_width
        lines = [Text()]
        for segment in self.segments:
            line = lines[-1]
            separator = _separator()
            if line and line.cell_len + separator.cell_len + segment.cell_len > width:
                lines.append(Text())
                line = lines[-1]
            if line:
                line.append_text(separator)
            line.append_text(segment)
        yield Text("\n").join(lines)


def _live(snapshot: Snapshot) -> bool:
    return bool(snapshot.info and snapshot.info.live_runner)


def _box_clock(snapshot: Snapshot, now: float) -> float:
    """The box's current time, from its last probe; avoids comparing two clocks."""
    probe = snapshot.probe or {}
    if probe.get("now") and snapshot.probe_at:
        return probe["now"] + (now - snapshot.probe_at)
    return now


def _uptime(snapshot: Snapshot, now: float) -> float | None:
    info = snapshot.info
    if not info or info.uptime_seconds is None:
        return None
    return info.uptime_seconds + (now - info.fetched_at)


def _idle(snapshot: Snapshot, box, now: float) -> IdleView | None:
    probe = snapshot.probe
    if not probe or not _live(snapshot):
        return None
    box_now = _box_clock(snapshot, now)
    uptime = _uptime(snapshot, now)
    started = box_now - uptime if uptime is not None else None
    claude_counts = bool(
        box.claude_enabled or (probe.get("claude") or {}).get("running")
    )
    return idle_view(probe, box.idle_minutes, started, claude_counts, box_now)


def header(snapshot: Snapshot, now: float) -> RenderableType:
    info = snapshot.info
    if info is None:
        return _panel(Text("looking up the box ...", style=MUTED), "fal dev")
    probe = snapshot.probe or {}
    num_gpus = len(probe.get("gpus") or []) or 1
    uptime = _uptime(snapshot, now)
    rate = price_per_hour(info.machine_type, num_gpus)
    if uptime is None or (_live(snapshot) and not probe):
        # Before the first probe the GPU count, and so the rate, is unknown.
        cost = Text("cost -", style=MUTED)
    elif rate is None:
        cost = Text("cost unknown", style=MUTED)
    else:
        spent = rate * (uptime or 0) / 3600
        cost = Text.styled(f"~${spent:.2f}", style="bold") + Text(
            f" at ${rate:.2f}/h list, est.", style=MUTED
        )
    state_style = STATE_STYLES.get(info.state, MUTED)
    line = _StatusLine(
        Text(f"{get_state_dot_icon()} {info.state}", style=f"bold {state_style}"),
        Text(info.app_name),
        Text(info.team or "personal", style=MUTED),
        Text(f"{num_gpus}x {info.machine_type or '?'}"),
        Text(f"up {duration(uptime)}"),
        cost,
    )
    title = Text.styled("fal dev", style=f"bold {ACCENT}") + Text(
        f"  {info.name}", style=TITLE
    )
    subtitle = Text(f"runner {info.runner_id or '-'}", style=f"default {MUTED}")
    return _panel(line, title, subtitle=subtitle, subtitle_align="right")


def gpu_panel(snapshot: Snapshot) -> RenderableType:
    probe = snapshot.probe or {}
    gpus = probe.get("gpus") or []
    if not gpus:
        detail = probe.get("gpu_error") or (
            "waiting for the box" if not probe else "no GPUs reported"
        )
        return _panel(Text(safe(detail), style=MUTED), "GPU")
    table = Table.grid(padding=(0, 1))
    for _ in range(5):
        table.add_column(no_wrap=True)
    for gpu in gpus:
        util = gpu.get("util_pct")
        used, total = gpu.get("mem_used_mib"), gpu.get("mem_total_mib")
        power = gpu.get("power_w")
        watts = round(power) if power is not None else "-"
        table.add_row(
            Text(f"#{gpu.get('index')}", style="dim"),
            _bar(util),
            Text(f"{util if util is not None else '-':>3}%"),
            Text(f"{(used or 0) / 1024:.1f}/{(total or 0) / 1024:.0f} GiB"),
            Text(
                f"{gpu.get('temp_c', '-')}{get_degrees_celsius_icon()} {watts}W",
                style="dim",
            ),
        )
    title = Text.styled("GPU", style=TITLE) + Text(
        f"  {safe(gpus[0].get('name') or '')}", style=f"default {MUTED}"
    )
    return _panel(table, title)


def idle_panel(snapshot: Snapshot, view: IdleView | None, now: float) -> RenderableType:
    probe = snapshot.probe
    info = snapshot.info
    if not probe or not info or view is None:
        return _waiting("Idle shutdown")
    box_now = _box_clock(snapshot, now)
    activity = probe.get("activity") or {}
    rows: list[tuple[str, RenderableType | str]] = []
    if info.state == "stopping":
        rows.append(
            ("auto-stop", Text("stopping now (scaled to 0)", style=f"bold {WARNING}"))
        )
    elif view.watchdog == "off":
        rows.append(("auto-stop", Text("off", style=f"bold {ERROR}")))
    elif view.remaining_seconds is None:
        rows.append(("auto-stop", Text("unknown (pass --idle-minutes)", style="dim")))
    else:
        style = f"bold {ERROR}" if view.remaining_seconds < 300 else "bold"
        label = Text.styled(f"in {duration(view.remaining_seconds)}", style=style)
        # An estimate can be up to a minute short: the watchdog looks once a minute.
        approx = " (up to 1m early)" if view.source == "estimated" else ""
        detail = f"{approx} {_dot()} {view.idle_minutes} min timeout"
        rows.append(("auto-stop", label + Text(detail, style=MUTED)))
    if view.active_now:
        rows.append(("active now", Text(", ".join(view.active_now), style=SUCCESS)))
    else:
        reason = safe("; ".join(str(r) for r in view.reasons) or "-")
        rows.append(("last activity", f"{reason}, {ago(view.last_active, box_now)}"))
    hold_until = activity.get("hold_until")
    if activity.get("held") and hold_until:
        rows.append(
            (
                "hold",
                Text(
                    f"until {_clock(hold_until)} "
                    f"({duration(hold_until - box_now)} left)",
                    style=ACCENT,
                ),
            )
        )
    else:
        rows.append(("hold", Text("none", style=MUTED)))
    return _panel(_kv(rows), "Idle shutdown")


def watchdog_banner(snapshot: Snapshot, view: IdleView | None) -> RenderableType | None:
    """A full-width alert bar, without a frame of its own."""
    if view is None or view.watchdog != "off":
        return None
    probe = snapshot.probe or {}
    lines = [w for w in probe.get("warnings") or [] if "idle shutdown" in str(w)]
    text = Text("IDLE SHUTDOWN IS OFF", style="bold")
    text.append("  this box bills until you stop it (press d)\n")
    text.append(safe(" ".join(lines) or "idle shutdown is off"))
    bar = Padding(text, (0, 1), style=f"white on {ERROR}", expand=True)
    return Group(bar, Text(""))


def _clock(epoch: float | None) -> str:
    return datetime.fromtimestamp(epoch).strftime("%H:%M") if epoch else "-"


def connections_panel(snapshot: Snapshot, now: float) -> RenderableType:
    probe = snapshot.probe
    if not probe:
        return _waiting("Connections")
    box_now = _box_clock(snapshot, now)
    activity = probe.get("activity") or {}
    ssh = activity.get("ssh_sessions") or 0
    claude = probe.get("claude") or {}
    hot = probe.get("hot") or {}
    if claude.get("running"):
        last = ago(activity.get("last_claude_activity"), box_now)
        claude_text = Text.styled("running", style=SUCCESS) + Text(
            f" {_dot()} last activity {last}", style=MUTED
        )
    else:
        claude_text = Text("not running", style=MUTED)
    rows: list[tuple[str, RenderableType | str]] = [
        ("ssh", Text(f"{ssh} attached", style=SUCCESS if ssh else MUTED)),
        ("claude remote", claude_text),
    ]
    if not hot.get("running"):
        rows.append(("hot reload", Text("not running", style=MUTED)))
    else:
        if hot.get("restarting"):
            state = Text.styled(
                f"restarting ({safe(hot['restarting'])})", style=WARNING
            )
        elif not hot.get("reachable", True):
            state = Text.styled("starting / not answering", style=WARNING)
        elif hot.get("ready"):
            state = Text.styled("ready", style=SUCCESS)
        else:
            state = Text.styled("loading model", style=WARNING)
        version = Text(f" {_dot()} code v{hot.get('code_version', '-')}", style=MUTED)
        rows.append(("hot reload", state + version))
        reload = hot.get("last_reload") or {}
        if reload:
            files = safe(", ".join(str(f) for f in reload.get("files") or []))
            ok = "" if reload.get("ok", True) else "  (import failed)"
            rows.append(("last reload", f"{files} in {reload.get('ms')} ms{ok}"))
        infer = hot.get("last_infer")
        if infer:
            at = infer.get("at")
            when = ago(at, box_now) if isinstance(at, (int, float)) else "-"
            rows.append(("last infer", f"{infer.get('ms')} ms, {when}"))
        if hot.get("error"):
            rows.append(("last error", Text(safe(hot["error"]), style=ERROR)))
    return _panel(_kv(rows), "Connections")


def work_panel(snapshot: Snapshot, now: float) -> RenderableType:
    probe = snapshot.probe
    if not probe:
        return _waiting("/work snapshot")
    box_now = _box_clock(snapshot, now)
    work = probe.get("work") or {}
    snap = work.get("snapshot") or {}
    lease = work.get("lease")
    dirty = work.get("dirty")
    if dirty is None:
        unsaved = Text("unknown", style=MUTED)
    else:
        unsaved = Text("yes", style=WARNING) if dirty else Text("no", style=SUCCESS)
    rows: list[tuple[str, RenderableType | str]] = [
        (
            "snapshot",
            f"{size(snap.get('size'))}, written {ago(snap.get('mtime'), box_now)}"
            if snap
            else "none yet",
        ),
        ("last save", ago(work.get("last_save"), box_now) + " (or restore)"),
        ("unsaved changes", unsaved),
    ]
    if lease:
        holder = "this runner" if lease.get("mine") else safe(lease.get("owner"))
        heartbeat = ago(lease.get("heartbeat"), box_now)
        rows.append(
            (
                "lease",
                Text.styled(holder, style=SUCCESS if lease.get("mine") else WARNING)
                + Text(
                    f" {_dot()} {safe(lease.get('state'))}, heartbeat {heartbeat}",
                    style=MUTED,
                ),
            )
        )
    else:
        rows.append(("lease", Text("none", style=MUTED)))
    return _panel(_kv(rows), "/work snapshot")


def _log_clock(stamp: str) -> str:
    """HH:MM:SS in local time from an ISO timestamp (the API sends UTC)."""
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return safe(stamp)[:8]
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone().strftime("%H:%M:%S")


def logs_panel(
    snapshot: Snapshot, lines: int, height: int | None = None
) -> RenderableType:
    """The log tail; with `height`, the panel fills exactly that many rows."""
    body = Text()
    logs = [
        log
        for log in snapshot.logs
        if not str(log.get("message", "")).startswith(NOISE_PREFIXES)
    ]
    for log in logs[-lines:]:
        level = str(log.get("level", "")).upper()
        style = (
            ERROR
            if level in ("ERROR", "CRITICAL")
            else WARNING
            if level == "WARNING"
            else ""
        )
        body += Text(f"{_log_clock(str(log.get('timestamp', '')))} ", style=MUTED)
        body += Text(
            safe(log.get("message", "")), style=style, overflow="ellipsis", no_wrap=True
        )
        body += "\n"
    body.rstrip()
    if not logs:
        body = Text("no recent logs", style=MUTED)
    title = Text.styled("Logs", style=TITLE) + Text(
        "  last 30 min", style=f"default {MUTED}"
    )
    return _panel(body, title, height=height)


def footer(snapshot: Snapshot, now: float) -> RenderableType:
    live = _live(snapshot)
    hints = Text(" ", no_wrap=True, overflow="ellipsis")
    for i, (key, label, needs_runner) in enumerate(KEY_HINTS):
        usable = live or not needs_runner
        if i:
            hints += Text("   ")
        hints += Text(key, style=f"bold {ACCENT}" if usable else MUTED)
        hints += Text(f" {label}", style="" if usable else MUTED)
    status = []
    if snapshot.probe_at and snapshot.probe_ms is not None:
        status.append(
            f"updated {duration(now - snapshot.probe_at)} ago "
            f"({snapshot.probe_ms:.0f} ms)"
        )
    for source, error in sorted(snapshot.errors.items()):
        status.append(f"{source} error: {safe(error)}")
    line = Text(
        "  ".join(status),
        style=ERROR if snapshot.errors else MUTED,
        overflow="ellipsis",
        no_wrap=True,
    )
    message = (
        snapshot.message if snapshot.message and now < snapshot.message_until else ""
    )
    # The action message comes before the status line, which may be cut short.
    return Group(
        hints,
        Text(safe(message), style=f"bold {ACCENT}", overflow="ellipsis", no_wrap=True),
        line,
    )


class _Panels:
    """Panels two per row on a wide terminal, stacked on a narrow one."""

    def __init__(self, *panels: RenderableType, stacked: tuple[int, ...] = ()) -> None:
        self.panels = panels
        # Order when stacked; what matters most goes first, as the bottom may be cut.
        self.stacked = stacked or tuple(range(len(panels)))

    def __rich_console__(self, console, options):
        if options.max_width < TWO_COLUMN_WIDTH:
            yield from (self.panels[i] for i in self.stacked)
            return
        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        for i in range(0, len(self.panels), 2):
            grid.add_row(*self.panels[i : i + 2])
        yield grid


class _Fit:
    """Fills the height it is given.

    `fill(rows)` builds the stretchy last part (the logs) for the rows left
    over; it is dropped when fewer than MIN_FILL rows are free, and anything
    that still does not fit is cut with a note saying so.
    """

    def __init__(
        self,
        parts: list[RenderableType],
        fill: Callable[[int | None], RenderableType] | None = None,
    ):
        self.parts = parts
        self.fill = fill

    def __rich_console__(self, console, options):
        height = options.height
        if height is None:
            yield Group(*self.parts, *([self.fill(None)] if self.fill else []))
            return
        unbounded = options.reset_height()
        lines = console.render_lines(Group(*self.parts), unbounded)
        free = height - len(lines)
        note = None
        if self.fill and free >= MIN_FILL:
            lines += console.render_lines(self.fill(free), unbounded)
        elif self.fill:
            note = "logs hidden; enlarge the terminal to see them"
        if len(lines) > height:
            hidden = len(lines) - height + 1
            lines = lines[: height - 1]
            note = f"{hidden} more lines hidden; enlarge the terminal"
        elif note and len(lines) >= height:
            note = None
        for line in lines:
            yield from line
            yield Segment.line()
        if note:
            yield Text(note, style=f"{MUTED} italic")


def body(snapshot: Snapshot, box, now: float, log_lines: int) -> RenderableType:
    parts: list[RenderableType] = [header(snapshot, now)]
    view = _idle(snapshot, box, now)
    banner = watchdog_banner(snapshot, view)
    if banner is not None:
        parts.append(banner)
    if snapshot.info and not _live(snapshot):
        hint = {
            "stopped": "The box is stopped. Start it with `fal dev up`.",
            "starting": "The box is starting; details appear once it is ready.",
            "failed": "The box failed to start; see its logs.",
        }.get(snapshot.info.state, "")
        parts.append(_panel(Text(hint, style=MUTED), ""))
    elif snapshot.info:
        parts.append(
            _Panels(
                gpu_panel(snapshot),
                idle_panel(snapshot, view, now),
                connections_panel(snapshot, now),
                work_panel(snapshot, now),
                # /work before Connections: it is what `d` acts on.
                stacked=(0, 1, 3, 2),
            )
        )

        def logs(rows: int | None) -> RenderableType:
            if rows is None:
                return logs_panel(snapshot, log_lines)
            return logs_panel(snapshot, rows - 2, height=rows)

        return _Fit(parts, logs)
    return _Fit(parts)


def render(
    snapshot: Snapshot, box, now: float | None = None, log_lines: int = PRINT_LOG_LINES
) -> RenderableType:
    """One frame as a plain renderable, for printing."""
    now = time.time() if now is None else now
    return Group(body(snapshot, box, now, log_lines), footer(snapshot, now))


def render_screen(snapshot: Snapshot, box, now: float | None = None) -> Layout:
    """One full-screen frame: the footer stays visible when the body is cropped."""
    now = time.time() if now is None else now
    layout = Layout()
    layout.split_column(
        Layout(body(snapshot, box, now, log_lines=PRINT_LOG_LINES), name="body"),
        Layout(footer(snapshot, now), name="footer", size=FOOTER_LINES),
    )
    return layout
