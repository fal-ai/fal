"""Rendering for `fal dev dash`: a Snapshot in, a rich renderable out."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.segment import Segment
from rich.table import Table
from rich.text import Text

from fal.dev.dash import (
    _CONTROL,
    HOLD_HOURS,
    IdleView,
    Snapshot,
    idle_view,
    price_per_hour,
)

STATE_STYLES = {
    "running": "bold green",
    "starting": "bold yellow",
    "stopping": "bold yellow",
    "stopped": "bold red",
    "failed": "bold red",
    "unknown": "dim",
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
SCREEN_LOG_LINES = 12
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
    style = "red" if pct >= 90 else "yellow" if pct >= 60 else "green"
    return Text("█" * filled, style=style) + Text("░" * (width - filled), style="dim")


def _kv(rows: list[tuple[str, RenderableType | str]]) -> Table:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    for key, value in rows:
        # Plain strings become Text so that box text is never parsed as markup.
        table.add_row(key, Text(value) if isinstance(value, str) else value)
    return table


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
        return Panel(Text("looking up the box ...", style="dim"), title="fal dev")
    probe = snapshot.probe or {}
    num_gpus = len(probe.get("gpus") or []) or 1
    uptime = _uptime(snapshot, now)
    rate = price_per_hour(info.machine_type, num_gpus)
    if uptime is None or (_live(snapshot) and not probe):
        # Before the first probe the GPU count, and so the rate, is unknown.
        cost = Text("-", style="dim")
    elif rate is None:
        cost = Text("unknown rate", style="dim")
    else:
        spent = rate * (uptime or 0) / 3600
        cost = Text(f"~${spent:.2f}", style="bold") + Text(
            f"  (${rate:.2f}/h list price, estimate)", style="dim"
        )
    title = Text(f"fal dev  {info.name}  ", style="bold") + Text(
        f" {info.state.upper()} ",
        style=f"reverse {STATE_STYLES.get(info.state, 'dim')}",
    )
    left = _kv(
        [
            ("app", info.app_name),
            ("team", info.team or "personal"),
            ("gpu", f"{num_gpus}x {info.machine_type or '?'}"),
        ]
    )
    right = _kv(
        [
            ("runner", info.runner_id or "-"),
            ("uptime", duration(uptime)),
            ("cost so far", cost),
        ]
    )
    grid = Table.grid(expand=True, padding=(0, 4))
    grid.add_column(ratio=1)
    grid.add_column(ratio=2)
    grid.add_row(left, right)
    return Panel(
        grid,
        title=title,
        title_align="left",
        border_style=STATE_STYLES.get(info.state, "dim"),
    )


def gpu_panel(snapshot: Snapshot) -> RenderableType:
    probe = snapshot.probe or {}
    gpus = probe.get("gpus") or []
    if not gpus:
        detail = probe.get("gpu_error") or (
            "waiting for the box" if not probe else "no GPUs reported"
        )
        return Panel(Text(safe(detail), style="dim"), title="GPU", title_align="left")
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
                f"{gpu.get('temp_c', '-')}°C {watts}W",
                style="dim",
            ),
        )
    title = Text("GPU  ") + Text(safe(gpus[0].get("name") or ""), style="dim")
    return Panel(table, title=title, title_align="left")


def idle_panel(snapshot: Snapshot, view: IdleView | None, now: float) -> RenderableType:
    probe = snapshot.probe
    info = snapshot.info
    if not probe or not info or view is None:
        return Panel(
            Text("waiting for the box", style="dim"),
            title="Idle shutdown",
            title_align="left",
        )
    box_now = _box_clock(snapshot, now)
    activity = probe.get("activity") or {}
    rows: list[tuple[str, RenderableType | str]] = []
    if info.state == "stopping":
        rows.append(
            ("auto-stop", Text("stopping now (scaled to 0)", style="bold yellow"))
        )
    elif view.watchdog == "off":
        rows.append(("auto-stop", Text("OFF", style="bold red")))
    elif view.remaining_seconds is None:
        rows.append(("auto-stop", Text("unknown (pass --idle-minutes)", style="dim")))
    else:
        style = "bold red" if view.remaining_seconds < 300 else "bold"
        label = Text(f"in {duration(view.remaining_seconds)}", style=style)
        detail = f"  after {view.idle_minutes} min idle"
        if view.source == "estimated":
            # The watchdog only looks once a minute.
            detail += ", estimate, +1 min"
        rows.append(("auto-stop", label + Text(detail, style="dim")))
    if view.active_now:
        rows.append(("active now", Text(", ".join(view.active_now), style="green")))
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
                    style="cyan",
                ),
            )
        )
    else:
        rows.append(("hold", Text("none", style="dim")))
    return Panel(_kv(rows), title="Idle shutdown", title_align="left")


def watchdog_banner(snapshot: Snapshot, view: IdleView | None) -> RenderableType | None:
    if view is None or view.watchdog != "off":
        return None
    probe = snapshot.probe or {}
    lines = [w for w in probe.get("warnings") or [] if "idle shutdown" in str(w)]
    text = Text(
        "IDLE SHUTDOWN IS OFF: this box bills until you stop it (press d).",
        style="bold white on red",
    )
    for line in lines or ["idle shutdown is off"]:
        text += Text(f"\n{safe(line)}", style="red")
    return Panel(text, border_style="bold red")


def _clock(epoch: float | None) -> str:
    return datetime.fromtimestamp(epoch).strftime("%H:%M") if epoch else "-"


def connections_panel(snapshot: Snapshot, now: float) -> RenderableType:
    probe = snapshot.probe
    if not probe:
        return Panel(
            Text("waiting for the box", style="dim"),
            title="Connections",
            title_align="left",
        )
    box_now = _box_clock(snapshot, now)
    activity = probe.get("activity") or {}
    ssh = activity.get("ssh_sessions") or 0
    claude = probe.get("claude") or {}
    hot = probe.get("hot") or {}
    if claude.get("running"):
        last = ago(activity.get("last_claude_activity"), box_now)
        claude_text = Text("running", style="green") + Text(
            f"  last activity {last}", style="dim"
        )
    else:
        claude_text = Text("not running", style="dim")
    rows: list[tuple[str, RenderableType | str]] = [
        ("ssh", Text(f"{ssh} attached", style="green" if ssh else "dim")),
        ("claude remote", claude_text),
    ]
    if not hot.get("running"):
        rows.append(("hot reload", Text("not running", style="dim")))
    else:
        if hot.get("restarting"):
            state = Text(f"restarting ({safe(hot['restarting'])})", style="yellow")
        elif not hot.get("reachable", True):
            state = Text("starting / not answering", style="yellow")
        elif hot.get("ready"):
            state = Text("ready", style="green")
        else:
            state = Text("loading model", style="yellow")
        version = Text(f"  code v{hot.get('code_version', '-')}", style="dim")
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
            rows.append(("last error", Text(safe(hot["error"]), style="red")))
    return Panel(_kv(rows), title="Connections", title_align="left")


def work_panel(snapshot: Snapshot, now: float) -> RenderableType:
    probe = snapshot.probe
    if not probe:
        return Panel(
            Text("waiting for the box", style="dim"),
            title="/work snapshot",
            title_align="left",
        )
    box_now = _box_clock(snapshot, now)
    work = probe.get("work") or {}
    snap = work.get("snapshot") or {}
    lease = work.get("lease")
    dirty = work.get("dirty")
    if dirty is None:
        unsaved = Text("unknown", style="dim")
    else:
        unsaved = Text("yes", style="yellow") if dirty else Text("no", style="green")
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
                Text(holder, style="green" if lease.get("mine") else "yellow")
                + Text(
                    f"  {safe(lease.get('state'))}, heartbeat {heartbeat}", style="dim"
                ),
            )
        )
    else:
        rows.append(("lease", Text("none", style="dim")))
    return Panel(_kv(rows), title="/work snapshot", title_align="left")


def _log_clock(stamp: str) -> str:
    """HH:MM:SS in local time from an ISO timestamp (the API sends UTC)."""
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return safe(stamp)[:8]
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone().strftime("%H:%M:%S")


def logs_panel(snapshot: Snapshot, lines: int) -> RenderableType:
    body = Text()
    logs = [
        log
        for log in snapshot.logs
        if not str(log.get("message", "")).startswith(NOISE_PREFIXES)
    ]
    for log in logs[-lines:]:
        level = str(log.get("level", "")).upper()
        style = (
            "red"
            if level in ("ERROR", "CRITICAL")
            else "yellow"
            if level == "WARNING"
            else ""
        )
        body += Text(f"{_log_clock(str(log.get('timestamp', '')))} ", style="dim")
        body += Text(
            safe(log.get("message", "")), style=style, overflow="ellipsis", no_wrap=True
        )
        body += "\n"
    body.rstrip()
    if not logs:
        body = Text("no recent logs", style="dim")
    return Panel(body, title="Logs (last 30 min)", title_align="left")


def footer(snapshot: Snapshot, now: float) -> RenderableType:
    live = _live(snapshot)
    hints = Text(no_wrap=True, overflow="ellipsis")
    for key, label, needs_runner in KEY_HINTS:
        usable = live or not needs_runner
        hints += Text(f" {key} ", style="reverse bold" if usable else "reverse dim")
        hints += Text(f" {label}   ", style="" if usable else "dim")
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
        style="red" if snapshot.errors else "dim",
        overflow="ellipsis",
        no_wrap=True,
    )
    message = (
        snapshot.message if snapshot.message and now < snapshot.message_until else ""
    )
    # The action message comes before the status line, which may be cut short.
    return Group(
        hints,
        Text(safe(message), style="bold cyan", overflow="ellipsis", no_wrap=True),
        line,
    )


class _Panels:
    """Panels two per row on a wide terminal, stacked on a narrow one."""

    def __init__(self, *panels: RenderableType) -> None:
        self.panels = panels

    def __rich_console__(self, console, options):
        if options.max_width < TWO_COLUMN_WIDTH:
            yield from self.panels
            return
        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        for i in range(0, len(self.panels), 2):
            grid.add_row(*self.panels[i : i + 2])
        yield grid


class _Fit:
    """Fills the height it is given: drops `optional` first, then says what is cut."""

    def __init__(self, parts: list[RenderableType], optional: list[RenderableType]):
        self.parts = parts
        self.optional = optional

    def __rich_console__(self, console, options):
        height = options.height
        if height is None:
            yield Group(*self.parts, *self.optional)
            return
        unbounded = options.reset_height()
        lines = console.render_lines(Group(*self.parts, *self.optional), unbounded)
        note = None
        if len(lines) > height and self.optional:
            lines = console.render_lines(Group(*self.parts), unbounded)
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
            yield Text(note, style="dim italic")


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
        parts.append(Panel(Text(hint, style="dim")))
    elif snapshot.info:
        parts.append(
            _Panels(
                gpu_panel(snapshot),
                idle_panel(snapshot, view, now),
                connections_panel(snapshot, now),
                work_panel(snapshot, now),
            )
        )
        return _Fit(parts, [logs_panel(snapshot, log_lines)])
    return _Fit(parts, [])


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
        Layout(body(snapshot, box, now, log_lines=SCREEN_LOG_LINES), name="body"),
        Layout(footer(snapshot, now), name="footer", size=FOOTER_LINES),
    )
    return layout
