"""Rendering for `fal dev dash`: a Snapshot in, a rich renderable out."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from rich.box import ROUNDED
from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.padding import Padding
from rich.panel import Panel
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
from fal.dev.box import Box
from fal.dev.dash.model import (
    HOLD_HOURS,
    LOG_MINUTES,
    Cost,
    IdleView,
    Snapshot,
    idle_view,
)

# One style per role: frames are all the same muted color, and color goes on
# titles, badges and values, never on borders. Named ANSI colors, so the
# terminal's own palette keeps them readable on light and dark backgrounds.
BORDER = "bright_black"
ACCENT = "cyan"
SUCCESS = "green"
WARNING = "yellow"
ERROR = "red"
MUTED = "dim"
TITLE = "bold default"

STATE_STYLES = {
    "running": SUCCESS,
    "starting": WARNING,
    "stopping": WARNING,
    "stopped": ERROR,
    "failed": ERROR,
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
LOGS_SHOWN = 8
# Lines the platform logs for every runner exec, including the dashboard's own probes.
NOISE_PREFIXES = ("Skipping authentication for /Isolate/",)
NOT_RUNNING_HINTS = {
    "stopped": "The box is stopped. Start it with `fal dev up`.",
    "starting": "The box is starting; details appear once it is ready.",
    "failed": "The box failed to start; see its logs.",
}


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


def size(num_bytes: float) -> str:
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
        return Text("-" * width, style=MUTED)
    filled = round(min(max(pct, 0), 100) / 100 * width)
    style = ERROR if pct >= 90 else WARNING if pct >= 60 else SUCCESS
    return Text.styled(get_bar_full_icon() * filled, style=style) + Text(
        get_bar_empty_icon() * (width - filled), style=BORDER
    )


def _kv(rows: list[tuple[str, RenderableType | str]]) -> Table:
    table = Table.grid(padding=(0, 1))
    table.add_column(style=MUTED, no_wrap=True)
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


def _live(snapshot: Snapshot) -> bool:
    return snapshot.runner_id is not None


def _box_clock(snapshot: Snapshot, now: float) -> float:
    """The box's current time, from its last probe; avoids comparing two clocks."""
    probe = snapshot.probe
    if probe and snapshot.probe_at:
        return probe["now"] + (now - snapshot.probe_at)
    return now


def _uptime(snapshot: Snapshot, now: float) -> float | None:
    state = snapshot.state
    if not (state and state.runner and snapshot.state_at):
        return None
    return state.runner.uptime.total_seconds() + (now - snapshot.state_at)


def _cost(cost: Cost, stale: bool) -> Text:
    if cost.per_hour is None:
        return Text("rate pending", style=MUTED)
    text = Text.styled(f"${cost.spent:.2f}", style="bold") + Text(
        f" at ${cost.per_hour:.2f}/h", style=MUTED
    )
    return text + Text(" (stale)", style=WARNING) if stale else text


def header(snapshot: Snapshot, box: Box, now: float) -> RenderableType:
    state = snapshot.state
    if state is None:
        return _panel(Text("looking up the box ...", style=MUTED), "fal dev")
    num_gpus = len(snapshot.probe["gpus"]) if snapshot.probe else 0
    segments = [
        Text(
            f"{get_state_dot_icon()} {state.code}",
            style=f"bold {STATE_STYLES[state.code]}",
        ),
        Text(box.app),
        Text(box.team or "personal", style=MUTED),
        Text(f"{num_gpus or 1}x {state.machine_type or '?'}"),
        Text(f"up {duration(_uptime(snapshot, now))}"),
    ]
    if _live(snapshot) and snapshot.cost and not snapshot.cost_hidden:
        segments.append(_cost(snapshot.cost, snapshot.cost_stale))
    line = Text(f" {get_column_separator_icon()} ", style=BORDER).join(segments)
    title = Text.styled("fal dev", style=f"bold {ACCENT}") + Text(
        f"  {box.name}", style=TITLE
    )
    runner = state.runner.runner_id if state.runner else "-"
    subtitle = Text(f"runner {runner}", style=f"default {MUTED}")
    return _panel(line, title, subtitle=subtitle, subtitle_align="right")


def gpu_panel(probe: dict) -> RenderableType:
    gpus = probe["gpus"]
    if not gpus:
        return _panel(
            Text(probe["gpu_error"] or "no GPUs reported", style=MUTED), "GPU"
        )
    table = Table.grid(padding=(0, 1))
    for _ in range(5):
        table.add_column(no_wrap=True)
    for gpu in gpus:
        util = gpu["util_pct"]
        power = gpu["power_w"]
        table.add_row(
            Text(f"#{gpu['index']}", style=MUTED),
            _bar(util),
            Text(f"{util if util is not None else '-':>3}%"),
            Text(
                f"{(gpu['mem_used_mib'] or 0) / 1024:.1f}"
                f"/{(gpu['mem_total_mib'] or 0) / 1024:.0f} GiB"
            ),
            Text(
                f"{gpu['temp_c'] if gpu['temp_c'] is not None else '-'}"
                f"{get_degrees_celsius_icon()} "
                f"{round(power) if power is not None else '-'}W",
                style=MUTED,
            ),
        )
    title = Text.styled("GPU", style=TITLE) + Text(
        f"  {gpus[0]['name']}", style=f"default {MUTED}"
    )
    return _panel(table, title)


def idle_panel(
    snapshot: Snapshot, probe: dict, view: IdleView, now: float
) -> RenderableType:
    box_now = _box_clock(snapshot, now)
    activity = probe["activity"]
    rows: list[tuple[str, RenderableType | str]] = []
    if snapshot.state and snapshot.state.code == "stopping":
        rows.append(
            ("auto-stop", Text("stopping now (scaled to 0)", style=f"bold {WARNING}"))
        )
    elif view.watchdog == "off":
        rows.append(("auto-stop", Text("off", style=f"bold {ERROR}")))
    elif view.remaining_seconds is None:
        rows.append(("auto-stop", Text("waiting for the first check", style=MUTED)))
    else:
        style = f"bold {ERROR}" if view.remaining_seconds < 300 else "bold"
        label = Text.styled(f"in {duration(view.remaining_seconds)}", style=style)
        detail = f" {get_inline_separator_icon()} {view.idle_minutes} min timeout"
        rows.append(("auto-stop", label + Text(detail, style=MUTED)))
    if view.active_now:
        rows.append(("active now", Text(", ".join(view.active_now), style=SUCCESS)))
    else:
        reason = "; ".join(str(r) for r in view.reasons) or "-"
        rows.append(("last activity", f"{reason}, {ago(view.last_active, box_now)}"))
    hold_until = activity["hold_until"]
    if activity["held"]:
        left = duration(hold_until - box_now)
        hold = Text(f"until {_clock(hold_until)} ({left} left)", style=ACCENT)
    else:
        hold = Text("none", style=MUTED)
    rows.append(("hold", hold))
    return _panel(_kv(rows), "Idle shutdown")


def watchdog_banner(probe: dict, view: IdleView) -> RenderableType | None:
    """A full-width alert bar, without a frame of its own."""
    if view.watchdog != "off":
        return None
    lines = [w for w in probe["warnings"] if "idle shutdown" in w]
    text = Text("IDLE SHUTDOWN IS OFF", style="bold")
    text.append("  this box bills until you stop it (press d)\n")
    text.append(" ".join(lines) or "idle shutdown is off")
    bar = Padding(text, (0, 1), style=f"white on {ERROR}", expand=True)
    return Group(bar, Text(""))


def _clock(epoch: float | None) -> str:
    return datetime.fromtimestamp(epoch).strftime("%H:%M") if epoch else "-"


def connections_panel(probe: dict, box_now: float) -> RenderableType:
    ssh = probe["activity"]["ssh_sessions"]
    hot = probe["hot"]
    if probe["claude_running"]:
        last = ago(probe["activity"]["last_claude_activity"], box_now)
        claude_text = Text.styled("running", style=SUCCESS) + Text(
            f" {get_inline_separator_icon()} last activity {last}", style=MUTED
        )
    else:
        claude_text = Text("not running", style=MUTED)
    rows: list[tuple[str, RenderableType | str]] = [
        ("ssh", Text(f"{ssh} attached", style=SUCCESS if ssh else MUTED)),
        ("claude remote", claude_text),
    ]
    if not hot["running"]:
        rows.append(("hot reload", Text("not running", style=MUTED)))
    else:
        if hot.get("restarting"):
            state = Text.styled(f"restarting ({hot['restarting']})", style=WARNING)
        elif not hot["reachable"]:
            state = Text.styled("starting / not answering", style=WARNING)
        elif hot.get("ready"):
            state = Text.styled("ready", style=SUCCESS)
        else:
            state = Text.styled("loading model", style=WARNING)
        version = Text(
            f" {get_inline_separator_icon()} code v{hot.get('code_version', '-')}",
            style=MUTED,
        )
        rows.append(("hot reload", state + version))
        if reload := hot.get("last_reload"):
            ok = "" if reload["ok"] else "  (import failed)"
            rows.append(
                (
                    "last reload",
                    f"{', '.join(reload['files'])} in {reload['ms']} ms{ok}",
                )
            )
        if infer := hot.get("last_infer"):
            rows.append(
                ("last infer", f"{infer['ms']} ms, {ago(infer['at'], box_now)}")
            )
        if hot.get("error"):
            rows.append(("last error", Text(hot["error"], style=ERROR)))
    return _panel(_kv(rows), "Connections")


def work_panel(probe: dict, box_now: float) -> RenderableType:
    work = probe["work"]
    snap = work["snapshot"]
    lease = work["lease"]
    dirty = work["dirty"]
    if dirty is None:
        unsaved = Text("unknown", style=MUTED)
    else:
        unsaved = Text("yes", style=WARNING) if dirty else Text("no", style=SUCCESS)
    rows: list[tuple[str, RenderableType | str]] = [
        (
            "snapshot",
            f"{size(snap['size'])}, written {ago(snap['mtime'], box_now)}"
            if snap
            else "none yet",
        ),
        ("last save", ago(work["last_save"], box_now) + " (or restore)"),
        ("unsaved changes", unsaved),
    ]
    if lease:
        holder = "this runner" if lease["mine"] else lease["owner"]
        heartbeat = ago(lease["heartbeat"], box_now)
        rows.append(
            (
                "lease",
                Text.styled(holder, style=SUCCESS if lease["mine"] else WARNING)
                + Text(
                    f" {get_inline_separator_icon()} {lease['state']}, "
                    f"heartbeat {heartbeat}",
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
        return stamp[:8]
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone().strftime("%H:%M:%S")


def logs_panel(snapshot: Snapshot) -> RenderableType:
    logs = [
        log
        for log in snapshot.logs
        if not str(log.get("message", "")).startswith(NOISE_PREFIXES)
    ]
    body = Text()
    for log in logs[-LOGS_SHOWN:]:
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
            str(log.get("message", "")), style=style, overflow="ellipsis", no_wrap=True
        )
        body += "\n"
    body.rstrip()
    if not logs:
        body = Text("no recent logs", style=MUTED)
    title = Text.styled("Logs", style=TITLE) + Text(
        f"  last {LOG_MINUTES} min", style=f"default {MUTED}"
    )
    return _panel(body, title)


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
        status.append(f"{source} error: {error}")
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
        Text(message, style=f"bold {ACCENT}", overflow="ellipsis", no_wrap=True),
        line,
    )


def body(snapshot: Snapshot, box: Box, now: float) -> RenderableType:
    parts: list[RenderableType] = [header(snapshot, box, now)]
    state = snapshot.state
    if state is None:
        return Group(*parts)
    probe = snapshot.probe
    if not _live(snapshot):
        hint = NOT_RUNNING_HINTS.get(state.code, "")
        return Group(*parts, _panel(Text(hint, style=MUTED), ""))
    if probe is None:
        panels = [
            _panel(Text("waiting for the box", style=MUTED), title)
            for title in ("GPU", "Idle shutdown", "/work snapshot", "Connections")
        ]
    else:
        view = idle_view(probe, _box_clock(snapshot, now))
        if banner := watchdog_banner(probe, view):
            parts.append(banner)
        panels = [
            gpu_panel(probe),
            idle_panel(snapshot, probe, view, now),
            # /work before Connections: it is what `d` acts on.
            work_panel(probe, _box_clock(snapshot, now)),
            connections_panel(probe, _box_clock(snapshot, now)),
        ]
    # Two panels per row when the terminal is wide enough, else stacked.
    parts.append(Columns(panels, equal=True))
    parts.append(logs_panel(snapshot))
    return Group(*parts)


def render(
    snapshot: Snapshot, box: Box, now: float | None = None, *, screen: bool = False
) -> RenderableType:
    """One frame; on a full screen the footer stays visible when the body is cut."""
    now = time.time() if now is None else now
    frame = body(snapshot, box, now)
    if not screen:
        return Group(frame, footer(snapshot, now))
    layout = Layout()
    layout.split_column(
        Layout(frame, name="body"),
        Layout(footer(snapshot, now), name="footer", size=FOOTER_LINES),
    )
    return layout
