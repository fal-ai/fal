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
    get_sparkline_icons,
    get_state_dot_icon,
)
from fal.dev import loading
from fal.dev.box import Box
from fal.dev.dash.model import (
    HOLD_HOURS,
    LOG_MINUTES,
    Cost,
    IdleView,
    Snapshot,
    estimate_cost,
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
    ("o", "open output", False),
    ("q", "quit", False),
]
FOOTER_LINES = 3
LOGS_SHOWN = 8  # minimum, and the whole tail when there is no screen
SPARK_WIDTH = 30
# Free space below these turns a disk row yellow, then red.
DISK_LOW_BYTES = (20 * 2**30, 5 * 2**30)
NETWORK_VOLUME_BYTES = 2**50
PROCESS_NAME_WIDTH = 20
# Lines the platform logs for every runner exec, including the dashboard's own probes.
NOISE_PREFIXES = ("Skipping authentication for /Isolate/",)
NOT_RUNNING_HINTS = {
    "stopped": "The box is stopped. Start it with `fal dev up`.",
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


def _estimate(cost: Cost) -> Text:
    """Uptime at list price, for logins that cannot read usage: marked as a guess."""
    return Text.styled(f"~${cost.spent:.2f} est.", style="bold") + Text(
        f" at ${cost.per_hour:.2f}/h", style=MUTED
    )


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
        Text(str(box.account), style=MUTED),
        *([Text(f"owner {owner}", style=MUTED)] if (owner := box.known_owner) else []),
        Text(f"{num_gpus or 1}x {state.machine_type or '?'}"),
        Text(f"up {duration(_uptime(snapshot, now))}"),
    ]
    if _live(snapshot) and snapshot.cost and not snapshot.cost_hidden:
        segments.append(_cost(snapshot.cost, snapshot.cost_stale))
    elif _live(snapshot) and snapshot.cost_hidden:
        guess = estimate_cost(state.machine_type, _uptime(snapshot, now))
        if guess:
            segments.append(_estimate(guess))
    line = Text(f" {get_column_separator_icon()} ", style=BORDER).join(segments)
    title = Text.styled("fal dev", style=f"bold {ACCENT}") + Text(
        f"  {box.name}", style=TITLE
    )
    if state.code == "starting":
        runner = "pending"
    else:
        runner = state.runner.runner_id if state.runner else "-"
    subtitle = Text(f"runner {runner}", style=f"default {MUTED}")
    return _panel(line, title, subtitle=subtitle, subtitle_align="right")


def _waited(snapshot: Snapshot, wait: str, now: float) -> float:
    return now - snapshot.since.get(wait, now)


def booting_panel(snapshot: Snapshot, now: float, animate: bool) -> RenderableType:
    """The GPU panel while the platform starts a runner: dust and the state."""
    state = snapshot.state
    label, expected = loading.STATES["gpu-runner"]
    runner = state.runner if state else None
    phase = loading.runner_phase(runner.state.value.lower()) if runner else "requested"
    # A runner's uptime covers a wait that began before the dash opened.
    waited = _uptime(snapshot, now)
    if waited is None:
        waited = _waited(snapshot, "starting", now)
    body = loading.dust_panel_body(
        waited, f"{label}: {phase}", expected=expected, animate=animate
    )
    return _panel(body, "GPU")


def hot_restart_panel(snapshot: Snapshot, now: float, animate: bool) -> RenderableType:
    """The hot server reloading its weights: what a restart waits on."""
    headline, expected = loading.STATES["hot-restart"]
    body = loading.dust_panel_body(
        _waited(snapshot, "hot-restart", now),
        "model load",
        expected=expected,
        headline=headline,
        animate=animate,
    )
    return _panel(body, "Hot server")


def sparkline(values: list[float], width: int = SPARK_WIDTH) -> Text:
    """Percent readings as block characters, averaged down to `width` when longer."""
    ramp = get_sparkline_icons()
    if len(values) > width:
        edges = [round(i * len(values) / width) for i in range(width + 1)]
        values = [sum(values[a:b]) / (b - a) for a, b in zip(edges, edges[1:])]
    top = len(ramp) - 1
    chars = "".join(ramp[round(min(max(v, 0), 100) / 100 * top)] for v in values)
    return Text(chars, style=ACCENT)


def _history_rows(
    history: dict[str, list[tuple[float, float]]] | None, name: str
) -> Table | None:
    """Last few minutes of one reading ("gpu" or "cpu"), under its bars."""
    points = (history or {}).get(name) or []
    if len(points) < 2:
        return None
    rows = [(name, points)]
    table = Table.grid(padding=(0, 1))
    for _ in range(3):
        table.add_column(no_wrap=True)
    for name, points in rows:
        span = points[-1][0] - points[0][0]
        table.add_row(
            Text(name, style=MUTED),
            sparkline([value for _, value in points]),
            Text(duration(span), style=MUTED),
        )
    return table


def gpu_panel(probe: dict, history: dict | None = None) -> RenderableType:
    gpus = probe["gpus"]
    table = Table.grid(padding=(0, 1))
    for _ in range(5):
        table.add_column(no_wrap=True)
    if not gpus:
        note = Text(probe["gpu_error"] or "no GPUs reported", style=MUTED)
        return _panel(note, "GPU")
    for gpu in gpus:
        # Rows laid out like the Box panel's; numbered only when there are several.
        prefix = f"#{gpu['index']} " if len(gpus) > 1 else ""
        util = gpu["util_pct"]
        used, total = gpu["mem_used_mib"], gpu["mem_total_mib"]
        vram = used / total * 100 if used is not None and total else None
        temp, power = gpu["temp_c"], gpu["power_w"]
        table.add_row(
            Text(f"{prefix}util", style=MUTED),
            _bar(util),
            Text(f"{round(util) if util is not None else '-':>3}%"),
            Text(""),
            Text(""),
        )
        table.add_row(
            Text(f"{prefix}vram", style=MUTED),
            _bar(vram),
            Text(f"{round(vram) if vram is not None else '-':>3}%"),
            Text(
                f"{used / 1024:.1f}/{total / 1024:.0f} GiB" if vram is not None else "-"
            ),
            Text(""),
        )
        table.add_row(
            Text(f"{prefix}temp", style=MUTED),
            Text(""),
            Text(""),
            Text(
                f"{temp if temp is not None else '-'}{get_degrees_celsius_icon()}"
                f"  {round(power) if power is not None else '-'}W"
            ),
            Text(""),
        )
    title = Text.styled("GPU", style=TITLE) + Text(
        f"  {gpus[0]['name']}", style=f"default {MUTED}"
    )
    return _panel(Group(table, *_trend(history, "gpu")), title)


def system_panel(probe: dict, history: dict | None = None) -> RenderableType | None:
    """The box's CPU, memory, network and disks; None from an older probe."""
    table = Table.grid(padding=(0, 1))
    for _ in range(5):
        table.add_column(no_wrap=True)
    _add_system_rows(table, probe)
    if not table.row_count:
        return None
    return _panel(Group(table, *_trend(history, "cpu")), "Box")


def _trend(history: dict | None, name: str) -> list[RenderableType]:
    trend = _history_rows(history, name)
    return [trend] if trend else []


def _rate(bytes_per_second: float | None) -> Text:
    return Text("-" if bytes_per_second is None else f"{size(bytes_per_second)}/s")


def _add_system_rows(table: Table, probe: dict) -> None:
    """CPU, memory, network and disks.

    Each is absent on a box deployed before the probe read it.
    """
    if system := probe.get("system"):
        _add_cpu_ram_net_rows(table, system)
    # Free space only: a box's root disk is shared with the node and /data is a
    # network volume, so their used and total are not this box's.
    for disk in probe.get("disks") or []:
        free = disk["free_bytes"]
        warn, low = DISK_LOW_BYTES
        style = ERROR if free < low else WARNING if free < warn else ""
        # A network volume reports its whole backend (petabytes), not a quota.
        label = (
            Text("network volume", style=MUTED)
            if free >= NETWORK_VOLUME_BYTES
            else Text(f"{size(free)} free", style=style)
        )
        table.add_row(
            Text("+".join(disk["paths"]), style=MUTED),
            Text(""),
            Text(""),
            label,
            Text(""),
        )


def _add_cpu_ram_net_rows(table: Table, system: dict) -> None:
    cpu = system.get("cpu_pct")
    cpus = system.get("cpus")
    table.add_row(
        Text("cpu", style=MUTED),
        _bar(cpu),
        Text(f"{round(cpu) if cpu is not None else '-':>3}%"),
        Text(f"{cpus:g} cores" if cpus else "", style=MUTED),
        Text(""),
    )
    used = system.get("mem_used_bytes")
    total = system.get("mem_total_bytes")
    known = used is not None and bool(total)
    pct = used / total * 100 if known else None
    table.add_row(
        Text("ram", style=MUTED),
        _bar(pct),
        Text(f"{round(pct) if pct is not None else '-':>3}%"),
        Text(f"{used / 2**30:.1f}/{total / 2**30:.0f} GiB" if known else "-"),
        Text(""),
    )
    if "net_in_bps" in system:
        table.add_row(
            Text("net", style=MUTED),
            Text("in ") + _rate(system["net_in_bps"]),
            Text(""),
            Text("out ") + _rate(system.get("net_out_bps")),
            Text(""),
        )


def processes_panel(probe: dict) -> RenderableType | None:
    """The busiest processes: CPU of one core, resident memory, GPU memory."""
    processes = (probe.get("system") or {}).get("processes")
    if not processes:
        return None
    table = Table.grid(padding=(0, 1))
    table.add_column(no_wrap=True, max_width=PROCESS_NAME_WIDTH, overflow="ellipsis")
    for _ in range(3):
        table.add_column(no_wrap=True, justify="right")
    table.add_row(
        Text("command", style=MUTED),
        Text("cpu", style=MUTED),
        Text("ram", style=MUTED),
        Text("gpu", style=MUTED),
    )
    for process in processes:
        gpu = process.get("gpu_mib")
        table.add_row(
            Text(process["name"]),
            Text(f"{process['cpu_pct']:.0f}%"),
            Text(size(process["rss_bytes"])),
            Text("-" if gpu is None else size(gpu * 2**20), style="" if gpu else MUTED),
        )
    return _panel(table, "Top processes")


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
    if not probe["hot"]["running"]:
        rows.append(("hot reload", Text("not running", style=MUTED)))
    return _panel(_kv(rows), "Connections")


def hot_panel(probe: dict, box_now: float) -> RenderableType | None:
    """The hot server while it runs: its state, last reload, last request, error."""
    hot = probe["hot"]
    if not hot["running"]:
        return None
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
    rows: list[tuple[str, RenderableType | str]] = [("state", state + version)]
    if reload := hot.get("last_reload"):
        ok = "" if reload["ok"] else "  (import failed)"
        when = f", {ago(reload['at'], box_now)}" if reload.get("at") else ""
        rows.append(
            (
                "last reload",
                f"{', '.join(reload['files'])} in {reload['ms']} ms{when}{ok}",
            )
        )
    if infer := hot.get("last_infer"):
        rows.append(("last infer", f"{infer['ms']} ms, {ago(infer['at'], box_now)}"))
    if hot.get("error"):
        rows.append(("last error", Text(hot["error"], style=ERROR)))
    return _panel(_kv(rows), "Hot reload")


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


def logs_panel(snapshot: Snapshot, shown: int = LOGS_SHOWN) -> RenderableType:
    logs = [
        log
        for log in snapshot.logs
        if not str(log.get("message", "")).startswith(NOISE_PREFIXES)
    ]
    body = Text()
    for log in logs[-shown:]:
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


class WithLogs:
    """The panels above, then a Logs panel that fills the rows left below them.

    On a screen the height is known, so the tail grows with the terminal (never
    below LOGS_SHOWN); without one it stays at LOGS_SHOWN lines.
    """

    def __init__(self, parts: list[RenderableType], snapshot: Snapshot):
        self.parts = parts
        self.snapshot = snapshot

    def __rich_console__(self, console, options):
        above = Group(*self.parts)
        shown = LOGS_SHOWN
        if options.height is not None:
            lines = console.render_lines(above, options.update(height=None), pad=False)
            # Two rows go to the panel's own border.
            shown = max(LOGS_SHOWN, options.height - len(lines) - 2)
        yield above
        yield logs_panel(self.snapshot, shown)


def footer(
    snapshot: Snapshot, now: float, key_hints: list = KEY_HINTS
) -> RenderableType:
    live = _live(snapshot)
    hints = Text(" ", no_wrap=True, overflow="ellipsis")
    for i, (key, label, needs_runner) in enumerate(key_hints):
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


def body(
    snapshot: Snapshot, box: Box, now: float, animate: bool = False
) -> RenderableType:
    parts: list[RenderableType] = [header(snapshot, box, now)]
    state = snapshot.state
    if state is None:
        return Group(*parts)
    probe = snapshot.probe
    if state.code == "starting":
        return Group(*parts, booting_panel(snapshot, now, animate))
    if not _live(snapshot):
        hint = NOT_RUNNING_HINTS.get(state.code, "")
        return Group(*parts, _panel(Text(hint, style=MUTED), ""))
    if probe is None:
        panels = [
            _panel(Text("waiting for the box", style=MUTED), title)
            for title in (
                "GPU",
                "Box",
                "Idle shutdown",
                "/work snapshot",
                "Connections",
            )
        ]
    else:
        view = idle_view(probe, _box_clock(snapshot, now))
        if banner := watchdog_banner(probe, view):
            parts.append(banner)
        if "hot-restart" in snapshot.since:
            parts.append(hot_restart_panel(snapshot, now, animate))
        box_now = _box_clock(snapshot, now)
        optional = [processes_panel(probe), hot_panel(probe, box_now)]
        system = system_panel(probe, snapshot.history)
        panels = [
            gpu_panel(probe, snapshot.history),
            *([system] if system else []),
            idle_panel(snapshot, probe, view, now),
            *[panel for panel in optional if panel],
            # /work before Connections: it is what `d` acts on.
            work_panel(probe, box_now),
            connections_panel(probe, box_now),
        ]
    # Two panels per row when the terminal is wide enough, else stacked.
    parts.append(Columns(panels, equal=True))
    return WithLogs(parts, snapshot)


def render(
    snapshot: Snapshot,
    box: Box,
    now: float | None = None,
    *,
    screen: bool = False,
    animate: bool = False,
    body_of=body,
    key_hints: list = KEY_HINTS,
) -> RenderableType:
    """One frame; on a full screen the footer stays visible when the body is cut.

    `animate` draws dust and spinners in loading states (loading's switches).
    `body_of` and `key_hints` let another kind of box (`fal dev app dash`)
    draw its own panels and keys in the same frame.
    """
    now = time.time() if now is None else now
    frame = body_of(snapshot, box, now, animate)
    bottom = footer(snapshot, now, key_hints)
    if not screen:
        return Group(frame, bottom)
    layout = Layout()
    layout.split_column(
        Layout(frame, name="body"),
        Layout(bottom, name="footer", size=FOOTER_LINES),
    )
    return layout
