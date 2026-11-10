"""A box in one place: the ready card after `fal dev up`, and `fal dev status`.

Both read the same Snapshot the dash draws (one probe of the box, plus the
run's cost when this login may read it).
"""

from __future__ import annotations

import time

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from fal.console.icons import get_inline_separator_icon
from fal.dev import config
from fal.dev.box import Box
from fal.dev.dash.model import IDLE_WARNING_SECONDS, Poller, Snapshot, idle_view
from fal.dev.dash.view import STATE_STYLES, ago, duration
from fal.dev.errors import DevError
from fal.dev.loading import format_elapsed, panel, runner_phase

MUTED = "dim"


def snapshot(box: Box) -> Snapshot:
    """What the dash would show now: runner state, one probe and the cost."""
    poller = Poller(box)
    poller.prime()
    snap = poller.snapshot
    if snap.state is None:
        raise DevError(
            snap.errors.get("state") or "no answer from the platform",
            title="could not read the box's state",
            fixes=[("fal dev status", "try again")],
        )
    return snap


def _box_now(snap: Snapshot, now: float) -> float:
    if snap.probe and snap.probe_at:
        return snap.probe["now"] + (now - snap.probe_at)
    return now


def _uptime(snap: Snapshot, now: float) -> float | None:
    state = snap.state
    if not (state and state.runner and snap.state_at):
        return None
    return state.runner.uptime.total_seconds() + (now - snap.state_at)


def _machine(snap: Snapshot) -> str:
    """'1x NVIDIA H100' on a GPU box, else the machine type, such as 'M'."""
    probe = snap.probe
    gpus = probe["gpus"] if probe else []
    if gpus:
        return f"{len(gpus)}x {gpus[0]['name']}"
    machine_type = snap.state.machine_type if snap.state else None
    if not machine_type:
        return "1x ?"
    return f"1x {machine_type}" if config.is_gpu(machine_type) else machine_type


def _idle(snap: Snapshot, now: float, *, short: bool = False) -> Text:
    """'stops itself after 60 min idle (in 38m)', or why it will not."""
    if not snap.probe:
        return Text("idle shutdown: waiting for the box", style=MUTED)
    view = idle_view(snap.probe, _box_now(snap, now))
    if view.watchdog == "off":
        return Text("idle shutdown is OFF: stop it with fal dev down", style="bold red")
    if view.remaining_seconds is None:
        minutes = view.idle_minutes
        return Text(
            f"stops itself after {minutes} min idle" if minutes else "idle shutdown on"
        )
    style = "bold red" if view.remaining_seconds < IDLE_WARNING_SECONDS else ""
    left = duration(view.remaining_seconds)
    if short:
        return Text(f"idle stop in {left}", style=style)
    text = Text(f"stops itself after {view.idle_minutes} min idle ")
    text.append(f"(in {left})", style=style or MUTED)
    return text


def _hold(snap: Snapshot, now: float, *, short: bool = False) -> Text | None:
    """'hold until 14:05 (1h 20m left)' while `fal dev hold` is in effect."""
    activity = snap.probe["activity"] if snap.probe else None
    if not activity or not activity["held"]:
        return None
    until = activity["hold_until"]
    left = duration(until - _box_now(snap, now))
    clock = time.strftime("%H:%M", time.localtime(until))
    text = f"held until {clock} ({left} left)"
    if not short:
        text += "; clear with: fal dev hold --off"
    return Text(text, style="cyan")


def _cost(snap: Snapshot, *, short: bool = False) -> Text | None:
    cost = snap.cost
    if not cost or snap.cost_hidden:
        return None
    text = Text(f"${cost.spent:.2f}" if short else f"${cost.spent:.2f} so far")
    if cost.per_hour is not None:
        rate = f" (${cost.per_hour:.2f}/h)" if short else f" at ${cost.per_hour:.2f}/h"
        text.append(rate, style=MUTED)
    return text


def _hot(snap: Snapshot) -> Text | None:
    hot = snap.probe["hot"] if snap.probe else None
    if not hot or not hot["running"]:
        return None
    version = f"hot v{hot.get('code_version', '-')}"
    if hot.get("restarting"):
        return Text(f"{version} restarting", style="yellow")
    if hot.get("ready"):
        return Text(f"{version} ready", style="green")
    return Text(f"{version} loading", style="yellow")


def _secrets(snap: Snapshot) -> Text | None:
    """The fal secrets the box was given, by name only."""
    names = ((snap.probe or {}).get("user_env") or {}).get("secrets")
    return Text(f"secrets: {', '.join(names)}", style=MUTED) if names else None


def status_line(snap: Snapshot, box: Box, now: float | None = None) -> Text:
    """`fal dev status`: the box in one line."""
    now = time.time() if now is None else now
    state = snap.state
    code = state.code if state else "unknown"
    parts = [
        Text(box.app, style="bold"),
        Text(code.upper(), style=f"bold {STATE_STYLES.get(code, '')}"),
    ]
    if owner := box.owner():
        parts.append(Text(f"owner {owner}", style=MUTED))
    if code == "starting":
        runner = state.runner if state else None
        phase = runner_phase(runner.state.value.lower()) if runner else ""
        parts.append(Text(f"waiting for a runner{f' ({phase})' if phase else ''}"))
        parts.append(Text("cancel with: fal dev down", style=MUTED))
    elif code == "failed":
        parts.append(Text("see why with: fal dev logs --all", style=MUTED))
    elif code == "stopped":
        if state and state.app is None:
            parts.append(Text("not deployed; create it with: fal dev up", style=MUTED))
        else:
            parts.append(Text("start it with: fal dev up", style=MUTED))
    elif code == "stopping":
        parts.append(Text("saving /work and scaling to 0", style=MUTED))
        if state and state.runner:
            parts.append(Text("keep it with: fal dev up", style=MUTED))
    else:
        parts += [Text(_machine(snap)), Text(f"up {duration(_uptime(snap, now))}")]
        for extra in (
            _cost(snap, short=True),
            _idle(snap, now, short=True),
            _hold(snap, now, short=True),
            _hot(snap),
            _secrets(snap),
        ):
            if extra is not None:
                parts.append(extra)
        if snap.probe and snap.probe["work"]["last_save"]:
            box_now = _box_now(snap, now)
            saved = ago(snap.probe["work"]["last_save"], box_now)
            parts.append(Text(f"/work saved {saved}", style=MUTED))
    return Text("  ").join(parts)


def ready_card(
    snap: Snapshot, box: Box, waited: float | None = None, now: float | None = None
) -> RenderableType:
    """What `fal dev up` ends with: the box, its cost and idle rule, how to connect."""
    now = time.time() if now is None else now
    runner = snap.state.runner if snap.state else None
    sep = f" {get_inline_separator_icon()} "
    facts = Table.grid(padding=(0, 2))
    facts.add_column(style=MUTED, no_wrap=True)
    facts.add_column()
    machine = Text(_machine(snap))
    if runner:
        machine.append(f"{sep}runner {runner.runner_id}", style=MUTED)
    facts.add_row("machine", machine)
    if owner := box.owner():
        facts.add_row("owner", Text(owner))
    if cost := _cost(snap):
        facts.add_row("cost", cost)
    facts.add_row("idle", _idle(snap, now))
    if hold := _hold(snap, now):
        facts.add_row("hold", hold)
    if hot := _hot(snap):
        facts.add_row("hot", hot)
    if secrets := _secrets(snap):
        facts.add_row("secrets", Text(secrets.plain.removeprefix("secrets: ")))

    commands = Table.grid(padding=(0, 2))
    commands.add_column(style=MUTED, no_wrap=True)
    commands.add_column(style="bold", no_wrap=True)
    commands.add_column(style=MUTED)
    commands.add_row("shell", "fal dev shell", "")
    commands.add_row("watch", "fal dev dash", "usage, idle shutdown, logs")
    commands.add_row(
        "editor", f"ssh {box.app}", "after: fal dev ssh-config >> ~/.ssh/config"
    )
    commands.add_row("hot", "fal dev hot start ./project", "keep a model loaded")
    commands.add_row("stop", "fal dev down", "saves /work, stops billing")

    title = Text.styled("fal dev", style="bold cyan") + Text(
        f"  {box.name}  ", style="bold default"
    )
    title.append("READY", style="bold green")
    subtitle = (
        Text(f"ready in {format_elapsed(waited)}", style=MUTED) if waited else None
    )
    return panel(
        Group(facts, Text(""), commands),
        title,
        subtitle=subtitle,
        subtitle_align="right",
        width=100,
    )
