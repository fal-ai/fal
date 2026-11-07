"""Live terminal dashboard for a running dev box: `fal dev dash`.

Each refresh costs one exec round trip: `remote/probe.py` runs on the box and
returns everything as one JSON document. Runner state and logs are polled less
often. Polling never counts as activity for the idle watchdog (see the probe).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from fal.dev import config
from fal.dev.box import Box, BoxState
from fal.dev.errors import DevError

HOLD_HOURS = 1.0
# The dash rings and warns when an idle box is this close to stopping itself.
IDLE_WARNING_SECONDS = 600
LOG_MINUTES = 30
LOG_LINES = 30
# The history request is several times this (the app dash filters its own polling).
LOG_ROWS_MAX = 200
STATE_INTERVAL = 15.0
LOG_INTERVAL = 10.0
COST_INTERVAL = 60.0
HISTORY_SECONDS = 300.0


def log_rows() -> int:
    """How many log lines to fetch: enough to fill a full-height Logs panel,
    read at each poll so a resized terminal is followed."""
    return min(LOG_ROWS_MAX, max(LOG_LINES, shutil.get_terminal_size().lines))


# Output images `fal dev hot run` saves next to the project, newest last.
OUTPUT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
# List price in USD per runner-hour by machine type, for the cost estimate shown
# when this login cannot read usage. The SDK ships no price list and none is
# invented here: a machine type missing from this table gets no estimate.
LIST_PRICES: dict[str, float] = {}
# Warnings the box writes when its idle watchdog cannot stop it.
WATCHDOG_OFF_PATTERNS = (
    "idle shutdown is off",
    "idle shutdown may not work",
    "idle shutdown failed",
)
# Control characters, including ESC: text from the box must not drive the terminal.
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def clean(value: Any) -> Any:
    """Box-originated JSON with control characters removed from every string."""
    if isinstance(value, str):
        return _CONTROL.sub("", value)
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, dict):
        return {clean(k): clean(v) for k, v in value.items()}
    return value


def probe_command(box: Box, check_dirty: bool) -> list[str]:
    settings = {
        "workspace": f"{config.WORKSPACE_ROOT}/{box.name}",
        "check_dirty": check_dirty,
    }
    path = f"{config.PROBE_DIR}/probe.py"
    # A box deployed before the probe was installed has no probe.py: say so at
    # once instead of letting python3 fail and the read wait for the end marker.
    script = (
        f'test -f "$0" || {{ echo {config.PROBE_MISSING}; echo {config.PROBE_END}; '
        'exit 1; }; exec python3 "$0" "$1"'
    )
    return ["sh", "-c", script, path, json.dumps(settings)]


MISSING = object()
# The keys the view reads, and the keys it reads inside some of them.
PROBE_KEYS: dict[str, set[str]] = {
    "now": set(),
    "gpus": set(),
    "gpu_error": set(),
    "idle_state": set(),
    "admin_key_set": set(),
    "warnings": set(),
    "claude_running": set(),
    "hot": {"running"},
    "activity": {
        "ssh_sessions",
        "gpu_busy",
        "hold_until",
        "held",
        "last_claude_activity",
    },
    "work": {"snapshot", "last_save", "lease", "dirty"},
}


def parse_probe(output: str) -> dict:
    """The probe's JSON document from exec output that may carry other lines."""
    if config.PROBE_MISSING in output:
        raise ValueError(
            "this box predates the dashboard probe; redeploy with: fal dev deploy"
        )
    text = output.split(config.PROBE_END, 1)[0]
    lines = [line for line in text.splitlines() if line.startswith("{")]
    if not lines:
        raise ValueError(f"no status in the probe output: {text.strip()[-300:]!r}")
    probe = clean(json.loads(lines[-1]))
    if "probe_error" in probe:
        raise ValueError(f"the probe failed on the box: {probe['probe_error']}")
    for key, subkeys in PROBE_KEYS.items():
        value = probe.get(key, MISSING)
        if value is MISSING or (subkeys and not subkeys <= value.keys()):
            raise ValueError(
                "the box's probe is from another fal version; "
                "redeploy with: fal dev deploy"
            )
    return probe


@dataclass
class IdleView:
    watchdog: str  # on, off, unknown
    idle_minutes: int | None
    last_active: float | None  # box clock
    remaining_seconds: float | None
    active_now: list[str]
    reasons: list[str]


def idle_view(probe: dict, now: float) -> IdleView:
    """When the idle watchdog will stop the box, and why it is awake.

    `now` and every timestamp are on the box's clock. The watchdog keeps its
    last-activity time in memory and publishes it every tick; before its first
    tick there is nothing to show but the signals read live by the probe.
    """
    activity = probe["activity"]
    warnings = " ".join(probe["warnings"]).lower()
    if any(pattern in warnings for pattern in WATCHDOG_OFF_PATTERNS):
        watchdog = "off"
    elif probe["admin_key_set"]:
        watchdog = "on"
    else:
        watchdog = "unknown"

    active_now = []
    if activity["ssh_sessions"]:
        active_now.append(f"{activity['ssh_sessions']} SSH session(s)")
    if activity["gpu_busy"]:
        active_now.append("GPU busy")
    if activity["held"]:
        active_now.append("hold")

    published = probe["idle_state"] or {}
    last_active = published.get("last_active")
    idle_minutes = published.get("idle_minutes")
    remaining = None
    if idle_minutes and last_active is not None:
        remaining = max(0.0, idle_minutes * 60 - (now - last_active))
    return IdleView(
        watchdog,
        idle_minutes,
        last_active,
        remaining,
        active_now,
        published.get("reasons") or active_now,
    )


@dataclass(frozen=True)
class Cost:
    """What the box's current run has cost so far."""

    spent: float
    per_hour: float | None  # None until the platform has usage rows for the run


def fetch_cost(box: Box, runner_uptime: float) -> Cost | None:
    """The run's cost from the platform's Usage API; None if this login may not read it.

    Needs billing access (an ADMIN-scoped key), which the platform answers with
    401 or 403 otherwise. Usage is aggregated by the platform, so it can lag the
    box by a while.
    """
    from fal import flags

    started = datetime.now(timezone.utc) - timedelta(seconds=runner_uptime)
    response = httpx.get(
        f"{flags.PLATFORM_URL}/serverless/usage",
        params={
            "app": box.app,
            "start": started.isoformat(),
            "expand": "summary",
            "bound_to_timeframe": "false",
        },
        headers=box.auth_headers(),
        timeout=10,
    )
    if response.status_code in (401, 403):
        return None
    response.raise_for_status()
    rows = response.json()["summary"]
    # Surge and regular usage of the same machine type are separate rows.
    spent = sum(row["cost"] for row in rows)
    rates = [row["unit_price"] * 3600 for row in rows if not row["is_surge"]]
    return Cost(spent, rates[0] if rates else None)


def estimate_cost(machine_type: str | None, uptime: float | None) -> Cost | None:
    """Runner uptime at the machine type's list price; None without a known price."""
    price = LIST_PRICES.get(machine_type or "")
    if price is None or uptime is None:
        return None
    return Cost(price * uptime / 3600, price)


# Rolling readings, by series name: (local time, percent), oldest first.
History = dict[str, list[tuple[float, float]]]


def record_history(history: History, probe: dict, at: float) -> History:
    """`history` plus the probe's GPU and CPU readings, trimmed to the last 5 minutes.

    Returns a new dict: the view reads snapshots from another thread.
    """
    readings: dict[str, float | None] = {}
    utils = [g["util_pct"] for g in probe["gpus"] if g["util_pct"] is not None]
    readings["gpu"] = max(utils) if utils else None
    readings["cpu"] = (probe.get("system") or {}).get("cpu_pct")
    cutoff = at - HISTORY_SECONDS
    kept: History = {}
    for name, value in readings.items():
        points = [p for p in history.get(name, []) if p[0] >= cutoff]
        if value is not None:
            points.append((at, value))
        if points:
            kept[name] = points
    return kept


@dataclass
class Snapshot:
    """Everything the view draws; replaced as a whole by the poller."""

    state: BoxState | None = None
    state_at: float | None = None  # local clock
    probe: dict | None = None
    probe_at: float | None = None  # local clock
    probe_ms: float | None = None
    # GPU and CPU percent of the current runner over the last HISTORY_SECONDS.
    history: History = field(default_factory=dict)
    logs: list[dict] = field(default_factory=list)
    # The cost of cost_runner's run; None until the platform has answered.
    cost: Cost | None = None
    cost_runner: str | None = None
    cost_hidden: bool = False  # this login may not read usage
    cost_stale: bool = False  # the last attempt to update the cost failed
    errors: dict[str, str] = field(default_factory=dict)
    message: str | None = None
    message_until: float = 0.0  # local clock; the message is hidden after this
    # Local clock when each wait in `waits()` began, while it lasts.
    since: dict[str, float] = field(default_factory=dict)

    @property
    def runner_id(self) -> str | None:
        """The runner to talk to, while one is up (running or being stopped)."""
        state = self.state
        if state and state.runner and state.code in ("running", "stopping"):
            return state.runner.runner_id
        return None


def waits(snapshot: Snapshot) -> set[str]:
    """The slow things the dash shows a loading state for right now."""
    found = set()
    if snapshot.state and snapshot.state.code == "starting":
        found.add("starting")
    probe = snapshot.probe
    # A probe that stopped answering is stale: do not animate its last answer.
    fresh = "probe" not in snapshot.errors
    if (
        snapshot.runner_id
        and probe
        and fresh
        and (probe.get("hot") or {}).get("restarting")
    ):
        found.add("hot-restart")
    return found


class IdleWarning:
    """Says once per idle stretch that the box is about to stop itself."""

    def __init__(self, before: float = IDLE_WARNING_SECONDS) -> None:
        self.before = before
        self.warned = False

    def check(self, snapshot: Snapshot, now: float) -> str | None:
        probe = snapshot.probe
        if not (probe and snapshot.probe_at and snapshot.runner_id):
            return None
        box_now = probe["now"] + (now - snapshot.probe_at)
        view = idle_view(probe, box_now)
        remaining = view.remaining_seconds
        if view.watchdog != "on" or remaining is None or remaining > self.before:
            self.warned = False  # activity came back: warn again next time
            return None
        if self.warned:
            return None
        self.warned = True
        minutes = max(1, round(remaining / 60))
        return (
            f"idle: this box stops itself in about {minutes} min and saves /work; "
            f"press h to hold it {HOLD_HOURS:g}h"
        )


class Poller:
    """Refreshes a Snapshot in a background thread: one exec per `interval`."""

    def __init__(
        self,
        box: Box,
        interval: float = 3.0,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.box = box
        self.interval = interval
        self.on_change = on_change or (lambda: None)
        self.snapshot = Snapshot()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._polls = 0
        self._next_state = 0.0
        self._next_logs = 0.0
        self._next_cost = 0.0
        self._state_requested = False

    def update(
        self,
        errors: dict[str, str | None] | None = None,
        *,
        if_runner: str | None = None,
        **changes,
    ) -> None:
        """Replace snapshot fields; a None in `errors` clears that source's error.

        With `if_runner`, nothing changes unless that is still the runner.
        """
        with self._lock:
            if if_runner and self.snapshot.runner_id != if_runner:
                return
            merged: dict[str, str | None] = {**self.snapshot.errors, **(errors or {})}
            snapshot = replace(
                self.snapshot,
                errors={source: text for source, text in merged.items() if text},
                **changes,
            )
            active = waits(snapshot)
            since = {k: v for k, v in snapshot.since.items() if k in active}
            for wait in active - since.keys():
                since[wait] = time.time()
            self.snapshot = replace(snapshot, since=since)
        self.on_change()

    def say(self, message: str, seconds: float = 10.0) -> None:
        """Show a status line in the footer for a while."""
        self.update(message=clean(message), message_until=time.time() + seconds)

    def refresh_state(self) -> None:
        """Ask for runner state on the next tick (after an action)."""
        self._state_requested = True
        self._wake.set()

    def poll_once(self, cost: bool = True) -> None:
        """Refresh what is due; `cost` starts the usage lookup in a thread when due."""
        now = time.time()
        if now >= self._next_state or self._state_requested:
            self._state_requested = False
            try:
                self.update(
                    state=self.box.state(), state_at=now, errors={"state": None}
                )
            except Exception as exc:
                self.update(errors={"state": _short(exc)})
            self._next_state = now + STATE_INTERVAL
        runner_id = self.snapshot.runner_id
        if self.snapshot.cost_runner != runner_id:
            # Another runner (or none): the old run's cost is not this one's.
            self.update(
                cost=None,
                cost_runner=runner_id,
                cost_hidden=False,
                cost_stale=False,
                history={},
                errors={"cost": None},
            )
            self._next_cost = 0.0
        if not runner_id:
            if self.snapshot.probe is not None or self.snapshot.logs:
                # A stopped box has no live state; drop the last runner's.
                self.update(probe=None, probe_at=None, logs=[], history={})
            return
        start = time.time()
        try:
            probe = self.probe(runner_id)
            self.update(
                probe=probe,
                probe_at=time.time(),
                probe_ms=(time.time() - start) * 1000,
                history=record_history(self.snapshot.history, probe, time.time()),
                errors={"probe": None},
            )
        except Exception as exc:
            self.update(errors={"probe": _short(exc)})
        if now >= self._next_logs:
            since = datetime.now(timezone.utc) - timedelta(minutes=LOG_MINUTES)
            try:
                logs = self.box.logs(runner_id, log_rows(), since)
                self.update(logs=clean(logs), errors={"logs": None})
            except Exception as exc:
                self.update(errors={"logs": _short(exc)})
            self._next_logs = now + LOG_INTERVAL
        if cost and now >= self._next_cost:
            self._next_cost = now + COST_INTERVAL
            # In its own thread: a slow answer must not hold up the probe.
            threading.Thread(target=self.refresh_cost, daemon=True).start()

    def probe(self, runner_id: str) -> dict:
        """One run of the box's probe, parsed."""
        # Walking /work for unsaved changes is the probe's only costly part;
        # do it every tenth refresh.
        check_dirty = self._polls % 10 == 0
        self._polls += 1
        _, output = self.box.run(
            runner_id,
            probe_command(self.box, check_dirty),
            timeout=30,
            until=config.PROBE_END.encode(),
        )
        probe = parse_probe(output)
        last = self.snapshot.probe
        if probe["work"]["dirty"] is None and last:
            # Keep the last answer between dirty checks.
            probe["work"]["dirty"] = last["work"]["dirty"]
        return probe

    def refresh_cost(self) -> None:
        """Update the run's cost; a failure keeps the last cost, marked stale."""
        state, state_at = self.snapshot.state, self.snapshot.state_at
        runner_id = self.snapshot.runner_id
        if not (state and state.runner and state_at and runner_id):
            return
        uptime = state.runner.uptime.total_seconds() + (time.time() - state_at)
        try:
            cost = fetch_cost(self.box, uptime)
        except Exception as exc:
            reason = _short(exc)
            if self.box.host:
                reason += (
                    " (on another host, point FAL_PLATFORM_URL at its platform API)"
                )
            changes: dict[str, Any] = {"cost_stale": True, "errors": {"cost": reason}}
        else:
            changes = {
                "cost": cost,
                "cost_hidden": cost is None,
                "cost_stale": False,
                "errors": {"cost": None},
            }
        # The runner may have been replaced while the platform was answering.
        self.update(if_runner=runner_id, **changes)

    def prime(self) -> None:
        """Poll until the first probe: the first poll may only find the runner."""
        self.poll_once(cost=False)
        if self.snapshot.probe is None:
            self.poll_once(cost=False)
        self.refresh_cost()

    def run(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._wake.wait(self.interval)
            self._wake.clear()

    def start(self) -> threading.Thread:
        thread = threading.Thread(
            target=self.run, daemon=True, name="fal-dev-dash-poller"
        )
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()


def _short(exc: BaseException) -> str:
    text = str(exc) or exc.__class__.__name__
    return clean(" ".join(text.split())[:300])


def hold(box: Box) -> str:
    if box.hold(HOLD_HOURS):
        return f"idle shutdown held for {HOLD_HOURS:g}h"
    return "a longer hold is already in place; kept it"


def down(box: Box) -> str:
    box.down()
    return "stopped: /work saved, scaled to 0"


def hot_restart(box: Box, runner_id: str) -> str:
    # On the box, not through the app's /hot route: that would count as activity.
    url = f"http://127.0.0.1:{config.HOT_PORT}/restart"
    code, _ = box.run(runner_id, ["curl", "-fsS", "-X", "POST", "-d", "{}", url])
    if code != 0:
        raise DevError("is the hot server running?")
    return "hot server restarting; the model reloads"


def latest_output(box: Box, cwd: Path | None = None) -> Path | None:
    """The newest image `fal dev hot run` saved, None if there is none.

    Results come back over HTTP and are written on this machine, never on the
    box: in hot-out/ of the folder the dash runs in, and of the project that
    folder last used `fal dev hot` with.
    """
    from fal.dev import hot, state

    cwd = cwd or Path.cwd()
    folders = [cwd]
    with contextlib.suppress(OSError):
        project = state.load(cwd, box.host).get("hot_local")
        if project:
            folders.append(Path(project))
    newest: tuple[float, Path] | None = None
    for folder in folders:
        try:
            candidates = list((folder / hot.OUTPUT_DIR).iterdir())
        except OSError:
            continue
        for path in candidates:
            if path.suffix.lower() not in OUTPUT_SUFFIXES:
                continue
            try:
                stamp = path.stat().st_mtime
            except OSError:
                continue
            if newest is None or stamp > newest[0]:
                newest = (stamp, path)
    return newest[1] if newest else None


def open_externally(path: Path) -> None:
    """Open a file in the system's viewer, for terminals that cannot draw images."""
    if sys.platform == "darwin":
        command = ["open", str(path)]
    elif sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
        return
    else:
        command = ["xdg-open", str(path)]
    try:
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as exc:
        raise DevError(f"could not open {path.name}: {exc}") from exc


def open_shell(box: Box, runner_id: str) -> None:
    """Open a shell in a child process.

    `fal runners shell` leaves a thread blocked reading stdin, which in this
    process would swallow the next key.
    """
    command = [sys.executable, "-m", "fal", "runners", "shell", runner_id]
    if box.account.fal_team:
        command += ["--team", box.account.fal_team]
    if box.host:
        command += ["--host", box.host]
    subprocess.run(command, check=False)
