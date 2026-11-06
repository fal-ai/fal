"""Live terminal dashboard for a running dev box: `fal dev dash`.

The dashboard talks to a box through the small `Box` interface below, so it
works with any box implementation that can list its runner and run a command
on it. `FalBox` implements it with the fal SDK for boxes laid out like the
fal-devbox app (workspace under /data/devboxes/<name>, /work snapshot).

Each refresh costs one exec round trip: `dash_probe.py` runs on the box and
returns everything as one JSON document. Runner state and logs are polled less
often. Polling never counts as activity for the idle watchdog (see the probe).
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from fal.dev.dash_probe import DEFAULTS as PROBE_DEFAULTS
from fal.dev.dash_probe import END_MARKER

PROBE_SOURCE = Path(__file__).with_name("dash_probe.py").read_text()

# Runner states, lowercased.
LIVE_RUNNER_STATES = {"running", "idle"}
FINAL_STATES = {"dead", "terminated", "terminating"}
STOPPING_STATES = {"draining"}
CRASH_STATE = "crash_backoff"
HOLD_HOURS = 1.0
# Control characters, including ESC: text from the box must not drive the terminal.
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")

# Public per-GPU list prices (fal.ai/pricing). Teams on custom rates pay less;
# the platform does not expose a team's own rate to API keys.
LIST_PRICE_PER_GPU_HOUR = {
    "GPU-B300": 8.50,
    "GPU-B200": 6.25,
    "GPU-H200": 4.50,
    "GPU-H100": 4.50,
    "GPU-RTX-PRO-6000": 2.99,
}

WATCHDOG_INTERVAL_SECONDS = 60
# Warnings the box writes when its idle watchdog cannot stop it.
WATCHDOG_OFF_PATTERNS = (
    "idle shutdown is off",
    "idle shutdown may not work",
    "idle shutdown failed",
)


@dataclass
class BoxInfo:
    """What the platform says about the box (not the box itself)."""

    name: str
    app_name: str
    team: str | None = None
    machine_type: str | None = None
    state: str = "unknown"  # starting, running, stopping, stopped, failed, unknown
    runner_id: str | None = None
    runner_state: str | None = None
    uptime_seconds: float | None = None
    min_concurrency: int | None = None
    fetched_at: float = field(default_factory=time.time)

    @property
    def live_runner(self) -> str | None:
        """The runner to talk to, while one is up (running or being stopped)."""
        return self.runner_id if self.state in ("running", "stopping") else None


class Box(Protocol):
    """A dev box as the dashboard sees it."""

    name: str
    # Minutes without activity before the box stops itself; None if unknown.
    idle_minutes: int | None
    # Whether Claude sessions count as activity; None if unknown.
    claude_enabled: bool | None
    # Overrides for the probe's paths and settings (see dash_probe.DEFAULTS).
    probe_config: dict[str, Any]

    def info(self) -> BoxInfo: ...

    def exec(
        self, runner_id: str, command: list[str], timeout: float, probe: bool = False
    ) -> tuple[int | None, bytes]:
        """Run a command without a terminal; (exit code, stdout).

        `probe` means the command ends its output with END_MARKER, which may
        arrive after the exit code. The exit code is None if none arrived.
        """
        ...

    def logs(self, runner_id: str, lines: int) -> list[dict]:
        """Newest log lines, oldest first: dicts with timestamp, level, message."""
        ...

    def shell(self, runner_id: str) -> None: ...

    def hold(self, runner_id: str, hours: float) -> str: ...

    def down(self) -> str: ...

    def hot_restart(self, runner_id: str) -> str: ...


def box_state(
    runners: list[dict], min_concurrency: int | None
) -> tuple[str, dict | None]:
    """(state, the runner that is the box) from runner dicts.

    Each runner dict has a lowercased `state` and `uptime_seconds`.
    """
    live = sorted(
        (r for r in runners if r["state"] in LIVE_RUNNER_STATES),
        key=lambda r: r.get("uptime_seconds") or 0,
    )
    if live:
        # min_concurrency None means the app lookup failed: assume still armed.
        return ("stopping" if min_concurrency == 0 else "running"), live[0]
    pending = [r for r in runners if r["state"] not in FINAL_STATES]
    if not pending:
        return "stopped", None
    newest = min(pending, key=lambda r: r.get("uptime_seconds") or 0)
    states = {r["state"] for r in pending}
    if CRASH_STATE in states:
        return "failed", newest
    if states <= STOPPING_STATES:
        return "stopping", newest
    return "starting", newest


def _clean(value: Any) -> Any:
    """Probe JSON with control characters removed from every string."""
    if isinstance(value, str):
        return _CONTROL.sub("", value)
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {_clean(k): _clean(v) for k, v in value.items()}
    return value


def parse_probe(stdout: bytes) -> dict:
    """The probe's JSON document from exec output that may carry other lines."""
    text = stdout.decode(errors="replace")
    if END_MARKER in text:
        text = text.split(END_MARKER, 1)[0]
    for raw in reversed(text.strip().splitlines()):
        line = raw.strip()
        if line.startswith("{"):
            probe = _clean(json.loads(line))
            if "probe_error" in probe:
                raise ValueError(f"the probe failed on the box: {probe['probe_error']}")
            return probe
    raise ValueError(f"no status in the probe output: {text.strip()[-300:]!r}")


def price_per_hour(machine_type: str | None, num_gpus: int) -> float | None:
    rate = LIST_PRICE_PER_GPU_HOUR.get(machine_type or "")
    return rate * max(num_gpus, 1) if rate is not None else None


@dataclass
class IdleView:
    watchdog: str  # on, off, unknown
    source: str  # watchdog (published by the box) or estimated
    idle_minutes: int | None
    last_active: float | None  # box clock
    remaining_seconds: float | None
    active_now: list[str]
    reasons: list[str]


def idle_view(
    probe: dict,
    idle_minutes: int | None,
    runner_started: float | None,
    claude_counts: bool,
    now: float,
) -> IdleView:
    """When the idle watchdog will stop the box, and why it is awake.

    `now` and every timestamp are on the box's clock. The watchdog keeps its
    last-activity time in memory; a box that publishes it (idle_state_file) is
    exact, otherwise it is rebuilt from the same signals the watchdog reads.
    """
    activity = probe.get("activity") or {}
    warnings = " ".join(probe.get("warnings") or []).lower()
    if any(p in warnings for p in WATCHDOG_OFF_PATTERNS):
        watchdog = "off"
    elif probe.get("admin_key_set"):
        watchdog = "on"
    else:
        watchdog = "unknown"

    active_now = []
    if activity.get("ssh_sessions"):
        active_now.append(f"{activity['ssh_sessions']} SSH session(s)")
    if activity.get("gpu_busy"):
        active_now.append("GPU busy")
    if activity.get("held"):
        active_now.append("hold")
    reasons = list(active_now)

    published = probe.get("idle_state")
    if not isinstance(published, dict) or not isinstance(
        published.get("last_active"), (int, float)
    ):
        published = None  # missing or malformed: estimate instead
    if published:
        source = "watchdog"
        last_active: float | None = float(published["last_active"])
        idle_minutes = published.get("idle_minutes", idle_minutes)
        if published.get("watchdog") in ("on", "off"):
            watchdog = published["watchdog"]
        reasons = published.get("reasons") or reasons
    else:
        source = "estimated"
        candidates = {
            "start": runner_started,
            "terminal input": activity.get("last_tty_input"),
        }
        if claude_counts:
            candidates["Claude session"] = activity.get("last_claude_activity")
        # Activity this dashboard saw earlier (SSH, GPU, hold) still counts.
        candidates["activity seen"] = probe.get("seen_active")
        infer_at = ((probe.get("hot") or {}).get("last_infer") or {}).get("at")
        if isinstance(infer_at, (int, float)):
            candidates["hot request"] = infer_at
        hold_until = activity.get("hold_until")
        if hold_until and hold_until <= now:
            # The watchdog counted the hold as activity until it ended.
            candidates["hold ended"] = hold_until
        known = {k: v for k, v in candidates.items() if v}
        last_active = max(known.values()) if known else None
        if active_now:
            last_active = now
        elif known:
            reasons = [max(known, key=lambda k: known[k])]

    remaining = None
    if idle_minutes and last_active is not None:
        remaining = max(0.0, idle_minutes * 60 - (now - last_active))
    return IdleView(
        watchdog, source, idle_minutes, last_active, remaining, active_now, reasons
    )


@dataclass
class Snapshot:
    """Everything the view draws; replaced as a whole by the poller."""

    info: BoxInfo | None = None
    probe: dict | None = None
    probe_at: float | None = None  # local clock
    probe_ms: float | None = None
    logs: list[dict] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    message: str | None = None
    message_until: float = 0.0  # local clock; the message is hidden after this


class Poller:
    """Refreshes a Snapshot in a background thread: one exec per `interval`."""

    def __init__(
        self,
        box: Box,
        interval: float = 3.0,
        info_interval: float = 15.0,
        log_interval: float = 10.0,
        log_lines: int = 30,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.box = box
        self.interval = interval
        self.info_interval = info_interval
        self.log_interval = log_interval
        self.log_lines = log_lines
        self.on_change = on_change or (lambda: None)
        self.snapshot = Snapshot()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._polls = 0
        self._next_info = 0.0
        self._next_logs = 0.0
        self._info_requested = False

    def update(self, **changes) -> None:
        with self._lock:
            current = self.snapshot
            errors = dict(current.errors)
            for key, value in (changes.pop("errors", None) or {}).items():
                if value is None:
                    errors.pop(key, None)
                else:
                    errors[key] = value
            self.snapshot = replace(current, **changes, errors=errors)
        self.on_change()

    def say(self, message: str, seconds: float = 10.0) -> None:
        """Show a status line in the footer for a while."""
        self.update(message=message, message_until=time.time() + seconds)

    def refresh_info(self) -> None:
        """Ask for runner state on the next tick (after an action)."""
        self._info_requested = True
        self._wake.set()

    def poll_once(self) -> None:
        now = time.time()
        if now >= self._next_info or self._info_requested:
            self._info_requested = False
            try:
                self.update(info=self.box.info(), errors={"info": None})
            except Exception as exc:
                self.update(errors={"info": _short(exc)})
            self._next_info = now + self.info_interval
        info = self.snapshot.info
        runner_id = info.live_runner if info else None
        if not runner_id:
            if self.snapshot.probe is not None or self.snapshot.logs:
                # A stopped box has no live state; drop the last runner's.
                self.update(probe=None, probe_at=None, logs=[])
            return
        config = dict(self.box.probe_config)
        # Walking /work for unsaved changes is the probe's only costly part;
        # do it every tenth refresh.
        config["check_dirty"] = self._polls % 10 == 0
        self._polls += 1
        start = time.time()
        try:
            _, out = self.box.exec(
                runner_id,
                ["python3", "-c", PROBE_SOURCE, json.dumps(config)],
                timeout=30,
                probe=True,
            )
            probe = parse_probe(out)
            activity = probe.get("activity") or {}
            last = self.snapshot.probe or {}
            if (
                activity.get("ssh_sessions")
                or activity.get("gpu_busy")
                or activity.get("held")
            ):
                probe["seen_active"] = probe.get("now")
            else:
                probe["seen_active"] = last.get("seen_active")
            if (probe.get("work") or {}).get("dirty") is None and last:
                # Keep the last answer between dirty checks.
                last_dirty = (last.get("work") or {}).get("dirty")
                probe["work"] = {**(probe.get("work") or {}), "dirty": last_dirty}
            self.update(
                probe=probe,
                probe_at=time.time(),
                probe_ms=(time.time() - start) * 1000,
                errors={"probe": None},
            )
        except Exception as exc:
            self.update(errors={"probe": _short(exc)})
        if now >= self._next_logs:
            try:
                self.update(
                    logs=self.box.logs(runner_id, self.log_lines), errors={"logs": None}
                )
            except Exception as exc:
                self.update(errors={"logs": _short(exc)})
            self._next_logs = now + self.log_interval

    def prime(self) -> None:
        """Poll until the first probe: the first poll may only find the runner."""
        self.poll_once()
        if self.snapshot.probe is None:
            self.poll_once()

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
    return " ".join(text.split())[:300]


# --- fal SDK implementation ---


def _ready_settings(message: str) -> dict:
    """Settings from the box's ready line.

    Example: `[devbox] ready: workspace=... idle_minutes=60 claude=True`.
    """
    return dict(re.findall(r"(\w+)=(\S+)", message.split("ready:", 1)[-1]))


class FalBox:
    """A box deployed as a fal app with the fal-devbox layout."""

    def __init__(
        self,
        name: str,
        app_name: str,
        *,
        team: str | None = None,
        host: str | None = None,
        idle_minutes: int | None = None,
        workspace: str | None = None,
        save_command: str = "/usr/local/bin/devbox-save",
    ) -> None:
        """`save_command` is trusted shell, run on the box before stopping it."""
        from fal.api.client import SyncServerlessClient

        self.name = name
        self.app_name = app_name
        self.team = team
        self.host = host
        self.idle_minutes = idle_minutes
        self.save_command = save_command
        self.claude_enabled: bool | None = None
        self.probe_config: dict[str, Any] = {
            "workspace": workspace or f"/data/devboxes/{name}"
        }
        self._client = SyncServerlessClient(host=host, team=team)
        self._stub = None
        self._settings_runner: str | None = None
        self._settings_attempts = 0

    def _config(self, key: str) -> Any:
        return self.probe_config.get(key, PROBE_DEFAULTS[key])

    def _shell_stub(self):
        stub = self._stub
        if stub is None:
            stub = self._stub = self._client._create_host()._connection.stub
        return stub

    def info(self) -> BoxInfo:
        aliases = [
            a
            for a in self._client.apps.list(filter=self.app_name)
            if a.alias == self.app_name
        ]
        alias = aliases[0] if aliases else None
        runners = [
            {
                "runner_id": r.runner_id,
                "state": r.state.value.lower(),
                "uptime_seconds": r.uptime.total_seconds(),
                "machine_type": r.machine_type,
            }
            for r in self._client.apps.runners(self.app_name)
        ]
        min_concurrency = alias.min_concurrency if alias else None
        state, runner = box_state(runners, min_concurrency)
        machine_type = (runner or {}).get("machine_type")
        if not machine_type and alias and alias.machine_types:
            machine_type = alias.machine_types[0]
        info = BoxInfo(
            name=self.name,
            app_name=self.app_name,
            team=self.team,
            machine_type=machine_type,
            state=state,
            runner_id=(runner or {}).get("runner_id"),
            runner_state=(runner or {}).get("state"),
            uptime_seconds=(runner or {}).get("uptime_seconds"),
            min_concurrency=min_concurrency,
        )
        if info.runner_id != self._settings_runner:
            self._settings_attempts = 0
        if info.runner_id and info.state == "running" and self._settings_attempts < 5:
            self._learn_settings(info.runner_id)
        return info

    def _learn_settings(self, runner_id: str) -> None:
        """Read idle_minutes and claude from the box's ready line, once per runner.

        The line can take a moment to be indexed, so a miss is retried a few times.
        """
        self._settings_runner = runner_id
        self._settings_attempts += 1
        try:
            lines = self._history(
                {"job_id": runner_id, "search": "[devbox] ready:"}, None, 5
            )
        except Exception:
            return
        for log in lines:
            settings = _ready_settings(log.get("message", ""))
            minutes = settings.get("idle_minutes", "")
            if minutes.isascii() and minutes.isdecimal():
                if self.idle_minutes is None:
                    self.idle_minutes = int(settings["idle_minutes"])
                self.claude_enabled = settings.get("claude") == "True"
                self._settings_attempts = 5  # found; stop looking
                return

    def exec(
        self, runner_id: str, command: list[str], timeout: float, probe: bool = False
    ) -> tuple[int | None, bytes]:
        import grpc
        import isolate_proto

        def inputs():
            yield isolate_proto.ShellRunnerInput(
                runner_id=runner_id, command=command, tty=False
            )
            yield isolate_proto.ShellRunnerInput(close=True)

        try:
            return self._exec_once(inputs, probe, timeout)
        except grpc.RpcError as exc:
            # The cached channel carries the access token it was built with.
            if exc.code() != grpc.StatusCode.UNAUTHENTICATED:
                raise
            self._stub = None
            return self._exec_once(inputs, probe, timeout)

    def _exec_once(
        self, inputs, probe: bool, timeout: float
    ) -> tuple[int | None, bytes]:
        call = self._shell_stub().ShellRunner(inputs(), timeout=timeout)
        out = bytearray()
        code = None
        marker = END_MARKER.encode()
        try:
            for message in call:
                if message.data and not (
                    message.HasField("stream") and message.stream == 2
                ):
                    out += message.data
                if message.HasField("exit_code"):
                    code = message.exit_code
                # Output can trail the exit code, so a probe is read up to its
                # end marker. Unlike fal.cli.runners._shell_session, this
                # captures output instead of printing it.
                if message.close or (marker in out if probe else code is not None):
                    break
        finally:
            call.cancel()
        return code, bytes(out)

    def _history(
        self, params: dict, since: datetime | None, page_size: int
    ) -> list[dict]:
        import httpx

        from fal.cli.runners import _post_history

        rest = self._client._create_rest_client()
        with httpx.Client(
            base_url=rest.base_url, headers=rest.get_headers(), timeout=15
        ) as http:
            items, _ = _post_history(http, params, since, None, page_size)
        return list(reversed(items))  # the API returns newest first

    def logs(self, runner_id: str, lines: int) -> list[dict]:
        since = datetime.now(timezone.utc) - timedelta(minutes=30)
        return self._history({"job_id": runner_id}, since, lines)

    def shell(self, runner_id: str) -> None:
        # A child process: `fal runners shell` leaves a thread blocked reading
        # stdin, which in this process would swallow the next key.
        command = [sys.executable, "-m", "fal", "runners", "shell", runner_id]
        if self.team:
            command += ["--team", self.team]
        if self.host:
            command += ["--host", self.host]
        subprocess.run(command, check=False)

    def _scale(self, min_concurrency: int) -> None:
        self._client.apps.scale(self.app_name, min_concurrency=min_concurrency)

    def hold(self, runner_id: str, hours: float) -> str:
        # Re-arming a box that has already stopped would start a new runner.
        if self.info().live_runner != runner_id:
            return "the box is no longer running; nothing held"
        hold_file = shlex.quote(self._config("hold_file"))
        # Re-arm first, in case the watchdog has just started stopping the box.
        self._scale(1)
        until = int(time.time() + hours * 3600)
        # Never shorten a longer hold, e.g. one set for an unattended job.
        script = (
            f"old=$(cat {hold_file} 2>/dev/null || echo 0); "
            f'if [ "${{old%.*}}" -gt {until} ] 2>/dev/null; then echo kept; '
            f"else echo {until} > {hold_file}; fi"
        )
        code, out = self.exec(runner_id, ["sh", "-c", script], timeout=30)
        if code != 0:
            return f"hold failed (exit {code}); the box is armed but not held"
        if b"kept" in out:
            return "a longer hold is already in place; kept it"
        # An idle shutdown already under way may have scaled to 0 after the re-arm.
        if not self.info().min_concurrency:
            self._scale(1)
        return f"idle shutdown held for {hours:g}h"

    def down(self) -> str:
        info = self.info()
        saved = "nothing to save"
        # Save whenever a runner is up, even if its state looks like stopping.
        runner_id = info.live_runner
        if runner_id:
            code, out = self.exec(
                runner_id, ["sh", "-c", f"{self.save_command} 2>&1"], timeout=1800
            )
            if code != 0:
                tail = out.decode(errors="replace").strip()[-200:]
                return f"save failed (exit {code}), box left running: {tail}"
            saved = "/work saved"
        self._scale(0)
        for runner in self._client.apps.runners(self.app_name):
            if runner.state.value.lower() not in FINAL_STATES:
                try:
                    self._client.runners.stop(runner.runner_id)
                except Exception:
                    pass  # scale 0 releases a runner that refuses
        return f"stopping: {saved}, scaled to 0"

    def hot_restart(self, runner_id: str) -> str:
        port = int(self._config("hot_port"))
        script = (
            "import urllib.request as u;"
            f"r=u.Request('http://127.0.0.1:{port}/restart',data=b'{{}}',method='POST');"
            "print(u.urlopen(r,timeout=10).read().decode())"
        )
        code, _ = self.exec(runner_id, ["python3", "-c", script], timeout=30)
        if code != 0:
            return "hot restart failed: is the hot server running?"
        return "hot server restarting; the model reloads"
