"""`fal dev app dash`: the live dashboard for a `fal dev app` box.

Same screen and loop as `fal dev dash`, with the panels an app box has: the
GPU, when the platform will scale the box to zero (its keep_alive), the app
process and its last reload, and, for an LLM app, the inference engine.

The box runs the user's own image, so the probe (`remote/app_probe.py`, with
`remote/signals.py`) is sent as source over the runner exec on every refresh.
Nothing it does is a request to the box, so watching never keeps it up.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from pathlib import Path

from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.text import Text

from fal.console.icons import get_inline_separator_icon
from fal.dev import config, loading
from fal.dev.app import live
from fal.dev.app.session import Session
from fal.dev.box import Box
from fal.dev.dash import view, waiting
from fal.dev.dash.model import Poller, Snapshot, clean
from fal.dev.dash.tui import Controls
from fal.dev.remote.protocol import (
    BOX_STATE_ROOT,
    CHILD_PORT,
    CHILD_PREFIX,
)

REMOTE = Path(__file__).resolve().parent.parent / "remote"
# Runs the probe with signals.py as a module, from sources given as arguments.
BOOTSTRAP = (
    "import sys, types\n"
    "signals = types.ModuleType('signals')\n"
    "exec(compile(sys.argv[1], 'signals.py', 'exec'), signals.__dict__)\n"
    "sys.modules['signals'] = signals\n"
    "source = sys.argv[2]\n"
    "del sys.argv[1:3]\n"
    "exec(compile(source, 'app_probe.py', 'exec'), {'__name__': '__main__'})\n"
)
# An app that serves these is an LLM app: its engine gets a panel.
LLM_ROUTE = "/v1/chat/completions"
ENGINE_STATS_ROUTE = "/engine-stats"
# The CLI's own polling of the box (`start` waiting for setup, `logs -f`):
# routine, and it would push the app's lines out of the log panel.
POLLING_LOG = re.compile(r'"POST /__fal_dev/(status|logs) HTTP')
# Token rates are averaged over this window of probes.
RATE_WINDOW_SECONDS = 15.0
KEY_HINTS = [
    ("s", "shell", True),
    ("r", "restart app", True),
    ("d", "stop box", False),
    ("q", "quit", False),
]
NOT_RUNNING_HINTS = {
    "stopped": "No runner: the box scaled to zero. The next `fal dev app` "
    "command (or a proxied request) starts it and runs setup() again.",
    "starting": "A runner is starting; details appear once it is up.",
    "failed": "The runner failed to start; see `fal dev app logs`.",
}


class AppBox(Box):
    """A `fal dev app` box, seen through the platform like a `fal dev up` one."""

    def __init__(self, session: Session) -> None:
        super().__init__(session.name, host=session.host, account=session.account)
        self.app = session.name
        self.session = session

    def _live_code(self, app) -> str:
        # Never armed: the box runs on requests and scales to zero after them.
        return "running"

    def logs(self, runner_id: str, count: int, since) -> list[dict]:
        lines = super().logs(runner_id, count * 4, since)
        kept = [
            line
            for line in lines
            if not POLLING_LOG.search(str(line.get("message", "")))
        ]
        return kept[-count:]


def probe_command(session: Session) -> list[str]:
    settings = {
        "session_dir": f"{BOX_STATE_ROOT}/{session.name}",
        "child_port": CHILD_PORT,
        "child_status": f"{CHILD_PREFIX}status",
        "child_marker": "fal.dev.remote.child",
        "llm_route": LLM_ROUTE,
        "engine_stats": ENGINE_STATS_ROUTE,
        "end": config.PROBE_END,
    }
    return [
        "python3",
        "-c",
        BOOTSTRAP,
        (REMOTE / "signals.py").read_text(),
        (REMOTE / "app_probe.py").read_text(),
        json.dumps(settings),
    ]


def parse(output: str) -> dict:
    text = output.split(config.PROBE_END, 1)[0]
    lines = [line for line in text.splitlines() if line.startswith("{")]
    if not lines:
        raise ValueError(f"no status in the probe output: {text.strip()[-300:]!r}")
    probe = clean(json.loads(lines[-1]))
    if "probe_error" in probe:
        raise ValueError(f"the probe failed on the box: {probe['probe_error']}")
    return probe


class AppPoller(Poller):
    box: AppBox

    def __init__(self, box: AppBox, *args, **kwargs) -> None:
        super().__init__(box, *args, **kwargs)
        # (box clock, prompt tokens, completion tokens) per probe.
        self._tokens: deque[tuple[float, int, int]] = deque()

    def probe(self, runner_id: str) -> dict:
        _, output = self.box.run(
            runner_id,
            probe_command(self.box.session),
            timeout=30,
            until=config.PROBE_END.encode(),
        )
        probe = parse(output)
        probe["rates"] = self._rates(probe)
        return probe

    def _rates(self, probe: dict) -> dict | None:
        """Tokens per second in and out, over the last few probes."""
        totals = (probe.get("engine") or {}).get("usage_totals")
        if totals is None:
            self._tokens.clear()
            return None
        now = probe["now"]
        sample = (
            now,
            int(totals.get("prompt_tokens", 0)),
            int(totals.get("completion_tokens", 0)),
        )
        if self._tokens and sample[1] < self._tokens[-1][1]:
            # The app restarted and its totals with it.
            self._tokens.clear()
        self._tokens.append(sample)
        while len(self._tokens) > 2 and now - self._tokens[0][0] > RATE_WINDOW_SECONDS:
            self._tokens.popleft()
        first = self._tokens[0]
        elapsed = now - first[0]
        if elapsed <= 0:
            return None
        return {
            "in_per_s": (sample[1] - first[1]) / elapsed,
            "out_per_s": (sample[2] - first[2]) / elapsed,
        }


def keep_alive_panel(probe: dict, box_now: float) -> RenderableType:
    activity = probe.get("activity")
    if not activity:
        return view._panel(Text("no box requests yet", style=view.MUTED), "Keep alive")
    keep_alive = activity.get("keep_alive") or 0
    last = activity.get("last_request")
    rows: list[tuple[str, RenderableType | str]] = []
    if activity.get("inflight"):
        busy = f"busy, {activity['inflight']} request(s) in flight"
        rows.append(("scale to 0", Text(busy, style=f"bold {view.SUCCESS}")))
    elif last is not None:
        remaining = max(0.0, keep_alive - (box_now - last))
        style = f"bold {view.ERROR}" if remaining < 120 else "bold"
        rows.append(
            (
                "scale to 0",
                Text.styled(f"in {view.duration(remaining)}", style=style)
                + Text(
                    f" {get_inline_separator_icon()} "
                    f"{view.duration(keep_alive)} without requests",
                    style=view.MUTED,
                ),
            )
        )
    rows.append(("last request", view.ago(last, box_now)))
    return view._panel(view._kv(rows), "Keep alive")


def sync_row(sync: dict, now: float) -> tuple[str, RenderableType]:
    """The hot reload row: the last local sync, and whether --watch is running."""
    last = sync.get("last")
    watching = sync.get("watching")
    state = (
        Text("watching", style=view.SUCCESS)
        if watching
        else Text("not watching: `fal dev app sync --watch`", style=view.MUTED)
    )
    if not last:
        return "hot reload", Text.assemble(state, ("  no sync yet", view.MUTED))
    action = last.get("action")
    how = {
        "reload": "reloaded in place",
        "restart": "process restart",
        "pending": "saved, reloads when ready",
    }.get(action, f"not reloaded ({action})" if action else "no reload")
    count = last.get("files", 0)
    parts = [f"{count} file{'s' if count != 1 else ''}", how]
    if last.get("ms") is not None and action == "reload":
        parts.append(f"{last['ms']:.0f} ms")
    parts.append(f"synced in {last.get('seconds', 0):.1f}s")
    detail = Text(", ".join(parts))
    if action not in ("reload", "restart", "pending"):
        detail.stylize(view.ERROR)
    return "hot reload", Text.assemble(
        state,
        "\n",
        detail,
        (f" ({view.duration(max(0, now - last['at']))} ago)", view.MUTED),
    )


def app_panel(
    probe: dict,
    box_now: float,
    session: Session,
    sync: dict | None = None,
    now: float | None = None,
) -> RenderableType:
    app = probe.get("app")
    process = probe.get("process")
    rows: list[tuple[str, RenderableType | str]] = [
        ("app", f"{Path(session.file_path).name}::{session.class_name}")
    ]
    if app:
        state = Text.styled("ready", style=view.SUCCESS) + Text(
            f" {get_inline_separator_icon()} code v{app.get('code_version', 0)}"
            f", pid {app.get('pid')}",
            style=view.MUTED,
        )
        rows.append(("process", state))
        if app.get("setup_seconds"):
            rows.append(("setup()", f"{app['setup_seconds']:.0f}s"))
        reload = app.get("last_reload") or {}
        # Without a local record (another machine synced), the box's own note.
        if reload and not (sync or {}).get("last"):
            files = ", ".join(reload.get("files") or []) or "-"
            rows.append(
                (
                    "last sync",
                    f"{reload.get('action')}: {files} in {reload.get('ms')} ms",
                )
            )
        if app.get("error"):
            last_line = str(app["error"]).strip().splitlines()[-1:]
            rows.append(("last error", Text(" ".join(last_line), style=view.ERROR)))
    elif process:
        since = view.ago(process.get("started"), box_now).replace(" ago", "")
        rows.append(
            ("process", Text(f"running setup() for {since}", style=view.WARNING))
        )
    else:
        rows.append(("process", Text("not started", style=view.MUTED)))
    if sync is not None:
        rows.append(sync_row(sync, time.time() if now is None else now))
    return view._panel(view._kv(rows), "App")


def _pct(value: float | None) -> str:
    return f"{value * 100:.0f}%" if value is not None else "-"


def engine_panel(probe: dict) -> RenderableType | None:
    engine = probe.get("engine")
    if not engine:
        return None
    stats = engine.get("stats") or {}
    totals = engine.get("usage_totals") or {}
    rates = probe.get("rates") or {}
    running = int(stats.get("vllm:num_requests_running", 0))
    waiting = int(stats.get("vllm:num_requests_waiting", 0))
    kv = stats.get("vllm:kv_cache_usage_perc", stats.get("vllm:gpu_cache_usage_perc"))
    queries = stats.get("vllm:prefix_cache_queries_total") or 0
    hits = stats.get("vllm:prefix_cache_hits_total") or 0
    rows: list[tuple[str, RenderableType | str]] = [
        (
            "requests",
            Text.styled(f"{running} running", style=view.SUCCESS if running else "")
            + Text(
                f", {waiting} waiting, {totals.get('requests', 0)} done",
                style=view.MUTED,
            ),
        ),
        (
            "KV cache",
            Text.assemble(
                view._bar(kv * 100 if kv is not None else None), f" {_pct(kv)}"
            ),
        ),
        ("prefix hits", _pct(hits / queries if queries else None)),
    ]
    if rates:
        rows.append(
            (
                "tokens/s",
                f"{rates['out_per_s']:.0f} out, {rates['in_per_s']:.0f} in "
                f"(last {RATE_WINDOW_SECONDS:.0f}s)",
            )
        )
    title = Text.styled("Engine", style=view.TITLE) + Text(
        f"  {engine.get('engine', '')} {get_inline_separator_icon()} "
        f"{engine.get('model', '')}",
        style=f"default {view.MUTED}",
    )
    return view._panel(view._kv(rows), title)


def _setup_done(snapshot: Snapshot) -> bool:
    """The probe saw the app ready, so a leftover phase note is stale."""
    app = (snapshot.probe or {}).get("app")
    return bool(app and app.get("ready"))


def _degraded(draw, title: str) -> RenderableType | None:
    """One panel; a probe field it needs being absent (a box on another fal
    version) shows a dash on that panel only, not on all of them."""
    try:
        return draw()
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        reason = f"- (probe has no {exc})" if isinstance(exc, KeyError) else "-"
        return view._panel(Text(reason, style=view.MUTED), title)


def body(
    snapshot: Snapshot, box: AppBox, now: float, animate: bool = False
) -> RenderableType:
    parts: list[RenderableType] = [view.header(snapshot, box, now)]
    state = snapshot.state
    if state is None:
        return Group(*parts)
    phase = next((s for s in live.starting() if s["name"] == box.name), None)
    if phase is not None and not _setup_done(snapshot):
        text = waiting.PHASE_TEXT[phase["phase"]]
        elapsed = loading.format_elapsed(now - phase["since"])
        parts.append(
            view._panel(
                Text.assemble(
                    "starting: ", (text, "bold"), (f" ({elapsed})", view.MUTED)
                ),
                "Start",
            )
        )
    if not view._live(snapshot):
        hint = NOT_RUNNING_HINTS.get(state.code, "")
        return Group(*parts, view._panel(Text(hint, style=view.MUTED), ""))
    probe = snapshot.probe
    if probe is None:
        panels: list[RenderableType] = [
            view._panel(Text("waiting for the box", style=view.MUTED), title)
            for title in ("GPU", "Keep alive", "App")
        ]
    else:
        box_now = view._box_clock(snapshot, now)
        panels = [
            _degraded(lambda: view.gpu_panel(probe), "GPU"),
            _degraded(lambda: keep_alive_panel(probe, box_now), "Keep alive"),
            _degraded(
                lambda: app_panel(
                    probe, box_now, box.session, live.read_sync(box.session.name), now
                ),
                "App",
            ),
        ]
        if engine := _degraded(lambda: engine_panel(probe), "Engine"):
            panels.append(engine)
    parts.append(Columns(panels, equal=True))
    return view.WithLogs(parts, snapshot)


def render(
    snapshot: Snapshot,
    box: AppBox,
    now: float | None = None,
    *,
    screen: bool = False,
    animate: bool = False,
):
    return view.render(
        snapshot,
        box,
        now,
        screen=screen,
        animate=animate,
        body_of=body,
        key_hints=KEY_HINTS,
    )


def restart(session: Session) -> str:
    from fal.dev.app import client

    started = time.perf_counter()
    client.restart(session, sync_first=False)
    return f"app restarted: setup() ran again in {time.perf_counter() - started:.0f}s"


def stop(session: Session) -> str:
    from fal.dev.app import deploy

    killed, _ = deploy.stop_box(session)
    session.delete()
    return f"stopped {session.name}: app deleted, {len(killed)} runner(s) killed"


def controls(box: AppBox) -> Controls:
    session = box.session
    return Controls(
        poller=AppPoller,
        render=render,
        actions={
            "r": (
                "restarting the app (setup() runs again)",
                lambda _: restart(session),
            ),
        },
        down=lambda: stop(session),
        down_prompt="press d again to stop the box and delete its app",
        down_label="stopping the box",
        gone=lambda: not session.path.exists(),
        idle_warning=False,
    )
