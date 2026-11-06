import io
import json
import re
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from rich.console import Console

from fal import flags
from fal.dev import config
from fal.dev.account import Account
from fal.dev.box import BoxState
from fal.dev.dash import model
from fal.dev.dash.model import (
    Cost,
    Poller,
    Snapshot,
    fetch_cost,
    idle_view,
    parse_probe,
    probe_command,
)
from fal.dev.dash.view import render
from fal.sdk import RunnerState

FIXTURES = Path(__file__).with_name("fixtures")
REMOTE = Path(model.__file__).parents[1] / "remote"
NOW = 1762380000.0


def _console_with_encoding(encoding: str) -> Console:
    return Console(file=io.TextIOWrapper(io.BytesIO(), encoding=encoding))


@pytest.fixture(autouse=True)
def unicode_icons(monkeypatch):
    """Glyphs follow the default console's encoding; make it UTF-8 here."""
    from fal.console import icons

    monkeypatch.setattr(icons, "default_console", _console_with_encoding("utf-8"))


@pytest.fixture(autouse=True)
def no_usage_calls(monkeypatch):
    """The poller asks the platform for the cost in a thread; never in tests."""
    monkeypatch.setattr(model, "fetch_cost", lambda box, uptime: None)


def _probe(**changes) -> dict:
    probe = json.loads((FIXTURES / "probe_running.json").read_text())
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(probe.get(key), dict):
            probe[key] = {**probe[key], **value}
        else:
            probe[key] = value
    return probe


def _runner(state=RunnerState.RUNNING, uptime=3600, runner_id="runner-abc"):
    runner = MagicMock()
    runner.runner_id = runner_id
    runner.state = state
    runner.uptime = timedelta(seconds=uptime)
    runner.machine_type = "GPU-H100"
    return runner


def _state(code="running", runner_id="runner-abc") -> BoxState:
    app = MagicMock(min_concurrency=1, machine_types=["GPU-H100"])
    return BoxState(
        code, app, None if code == "stopped" else _runner(runner_id=runner_id)
    )


class FakeDashBox:
    name = "demo"
    app = "devbox-demo"
    account = Account(None, False, via_key=True)
    host = None

    def __init__(self, probe=None, code="running"):
        self.probe = probe or _probe()
        self.code = code
        self.runner_id = "runner-abc"
        self.run_calls = []
        self.log_calls = 0

    def auth_headers(self):
        return {"Authorization": "Key k"}

    def state(self):
        return _state(self.code, self.runner_id)

    def run(self, runner_id, command, *, timeout=None, until=None):
        self.run_calls.append((command, until))
        out = f"notice from the shell\n{json.dumps(self.probe)}\n{config.PROBE_END}\n"
        return 0, out

    def logs(self, runner_id, count, since):
        self.log_calls += 1
        return [
            {
                "timestamp": "2026-11-05T21:58:00.000Z",
                "level": "INFO",
                "message": "[devbox] ready: idle_minutes=60",
            },
            {
                "timestamp": "2026-11-05T21:59:00.000Z",
                "level": "ERROR",
                "message": "[hot] boom",
            },
        ]


def _snapshot(probe=None, code="running", **changes) -> Snapshot:
    box = FakeDashBox(probe, code)
    return Snapshot(
        state=box.state(),
        state_at=NOW,
        probe=box.probe if code == "running" else None,
        probe_at=NOW,
        probe_ms=420,
        **changes,
    )


def _text(renderable, width=150) -> str:
    console = Console(record=True, width=width, force_terminal=False, color_system=None)
    console.print(renderable)
    return console.export_text()


def _frame(snapshot, box=None, **kwargs) -> str:
    return _text(render(snapshot, box or FakeDashBox(), now=NOW), **kwargs)


def test_render_falls_back_to_ascii(monkeypatch):
    from fal.console import icons

    monkeypatch.setattr(icons, "default_console", _console_with_encoding("ascii"))
    text = _frame(_snapshot())
    assert "* running" in text
    assert "41 C" in text


def test_parse_probe_skips_noise_and_stops_at_marker():
    probe = _probe()
    out = f'banner\n{json.dumps(probe)}\n{config.PROBE_END}\n{{"late": true}}\n'
    assert parse_probe(out) == probe
    with pytest.raises(ValueError):
        parse_probe("no json here")


def test_parse_probe_raises_on_probe_error():
    with pytest.raises(ValueError, match="probe failed"):
        parse_probe('{"probe_error": "boom"}\n' + config.PROBE_END)


def test_idle_view_reads_the_watchdogs_published_state():
    view = idle_view(_probe(), NOW)
    assert view.watchdog == "on"
    # The watchdog last saw a hot request 2 minutes ago.
    assert view.last_active == NOW - 120
    assert view.remaining_seconds == 58 * 60
    assert view.reasons == ["hot request"]
    assert view.active_now == []


def test_idle_view_shows_what_is_awake_right_now():
    probe = _probe(
        activity={"ssh_sessions": 2, "gpu_busy": True},
        idle_state={"last_active": NOW - 600, "idle_minutes": 30, "reasons": []},
    )
    view = idle_view(probe, NOW)
    assert view.active_now == ["2 SSH session(s)", "GPU busy"]
    assert view.remaining_seconds == 20 * 60


def test_idle_view_has_no_countdown_before_the_first_watchdog_tick():
    view = idle_view(_probe(idle_state=None), NOW)
    assert view.remaining_seconds is None


def test_idle_view_detects_watchdog_off():
    probe = _probe(
        admin_key_set=False,
        warnings=["DEVBOX_ADMIN_KEY is not set, so idle shutdown is off."],
    )
    assert idle_view(probe, NOW).watchdog == "off"


def test_render_running_box():
    snapshot = _snapshot(cost=Cost(spent=4.5, per_hour=4.5))
    text = _frame(snapshot)
    for expected in [
        "fal dev  demo",
        "● running",
        "devbox-demo",
        "1x GPU-H100",
        "runner-abc",
        "1h 00m",
        "$4.50 at $4.50/h",
        "37%",
        "21.0/80 GiB",
        "41°C",
        "in 58m 00s",
        "· 60 min timeout",
        "hot request, 2m 00s ago",
        "hold",
        "code v3",
        "infer.py in 9.4 ms",
        "571 ms, 2m 00s ago",
        "60.0 MB",
        "this runner",
        "no recent logs",
        " s ",
        "quit",
    ]:
        assert expected in text, expected
    assert "IDLE SHUTDOWN IS OFF" not in text


def test_render_cost_states():
    assert "rate pending" in _frame(_snapshot(cost=Cost(spent=0.0, per_hour=None)))
    # Nothing is shown until the platform has answered.
    unanswered = _frame(_snapshot())
    assert "rate pending" not in unanswered
    assert "$" not in unanswered
    hidden = _frame(_snapshot(cost_hidden=True))
    assert "rate pending" not in hidden
    assert "$" not in hidden


def test_render_warns_loudly_when_watchdog_is_off():
    probe = _probe(
        admin_key_set=False,
        warnings=["DEVBOX_ADMIN_KEY is not set, so idle shutdown is off."],
    )
    text = _frame(_snapshot(probe), FakeDashBox(probe))
    assert "IDLE SHUTDOWN IS OFF" in text
    assert re.search(r"auto-stop\s+off", text)


def test_render_hot_error_and_hold():
    probe = _probe(
        hot={"error": "NameError: name 'pipe' is not defined", "ready": True},
        activity={"held": True, "hold_until": NOW + 1800},
    )
    text = _frame(_snapshot(probe), FakeDashBox(probe))
    assert "NameError: name 'pipe' is not defined" in text
    assert "30m 00s left" in text


def test_render_stopped_box():
    text = _frame(_snapshot(code="stopped"), FakeDashBox(code="stopped"))
    assert "● stopped" in text
    assert "fal dev up" in text
    assert "rate pending" not in text
    assert "IDLE SHUTDOWN IS OFF" not in text


def test_render_hides_box_text_control_characters():
    out = json.dumps(_probe(hot={"error": "bad\x1b]0;title\x07 thing"}))
    snapshot = _snapshot(parse_probe(out))
    snapshot.logs = model.clean(
        [{"timestamp": "x", "level": "INFO", "message": "a\x1b[2Jb"}]
    )
    text = _frame(snapshot)
    assert "\x1b" not in text
    assert "bad]0;title thing" in text
    assert "a[2Jb" in text


def test_probe_text_cannot_inject_markup_or_escapes():
    out = json.dumps(
        _probe(
            hot={
                "last_reload": {
                    "files": ["[/x]", "[link=https://e.x]y[/link]"],
                    "ms": 1,
                    "ok": True,
                }
            },
            gpus=[
                {
                    "index": 0,
                    "name": "g",
                    "util_pct": 1,
                    "mem_used_mib": 1,
                    "mem_total_mib": 2,
                    "temp_c": "\x1b]0;pwn\x07x",
                    "power_w": 1,
                }
            ],
        )
    )
    probe = parse_probe(out)
    assert probe["gpus"][0]["temp_c"] == "]0;pwnx"
    console = Console(record=True, width=150, force_terminal=True)
    console.print(
        render(_snapshot(probe), FakeDashBox(probe), now=NOW, screen=True), height=60
    )
    assert "[/x], [link=https://e.x]y[/link]" in console.export_text(styles=False)
    assert "\x1b]8" not in console.export_text(styles=True)


def test_footer_message_expires():
    snapshot = _snapshot()
    snapshot.message, snapshot.message_until = "press d again", NOW + 5
    assert "press d again" in _text(render(snapshot, FakeDashBox(), now=NOW))
    assert "press d again" not in _text(render(snapshot, FakeDashBox(), now=NOW + 6))


def test_panels_stack_on_narrow_terminals():
    narrow = _frame(_snapshot(), width=80)
    # Stacked, the GPU row keeps its percent, memory and power.
    assert "37%" in narrow
    assert "212W" in narrow


@pytest.mark.parametrize("height", [20, 40])
def test_screen_keeps_the_footer_visible(height):
    console = Console(
        record=True, width=80, height=height, force_terminal=False, color_system=None
    )
    console.print(
        render(_snapshot(), FakeDashBox(), now=NOW, screen=True), height=height
    )
    assert "quit" in console.export_text()


def test_poller_runs_one_exec_per_refresh():
    box = FakeDashBox()
    poller = Poller(box)
    poller.poll_once()
    poller.poll_once()
    assert len(box.run_calls) == 2
    command, until = box.run_calls[0]
    assert command[:2] == ["sh", "-c"]
    assert command[3] == f"{config.PROBE_DIR}/probe.py"
    assert json.loads(command[4]) == {
        "workspace": "/data/devboxes/demo",
        "check_dirty": True,
    }
    assert until == config.PROBE_END.encode()
    assert json.loads(box.run_calls[1][0][4])["check_dirty"] is False
    # Logs are polled less often than the probe.
    assert box.log_calls == 1
    snapshot = poller.snapshot
    assert snapshot.probe["hot"]["code_version"] == 3
    assert snapshot.logs[-1]["message"] == "[hot] boom"
    assert snapshot.errors == {}


def test_poller_keeps_last_dirty_answer_between_checks():
    box = FakeDashBox()
    poller = Poller(box)
    poller.poll_once()
    box.probe = _probe(work={"dirty": None})
    poller.poll_once()
    assert poller.snapshot.probe["work"]["dirty"] is True


def test_poller_reports_probe_errors_without_crashing():
    box = FakeDashBox()
    box.run = lambda *a, **k: (1, "python3: not found")
    poller = Poller(box)
    poller.poll_once()
    assert "no status" in poller.snapshot.errors["probe"]
    assert "probe error" in _text(render(poller.snapshot, box, now=NOW))


def test_poller_skips_exec_when_box_is_stopped():
    box = FakeDashBox(code="stopped")
    poller = Poller(box)
    poller.poll_once()
    assert box.run_calls == []


def test_poller_drops_probe_when_box_stops():
    box = FakeDashBox()
    poller = Poller(box)
    poller.poll_once()
    box.code = "stopped"
    poller.refresh_state()
    poller.poll_once()
    assert poller.snapshot.probe is None
    assert poller.snapshot.logs == []


def test_probe_command_names_the_boxs_workspace():
    command = probe_command(FakeDashBox(), check_dirty=False)

    assert command[3] == "/usr/local/lib/devbox-probe/probe.py"
    assert json.loads(command[4])["workspace"] == "/data/devboxes/demo"


def test_probe_script_reads_box_state(tmp_path):
    """Run the real probe, installed like the box does, against a fake box layout."""
    workspace = tmp_path / "devboxes" / "demo"
    transcripts = workspace / ".home" / ".claude" / "projects" / "p"
    transcripts.mkdir(parents=True)
    (transcripts / "s.jsonl").write_text("{}\n")
    (workspace / "work.tar.zst").write_bytes(b"x" * 2048)
    (workspace / "work.lease").write_text("someone-else active 1762380000\n")
    work = tmp_path / "work"
    work.mkdir()
    stamp = tmp_path / "save-stamp"
    stamp.touch()
    (work / "new.py").write_text("print(1)\n")
    hold = tmp_path / "hold"
    hold.write_text("9999999999\n")
    warning = tmp_path / "warning"
    warning.write_text("devbox: DEVBOX_ADMIN_KEY is not set, idle shutdown is off.\n")
    overrides = {
        "LOCAL_WORK": work,
        "SAVE_STAMP": stamp,
        "HOLD_FILE": hold,
        "WARNING_FILE": warning,
        "IDLE_STATE_FILE": tmp_path / "missing.json",
        "ADMIN_KEY_SECRET": "FAL_DEV_DASH_TEST_UNSET_KEY",
        "CLAUDE_TMUX_SESSION": "fal-dev-dash-test-no-such-session",
        "HOT_TMUX_SESSION": "fal-dev-dash-test-no-such-session",
    }
    install = tmp_path / "probe"
    install.mkdir()
    (install / "probe.py").write_text((REMOTE / "probe.py").read_text())
    (install / "signals.py").write_text((REMOTE / "signals.py").read_text())
    (install / "config.py").write_text(
        Path(config.__file__).read_text()
        + "".join(f"{name} = {str(value)!r}\n" for name, value in overrides.items())
    )
    settings = {"workspace": str(workspace), "check_dirty": True}
    out = subprocess.run(
        [sys.executable, str(install / "probe.py"), json.dumps(settings)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout
    probe = parse_probe(out)
    assert probe["activity"]["held"] is True
    assert probe["activity"]["last_claude_activity"]
    assert probe["admin_key_set"] is False
    assert probe["warnings"] == ["DEVBOX_ADMIN_KEY is not set, idle shutdown is off."]
    assert probe["claude_running"] is False
    assert probe["hot"] == {"running": False}
    assert probe["idle_state"] is None
    assert probe["work"]["snapshot"]["size"] == 2048
    assert probe["work"]["lease"]["owner"] == "someone-else"
    assert probe["work"]["lease"]["mine"] is False
    assert probe["work"]["dirty"] is True
    assert idle_view(probe, probe["now"]).watchdog == "off"


def test_published_idle_state_is_what_the_dash_reads(monkeypatch, tmp_path):
    import runpy

    from fal.dev.box import BOX_APP

    module = runpy.run_path(str(BOX_APP))
    path = tmp_path / "idle.json"
    monkeypatch.setattr(config, "IDLE_STATE_FILE", str(path))
    module["_publish_idle_state"](NOW - 600, ["hot request"])

    view = idle_view(_probe(idle_state=json.loads(path.read_text())), NOW)
    assert view.reasons == ["hot request"]
    assert view.remaining_seconds == 60 * 60 - 600


def test_fetch_cost_sums_the_runs_usage(monkeypatch):
    seen = {}

    def get(url, **kwargs):
        seen.update(url=url, **kwargs)
        rows = [
            {"cost": 3.0, "unit_price": 0.00125, "is_surge": False},
            {"cost": 0.5, "unit_price": 0.0025, "is_surge": True},
        ]
        return httpx.Response(
            200, json={"summary": rows}, request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", get)
    box = FakeDashBox()

    cost = fetch_cost(box, runner_uptime=7200)

    assert cost == Cost(spent=3.5, per_hour=pytest.approx(4.5))
    assert seen["url"] == f"{flags.PLATFORM_URL}/serverless/usage"
    assert seen["params"]["app"] == "devbox-demo"
    assert seen["params"]["expand"] == "summary"
    assert seen["headers"] == {"Authorization": "Key k"}


def test_fetch_cost_has_no_rate_before_usage_shows_up(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: httpx.Response(
            200, json={"summary": []}, request=httpx.Request("GET", url)
        ),
    )

    assert fetch_cost(FakeDashBox(), 60) == Cost(spent=0, per_hour=None)


@pytest.mark.parametrize("status", [401, 403])
def test_fetch_cost_is_none_without_billing_access(monkeypatch, status):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: httpx.Response(status, request=httpx.Request("GET", url)),
    )

    assert fetch_cost(FakeDashBox(), 60) is None


def test_refresh_cost_hides_the_line_without_access_and_survives_errors(monkeypatch):
    poller = Poller(FakeDashBox())
    poller.update(state=_state(), state_at=time.time())

    monkeypatch.setattr(model, "fetch_cost", lambda box, uptime: None)
    poller.refresh_cost()
    assert poller.snapshot.cost_hidden

    monkeypatch.setattr(model, "fetch_cost", lambda box, uptime: Cost(1.0, 2.0))
    poller.refresh_cost()
    assert poller.snapshot.cost == Cost(1.0, 2.0)
    assert not poller.snapshot.cost_hidden

    def fail(box, uptime):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(model, "fetch_cost", fail)
    poller.refresh_cost()
    assert poller.snapshot.cost == Cost(1.0, 2.0)
    assert poller.snapshot.cost_stale
    assert "down" in poller.snapshot.errors["cost"]
    assert "(stale)" in _frame(poller.snapshot)

    # Asking again a minute later recovers, also from a hidden line.
    monkeypatch.setattr(model, "fetch_cost", lambda box, uptime: Cost(2.0, 2.0))
    poller.refresh_cost()
    assert not poller.snapshot.cost_stale
    assert poller.snapshot.errors == {}


def test_hold_reports_whether_a_longer_hold_was_kept():
    box = MagicMock()
    box.hold.return_value = True
    assert model.hold(box) == "idle shutdown held for 1h"
    box.hold.return_value = False
    assert "longer hold" in model.hold(box)


def test_hot_restart_runs_on_the_box_not_through_the_app():
    box = MagicMock()
    box.run.return_value = (0, "")

    model.hot_restart(box, "runner-abc")

    command = box.run.call_args.args[1]
    assert command[0] == "curl"
    assert command[-1] == f"http://127.0.0.1:{config.HOT_PORT}/restart"
    box.run.return_value = (7, "")
    with pytest.raises(model.DevError, match="hot server running"):
        model.hot_restart(box, "runner-abc")


def test_a_failed_cost_lookup_on_another_host_names_the_platform_url(monkeypatch):
    box = FakeDashBox()
    box.host = "api.dev.example"
    poller = Poller(box)
    poller.poll_once(cost=False)

    def fail(box, uptime):
        raise RuntimeError("down")

    monkeypatch.setattr(model, "fetch_cost", fail)
    poller.refresh_cost()
    assert "FAL_PLATFORM_URL" in poller.snapshot.errors["cost"]


def test_a_failed_first_cost_lookup_shows_nothing_rather_than_stale():
    poller = Poller(FakeDashBox())
    poller.update(state=_state(), state_at=time.time())

    def fail(box, uptime):
        raise httpx.ConnectError("down")

    model.fetch_cost = fail
    poller.refresh_cost()

    assert poller.snapshot.cost is None
    assert "stale" not in _frame(poller.snapshot)
    assert "cost error" in _frame(poller.snapshot)


def test_cost_follows_the_runner_across_a_down_and_up(monkeypatch):
    box = FakeDashBox()
    poller = Poller(box)
    monkeypatch.setattr(model, "fetch_cost", lambda box, uptime: Cost(9.0, 4.5))
    poller.poll_once(cost=False)
    poller.refresh_cost()
    assert poller.snapshot.cost == Cost(9.0, 4.5)

    # Down: the cost of the old run goes with it.
    box.code = "stopped"
    poller.refresh_state()
    poller.poll_once(cost=False)
    assert poller.snapshot.cost is None

    # Up again on another runner: an answer for the old runner is discarded.
    def answer_for_the_old_runner(box_, uptime):
        box.code = "running"
        box.runner_id = "runner-new"
        poller.refresh_state()
        poller.poll_once(cost=False)
        return Cost(99.0, 4.5)

    box.code = "running"
    box.runner_id = "runner-old"
    poller.refresh_state()
    poller.poll_once(cost=False)
    monkeypatch.setattr(model, "fetch_cost", answer_for_the_old_runner)
    poller.refresh_cost()

    assert poller.snapshot.runner_id == "runner-new"
    assert poller.snapshot.cost is None


def test_prime_fetches_the_cost_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        model, "fetch_cost", lambda box, uptime: calls.append(1) or Cost(1.0, 2.0)
    )
    poller = Poller(FakeDashBox())

    poller.prime()

    assert len(calls) == 1
    assert poller.snapshot.cost == Cost(1.0, 2.0)


def test_action_messages_cannot_drive_the_terminal():
    poller = Poller(FakeDashBox())

    poller.say("failed: \x1b]0;pwn\x07boom")

    assert poller.snapshot.message == "failed: ]0;pwnboom"
    assert "\x1b" not in model._short(RuntimeError("a\x1b[2Jb"))


def test_a_box_without_the_probe_answers_at_once_and_says_to_redeploy(tmp_path):
    command = probe_command(FakeDashBox(), check_dirty=False)
    command[3] = str(tmp_path / "no-such-probe.py")

    done = subprocess.run(
        command, capture_output=True, text=True, timeout=10, check=False
    )

    assert done.returncode == 1
    assert config.PROBE_END in done.stdout
    with pytest.raises(ValueError, match="predates the dashboard probe"):
        parse_probe(done.stdout)


def test_probe_output_from_another_version_is_an_error_not_a_crash():
    old = _probe()
    del old["claude_running"]
    with pytest.raises(ValueError, match="another fal version"):
        parse_probe(json.dumps(old))
    short = _probe()
    del short["activity"]["held"]
    with pytest.raises(ValueError, match="another fal version"):
        parse_probe(json.dumps(short))


def test_header_says_runner_pending_until_a_runner_is_up():
    snapshot = Snapshot(state=_state("starting"), state_at=NOW)
    assert "runner pending" in _frame(snapshot)
    assert "runner runner-abc" in _frame(_snapshot())
    assert "runner -" in _frame(Snapshot(state=_state("stopped"), state_at=NOW))


def test_booting_box_shows_the_dust_panel_with_the_runner_state():
    state = _state("starting")
    state.runner.state = RunnerState.PENDING
    state.runner.uptime = timedelta(seconds=42)
    text = _frame(Snapshot(state=state, state_at=NOW))
    assert "GPU" in text
    assert "waiting for a GPU runner: queued  42s  typically ~1m40s" in text


def test_hot_restart_shows_the_hot_server_panel():
    probe = _probe(hot={"running": True, "restarting": "load.py changed"})
    snapshot = _snapshot(probe, since={"hot-restart": NOW - 12})
    text = _frame(snapshot)
    assert "Hot server" in text
    assert "restarting the hot server: loading weights" in text
    assert "model load  12s  typically ~35s" in text
    assert "Hot server" not in _frame(_snapshot())


def test_poller_times_waits_from_when_they_began():
    box = FakeDashBox(_probe(hot={"running": True, "restarting": "manual"}))
    poller = Poller(box)
    poller.poll_once(cost=False)
    began = poller.snapshot.since["hot-restart"]
    poller.poll_once(cost=False)
    assert poller.snapshot.since["hot-restart"] == began
    box.probe = _probe()
    poller.poll_once(cost=False)
    assert "hot-restart" not in poller.snapshot.since


def test_a_stale_probe_does_not_keep_animating():
    probe = _probe(hot={"running": True, "restarting": "manual"})
    assert model.waits(_snapshot(probe)) == {"hot-restart"}
    assert model.waits(_snapshot(probe, errors={"probe": "timed out"})) == set()


def test_dash_waits_draw_dust_only_when_animating():
    snapshot = Snapshot(state=_state("starting"), state_at=NOW)

    def braille(animate):
        frame = _text(render(snapshot, FakeDashBox(), now=NOW, animate=animate))
        return any(0x2800 <= ord(ch) <= 0x28FF for ch in frame)

    assert braille(True)
    assert not braille(False)


def test_dashboard_decides_on_animation_before_live_replaces_stdout(monkeypatch):
    from fal.dev.dash import tui

    seen = []

    class _Live:
        def __init__(self, renderable, **kwargs):
            seen.append(renderable)

        def start(self):
            raise KeyboardInterrupt  # end the loop after the first frame

        def stop(self):
            pass

    monkeypatch.setattr(tui.os, "name", "posix")
    monkeypatch.setattr(tui.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(tui.sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(tui.sys.stdin, "fileno", lambda: 0, raising=False)
    monkeypatch.setattr(
        tui, "_cbreak", lambda fd: __import__("contextlib").nullcontext()
    )
    monkeypatch.setattr(tui, "Live", _Live)
    monkeypatch.setattr(tui.model.Poller, "start", lambda self: None)
    calls = []
    monkeypatch.setattr(
        tui.loading, "animation_enabled", lambda stream: calls.append(stream) or True
    )
    monkeypatch.setattr(tui, "render", lambda *a, **k: k["animate"])
    tui.run_dashboard(FakeDashBox())
    assert seen == [True] and len(calls) == 1


def test_gpu_panel_shows_cpu_and_memory():
    system = {
        "cpu_pct": 37.5,
        "cpus": 16,
        "mem_used_bytes": 24 * 2**30,
        "mem_total_bytes": 128 * 2**30,
    }
    text = _frame(_snapshot(probe=_probe(system=system)))
    assert re.search(r"cpu .* 38% +16 cores", text)
    assert re.search(r"ram .* 19% +24\.0/128 GiB", text)


def test_gpu_panel_without_system_readings_from_an_older_probe():
    probe = _probe()
    probe.pop("system", None)
    text = _frame(_snapshot(probe=probe))
    assert "cores" not in text
    assert "GPU" in text


def test_unknown_memory_use_is_not_shown_as_zero():
    system = {
        "cpu_pct": None,
        "cpus": 4,
        "mem_used_bytes": None,
        "mem_total_bytes": 2**34,
    }
    text = _frame(_snapshot(probe=_probe(system=system)))
    assert re.search(r"ram .* -% +-", text)
    assert "0.0/16 GiB" not in text


def test_cpu_and_memory_still_show_when_the_gpus_cannot_be_read():
    system = {
        "cpu_pct": 50.0,
        "cpus": 8,
        "mem_used_bytes": 2**33,
        "mem_total_bytes": 2**35,
    }
    probe = _probe(gpus=[], gpu_error="nvidia-smi failed", system=system)
    text = _frame(_snapshot(probe=probe))
    assert "nvidia-smi failed" in text
    assert re.search(r"cpu .* 50% +8 cores", text)
    assert re.search(r"ram .* 25% +8\.0/32 GiB", text)


def _system(**changes) -> dict:
    return {
        "cpu_pct": 10.0,
        "cpus": 8,
        "mem_used_bytes": 2**33,
        "mem_total_bytes": 2**35,
        **changes,
    }


def test_disks_show_free_space_and_warn_when_low():
    from fal.dev.dash import view

    disks = [
        {
            "paths": ["/work", "/"],
            "used_bytes": 6 * 2**40,
            "total_bytes": 7 * 2**40,
            "free_bytes": 3 * 2**30,
        },
        {
            "paths": ["/data"],
            "used_bytes": 1,
            "total_bytes": 2**53,
            "free_bytes": 500 * 2**30,
        },
    ]
    probe = _probe(system=_system(), disks=disks)
    text = _frame(_snapshot(probe=probe))
    assert re.search(r"/work\+/ +3\.0 GB free", text)
    assert re.search(r"/data +500\.0 GB free", text)
    assert "%" not in "".join(line for line in text.splitlines() if "free" in line)

    table = view.Table.grid()
    for _ in range(5):
        table.add_column()
    view._add_system_rows(table, {"disks": disks})
    styles = [cell.style for cell in table.columns[3].cells]
    assert styles == [view.ERROR, ""]


def test_no_disk_rows_from_an_older_probe():
    probe = _probe(system=_system())
    probe.pop("disks", None)
    assert "/data" not in _frame(_snapshot(probe=probe))


def test_network_rates_show_next_to_cpu_and_memory():
    system = _system(net_in_bps=2_621_440, net_out_bps=512)
    text = _frame(_snapshot(probe=_probe(system=system)))
    assert re.search(r"net +in 2\.5 MB/s +out 512 B/s", text)
    assert "net" not in _frame(_snapshot(probe=_probe(system=_system())))


def test_top_processes_show_cpu_memory_and_gpu_memory():
    processes = [
        {
            "pid": 1,
            "name": "serve.py",
            "cpu_pct": 212.4,
            "rss_bytes": 3 * 2**30,
            "gpu_mib": 20480,
        },
        {"pid": 2, "name": "[/x]", "cpu_pct": 0.0, "rss_bytes": 4096},
    ]
    text = _frame(_snapshot(probe=_probe(system=_system(processes=processes))))
    assert "Top processes" in text
    assert re.search(r"serve\.py +212% +3\.0 GB +20\.0 GB", text)
    assert re.search(r"\[/x\] +0% +4\.0 KB +-", text)
    assert "Top processes" not in _frame(_snapshot(probe=_probe(system=_system())))


def test_sparkline_scales_percent_to_eight_levels():
    from fal.dev.dash import view

    assert view.sparkline([0, 50, 100]).plain == "▁▅█"
    assert view.sparkline([200, -5]).plain == "█▁"
    # More readings than columns are averaged down, newest last.
    long = view.sparkline([0.0] * 50 + [100.0] * 50, width=10).plain
    assert long == "▁" * 5 + "█" * 5


def test_sparkline_falls_back_to_ascii(monkeypatch):
    from fal.console import icons
    from fal.dev.dash import view

    monkeypatch.setattr(icons, "default_console", _console_with_encoding("ascii"))
    assert view.sparkline([0, 100]).plain == "_#"


def test_history_keeps_five_minutes_and_skips_unknown_readings():
    probe = _probe(system=_system(cpu_pct=40.0))
    history = model.record_history({}, probe, NOW)
    assert history == {"gpu": [(NOW, 37)], "cpu": [(NOW, 40.0)]}
    later = model.record_history(history, probe, NOW + 301)
    assert later["gpu"] == [(NOW + 301, 37)]
    # Not mutated: the view reads the old dict while the poller builds the new.
    assert history["gpu"] == [(NOW, 37)]
    no_cpu = model.record_history({}, _probe(system=_system(cpu_pct=None)), NOW)
    assert "cpu" not in no_cpu


def test_history_uses_the_busiest_gpu():
    gpus = [{"util_pct": 5}, {"util_pct": 90}, {"util_pct": None}]
    assert model.record_history({}, _probe(gpus=gpus), NOW)["gpu"] == [(NOW, 90)]


def test_panel_draws_sparklines_once_there_is_a_trend():
    history = {"gpu": [(NOW - 60, 0), (NOW, 100)], "cpu": [(NOW, 50)]}
    text = _frame(_snapshot(probe=_probe(system=_system()), history=history))
    assert re.search(r"gpu +▁█ +1m 00s", text)
    # One reading is not a trend.
    assert "cpu ▅" not in text


def test_poller_records_history_and_forgets_it_with_the_runner():
    box = FakeDashBox(_probe(system=_system(cpu_pct=30.0)))
    poller = Poller(box)
    poller.poll_once()
    poller.poll_once()
    assert len(poller.snapshot.history["gpu"]) == 2
    assert len(poller.snapshot.history["cpu"]) == 2
    box.runner_id = "runner-new"
    poller.refresh_state()
    poller.poll_once()
    assert len(poller.snapshot.history["gpu"]) == 1
    box.code = "stopped"
    poller.refresh_state()
    poller.poll_once()
    assert poller.snapshot.history == {}


def test_estimated_cost_is_marked_and_needs_a_known_price(monkeypatch):
    hidden = _snapshot(cost_hidden=True)
    assert "est." not in _frame(hidden)  # no price list ships: no guess
    monkeypatch.setattr(model, "LIST_PRICES", {"GPU-H100": 5.0})
    text = _frame(hidden)
    assert "~$5.00 est. at $5.00/h" in text
    # Usage the platform reports wins over a guess.
    real = _snapshot(cost=Cost(spent=4.5, per_hour=4.5))
    assert "est." not in _frame(real)
    monkeypatch.setattr(model, "LIST_PRICES", {"GPU-A100": 5.0})
    assert "est." not in _frame(hidden)


def test_estimate_cost_is_uptime_at_the_list_price(monkeypatch):
    monkeypatch.setattr(model, "LIST_PRICES", {"GPU-H100": 4.0})
    assert model.estimate_cost("GPU-H100", 1800) == Cost(2.0, 4.0)
    assert model.estimate_cost("GPU-H100", None) is None
    assert model.estimate_cost(None, 60) is None


def test_hot_panel_shows_state_reload_infer_and_error():
    hot = {
        "last_reload": {"files": ["infer.py"], "ms": 9.4, "ok": False, "at": NOW - 30},
        "last_infer": {"at": NOW - 90, "ms": 571},
        "error": "NameError: pipe",
    }
    text = _frame(_snapshot(_probe(hot=hot)))
    assert "Hot reload" in text
    assert "infer.py in 9.4 ms, 30s ago  (import failed)" in text
    assert "571 ms, 1m 30s ago" in text
    assert "NameError: pipe" in text
    assert re.search(r"state +ready · code v3", text)
    # Said once, in its own panel, not again under Connections.
    assert text.count("code v3") == 1


def test_hot_panel_copes_with_a_reload_that_has_no_time():
    text = _frame(_snapshot())
    assert "infer.py in 9.4 ms" in text
    assert "ms," not in text.split("last reload")[1].splitlines()[0]


def test_no_hot_panel_while_the_hot_server_is_not_running():
    text = _frame(_snapshot(_probe(hot={"running": False})))
    assert "Hot reload" not in text
    assert re.search(r"hot reload +not running", text)


def test_footer_lists_the_open_key_even_without_a_runner():
    text = _frame(_snapshot(code="stopped"))
    assert "o open output" in text


def test_latest_output_is_the_newest_image_in_hot_out(monkeypatch, tmp_path):
    import os

    from fal.dev import state

    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    out = tmp_path / "proj" / "hot-out"
    out.mkdir(parents=True)
    for name, age in [("old.png", 300), ("new.jpg", 10), ("notes.txt", 0)]:
        (out / name).write_text("x")
        os.utime(out / name, (NOW - age, NOW - age))
    assert model.latest_output(FakeDashBox(), tmp_path / "proj") == out / "new.jpg"
    assert model.latest_output(FakeDashBox(), tmp_path) is None


def test_latest_output_also_looks_in_the_remembered_hot_project(monkeypatch, tmp_path):
    from fal.dev import state

    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    project = tmp_path / "project"
    (project / "hot-out").mkdir(parents=True)
    (project / "hot-out" / "a.png").write_text("x")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    state.update(elsewhere, None, hot_local=str(project))
    assert model.latest_output(FakeDashBox(), elsewhere) == project / "hot-out/a.png"


@pytest.mark.parametrize(
    "platform,command",
    [("darwin", "open"), ("linux", "xdg-open")],
)
def test_open_externally_uses_the_systems_viewer(monkeypatch, platform, command):
    started = []
    monkeypatch.setattr(model.sys, "platform", platform)
    monkeypatch.setattr(
        model.subprocess, "Popen", lambda args, **kwargs: started.append(args)
    )
    model.open_externally(Path("/tmp/a.png"))
    assert started == [[command, "/tmp/a.png"]]


def _drive_dashboard(monkeypatch, keys, **patches):
    """Run the dash loop with scripted keys; returns what the footer was told."""
    import contextlib

    from fal.dev.dash import tui

    said = []

    class _Live:
        def __init__(self, renderable, **kwargs):
            self.stops = 0

        def start(self):
            pass

        def stop(self):
            self.stops += 1

        def update(self, frame, refresh=False):
            pass

    script = iter(keys)
    monkeypatch.setattr(tui.os, "name", "posix")
    monkeypatch.setattr(tui.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(tui.sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(tui.sys.stdin, "fileno", lambda: 0, raising=False)
    monkeypatch.setattr(tui, "_cbreak", lambda fd: contextlib.nullcontext())
    monkeypatch.setattr(tui, "_restored", lambda fd: contextlib.nullcontext())
    monkeypatch.setattr(tui, "Live", _Live)
    monkeypatch.setattr(tui, "_read_key", lambda fd, timeout: next(script))
    monkeypatch.setattr(tui, "render", lambda *a, **k: "frame")
    monkeypatch.setattr(tui.model.Poller, "start", lambda self: None)
    monkeypatch.setattr(
        tui.model.Poller, "say", lambda self, m, seconds=10: said.append(m)
    )
    for name, value in patches.items():
        monkeypatch.setattr(tui, name, value)
    tui.run_dashboard(FakeDashBox())
    return said


def test_o_opens_the_latest_output_in_the_system_viewer(monkeypatch):
    opened = []
    monkeypatch.setattr(model, "latest_output", lambda box: Path("hot-out/a.png"))
    monkeypatch.setattr(model, "open_externally", opened.append)
    said = _drive_dashboard(
        monkeypatch, ["o", "q"], _show_output=lambda path, console: False
    )
    assert opened == [Path("hot-out/a.png")]
    assert said == ["opened a.png"]


def test_o_previews_inline_when_the_terminal_can_draw_images(monkeypatch):
    shown, opened = [], []
    monkeypatch.setattr(model, "latest_output", lambda box: Path("hot-out/a.png"))
    monkeypatch.setattr(model, "open_externally", opened.append)
    monkeypatch.setattr(Console, "input", lambda self, prompt="": "")

    def show(path, console):
        shown.append(path)
        return True

    _drive_dashboard(monkeypatch, ["o", "q"], _show_output=show)
    assert shown == [Path("hot-out/a.png")] and opened == []


def test_o_says_where_images_come_from_when_there_are_none(monkeypatch):
    monkeypatch.setattr(model, "latest_output", lambda box: None)
    said = _drive_dashboard(monkeypatch, ["o", "q"])
    assert said and "hot-out/" in said[0] and "fal dev hot run" in said[0]


def test_o_reports_a_viewer_that_fails(monkeypatch):
    monkeypatch.setattr(model, "latest_output", lambda box: Path("hot-out/a.png"))

    def fail(path):
        raise model.DevError("no viewer")

    monkeypatch.setattr(model, "open_externally", fail)
    said = _drive_dashboard(
        monkeypatch, ["o", "q"], _show_output=lambda path, console: False
    )
    assert said == ["open failed: no viewer"]
