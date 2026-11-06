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


class FakeBox:
    name = "troy"
    app = "devbox-troy"
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
    box = FakeBox(probe, code)
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
    return _text(render(snapshot, box or FakeBox(), now=NOW), **kwargs)


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
        "fal dev  troy",
        "● running",
        "devbox-troy",
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
    text = _frame(_snapshot(probe), FakeBox(probe))
    assert "IDLE SHUTDOWN IS OFF" in text
    assert re.search(r"auto-stop\s+off", text)


def test_render_hot_error_and_hold():
    probe = _probe(
        hot={"error": "NameError: name 'pipe' is not defined", "ready": True},
        activity={"held": True, "hold_until": NOW + 1800},
    )
    text = _frame(_snapshot(probe), FakeBox(probe))
    assert "NameError: name 'pipe' is not defined" in text
    assert "30m 00s left" in text


def test_render_stopped_box():
    text = _frame(_snapshot(code="stopped"), FakeBox(code="stopped"))
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
        render(_snapshot(probe), FakeBox(probe), now=NOW, screen=True), height=60
    )
    assert "[/x], [link=https://e.x]y[/link]" in console.export_text(styles=False)
    assert "\x1b]8" not in console.export_text(styles=True)


def test_footer_message_expires():
    snapshot = _snapshot()
    snapshot.message, snapshot.message_until = "press d again", NOW + 5
    assert "press d again" in _text(render(snapshot, FakeBox(), now=NOW))
    assert "press d again" not in _text(render(snapshot, FakeBox(), now=NOW + 6))


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
    console.print(render(_snapshot(), FakeBox(), now=NOW, screen=True), height=height)
    assert "quit" in console.export_text()


def test_poller_runs_one_exec_per_refresh():
    box = FakeBox()
    poller = Poller(box)
    poller.poll_once()
    poller.poll_once()
    assert len(box.run_calls) == 2
    command, until = box.run_calls[0]
    assert command[:2] == ["sh", "-c"]
    assert command[3] == f"{config.PROBE_DIR}/probe.py"
    assert json.loads(command[4]) == {
        "workspace": "/data/devboxes/troy",
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
    box = FakeBox()
    poller = Poller(box)
    poller.poll_once()
    box.probe = _probe(work={"dirty": None})
    poller.poll_once()
    assert poller.snapshot.probe["work"]["dirty"] is True


def test_poller_reports_probe_errors_without_crashing():
    box = FakeBox()
    box.run = lambda *a, **k: (1, "python3: not found")
    poller = Poller(box)
    poller.poll_once()
    assert "no status" in poller.snapshot.errors["probe"]
    assert "probe error" in _text(render(poller.snapshot, box, now=NOW))


def test_poller_skips_exec_when_box_is_stopped():
    box = FakeBox(code="stopped")
    poller = Poller(box)
    poller.poll_once()
    assert box.run_calls == []


def test_poller_drops_probe_when_box_stops():
    box = FakeBox()
    poller = Poller(box)
    poller.poll_once()
    box.code = "stopped"
    poller.refresh_state()
    poller.poll_once()
    assert poller.snapshot.probe is None
    assert poller.snapshot.logs == []


def test_probe_command_names_the_boxs_workspace():
    command = probe_command(FakeBox(), check_dirty=False)

    assert command[3] == "/usr/local/lib/devbox-probe/probe.py"
    assert json.loads(command[4])["workspace"] == "/data/devboxes/troy"


def test_probe_script_reads_box_state(tmp_path):
    """Run the real probe, installed like the box does, against a fake box layout."""
    workspace = tmp_path / "devboxes" / "troy"
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
    box = FakeBox()

    cost = fetch_cost(box, runner_uptime=7200)

    assert cost == Cost(spent=3.5, per_hour=pytest.approx(4.5))
    assert seen["url"] == f"{flags.PLATFORM_URL}/serverless/usage"
    assert seen["params"]["app"] == "devbox-troy"
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

    assert fetch_cost(FakeBox(), 60) == Cost(spent=0, per_hour=None)


@pytest.mark.parametrize("status", [401, 403])
def test_fetch_cost_is_none_without_billing_access(monkeypatch, status):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: httpx.Response(status, request=httpx.Request("GET", url)),
    )

    assert fetch_cost(FakeBox(), 60) is None


def test_refresh_cost_hides_the_line_without_access_and_survives_errors(monkeypatch):
    poller = Poller(FakeBox())
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


def test_a_failed_first_cost_lookup_shows_nothing_rather_than_stale():
    poller = Poller(FakeBox())
    poller.update(state=_state(), state_at=time.time())

    def fail(box, uptime):
        raise httpx.ConnectError("down")

    model.fetch_cost = fail
    poller.refresh_cost()

    assert poller.snapshot.cost is None
    assert "stale" not in _frame(poller.snapshot)
    assert "cost error" in _frame(poller.snapshot)


def test_cost_follows_the_runner_across_a_down_and_up(monkeypatch):
    box = FakeBox()
    poller = Poller(box)
    poller._cost_in_background = False
    monkeypatch.setattr(model, "fetch_cost", lambda box, uptime: Cost(9.0, 4.5))
    poller.poll_once()
    poller.refresh_cost()
    assert poller.snapshot.cost == Cost(9.0, 4.5)

    # Down: the cost of the old run goes with it.
    box.code = "stopped"
    poller.refresh_state()
    poller.poll_once()
    assert poller.snapshot.cost is None

    # Up again on another runner: an answer for the old runner is discarded.
    def answer_for_the_old_runner(box_, uptime):
        box.code = "running"
        box.runner_id = "runner-new"
        poller.refresh_state()
        poller.poll_once()
        return Cost(99.0, 4.5)

    box.code = "running"
    box.runner_id = "runner-old"
    poller.refresh_state()
    poller.poll_once()
    monkeypatch.setattr(model, "fetch_cost", answer_for_the_old_runner)
    poller.refresh_cost()

    assert poller.snapshot.runner_id == "runner-new"
    assert poller.snapshot.cost is None


def test_prime_fetches_the_cost_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        model, "fetch_cost", lambda box, uptime: calls.append(1) or Cost(1.0, 2.0)
    )
    poller = Poller(FakeBox())

    poller.prime()

    assert len(calls) == 1
    assert poller.snapshot.cost == Cost(1.0, 2.0)


def test_action_messages_cannot_drive_the_terminal():
    poller = Poller(FakeBox())

    poller.say("failed: \x1b]0;pwn\x07boom")

    assert poller.snapshot.message == "failed: ]0;pwnboom"
    assert "\x1b" not in model._short(RuntimeError("a\x1b[2Jb"))


def test_a_box_without_the_probe_answers_at_once_and_says_to_redeploy(tmp_path):
    command = probe_command(FakeBox(), check_dirty=False)
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
