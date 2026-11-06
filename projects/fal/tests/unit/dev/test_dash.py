import json
import re
import subprocess
import sys
from pathlib import Path

import grpc
import pytest
from rich.console import Console

from fal.dev import dash
from fal.dev.dash import (
    END_MARKER,
    PROBE_SOURCE,
    BoxInfo,
    Poller,
    Snapshot,
    box_state,
    idle_view,
    parse_probe,
    price_per_hour,
)
from fal.dev.dash_view import render, render_screen

FIXTURES = Path(__file__).with_name("fixtures")
NOW = 1762380000.0


def _probe(**changes) -> dict:
    probe = json.loads((FIXTURES / "probe_running.json").read_text())
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(probe.get(key), dict):
            probe[key] = {**probe[key], **value}
        else:
            probe[key] = value
    return probe


class FakeBox:
    def __init__(self, probe=None, state="running", idle_minutes=60):
        self.name = "troy"
        self.idle_minutes = idle_minutes
        self.claude_enabled = True
        self.probe_config = {"workspace": "/data/devboxes/troy"}
        self.probe = probe or _probe()
        self.state = state
        self.exec_calls = []
        self.log_calls = 0

    def info(self):
        return BoxInfo(
            name="troy",
            app_name="devbox-troy",
            team=None,
            machine_type="GPU-H100",
            state=self.state,
            runner_id="runner-abc" if self.state != "stopped" else None,
            runner_state="running",
            uptime_seconds=3600,
            min_concurrency=1,
            fetched_at=NOW,
        )

    def exec(self, runner_id, command, timeout, probe=False):
        self.exec_calls.append(command)
        return 0, b"notice from the shell\n" + json.dumps(
            self.probe
        ).encode() + b"\n" + END_MARKER.encode() + b"\n"

    def logs(self, runner_id, lines):
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


def _snapshot(probe=None, state="running") -> Snapshot:
    box = FakeBox(probe, state)
    return Snapshot(
        info=box.info(),
        probe=box.probe if state == "running" else None,
        probe_at=NOW,
        probe_ms=420,
    )


def _text(renderable, width=150) -> str:
    console = Console(record=True, width=width, force_terminal=False, color_system=None)
    console.print(renderable)
    return console.export_text()


def test_parse_probe_skips_noise_and_stops_at_marker():
    out = b'banner\n{"now": 1}\n' + END_MARKER.encode() + b'\n{"late": true}\n'
    assert parse_probe(out) == {"now": 1}
    with pytest.raises(ValueError):
        parse_probe(b"no json here")


@pytest.mark.parametrize(
    "runners,min_concurrency,expected",
    [
        ([], 0, "stopped"),
        ([{"runner_id": "a", "state": "dead"}], 0, "stopped"),
        ([{"runner_id": "a", "state": "running", "uptime_seconds": 5}], 1, "running"),
        ([{"runner_id": "a", "state": "idle", "uptime_seconds": 5}], 0, "stopping"),
        ([{"runner_id": "a", "state": "docker_pull"}], 1, "starting"),
        ([{"runner_id": "a", "state": "draining"}], 0, "stopping"),
        ([{"runner_id": "a", "state": "crash_backoff"}], 1, "failed"),
    ],
)
def test_box_state(runners, min_concurrency, expected):
    assert box_state(runners, min_concurrency)[0] == expected


def test_box_state_picks_newest_live_runner():
    runners = [
        {"runner_id": "old", "state": "running", "uptime_seconds": 900},
        {"runner_id": "new", "state": "running", "uptime_seconds": 30},
    ]
    assert box_state(runners, 1)[1]["runner_id"] == "new"


def test_price_per_hour():
    assert price_per_hour("GPU-H100", 2) == 9.0
    assert price_per_hour("GPU-UNKNOWN", 1) is None


def test_idle_view_estimates_from_box_signals():
    view = idle_view(
        _probe(), 60, runner_started=NOW - 3600, claude_counts=True, now=NOW
    )
    assert view.watchdog == "on"
    assert view.source == "estimated"
    # Newest signal is a hot request 2 minutes ago (terminal input was 5).
    assert view.last_active == NOW - 120
    assert view.remaining_seconds == 58 * 60
    assert view.reasons == ["hot request"]


def test_idle_view_active_now_resets_the_clock():
    probe = _probe(activity={"ssh_sessions": 2, "gpu_busy": True})
    view = idle_view(probe, 60, NOW - 3600, False, NOW)
    assert view.remaining_seconds == 3600
    assert view.active_now == ["2 SSH session(s)", "GPU busy"]


def test_idle_view_prefers_the_watchdogs_published_state():
    probe = _probe(
        idle_state={
            "last_active": NOW - 600,
            "idle_minutes": 30,
            "reasons": ["hot request"],
        }
    )
    view = idle_view(probe, 60, NOW - 3600, False, NOW)
    assert view.source == "watchdog"
    assert view.remaining_seconds == 20 * 60
    assert view.reasons == ["hot request"]


def test_idle_view_detects_watchdog_off():
    probe = _probe(
        admin_key_set=False,
        warnings=[
            "DEVBOX_ADMIN_KEY is not set, so idle shutdown is off. "
            "Run `devbox down` when done."
        ],
    )
    assert idle_view(probe, 60, None, False, NOW).watchdog == "off"


def test_render_running_box():
    text = _text(render(_snapshot(), FakeBox(), now=NOW))
    for expected in [
        "fal dev  troy",
        "RUNNING",
        "devbox-troy",
        "1x GPU-H100",
        "runner-abc",
        "1h 00m",
        "~$4.50",
        "37%",
        "21.0/80 GiB",
        "41°C",
        "in 58m 00s",
        "estimate",
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


def test_render_warns_loudly_when_watchdog_is_off():
    probe = _probe(
        admin_key_set=False,
        warnings=["DEVBOX_ADMIN_KEY is not set, so idle shutdown is off."],
    )
    text = _text(render(_snapshot(probe), FakeBox(probe), now=NOW))
    assert "IDLE SHUTDOWN IS OFF" in text
    assert re.search(r"auto-stop\s+OFF", text)


def test_render_hot_error_and_hold():
    probe = _probe(
        hot={"error": "NameError: name 'pipe' is not defined", "ready": True},
        activity={"held": True, "hold_until": NOW + 1800},
    )
    text = _text(render(_snapshot(probe), FakeBox(probe), now=NOW))
    assert "NameError: name 'pipe' is not defined" in text
    assert "30m 00s left" in text


def test_render_stopped_box():
    text = _text(render(_snapshot(state="stopped"), FakeBox(state="stopped"), now=NOW))
    assert "STOPPED" in text
    assert "fal dev up" in text


def test_poller_runs_one_exec_per_refresh():
    box = FakeBox()
    poller = Poller(box)
    poller.poll_once()
    poller.poll_once()
    assert len(box.exec_calls) == 2
    command = box.exec_calls[0]
    assert command[:3] == ["python3", "-c", PROBE_SOURCE]
    assert json.loads(command[3])["check_dirty"] is True
    assert json.loads(box.exec_calls[1][3])["check_dirty"] is False
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
    box.exec = lambda *a, **k: (1, b"python3: not found")  # type: ignore
    poller = Poller(box)
    poller.poll_once()
    assert "no status" in poller.snapshot.errors["probe"]
    text = _text(render(poller.snapshot, box, now=NOW))
    assert "probe error" in text


def test_poller_skips_exec_when_box_is_stopped():
    box = FakeBox(state="stopped")
    poller = Poller(box)
    poller.poll_once()
    assert box.exec_calls == []


def test_ready_settings():
    message = (
        "[devbox] ready: workspace=/data/devboxes/troy idle_minutes=45 claude=True"
    )
    assert dash._ready_settings(message) == {
        "workspace": "/data/devboxes/troy",
        "idle_minutes": "45",
        "claude": "True",
    }


def test_probe_script_reads_box_state(tmp_path):
    """Run the real probe against a fake box layout (no GPU, no tmux sessions)."""
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
    warning.write_text(
        "devbox: DEVBOX_ADMIN_KEY is not set, so idle shutdown is off.\n"
    )
    config = {
        "workspace": str(workspace),
        "local_work": str(work),
        "save_stamp": str(stamp),
        "hold_file": str(hold),
        "warning_file": str(warning),
        "idle_state_file": str(tmp_path / "missing.json"),
        "admin_key_env": "FAL_DEV_DASH_TEST_UNSET_KEY",
        "claude_tmux": "fal-dev-dash-test-no-such-session",
        "hot_tmux": "fal-dev-dash-test-no-such-session",
        "hot_log": str(tmp_path / "hot.log"),
        "check_dirty": True,
    }
    out = subprocess.run(
        [sys.executable, "-c", PROBE_SOURCE, json.dumps(config)],
        capture_output=True,
        timeout=60,
        check=True,
    ).stdout
    probe = parse_probe(out)
    assert "probe_error" not in probe
    assert probe["activity"]["held"] is True
    assert probe["activity"]["last_claude_activity"]
    assert probe["admin_key_set"] is False
    assert probe["warnings"] == [
        "DEVBOX_ADMIN_KEY is not set, so idle shutdown is off."
    ]
    assert probe["claude"] == {"running": False}
    assert probe["hot"] == {"running": False}
    assert probe["work"]["snapshot"]["size"] == 2048
    assert probe["work"]["lease"]["owner"] == "someone-else"
    assert probe["work"]["lease"]["mine"] is False
    assert probe["work"]["dirty"] is True
    assert idle_view(probe, 60, None, False, probe["now"]).watchdog == "off"


def test_parse_probe_raises_on_probe_error():
    with pytest.raises(ValueError, match="probe failed"):
        parse_probe(b'{"probe_error": "boom"}\n' + END_MARKER.encode())


def test_idle_view_counts_an_expired_hold():
    probe = _probe(activity={"hold_until": NOW - 60, "last_tty_input": None})
    view = idle_view(probe, 60, NOW - 3600, False, NOW)
    assert view.reasons == ["hold ended"]
    assert view.remaining_seconds == 59 * 60


def test_render_hides_box_text_control_characters():
    probe = _probe(hot={"error": "bad\x1b]0;title\x07 thing"})
    snapshot = _snapshot(probe)
    snapshot.logs = [{"timestamp": "x", "level": "INFO", "message": "a\x1b[2Jb"}]
    console = Console(record=True, width=150, force_terminal=False, color_system=None)
    console.print(render(snapshot, FakeBox(probe), now=NOW))
    text = console.export_text()
    assert "\x1b" not in text
    assert "bad]0;title thing" in text
    assert "a[2Jb" in text


def test_no_watchdog_banner_once_stopped():
    probe = _probe(warnings=["DEVBOX_ADMIN_KEY is not set, so idle shutdown is off."])
    snapshot = _snapshot(probe)
    snapshot.info.state = "stopped"
    assert "IDLE SHUTDOWN IS OFF" not in _text(
        render(snapshot, FakeBox(probe), now=NOW)
    )


def test_footer_message_expires():
    snapshot = _snapshot()
    snapshot.message, snapshot.message_until = "press d again", NOW + 5
    assert "press d again" in _text(render(snapshot, FakeBox(), now=NOW))
    assert "press d again" not in _text(render(snapshot, FakeBox(), now=NOW + 6))


def test_screen_layout_keeps_footer_on_short_terminals():
    console = Console(
        record=True, width=80, height=20, force_terminal=False, color_system=None
    )
    console.print(render_screen(_snapshot(), FakeBox(), now=NOW), height=20)
    assert "quit" in console.export_text()


def test_poller_drops_probe_when_box_stops():
    box = FakeBox()
    poller = Poller(box)
    poller.poll_once()
    box.state = "stopped"
    poller.refresh_info()
    poller.poll_once()
    assert poller.snapshot.probe is None
    assert poller.snapshot.logs == []


def _fal_box(monkeypatch, exec_result):
    box = dash.FalBox("troy", "devbox-troy")
    scaled = []
    monkeypatch.setattr(box, "info", lambda: FakeBox().info())
    monkeypatch.setattr(box, "exec", lambda *a, **k: exec_result)
    monkeypatch.setattr(box, "_scale", scaled.append)
    return box, scaled


@pytest.mark.parametrize("exec_result", [(None, b""), (1, b"disk full")])
def test_down_keeps_the_box_when_the_save_is_not_confirmed(monkeypatch, exec_result):
    box, scaled = _fal_box(monkeypatch, exec_result)
    assert box.down().startswith("save failed")
    assert scaled == []


def test_hold_quotes_the_hold_file(monkeypatch):
    box, _ = _fal_box(monkeypatch, (0, b""))
    commands = []
    monkeypatch.setattr(
        box, "exec", lambda r, c, timeout: commands.append(c) or (0, b"")
    )
    box.probe_config["hold_file"] = "/tmp/a b;rm -rf x"
    assert box.hold("runner-abc", 1).startswith("idle shutdown held")
    assert "'/tmp/a b;rm -rf x'" in commands[0][2]


def test_poller_remembers_activity_it_saw():
    box = FakeBox(_probe(activity={"ssh_sessions": 1}))
    poller = Poller(box)
    poller.poll_once()
    assert poller.snapshot.probe["seen_active"] == NOW
    box.probe = _probe(now=NOW + 600, activity={"ssh_sessions": 0})
    poller.poll_once()
    probe = poller.snapshot.probe
    assert probe["seen_active"] == NOW
    view = idle_view(probe, 60, NOW - 3600, False, NOW + 600)
    assert view.reasons == ["activity seen"]
    assert view.remaining_seconds == 50 * 60


def test_probe_text_cannot_inject_markup_or_escapes():
    out = json.dumps(
        _probe(
            hot={
                "last_reload": {
                    "files": ["[/x]", "[link=https://e.x]y[/link]"],
                    "ms": 1,
                }
            },
            gpus=[{"index": 0, "name": "g", "temp_c": "\x1b]0;pwn\x07x"}],
        )
    ).encode()
    probe = parse_probe(out)
    assert probe["gpus"][0]["temp_c"] == "]0;pwnx"
    snapshot = _snapshot(probe)
    console = Console(record=True, width=150, force_terminal=True)
    console.print(render_screen(snapshot, FakeBox(probe), now=NOW), height=60)
    text = console.export_text(styles=False)
    assert "[/x], [link=https://e.x]y[/link]" in text
    assert "\x1b]8" not in console.export_text(styles=True)


def test_hold_refuses_when_the_runner_is_gone(monkeypatch):
    box, scaled = _fal_box(monkeypatch, (0, b""))
    assert box.hold("some-old-runner", 1).startswith("the box is no longer running")
    assert scaled == []


def test_panels_stack_on_narrow_terminals():
    snapshot = _snapshot()
    narrow = _text(render(snapshot, FakeBox(), now=NOW), width=80)
    # Stacked, the GPU row keeps its percent, memory and power.
    assert "37%" in narrow
    assert "212W" in narrow


def test_malformed_published_idle_state_falls_back_to_estimate():
    probe = _probe(idle_state={"last_active": "soon"})
    assert idle_view(probe, 60, NOW - 3600, False, NOW).source == "estimated"


def test_private_sdk_hooks_still_exist():
    """FalBox leans on these internals; fail loudly here if they move."""
    import inspect

    from fal.api.client import SyncServerlessClient
    from fal.cli import runners

    assert list(inspect.signature(runners._post_history).parameters) == [
        "client",
        "base_params",
        "since",
        "until",
        "page_size",
    ]
    assert hasattr(SyncServerlessClient, "_create_host")
    assert hasattr(SyncServerlessClient, "_create_rest_client")


def test_hold_never_shortens_a_longer_hold(monkeypatch, tmp_path):
    box, _ = _fal_box(monkeypatch, (0, b""))
    hold_file = tmp_path / "hold until"
    box.probe_config["hold_file"] = str(hold_file)

    def run_locally(runner_id, command, timeout):
        result = subprocess.run(command, capture_output=True, check=False)
        return result.returncode, result.stdout

    monkeypatch.setattr(box, "exec", run_locally)
    assert box.hold("runner-abc", 1) == "idle shutdown held for 1h"
    one_hour = int(hold_file.read_text())
    hold_file.write_text(f"{one_hour + 7 * 3600}\n")
    assert box.hold("runner-abc", 1).startswith("a longer hold is already in place")
    assert int(hold_file.read_text()) == one_hour + 7 * 3600


class _Unauthenticated(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAUTHENTICATED


class _Message:
    def __init__(self, data=b"", exit_code=None):
        self.data = data
        self.exit_code = exit_code
        self.close = False
        self.stream = 1

    def HasField(self, name):
        return name == "exit_code" and self.exit_code is not None


class _Call:
    def __init__(self, messages):
        self.messages = messages
        self.cancelled = False

    def __iter__(self):
        return iter(self.messages)

    def cancel(self):
        self.cancelled = True


def test_exec_rebuilds_the_channel_once_on_expired_token(monkeypatch):
    box = dash.FalBox("troy", "devbox-troy")
    stubs = []

    class Stub:
        def __init__(self, fail):
            self.fail = fail

        def ShellRunner(self, inputs, timeout):
            list(inputs)
            if self.fail:
                raise _Unauthenticated()
            body = b'{"now": 1}\n' + END_MARKER.encode() + b"\n"
            # The exit code arrives before the probe's output ends.
            return _Call([_Message(exit_code=0), _Message(body), _Message(b"late")])

    def new_stub():
        stubs.append(Stub(fail=not stubs))
        return stubs[-1]

    monkeypatch.setattr(box, "_shell_stub", lambda: box._stub or new_stub())
    code, out = box.exec("runner-abc", ["python3"], timeout=5, probe=True)
    assert len(stubs) == 2
    assert code == 0
    assert parse_probe(out) == {"now": 1}
    assert b"late" not in out


@pytest.mark.parametrize("width,height", [(80, 24), (60, 24)])
def test_screen_says_when_content_is_cut(width, height):
    console = Console(
        record=True, width=width, height=height, force_terminal=False, color_system=None
    )
    console.print(render_screen(_snapshot(), FakeBox(), now=NOW), height=height)
    text = console.export_text()
    assert "lines hidden; enlarge the terminal" in text
    assert "quit" in text or "q  …" in text


def test_screen_shows_everything_when_it_fits():
    console = Console(
        record=True, width=120, height=40, force_terminal=False, color_system=None
    )
    console.print(render_screen(_snapshot(), FakeBox(), now=NOW), height=40)
    text = console.export_text()
    assert "hidden" not in text
    assert "/work snapshot" in text
    assert "Logs" in text
