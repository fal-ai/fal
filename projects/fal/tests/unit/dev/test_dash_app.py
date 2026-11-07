import json
import os

import pytest
from rich.console import Console

from dev.app.fakes import make_session
from fal.dev import config
from fal.dev.dash import app as app_dash
from fal.dev.dash import model, view
from fal.dev.dash.model import Snapshot

from .test_dash import NOW, _state

pytestmark = pytest.mark.usefixtures("dev_account")


def _probe(**changes) -> dict:
    probe = {
        "now": NOW,
        "gpus": [
            {
                "index": 0,
                "name": "NVIDIA RTX PRO 6000",
                "util_pct": 87,
                "mem_used_mib": 90000,
                "mem_total_mib": 97000,
                "temp_c": 60,
                "power_w": 400,
                "power_limit_w": 600,
            }
        ],
        "gpu_error": None,
        "activity": {"last_request": NOW - 60, "inflight": 0, "keep_alive": 900},
        "process": {"pid": 12, "started": NOW - 300},
        "app": {
            "ready": True,
            "pid": 12,
            "code_version": 2,
            "setup_seconds": 133.2,
            "last_reload": {"action": "reload", "files": ["app.py"], "ms": 640},
            "error": None,
            "endpoints": ["/v1/chat/completions"],
        },
        "engine": {
            "engine": "vllm",
            "model": "openai/gpt-oss-120b",
            "stats": {
                "vllm:num_requests_running": 1,
                "vllm:num_requests_waiting": 0,
                "vllm:kv_cache_usage_perc": 0.05,
                "vllm:prefix_cache_queries_total": 1000,
                "vllm:prefix_cache_hits_total": 930,
            },
            "usage_totals": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "requests": 4,
            },
        },
    }
    probe.update(changes)
    return probe


def _frame(probe, code="running") -> str:
    box = app_dash.AppBox(make_session())
    snapshot = Snapshot(
        state=_state(code), state_at=NOW, probe=probe, probe_at=NOW, probe_ms=400
    )
    console = Console(record=True, width=150, force_terminal=False, color_system=None)
    console.print(app_dash.render(snapshot, box, NOW))
    return console.export_text()


def test_render_shows_keep_alive_app_and_engine():
    frame = _frame(_probe(rates={"in_per_s": 2000.0, "out_per_s": 140.0}))
    assert "scale to 0   in 14m 00s" in frame
    assert "ready" in frame and "code v2, pid 12" in frame
    assert "reload: app.py in 640 ms" in frame
    assert "1 running, 0 waiting, 4 done" in frame
    assert "prefix hits 93%" in frame
    assert "140 out, 2000 in" in frame
    assert "r restart app" in frame


def test_render_shows_setup_running_and_a_busy_box():
    probe = _probe(
        app=None,
        engine=None,
        activity={"last_request": NOW, "inflight": 2, "keep_alive": 900},
    )
    frame = _frame(probe)
    assert "running setup() for 5m 00s" in frame
    assert "busy, 2 request(s) in flight" in frame
    assert "Engine" not in frame


def test_probe_command_ships_the_probe_and_parse_reads_it():
    command = app_dash.probe_command(make_session())
    assert command[:2] == ["python3", "-c"]
    settings = json.loads(command[-1])
    assert settings["session_dir"] == "/tmp/fal-dev-app/dev-demo-me"
    output = f"noise\n{json.dumps(_probe())}\n{config.PROBE_END}\n"
    assert app_dash.parse(output)["app"]["pid"] == 12
    with pytest.raises(ValueError, match="probe failed"):
        app_dash.parse(json.dumps({"probe_error": "boom"}))


def test_token_rates_follow_the_engine_totals():
    poller = app_dash.AppPoller(app_dash.AppBox(make_session()))
    assert poller._rates(_probe()) is None
    later = _probe(now=NOW + 10)
    later["engine"]["usage_totals"] = {
        "prompt_tokens": 21000,
        "completion_tokens": 1500,
    }
    assert poller._rates(later) == {"in_per_s": 2000.0, "out_per_s": 140.0}
    # A restarted app starts its totals again.
    reset = _probe(now=NOW + 13)
    reset["engine"]["usage_totals"] = {"prompt_tokens": 5, "completion_tokens": 1}
    assert poller._rates(reset) is None


def test_logs_leave_out_the_clis_own_polling(monkeypatch):
    lines = [
        {"message": 'INFO: 1.2.3.4:1 - "POST /__fal_dev/status HTTP/1.1" 200 OK'},
        {"message": 'INFO: 1.2.3.4:1 - "POST /__fal_dev/proxy HTTP/1.1" 200 OK'},
        {"message": "[vllm] ready"},
    ]
    monkeypatch.setattr(app_dash.Box, "logs", lambda self, runner, count, since: lines)
    kept = app_dash.AppBox(make_session()).logs("r", 30, None)
    assert [line["message"] for line in kept] == [lines[1]["message"], "[vllm] ready"]


def test_app_box_is_running_while_it_has_a_runner():
    box = app_dash.AppBox(make_session())
    assert box.app == "dev-demo-me"
    assert box._live_code(None) == "running"


def test_the_probe_finds_the_app_process_not_itself(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(app_dash.REMOTE))
    import app_probe

    def proc(pid: int, *argv: str) -> None:
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "cmdline").write_bytes(
            b"\0".join(arg.encode() for arg in argv) + b"\0"
        )

    marker = "fal.dev.remote.child"
    # The probe carries the marker in its settings.
    proc(7, "python3", "-c", "...", json.dumps({"child_marker": marker}))
    assert app_probe._app_process(marker, str(tmp_path)) is None
    proc(12, "/usr/bin/python", "-m", marker, "--port", "8181")
    assert app_probe._app_process(marker, str(tmp_path))["pid"] == 12


def _screen_frame(height: int, logs: int = 80) -> str:
    box = app_dash.AppBox(make_session())
    snapshot = Snapshot(
        state=_state("running"),
        state_at=NOW,
        probe=_probe(),
        probe_at=NOW,
        probe_ms=400,
        logs=[
            {"timestamp": "", "level": "INFO", "message": f"line {i:03d}"}
            for i in range(logs)
        ],
    )
    console = Console(
        record=True, width=150, height=height, force_terminal=False, color_system=None
    )
    console.print(app_dash.render(snapshot, box, NOW, screen=True))
    return console.export_text()


def test_logs_panel_fills_the_screen_and_grows_with_it():
    short, tall = _screen_frame(50), _screen_frame(80)
    assert short.count("line 0") < tall.count("line 0")
    assert "line 079" in tall and "line 079" in short
    # The panel ends on the row above the footer: no gap, nothing pushed off.
    rows = tall.splitlines()
    assert rows[-4].startswith("╰") and "r restart app" in rows[-3]


def test_logs_without_a_screen_stay_at_the_fixed_tail():
    box = app_dash.AppBox(make_session())
    snapshot = Snapshot(
        state=_state("running"),
        state_at=NOW,
        probe=_probe(),
        probe_at=NOW,
        logs=[{"level": "INFO", "message": f"line {i:03d}"} for i in range(80)],
    )
    console = Console(record=True, width=150, force_terminal=False, color_system=None)
    console.print(app_dash.render(snapshot, box, NOW))
    assert console.export_text().count("line 0") == view.LOGS_SHOWN


def test_fetch_count_follows_the_terminal_height(monkeypatch):
    monkeypatch.delenv("LINES", raising=False)
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda *a, **k: os.terminal_size((150, 80))
    )
    assert model.log_rows() == 80
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda *a, **k: os.terminal_size((150, 20))
    )
    assert model.log_rows() == model.LOG_LINES
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda *a, **k: os.terminal_size((150, 500))
    )
    assert model.log_rows() == model.LOG_ROWS_MAX
