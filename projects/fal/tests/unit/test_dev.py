import io
import json
import runpy
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from fal.dev import box as box_module
from fal.dev import config, hot, state
from fal.dev.box import (
    BOX_APP,
    Box,
    DevError,
    _HoldUntilMarker,
    fal_spec,
    newest_live,
    ssh_config,
)
from fal.sdk import RunnerState


def _runner(runner_id, state=RunnerState.RUNNING, uptime=60, machine="GPU-H100"):
    runner = MagicMock()
    runner.runner_id = runner_id
    runner.state = state
    runner.uptime = timedelta(seconds=uptime)
    runner.machine_type = machine
    return runner


def _box(runners=(), min_concurrency=1, deployed=True):
    box = Box("klein")
    box.client = MagicMock()
    box.client.apps.runners.return_value = list(runners)
    app = MagicMock(alias="devbox-klein", min_concurrency=min_concurrency)
    app.machine_types = ["GPU-H100"]
    box.client.apps.list.return_value = [app] if deployed else []
    return box


# --- state ---


def test_state_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")

    state.update(tmp_path, "host", "team", name="klein", hot_dir="/work/k")
    state.update(tmp_path, "host", "team", hot_dir=None)

    data = state.load(tmp_path, "host", "team")
    assert data["name"] == "klein"
    assert "hot_dir" not in data
    assert state.load(tmp_path, "host", "other-team") == {}


def test_project_key_depends_on_project_and_connection(tmp_path):
    key = state.project_key(tmp_path, "host", "team")

    assert key == state.project_key(tmp_path, "host", "team")
    assert key != state.project_key(tmp_path, "other-host", "team")
    assert key != state.project_key(tmp_path, "host", "other-team")
    assert key != state.project_key(tmp_path / "sub", "host", "team")


def test_state_ignores_corrupt_file(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    path = state._path(tmp_path, None, None)
    path.parent.mkdir(parents=True)
    path.write_text("not json")

    assert state.load(tmp_path, None, None) == {}


# --- box ---


@pytest.mark.parametrize(
    "version, expected",
    [
        ("1.84.0", "fal==1.84.0"),
        ("1.84.1.dev3+g1234abc", "fal"),
        ("1.85.0rc1", "fal"),
        ("not-a-version", "fal"),
    ],
)
def test_fal_spec(version, expected):
    assert fal_spec(version) == expected


def test_newest_live_skips_stopped_runners():
    runners = [
        _runner("old", uptime=500),
        _runner("dead", RunnerState.DEAD, uptime=1),
        _runner("new", RunnerState.IDLE, uptime=10),
    ]

    assert [r.runner_id for r in newest_live(runners)] == ["new", "old"]


def test_require_runner():
    assert _box([_runner("r1")]).require_runner() == "r1"
    with pytest.raises(DevError, match="not running"):
        _box([]).require_runner()
    with pytest.raises(DevError, match="shutting down"):
        _box([_runner("r1")], min_concurrency=0).require_runner()


@pytest.mark.parametrize(
    "runners, min_concurrency, deployed, expected",
    [
        ([_runner("r1")], 1, True, "devbox-klein: running on GPU-H100"),
        ([_runner("r1")], 0, True, "devbox-klein: shutting down"),
        ([_runner("r1", RunnerState.DRAINING)], 0, True, "devbox-klein: stopping"),
        ([_runner("r1", RunnerState.PENDING)], 1, True, "devbox-klein: starting"),
        ([], 0, True, "devbox-klein: stopped"),
        ([], 0, False, "devbox-klein: not deployed"),
    ],
)
def test_status(runners, min_concurrency, deployed, expected):
    lines = _box(runners, min_concurrency, deployed).status()

    assert lines[0].startswith(expected)


def test_stop_scales_to_zero_and_stops_live_runners():
    box = _box([_runner("r1"), _runner("gone", RunnerState.TERMINATED)])
    box.client.apps.runners.side_effect = [
        [_runner("r1"), _runner("gone", RunnerState.TERMINATED)],
        [_runner("r1", RunnerState.TERMINATED)],
    ]

    assert box.stop()
    box.client.apps.scale.assert_called_once_with(
        "devbox-klein", min_concurrency=0, machine_types=None
    )
    box.client.runners.stop.assert_called_once_with("r1")


def test_wait_until_ready_raises_on_crash(monkeypatch):
    monkeypatch.setattr(box_module, "POLL_SECONDS", 0)
    box = _box([_runner("r1", RunnerState.CRASH_BACKOFF)])

    with pytest.raises(DevError, match="failed to start"):
        box.wait_until_ready(timeout=5)


def test_start_cancels_when_no_runner_comes_up(monkeypatch):
    monkeypatch.setattr(box_module, "POLL_SECONDS", 0)
    box = _box([])
    box.wait_until_ready = MagicMock(return_value=None)
    box.stop = MagicMock(return_value=True)

    with pytest.raises(DevError, match="cancelled"):
        box.start(timeout=1)
    box.stop.assert_called_once()


def test_save_work_raises_and_keeps_box_up_on_failure():
    box = _box([_runner("r1")])
    box.run = MagicMock(return_value=(1, "disk full"))

    with pytest.raises(DevError, match="disk full"):
        box.save_work()
    box.run.assert_called_once_with("r1", [config.SAVE_COMMAND])


def test_hold_rejects_out_of_range_hours():
    with pytest.raises(DevError, match="at most"):
        _box([_runner("r1")]).hold(config.MAX_HOLD_HOURS + 1)


def test_deploy_stops_a_running_box_first_and_restarts_it(monkeypatch):
    box = _box([_runner("r1")])
    box.save_work = MagicMock()
    box.stop = MagicMock(return_value=True)

    message = box.deploy(idle_minutes=30)

    box.save_work.assert_called_once()
    box.stop.assert_called_once()
    (ref,), kwargs = box.client.deploy.call_args
    assert ref == (str(BOX_APP), "DevBox")
    assert kwargs == {
        "app_name": "devbox-klein",
        "auth": "private",
        "reset_scale": True,
    }
    box.client.apps.scale.assert_called_once_with(
        "devbox-klein", min_concurrency=1, machine_types=None
    )
    assert "restarting on the new image" in message


def test_hold_until_marker_drops_setup_output():
    out = io.BytesIO()
    stream = _HoldUntilMarker(out)

    stream.write(b"noise\n__devbox_sshd")
    assert not stream.ready
    stream.write(b"_ready__\nSSH-2.0-OpenSSH\r\n")
    stream.write(b"more")

    assert stream.ready
    assert out.getvalue() == b"SSH-2.0-OpenSSH\r\nmore"


def test_ssh_config_entry():
    entry = ssh_config("klein", "/path to/fal", team="my-team")

    assert "Host devbox-klein" in entry
    assert (
        "ProxyCommand '/path to/fal' dev ssh-proxy --name klein --team my-team" in entry
    )


# --- hot ---


def test_parse_params():
    params = hot.parse_params(["steps=4", "name=fox", "size=[1, 2]"], "a fox")

    assert params == {"steps": 4, "name": "fox", "size": [1, 2], "prompt": "a fox"}
    with pytest.raises(DevError):
        hot.parse_params(["steps"], None)


def test_remote_dir(tmp_path):
    assert hot.remote_dir(None, "/work/x") == "/work/x"
    assert hot.remote_dir(str(tmp_path / "klein"), None) == "/work/klein"
    with pytest.raises(DevError):
        hot.remote_dir(None, None)


def test_scan_skips_hidden_venv_and_large_files(tmp_path):
    (tmp_path / "infer.py").write_text("x")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "util.py").write_text("x")
    for skipped in (".venv", "hot-out", ".git"):
        (tmp_path / skipped).mkdir()
        (tmp_path / skipped / "f.py").write_text("x")
    (tmp_path / ".env").write_text("x")
    (tmp_path / "big.bin").write_bytes(b"x" * (hot.MAX_FILE_BYTES + 1))

    assert sorted(hot.scan(tmp_path)) == ["infer.py", "pkg/util.py"]


def test_read_token_uses_cache_for_same_runner(monkeypatch, tmp_path):
    monkeypatch.setattr(hot, "STATE_DIR", tmp_path)
    box = _box([_runner("r1")])
    token = "a" * 43
    box.run = MagicMock(return_value=(0, f"{token}\n"))

    assert hot.read_token(box, "r1") == token
    assert hot.read_token(box, "r1") == token
    box.run.assert_called_once()

    box.run.return_value = (0, "cat: no such file\n")
    with pytest.raises(DevError, match="redeploy"):
        hot.read_token(box, "r2")


# --- what runs on the box ---


def test_box_app_loads_like_a_deploy(monkeypatch):
    """The deploy runs box_app.py as a file; its settings come from the environment."""
    monkeypatch.setenv(config.ENV_NAME, "klein")
    monkeypatch.setenv(config.ENV_MACHINE_TYPE, "GPU-B200")
    monkeypatch.setenv(config.ENV_PYTHON_VERSION, "3.12")
    monkeypatch.setenv(config.ENV_FAL_SPEC, "fal==1.84.0")
    module = runpy.run_path(str(BOX_APP))

    app = module["DevBox"]
    assert module["APP_NAME"] == "devbox-klein"
    assert app.machine_type == "GPU-B200"
    assert app.min_concurrency == 0
    assert app.image.build_args == {"PYTHON_VERSION": "3.12", "FAL_SPEC": "fal==1.84.0"}
    assert "FROM nvidia/cuda" in app.image.dockerfile_str
    assert "def main()" in module["HOT_SERVER_SOURCE"]
    assert "devbox-save" in module["_save_script"]()


def test_box_app_hot_dir_stays_under_work():
    module = runpy.run_path(str(BOX_APP))

    assert str(module["_hot_dir"]("/work/klein")) == "/work/klein"
    with pytest.raises(ValueError):
        module["_hot_dir"]("/work/../etc")
    with pytest.raises(ValueError):
        module["_hot_dir"]("/tmp")


def test_box_app_write_files_skips_unchanged(tmp_path):
    module = runpy.run_path(str(BOX_APP))
    module["_hot_dir"].__globals__["LOCAL_WORK"] = tmp_path
    project = tmp_path / "klein"
    project.mkdir()
    (project / "load.py").write_text("same")
    before = (project / "load.py").stat().st_mtime_ns

    files = {"load.py": "same", "infer.py": "new", "pkg/util.py": "x"}
    result = module["_hot_write_files"]({"dir": str(project), "files": files})

    assert sorted(result["written"]) == ["infer.py", "pkg/util.py"]
    assert (project / "load.py").stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="outside"):
        module["_hot_write_files"]({"dir": str(project), "files": {"../x": "y"}})


def test_project_paths_keeps_remembered_dir_for_same_project(tmp_path):
    a, b = tmp_path / "model-a", tmp_path / "model-b"
    remembered = {"hot_local": str(a.resolve()), "hot_dir": "/work/custom"}

    assert hot.project_paths(None, None, remembered) == (
        str(a.resolve()),
        "/work/custom",
    )
    assert hot.project_paths(str(a), None, remembered)[1] == "/work/custom"
    # A different project gets its own folder, not the remembered one.
    assert hot.project_paths(str(b), None, remembered) == (
        str(b.resolve()),
        "/work/model-b",
    )
    assert hot.project_paths(str(b), "/work/x", remembered)[1] == "/work/x"


def _hot_box(responses):
    box = hot.HotBox.__new__(hot.HotBox)
    box.post = MagicMock(side_effect=responses)
    return box


def test_request_reports_a_stopped_server_instead_of_waiting():
    body = json.dumps(
        {"error": "the hot server is not running (fal dev hot start), or is restarting"}
    ).encode()
    box = _hot_box([(409, {}, body)])
    box.wait_ready = MagicMock()

    assert box.request({}) is False
    box.wait_ready.assert_not_called()


def test_request_waits_through_a_full_restart(tmp_path):
    restarting = b'{"restarting": "load.py changed"}'
    box = _hot_box(
        [(409, {}, restarting), (200, {"content-type": "application/json"}, b"{}")]
    )
    box.wait_ready = MagicMock(return_value={"load_seconds": 1.0})

    assert box.request({}) is True
    box.wait_ready.assert_called_once()


def test_wait_ready_rides_out_a_network_blip(monkeypatch):
    monkeypatch.setattr(hot.time, "sleep", lambda s: None)
    ready = json.dumps({"ready": True, "load_seconds": 2}).encode()
    box = _hot_box([(0, {}, b'{"error": "timed out"}'), (200, {}, ready)])

    assert box.wait_ready(timeout=60)["load_seconds"] == 2


def test_state_update_skips_unchanged_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    state.update(tmp_path, None, None, name="klein")
    path = state._path(tmp_path, None, None)
    before = path.stat().st_mtime_ns

    state.update(tmp_path, None, None, name="klein")

    assert path.stat().st_mtime_ns == before


def test_start_never_stops_a_box_that_was_already_up():
    box = _box([_runner("r1")])
    box.client.apps.scale.side_effect = RuntimeError("transient")
    box.stop = MagicMock()

    with pytest.raises(RuntimeError):
        box.start(timeout=1)
    box.stop.assert_not_called()


def test_wait_ready_stops_when_the_server_waits_for_a_fix(monkeypatch):
    monkeypatch.setattr(hot.time, "sleep", lambda s: None)
    log_rows = [
        "00:00:01 [hot] loading the model (load.load()) ...",
        "00:00:02 [hot] load.load() failed; fix load.py and save to retry:",
        "Traceback (most recent call last):",
        *[f'  File "load.py", line {i}' for i in range(10)],
        "ModuleNotFoundError: No module named 'missing'",
    ]
    stuck = json.dumps({"error": "not running", "log": log_rows}).encode()
    box = _hot_box([(409, {}, stuck)])

    with pytest.raises(DevError, match="stopped on an error"):
        box.wait_ready(timeout=60)


def test_deploy_rejects_an_unsupported_python(monkeypatch):
    monkeypatch.setattr(box_module.sys, "version_info", (3, 14, 0))
    box = _box([])

    with pytest.raises(DevError, match="Python 3.10 to 3.13"):
        box.deploy()
    box.client.deploy.assert_not_called()


def test_waiting_for_fix_ignores_a_failure_followed_by_a_retry():
    assert not hot.waiting_for_fix(
        ["[hot] load.load() failed", "[hot] loading the model"]
    )
    assert hot.waiting_for_fix(["[hot] restarting", "[hot] server exited with code 1"])
    assert not hot.waiting_for_fix(["[hot] loading the model (load.load()) ..."])


def test_new_log_lines_prints_each_line_once():
    assert hot.new_log_lines([], ["a", "b"]) == ["a", "b"]
    assert hot.new_log_lines(["a", "b"], ["a", "b"]) == []
    assert hot.new_log_lines(["a", "b"], ["a", "b", "c"]) == ["c"]
    # A sliding tail: the oldest line dropped off the front.
    assert hot.new_log_lines(["a", "b", "c"], ["b", "c", "d"]) == ["d"]
    # A different log (the server restarted): print all of it.
    assert hot.new_log_lines(["a", "b"], ["x", "y"]) == ["x", "y"]


def test_box_name_must_match_fully():
    assert config.NAME_PATTERN.fullmatch("klein")
    assert not config.NAME_PATTERN.fullmatch("klein\n")


def test_restarting_reads_the_log_order():
    serving = "00:00:03 [hot] serving /work/klein on 127.0.0.1:7860"
    restart = "00:00:09 [hot] FULL RESTART: load.py changed; the weights reload"
    loading = "00:00:01 [hot] loading the model (load.load()) ..."

    assert hot.restarting([loading, serving, restart])
    # Stopped after serving: nothing to wait for.
    assert not hot.restarting([loading, serving])
    assert not hot.restarting([restart, "[hot] load.load() failed; fix load.py"])


def test_request_waits_when_the_server_already_exited_to_restart():
    log_rows = ["[hot] serving /work/klein", "[hot] FULL RESTART: load.py changed"]
    closed = json.dumps({"error": "not running", "log": log_rows}).encode()
    box = _hot_box([(409, {}, closed), (200, {"content-type": "text/plain"}, b"ok")])
    box.wait_ready = MagicMock(return_value={"load_seconds": 1.0})

    assert box.request({}) is True
    box.wait_ready.assert_called_once()


def test_push_reports_a_restarted_box(tmp_path):
    (tmp_path / "infer.py").write_text("x")
    box = _hot_box([(403, {}, b'{"error": "bad or missing hot token"}')])

    with pytest.raises(hot.HotTokenRejected, match="restarted"):
        box.push(tmp_path, "/work/x", ["infer.py"])
