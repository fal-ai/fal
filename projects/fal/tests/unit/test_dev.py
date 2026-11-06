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
    box.team_workspace_exists = MagicMock(return_value=False)
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


def test_dash_probe_defaults_match_the_box_layout():
    """dash_probe.py runs on the box as a standalone script, so it repeats these."""
    from fal.dev.remote.probe import DEFAULTS

    assert DEFAULTS["local_work"] == config.LOCAL_WORK
    assert DEFAULTS["snapshot"] == config.SNAPSHOT_FILE
    assert DEFAULTS["lease"] == config.LEASE_FILE
    assert DEFAULTS["hold_file"] == config.HOLD_FILE
    assert DEFAULTS["admin_key_env"] == config.ADMIN_KEY_SECRET
    assert DEFAULTS["claude_config_dir"] == config.CLAUDE_CONFIG_SUBDIR
    assert DEFAULTS["claude_tmux"] == config.CLAUDE_TMUX_SESSION
    assert DEFAULTS["hot_tmux"] == config.HOT_TMUX_SESSION
    assert DEFAULTS["hot_port"] == config.HOT_PORT
    assert DEFAULTS["hot_log"] == config.HOT_LOG
    assert DEFAULTS["gpu_busy_pct"] == config.GPU_BUSY_PCT
    assert DEFAULTS["idle_state_file"] == config.IDLE_STATE_FILE
    assert DEFAULTS["save_stamp"] == config.SAVE_STAMP
    assert DEFAULTS["warning_file"] == config.WARNING_FILE
    # A new key must be pinned here too (workspace is per box; check_dirty a flag).
    assert len(DEFAULTS) == 16


def test_dash_box_shares_the_box_client():
    from fal.dev.dash.model import FalBox

    box = _box([])
    dash_box = FalBox.from_box(box, idle_minutes=30)

    assert dash_box.app_name == "devbox-klein"
    assert dash_box._client is box.client
    assert dash_box.idle_minutes == 30
    assert dash_box.save_command == config.SAVE_COMMAND
    assert dash_box.probe_config["workspace"] == "/data/devboxes/klein"


def test_hold_command_never_shortens_a_longer_hold(tmp_path):
    import subprocess

    hold = tmp_path / "hold"
    run = lambda until: subprocess.run(  # noqa: E731
        ["sh", "-c", config.hold_command(until, str(hold))],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    assert run(100) == ""
    assert hold.read_text().strip() == "100"
    assert run(50).strip() == config.HOLD_KEPT
    assert hold.read_text().strip() == "100"
    assert run(200) == ""
    assert hold.read_text().strip() == "200"


def test_published_idle_state_is_what_the_dash_reads(monkeypatch, tmp_path):
    from fal.dev.dash.model import idle_view

    module = runpy.run_path(str(BOX_APP))
    path = tmp_path / "idle.json"
    monkeypatch.setattr(config, "IDLE_STATE_FILE", str(path))
    module["_publish_idle_state"](1000.0, ["hot request"])

    view = idle_view(
        {"idle_state": json.loads(path.read_text())},
        idle_minutes=None,
        runner_started=None,
        claude_counts=False,
        now=1060.0,
    )
    assert view.source == "watchdog"
    assert view.reasons == ["hot request"]
    # On/off comes from the box's warnings, never from this file.
    assert "watchdog" not in json.loads(path.read_text())
    assert view.remaining_seconds == 60 * 60 - 60


# --- storage ---


def test_volume_name_fits_the_controller_limit():
    from fal.volumes import canonical_volume_name

    assert config.volume_name("klein") == "devbox-klein"
    long_a, long_b = "a" * 40, "a" * 39 + "b"
    for name in (long_a, long_b, "x" * 25, "ab-" + "c" * 37):
        volume = config.volume_name(name)
        assert len(volume) <= config.VOLUME_NAME_MAX
        assert canonical_volume_name(volume) == volume
    assert config.volume_name(long_a) != config.volume_name(long_b)
    assert config.volume_name(long_a) == config.volume_name(long_a)


def test_deploy_mounts_the_box_volume(monkeypatch):
    box = _box(min_concurrency=0)
    box.stop = MagicMock(return_value=True)
    box.team_workspace_exists = MagicMock(return_value=False)
    seen = {}
    box.client.deploy.side_effect = lambda *a, **k: seen.update(
        volume=box_module.os.environ[config.ENV_VOLUME]
    )

    message = box.deploy()

    box.client.volumes.create.assert_called_once_with("devbox-klein")
    assert seen["volume"] == "devbox-klein"
    assert "volume devbox-klein" in message


def test_deploy_with_team_data_skips_the_volume():
    box = _box(min_concurrency=0)
    box.stop = MagicMock(return_value=True)
    seen = {}
    box.client.deploy.side_effect = lambda *a, **k: seen.update(
        volume=box_module.os.environ[config.ENV_VOLUME]
    )

    message = box.deploy(team_data=True)

    box.client.volumes.create.assert_not_called()
    assert seen["volume"] == ""
    assert "team-shared /data" in message


def test_ensure_volume_reuses_an_existing_volume():
    from fal.exceptions.volumes import VolumeAlreadyExistsError

    box = _box()
    box.client.volumes.create.side_effect = VolumeAlreadyExistsError("exists")
    box.team_workspace_exists = MagicMock()

    assert box.ensure_volume() == "devbox-klein"
    box.team_workspace_exists.assert_not_called()


def test_ensure_volume_points_ineligible_teams_at_team_data():
    from fal.exceptions.volumes import VolumeAccountNotEligibleError

    box = _box()
    box.client.volumes.create.side_effect = VolumeAccountNotEligibleError("no")

    with pytest.raises(DevError, match="--team-data"):
        box.ensure_volume()


def test_ensure_volume_warns_when_the_copy_needs_the_admin_key(capsys):
    box = _box()
    box.team_workspace_exists = MagicMock(return_value=True)
    box.client.secrets.list.return_value = []

    assert box.ensure_volume() == "devbox-klein"
    assert config.ADMIN_KEY_SECRET in capsys.readouterr().err


def test_box_app_mounts_its_volume_at_data(monkeypatch):
    monkeypatch.setenv(config.ENV_VOLUME, "devbox-klein")
    app = runpy.run_path(str(BOX_APP))["DevBox"]
    assert app.host_kwargs["volumes"] == {"/data": "devbox-klein"}

    monkeypatch.setenv(config.ENV_VOLUME, "")
    app = runpy.run_path(str(BOX_APP))["DevBox"]
    assert "volumes" not in app.host_kwargs


def _migration(monkeypatch, tmp_path, team_files, admin=True):
    """box_app's migration against a fake files API serving `team_files`."""
    import httpx

    module = runpy.run_path(str(BOX_APP))
    migrate = module["_migrate_from_team_data"]
    workspace = tmp_path / "data" / "devboxes" / "klein"
    g = migrate.__globals__
    g.update(
        VOLUME="devbox-klein",
        APP_NAME="devbox-klein",
        WORKSPACE=workspace,
        SNAPSHOT=workspace / config.SNAPSHOT_FILE,
        WARNING_FILE=tmp_path / "warning",
    )
    workspace.mkdir(parents=True)  # setup creates it before the copy
    root = str(workspace)
    files = {f"{root}/{rel}": body for rel, body in team_files.items()}
    downloads = []

    def handler(request):
        path = request.url.path
        if path.startswith("/files/list/"):
            folder = path[len("/files/list/") :].rstrip("/")
            entries = {}
            for name, body in files.items():
                if not name.startswith(folder + "/"):
                    continue
                child = name[len(folder) + 1 :].split("/")[0]
                full = f"{folder}/{child}"
                entries[full] = {
                    "path": full,
                    "is_file": full == name,
                    "size": len(body),
                }
            if not entries:
                return httpx.Response(404)
            return httpx.Response(200, json=list(entries.values()))
        name = path[len("/files/file/") :]
        downloads.append(name)
        if name not in files:
            return httpx.Response(404)
        start = int(request.headers.get("Range", "bytes=0-")[len("bytes=") : -1])
        size = len(files[name])
        if start >= size and start:
            return httpx.Response(416)
        if start:
            return httpx.Response(
                206,
                content=files[name][start:],
                headers={"Content-Range": f"bytes {start}-{size - 1}/{size}"},
            )
        return httpx.Response(200, content=files[name])

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    client = MagicMock()
    client._rest_url = "https://api.test"
    client._credentials.to_headers.return_value = {}
    client.apps.list.return_value = [MagicMock(alias="devbox-klein")]
    g["_admin_client"] = lambda: client if admin else None
    return migrate, workspace, downloads


def test_migration_copies_the_team_workspace_once(monkeypatch, tmp_path):
    team = {
        config.SNAPSHOT_FILE: b"snapshot",
        ".home/.ssh/id_ed25519": b"key",
        ".bash_history": b"ls",
        config.LEASE_FILE: b"runner active 1",
        f"{config.SNAPSHOT_FILE}.host.tmp": b"partial",
    }
    migrate, workspace, downloads = _migration(monkeypatch, tmp_path, team)
    (workspace / ".home").mkdir()
    (workspace / ".bash_history").write_bytes(b"newer")

    migrate()

    assert (workspace / config.SNAPSHOT_FILE).read_bytes() == b"snapshot"
    assert (workspace / ".home/.ssh/id_ed25519").stat().st_mode & 0o077 == 0
    # Never overwrites the volume, skips the lease and partial saves.
    assert (workspace / ".bash_history").read_bytes() == b"newer"
    assert not (workspace / config.LEASE_FILE).exists()
    assert downloads[-1].endswith(config.SNAPSHOT_FILE)
    assert len(downloads) == 2
    assert (workspace / config.MIGRATED_MARKER).exists()

    migrate()
    assert len(downloads) == 2


def test_migration_waits_for_the_admin_key(monkeypatch, tmp_path):
    team = {config.SNAPSHOT_FILE: b"snapshot"}
    migrate, workspace, downloads = _migration(monkeypatch, tmp_path, team, admin=False)

    migrate()

    assert not downloads
    assert not (workspace / config.MIGRATED_MARKER).exists()
    assert config.ADMIN_KEY_SECRET in (tmp_path / "warning").read_text()


def test_migration_with_no_team_copy_marks_the_volume(monkeypatch, tmp_path):
    migrate, workspace, downloads = _migration(monkeypatch, tmp_path, {})

    migrate()

    assert not downloads
    assert (workspace / config.MIGRATED_MARKER).read_text() == "0 files copied\n"


def test_migration_keeps_both_snapshots_after_an_early_save(monkeypatch, tmp_path):
    """A box that saved /work before it had the key still gets the team copy."""
    team = {config.SNAPSHOT_FILE: b"team", ".home/#odd?name": b"x"}
    migrate, workspace, downloads = _migration(monkeypatch, tmp_path, team)
    (workspace / config.SNAPSHOT_FILE).write_bytes(b"volume")

    migrate()

    assert (workspace / config.SNAPSHOT_FILE).read_bytes() == b"volume"
    side = workspace / f"{config.SNAPSHOT_FILE}.from-team-data"
    assert side.read_bytes() == b"team"
    assert (workspace / ".home/#odd?name").read_bytes() == b"x"
    assert (workspace / config.MIGRATED_MARKER).exists()


def test_migration_refuses_paths_outside_the_workspace(monkeypatch, tmp_path):
    team = {"../other/secret": b"x"}
    migrate, workspace, downloads = _migration(monkeypatch, tmp_path, team)

    migrate()

    assert not downloads
    assert not (workspace / config.MIGRATED_MARKER).exists()
    assert "outside the workspace" in (tmp_path / "warning").read_text()


def test_migration_restarts_a_finished_leftover_download(monkeypatch, tmp_path):
    team = {config.SNAPSHOT_FILE: b"0123"}
    migrate, workspace, downloads = _migration(monkeypatch, tmp_path, team)
    (workspace / f"{config.SNAPSHOT_FILE}.migrating.tmp").write_bytes(b"0123")

    migrate()

    assert (workspace / config.SNAPSHOT_FILE).read_bytes() == b"0123"
    assert (workspace / config.MIGRATED_MARKER).exists()


def test_migration_copies_the_rest_when_one_file_fails(monkeypatch, tmp_path):
    team = {"a.txt": b"a", config.SNAPSHOT_FILE: b"snap"}
    migrate, workspace, downloads = _migration(monkeypatch, tmp_path, team)
    (workspace / "a.txt.migrating.tmp").mkdir()  # makes the a.txt write fail

    migrate()

    assert (workspace / config.SNAPSHOT_FILE).read_bytes() == b"snap"
    assert not (workspace / config.MIGRATED_MARKER).exists()
    assert "1 files failed" in (tmp_path / "warning").read_text()

    # The retry finishes without a second copy of the snapshot.
    (workspace / "a.txt.migrating.tmp").rmdir()
    migrate()
    assert (workspace / "a.txt").read_bytes() == b"a"
    assert not (workspace / f"{config.SNAPSHOT_FILE}.from-team-data").exists()
    assert (workspace / config.MIGRATED_MARKER).exists()


def test_migration_resumes_an_interrupted_download(monkeypatch, tmp_path):
    team = {config.SNAPSHOT_FILE: b"0123456789"}
    migrate, workspace, downloads = _migration(monkeypatch, tmp_path, team)
    partial = workspace / f"{config.SNAPSHOT_FILE}.migrating.tmp"
    partial.write_bytes(b"01234")

    migrate()

    assert (workspace / config.SNAPSHOT_FILE).read_bytes() == b"0123456789"
    assert not partial.exists()


def test_migration_drops_a_download_of_the_wrong_size(monkeypatch, tmp_path):
    """A leftover from an older version of the file must not be spliced in."""
    team = {config.SNAPSHOT_FILE: b"0123456789"}
    migrate, workspace, downloads = _migration(monkeypatch, tmp_path, team)
    stale = workspace / f"{config.SNAPSHOT_FILE}.migrating.tmp"
    stale.write_bytes(b"x" * 12)

    migrate()

    assert (workspace / config.SNAPSHOT_FILE).read_bytes() == b"0123456789"


def test_deploy_waits_for_a_runner_still_saving(monkeypatch):
    box = _box([_runner("r1", state=RunnerState.DRAINING)], min_concurrency=0)
    box.team_workspace_exists = MagicMock(return_value=False)
    monkeypatch.setattr(box_module, "POLL_SECONDS", 0)
    monkeypatch.setattr(config, "LEASE_WAIT_SECONDS", 0)

    with pytest.raises(DevError, match="still saving"):
        box.deploy()

    box.client.deploy.assert_not_called()
    box.client.runners.stop.assert_not_called()
