import io
import json
import runpy
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fal.dev import box as box_module
from fal.dev import config, files, hot, state
from fal.dev.account import Account
from fal.dev.box import (
    BOX_APP,
    REMOTE,
    Box,
    _HoldUntilMarker,
    fal_spec,
    newest_live,
    ssh_config,
)
from fal.dev.errors import DevError
from fal.sdk import RunnerState


def _runner(runner_id, state=RunnerState.RUNNING, uptime=60, machine="GPU-H100"):
    runner = MagicMock()
    runner.runner_id = runner_id
    runner.state = state
    runner.uptime = timedelta(seconds=uptime)
    runner.machine_type = machine
    return runner


def _box(runners=(), min_concurrency=1, deployed=True):
    box = Box("klein", account=Account("my-team", False))
    box.client = MagicMock()
    box.client.apps.runners.return_value = list(runners)
    app = MagicMock(alias="devbox-klein", min_concurrency=min_concurrency)
    app.machine_types = ["GPU-H100"]
    box.client.apps.list.return_value = [app] if deployed else []
    return box


# --- state ---


def test_state_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")

    state.update(tmp_path, "host", name="klein", hot_dir="/work/k")
    state.update(tmp_path, "host", hot_dir=None)

    data = state.load(tmp_path, "host")
    assert data["name"] == "klein"
    assert "hot_dir" not in data
    assert state.load(tmp_path, "other-host") == {}
    assert state.load(tmp_path / "sub", "host") == {}


def test_box_state_is_kept_per_box_and_host(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")

    state.update_box("devbox-a", "host", team="red")
    state.update_box("devbox-b", "host", team="blue")

    assert state.load_box("devbox-a", "host")["team"] == "red"
    assert state.load_box("devbox-b", "host")["team"] == "blue"
    assert state.load_box("devbox-a", "other-host") == {}


def test_state_ignores_corrupt_file(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    path = state._folder(tmp_path, None)
    path.parent.mkdir(parents=True)
    path.write_text("not json")

    assert state.load(tmp_path, None) == {}


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


@pytest.mark.parametrize(
    "runners, min_concurrency, expected",
    [
        ([_runner("r1")], 1, "running"),
        ([_runner("r1", RunnerState.IDLE)], 0, "stopping"),
        ([_runner("r1", RunnerState.DRAINING)], 0, "stopping"),
        ([_runner("r1", RunnerState.DOCKER_PULL)], 1, "starting"),
        ([_runner("r1", RunnerState.CRASH_BACKOFF)], 1, "failed"),
        ([_runner("r1", RunnerState.DEAD)], 0, "stopped"),
        ([], 0, "stopped"),
    ],
)
def test_state_classifies_the_box(runners, min_concurrency, expected):
    assert _box(runners, min_concurrency).state().code == expected


def test_state_picks_the_newest_live_runner_and_its_machine_type():
    box = _box([_runner("old", uptime=900), _runner("new", uptime=30)])

    state = box.state()

    assert state.runner.runner_id == "new"
    assert state.machine_type == "GPU-H100"
    assert _box([]).state().machine_type == "GPU-H100"


def test_run_returns_the_exit_code_and_output(monkeypatch):
    def fake_session(client, runner_id, command, **kwargs):
        assert (runner_id, command, kwargs["interactive"]) == ("r1", ["ls"], False)
        assert kwargs["console"] is None
        kwargs["stdout"].write(b"a\xffb")
        return 3

    monkeypatch.setattr(box_module, "shell_session", fake_session)

    assert _box().run("r1", ["ls"]) == (3, "a\ufffdb")


def test_run_turns_connection_errors_into_a_dev_error(monkeypatch):
    import grpc

    class Failure(grpc.RpcError):
        def __init__(self, code):
            self._code = code

        def code(self):
            return self._code

        def details(self):
            return "no route"

    def fail(*args, **kwargs):
        raise Failure(grpc.StatusCode.UNAVAILABLE)

    monkeypatch.setattr(box_module, "shell_session", fail)

    with pytest.raises(DevError, match="no route"):
        _box().run("r1", ["ls"])


def test_run_reuses_its_connection_and_rebuilds_it_once_after_expiry(monkeypatch):
    import grpc

    class Expired(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNAUTHENTICATED

        def details(self):
            return "expired"

    box = _box()
    stubs = []
    box.client._create_host.side_effect = lambda: stubs.append(
        MagicMock()
    ) or MagicMock(_connection=MagicMock(stub=stubs[-1]))
    used = []

    def session(client, runner_id, command, **kwargs):
        used.append((kwargs["stub"], kwargs["timeout"]))
        if len(used) == 1:
            raise Expired()
        return 0

    monkeypatch.setattr(box_module, "shell_session", session)

    assert box.run("r1", ["ls"]) == (0, "")
    assert box.run("r1", ["ls"]) == (0, "")

    assert len(stubs) == 2
    assert [stub for stub, _ in used] == [stubs[0], stubs[1], stubs[1]]
    assert used[0][1] == box_module.RUN_TIMEOUT_SECONDS


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
        ([_runner("r1", RunnerState.CRASH_BACKOFF)], 1, True, "devbox-klein: failed"),
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
    box.run.assert_called_once_with(
        "r1", [config.SAVE_COMMAND], timeout=box_module.SAVE_TIMEOUT_SECONDS
    )


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


def test_ssh_config_proxy_reaches_the_boxs_host():
    entry = ssh_config("klein", "fal", host="api.shark.test")

    assert "dev ssh-proxy --name klein --host api.shark.test" in entry
    assert "--host" not in ssh_config("klein", "fal")


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


def test_scan_never_picks_up_secrets(tmp_path):
    (tmp_path / "infer.py").write_text("x")
    (tmp_path / "secrets.py").write_text("x")  # code, not a secret file
    (tmp_path / "keys").mkdir()
    for secret in (
        "server.pem",
        "credentials.json",
        "id_rsa",
        "secrets.yaml",
        "keys/token.json",
        ".env",
        ".env.production",
        "prod.env",
        "secrets/api.txt",
        "vars.tfvars",
        "state.tfstate",
        "service_account_prod.json",
    ):
        path = tmp_path / secret
        path.parent.mkdir(exist_ok=True)
        path.write_text("secret")

    assert sorted(hot.scan(tmp_path)) == ["infer.py", "secrets.py"]


def test_walk_leaves_out_skipped_folders_and_secrets(tmp_path):
    for name in ("a.py", "pkg/b.py", ".hidden/c.py", "node_modules/d.js", ".ssh/id_x"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    (tmp_path / ".dotfile").write_text("x")
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "private.py").write_text("x")
    (tmp_path / "linked.py").symlink_to(outside / "private.py")
    (tmp_path / "linked_dir").symlink_to(outside, target_is_directory=True)

    found = {p.as_posix() for p in files.walk(tmp_path)}
    assert found == {"a.py", "pkg/b.py", ".hidden/c.py", ".dotfile"}
    visible = {p.as_posix() for p in files.walk(tmp_path, hidden=False)}
    assert visible == {"a.py", "pkg/b.py"}


@pytest.mark.parametrize(
    "parts, secret",
    [
        ((".env",), True),
        (("app", ".env.local"), True),
        (("deploy", "key.PEM"), True),
        (("home", ".aws", "config"), True),
        (("secrets.json",), True),
        (("secrets.py",), False),
        (("src", "keyboard.py"), False),
    ],
)
def test_secret_like(parts, secret):
    assert files.secret_like(parts) is secret


def test_read_token_uses_cache_for_same_runner(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path)
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
    assert "devbox-save" in module["SAVE_SCRIPT"]
    assert sorted(module["PROBE_FILES"]) == ["config.py", "probe.py", "signals.py"]


def test_box_app_hot_dir_stays_under_work():
    module = runpy.run_path(str(BOX_APP))

    assert str(module["_hot_dir"]("/work/klein")) == "/work/klein"
    with pytest.raises(ValueError):
        module["_hot_dir"]("/work/../etc")
    with pytest.raises(ValueError):
        module["_hot_dir"]("/tmp")


def test_a_resync_of_unchanged_files_never_restarts_the_hot_server(tmp_path):
    """`hot sync --watch` pushes every file first; load.py must not restart it."""
    server = runpy.run_path(str(REMOTE / "hot_server.py"))
    box_app = runpy.run_path(str(BOX_APP))
    box_app["_hot_dir"].__globals__["LOCAL_WORK"] = tmp_path
    project = tmp_path / "klein"
    project.mkdir()
    sources = {"load.py": "def load(): ...", "infer.py": "def infer(state): ..."}
    for name, text in sources.items():
        (project / name).write_text(text)
    running = server["Project"](project)

    result = box_app["_hot_write_files"]({"dir": str(project), "files": sources})

    assert result == {"written": []}
    assert running.check() is None
    sources["load.py"] = "def load(): return 1"
    box_app["_hot_write_files"]({"dir": str(project), "files": sources})
    assert running.check() == "load.py changed"


@pytest.mark.parametrize("script", ["devbox-save.sh", "devbox-hot.sh", "bashrc.sh"])
def test_box_scripts_are_valid_bash_once_filled_in(script, tmp_path):
    import subprocess

    module = runpy.run_path(str(BOX_APP))
    text = module["_fill"](
        (REMOTE / "scripts" / script).read_text(),
        WORK="/work",
        WORK_PARENT="/",
        WORK_NAME="work",
        SNAPSHOT="/data/w.tar.zst",
        LEASE="/data/w.lease",
        STAMP="/tmp/stamp",
        INTERVAL=15,
        SERVER="/usr/local/lib/devbox-hot/hot_server.py",
        PORT=7860,
        RESTART_CODE=3,
        WORKSPACE="/data/devboxes/klein",
        CLAUDE_DIR="/data/c",
        WARNING_FILE="/tmp/w",
    )
    path = tmp_path / script
    path.write_text(text)

    done = subprocess.run(
        ["bash", "-n", str(path)], capture_output=True, text=True, check=False
    )

    assert done.returncode == 0, done.stderr
    # Bash's own variables survive; the placeholders do not.
    assert ("$stamp" in text) == (script == "devbox-save.sh")
    assert "$WORK" not in text
    assert "$PORT" not in text


def test_sshd_config_carries_the_hosts_environment(monkeypatch, tmp_path):
    module = runpy.run_path(str(BOX_APP))
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")
    monkeypatch.setenv(config.ADMIN_KEY_SECRET, "secret")

    text = module["_sshd_config"](tmp_path / "host_key")

    assert f"HostKey {tmp_path / 'host_key'}" in text
    assert '"CUDA_HOME=/usr/local/cuda"' in text
    assert "secret" not in text
    assert "$" not in text


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


def test_post_json_keeps_payloads_out_of_request_storage(monkeypatch):
    import httpx

    from fal.dev.transport import post_json

    seen = {}

    def fake_post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return httpx.Response(200)

    monkeypatch.setattr(httpx, "post", fake_post)

    post_json("https://fal.run/a/b/hot/files", {"x": 1}, {"Authorization": "Bearer t"})

    assert seen["json"] == {"x": 1}
    assert seen["headers"] == {"Authorization": "Bearer t", "X-Fal-Store-IO": "0"}


def test_box_builds_credentials_once_across_many_calls(monkeypatch):
    calls = []
    credentials = MagicMock()
    credentials.to_headers.return_value = {"Authorization": "Bearer t"}
    monkeypatch.setattr(
        "fal.sdk.get_credentials", lambda team: calls.append(team) or credentials
    )
    box = _box()

    assert [box.auth_headers() for _ in range(20)] == [
        {"Authorization": "Bearer t"}
    ] * 20
    assert calls == ["my-team"]


def _hot_box_with_auth(auth):
    box = hot.HotBox.__new__(hot.HotBox)
    box.base, box.token = "https://fal.run/me/devbox-klein/hot", "tok"
    box.box = MagicMock(auth_headers=auth)
    return box


def test_hot_box_posts_with_its_token_and_reports_failures_as_no_response(monkeypatch):
    import httpx

    calls = []

    def fake_post_json(url, payload, headers, timeout):
        calls.append((url, headers))
        if len(calls) == 2:
            raise httpx.ConnectError("down")
        return httpx.Response(409, headers={"X-Hot-Code-Version": "3"}, content=b"{}")

    monkeypatch.setattr(hot, "post_json", fake_post_json)
    box = _hot_box_with_auth(lambda: {"Authorization": "Bearer t"})

    status, headers, body = box.post("status")
    assert (status, headers["x-hot-code-version"], body) == (409, "3", b"{}")
    assert calls[0] == (
        "https://fal.run/me/devbox-klein/hot/status",
        {"Authorization": "Bearer t", config.HOT_TOKEN_HEADER: "tok"},
    )
    status, _, body = box.post("status")
    assert (status, json.loads(body)) == (0, {"error": "down"})


def test_a_credential_failure_is_retried_by_watch_and_wait_ready(monkeypatch, tmp_path):
    import httpx

    monkeypatch.setattr(hot.time, "sleep", lambda s: None)
    attempts = []

    def auth():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("login expired")
        return {}

    monkeypatch.setattr(
        hot,
        "post_json",
        lambda url, payload, headers, timeout: httpx.Response(
            200, json={"ready": True, "load_seconds": 1, "written": ["infer.py"]}
        ),
    )
    box = _hot_box_with_auth(auth)

    assert box.wait_ready(timeout=60)["load_seconds"] == 1
    assert len(attempts) == 2

    # A push that failed is sent again with the next save.
    (tmp_path / "infer.py").write_text("x")
    attempts.clear()
    with pytest.raises(DevError, match="sync failed"):
        box.push(tmp_path, "/work/x", ["infer.py"])
    assert box.push(tmp_path, "/work/x", ["infer.py"]) == ["infer.py"]


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
    state.update(tmp_path, None, name="klein")
    path = state._folder(tmp_path, None)
    before = path.stat().st_mtime_ns

    state.update(tmp_path, None, name="klein")

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


def test_hold_script_never_shortens_a_longer_hold(tmp_path):
    import subprocess

    script = (REMOTE / "scripts" / "hold.sh").read_text()
    hold = tmp_path / "a hold file"

    def run(until):
        return subprocess.run(
            [
                "sh",
                "-c",
                script,
                "sh",
                str(hold),
                str(until),
                str(config.HOLD_KEPT_EXIT),
            ],
            capture_output=True,
            check=False,
        ).returncode

    assert run(100) == 0
    assert hold.read_text().strip() == "100"
    assert run(50) == config.HOLD_KEPT_EXIT
    assert hold.read_text().strip() == "100"
    assert run(200) == 0
    assert hold.read_text().strip() == "200"


def test_hold_reports_a_kept_longer_hold():
    box = _box([_runner("r1")])
    box.run = MagicMock(return_value=(config.HOLD_KEPT_EXIT, ""))

    assert box.hold(1) is False
    box.run.return_value = (0, "")
    assert box.hold(1) is True
    box.run.return_value = (2, "sh: boom")
    with pytest.raises(DevError, match="boom"):
        box.hold(1)


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

    assert box.ensure_volume() == "devbox-klein"


def test_ensure_volume_points_ineligible_teams_at_team_data():
    from fal.exceptions.volumes import VolumeAccountNotEligibleError

    box = _box()
    box.client.volumes.create.side_effect = VolumeAccountNotEligibleError("no")

    with pytest.raises(DevError, match="--team-data"):
        box.ensure_volume()


def test_box_app_mounts_its_volume_at_data(monkeypatch):
    monkeypatch.setenv(config.ENV_VOLUME, "devbox-klein")
    app = runpy.run_path(str(BOX_APP))["DevBox"]
    assert app.host_kwargs["volumes"] == {"/data": "devbox-klein"}

    monkeypatch.setenv(config.ENV_VOLUME, "")
    app = runpy.run_path(str(BOX_APP))["DevBox"]
    assert "volumes" not in app.host_kwargs


def test_deploy_waits_for_a_runner_still_saving(monkeypatch):
    box = _box([_runner("r1", state=RunnerState.DRAINING)], min_concurrency=0)
    monkeypatch.setattr(box_module, "POLL_SECONDS", 0)
    monkeypatch.setattr(config, "LEASE_WAIT_SECONDS", 0)

    with pytest.raises(DevError, match="still saving"):
        box.deploy()

    box.client.deploy.assert_not_called()
    box.client.runners.stop.assert_not_called()


def test_a_failed_gpu_query_never_lets_a_busy_box_look_idle():
    from fal.dev.remote import signals

    module = runpy.run_path(str(BOX_APP))
    gpu_activity = module["_gpu_activity"]
    busy = {"util_pct": 90}
    idle = {"util_pct": 0}
    patches = {
        "busy": ([busy], None),
        "idle": ([idle], None),
        "failed": ([], "nvidia-smi timed out"),
    }
    original = signals.gpus
    try:
        results = {}
        for name, answer in patches.items():
            signals.gpus = lambda answer=answer: answer
            results[name] = gpu_activity()
    finally:
        signals.gpus = original

    assert results == {
        "busy": "GPU busy",
        "idle": None,
        "failed": "GPU status unknown",
    }


def test_ssh_proxy_retries_the_runner_and_arming_lookups_together(monkeypatch):
    monkeypatch.setattr(box_module.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        box_module,
        "shell_session",
        lambda *a, **k: k["stdout"].write(box_module.SSH_READY_MARKER) or 0,
    )
    monkeypatch.setattr(box_module.sys, "stdout", MagicMock())
    box = _box([_runner("r1")])
    box.client.apps.list.side_effect = [
        RuntimeError("blip"),
        box.client.apps.list.return_value,
    ]

    assert box.ssh_proxy(timeout=1) == 0


def test_a_gpu_that_cannot_be_read_is_warned_about_once_then_cleared(tmp_path):
    from fal.dev.remote import signals

    module = runpy.run_path(str(BOX_APP))
    warning = tmp_path / "warning"
    module["_warn"].__globals__["WARNING_FILE"] = warning
    original = signals.gpus
    try:
        signals.gpus = lambda: ([], "timed out")
        for _ in range(5):
            assert module["_gpu_activity"]() == "GPU status unknown"
        assert warning.read_text().count("GPU status is unknown") == 1
        signals.gpus = lambda: ([{"util_pct": 0}], None)
        assert module["_gpu_activity"]() is None
    finally:
        signals.gpus = original

    assert warning.read_text() == ""


@pytest.mark.parametrize(
    "name",
    [
        "terraform.tfstate.backup",
        "prod.tfvars.json",
        "my-adminsdk-x1.json",
        "deploy-key.json",
        "deploy_key.json",
        "credentials.yaml",
        "credentials",
        "client_secret_123.apps.json",
        "serviceAccount.json",
        "ServiceAccountKey.JSON",
    ],
)
def test_more_secret_file_names(name):
    assert files.secret_like(("cfg", name))


def test_save_work_failure_says_the_save_may_still_run():
    box = _box([_runner("r1")])
    box.run = MagicMock(side_effect=DevError("timed out"))

    with pytest.raises(DevError, match="may still be in progress.*--no-save"):
        box.save_work()


def test_probe_survives_a_whitespace_only_hot_error(monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(REMOTE))
    monkeypatch.syspath_prepend(str(Path(config.__file__).parent))
    probe = importlib.import_module("probe")
    monkeypatch.setattr(probe.signals, "tmux_session", lambda name: True)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"error": "  \\n "}'

    monkeypatch.setattr(probe.urllib.request, "urlopen", lambda *a, **k: Response())

    assert probe._hot()["error"] == ""


def test_run_starts_each_attempt_with_an_empty_buffer(monkeypatch):
    import grpc

    class Expired(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNAUTHENTICATED

    box = _box()
    calls = []

    def session(client, runner_id, command, **kwargs):
        calls.append(1)
        kwargs["stdout"].write(b"partial")
        if len(calls) == 1:
            raise Expired()
        return 0

    monkeypatch.setattr(box_module, "shell_session", session)

    assert box.run("r1", ["ls"]) == (0, "partial")


def test_push_leaves_out_a_file_that_became_a_symlink(tmp_path):
    (tmp_path / "infer.py").write_text("x")
    (tmp_path / "other.py").write_text("y")
    outside = tmp_path.parent / f"{tmp_path.name}-secret.py"
    outside.write_text("secret")
    (tmp_path / "other.py").unlink()
    (tmp_path / "other.py").symlink_to(outside)
    box = _hot_box([(200, {}, b'{"written": ["infer.py"]}')])

    box.push(tmp_path, "/work/x", ["infer.py", "other.py"])

    assert list(box.post.call_args.args[1]["files"]) == ["infer.py"]
