import io
from unittest.mock import MagicMock

import pytest

from dev.fakes import make_box, make_runner
from fal.dev import box as box_module
from fal.dev import config
from fal.dev.box import BOX_APP, _HoldUntilMarker, fal_spec, newest_live, ssh_config
from fal.dev.errors import DevError
from fal.sdk import RunnerState


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
        make_runner("old", uptime=500),
        make_runner("dead", RunnerState.DEAD, uptime=1),
        make_runner("new", RunnerState.IDLE, uptime=10),
    ]

    assert [r.runner_id for r in newest_live(runners)] == ["new", "old"]


@pytest.mark.parametrize(
    "runners, min_concurrency, expected",
    [
        ([make_runner("r1")], 1, "running"),
        ([make_runner("r1", RunnerState.IDLE)], 0, "stopping"),
        ([make_runner("r1", RunnerState.DRAINING)], 0, "stopping"),
        ([make_runner("r1", RunnerState.DOCKER_PULL)], 1, "starting"),
        ([make_runner("r1", RunnerState.CRASH_BACKOFF)], 1, "failed"),
        ([make_runner("r1", RunnerState.DEAD)], 0, "stopped"),
        ([], 0, "stopped"),
    ],
)
def test_state_classifies_the_box(runners, min_concurrency, expected):
    assert make_box(runners, min_concurrency).state().code == expected


def test_state_picks_the_newest_live_runner_and_its_machine_type():
    box = make_box([make_runner("old", uptime=900), make_runner("new", uptime=30)])

    state = box.state()

    assert state.runner.runner_id == "new"
    assert state.machine_type == "GPU-H100"
    assert make_box([]).state().machine_type == "GPU-H100"


def test_run_returns_the_exit_code_and_output(monkeypatch):
    def fake_session(client, runner_id, command, **kwargs):
        assert (runner_id, command, kwargs["interactive"]) == ("r1", ["ls"], False)
        assert kwargs["console"] is None
        kwargs["stdout"].write(b"a\xffb")
        return 3

    monkeypatch.setattr(box_module, "shell_session", fake_session)

    assert make_box().run("r1", ["ls"]) == (3, "a\ufffdb")


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
        make_box().run("r1", ["ls"])


def test_run_reuses_its_connection_and_rebuilds_it_once_after_expiry(monkeypatch):
    import grpc

    class Expired(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNAUTHENTICATED

        def details(self):
            return "expired"

    box = make_box()
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
    assert make_box([make_runner("r1")]).require_runner() == "r1"
    with pytest.raises(DevError, match="not running"):
        make_box([]).require_runner()
    with pytest.raises(DevError, match="shutting down"):
        make_box([make_runner("r1")], min_concurrency=0).require_runner()


@pytest.mark.parametrize(
    "runners, min_concurrency, deployed, expected",
    [
        ([make_runner("r1")], 1, True, "devbox-demo: running on GPU-H100"),
        ([make_runner("r1")], 0, True, "devbox-demo: shutting down"),
        ([make_runner("r1", RunnerState.DRAINING)], 0, True, "devbox-demo: stopping"),
        ([make_runner("r1", RunnerState.PENDING)], 1, True, "devbox-demo: starting"),
        (
            [make_runner("r1", RunnerState.CRASH_BACKOFF)],
            1,
            True,
            "devbox-demo: failed",
        ),
        ([], 0, True, "devbox-demo: stopped"),
        ([], 0, False, "devbox-demo: not deployed"),
    ],
)
def test_status(runners, min_concurrency, deployed, expected):
    lines = make_box(runners, min_concurrency, deployed).status()

    assert lines[0].startswith(expected)


def test_stop_scales_to_zero_and_stops_live_runners():
    box = make_box([make_runner("r1"), make_runner("gone", RunnerState.TERMINATED)])
    box.client.apps.runners.side_effect = [
        [make_runner("r1"), make_runner("gone", RunnerState.TERMINATED)],
        [make_runner("r1", RunnerState.TERMINATED)],
    ]

    assert box.stop()
    box.client.apps.scale.assert_called_once_with(
        "devbox-demo", min_concurrency=0, machine_types=None
    )
    box.client.runners.stop.assert_called_once_with("r1")


def test_wait_until_ready_raises_on_crash(monkeypatch):
    monkeypatch.setattr(box_module, "POLL_SECONDS", 0)
    box = make_box([make_runner("r1", RunnerState.CRASH_BACKOFF)])

    with pytest.raises(DevError, match="failed to start"):
        box.wait_until_ready(timeout=5)


def test_start_cancels_when_no_runner_comes_up(monkeypatch):
    monkeypatch.setattr(box_module, "POLL_SECONDS", 0)
    box = make_box([])
    box.wait_until_ready = MagicMock(return_value=None)
    box.stop = MagicMock(return_value=True)

    with pytest.raises(DevError, match="cancelled"):
        box.start(timeout=1)
    box.stop.assert_called_once()


def test_save_work_raises_and_keeps_box_up_on_failure():
    box = make_box([make_runner("r1")])
    box.run = MagicMock(return_value=(1, "disk full"))

    with pytest.raises(DevError, match="disk full"):
        box.save_work()
    box.run.assert_called_once_with(
        "r1", [config.SAVE_COMMAND], timeout=box_module.SAVE_TIMEOUT_SECONDS
    )


def test_hold_rejects_out_of_range_hours():
    with pytest.raises(DevError, match="at most"):
        make_box([make_runner("r1")]).hold(config.MAX_HOLD_HOURS + 1)


def test_deploy_stops_a_running_box_first_and_restarts_it(monkeypatch):
    box = make_box([make_runner("r1")])
    box.save_work = MagicMock()
    box.stop = MagicMock(return_value=True)

    message = box.deploy(idle_minutes=30)

    box.save_work.assert_called_once()
    box.stop.assert_called_once()
    (ref,), kwargs = box.client.deploy.call_args
    assert ref == (str(BOX_APP), "DevBox")
    assert kwargs == {
        "app_name": "devbox-demo",
        "auth": "private",
        "reset_scale": True,
    }
    box.client.apps.scale.assert_called_once_with(
        "devbox-demo", min_concurrency=1, machine_types=None
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
    entry = ssh_config("demo", "/path to/fal", team="my-team")

    assert "Host devbox-demo" in entry
    assert (
        "ProxyCommand '/path to/fal' dev ssh-proxy --name demo --team my-team" in entry
    )


def test_ssh_config_proxy_reaches_the_boxs_host():
    entry = ssh_config("demo", "fal", host="api.example.test")

    assert "dev ssh-proxy --name demo --host api.example.test" in entry
    assert "--host" not in ssh_config("demo", "fal")


def test_box_builds_credentials_once_across_many_calls(monkeypatch):
    calls = []
    credentials = MagicMock()
    credentials.to_headers.return_value = {"Authorization": "Bearer t"}
    monkeypatch.setattr(
        "fal.sdk.get_credentials", lambda team: calls.append(team) or credentials
    )
    box = make_box()

    assert [box.auth_headers() for _ in range(20)] == [
        {"Authorization": "Bearer t"}
    ] * 20
    assert calls == ["my-team"]


def test_start_never_stops_a_box_that_was_already_up():
    box = make_box([make_runner("r1")])
    box.client.apps.scale.side_effect = RuntimeError("transient")
    box.stop = MagicMock()

    with pytest.raises(RuntimeError):
        box.start(timeout=1)
    box.stop.assert_not_called()


def test_deploy_rejects_an_unsupported_python(monkeypatch):
    monkeypatch.setattr(box_module.sys, "version_info", (3, 14, 0))
    box = make_box([])

    with pytest.raises(DevError, match="Python 3.10 to 3.13"):
        box.deploy()
    box.client.deploy.assert_not_called()


def test_box_name_must_match_fully():
    assert config.NAME_PATTERN.fullmatch("demo")
    assert not config.NAME_PATTERN.fullmatch("demo\n")


def test_hold_reports_a_kept_longer_hold():
    box = make_box([make_runner("r1")])
    box.run = MagicMock(return_value=(config.HOLD_KEPT_EXIT, ""))

    assert box.hold(1) is False
    box.run.return_value = (0, "")
    assert box.hold(1) is True
    box.run.return_value = (2, "sh: boom")
    with pytest.raises(DevError, match="boom"):
        box.hold(1)


def test_volume_name_fits_the_controller_limit():
    from fal.volumes import canonical_volume_name

    assert config.volume_name("demo") == "devbox-demo"
    long_a, long_b = "a" * 40, "a" * 39 + "b"
    for name in (long_a, long_b, "x" * 25, "ab-" + "c" * 37):
        volume = config.volume_name(name)
        assert len(volume) <= config.VOLUME_NAME_MAX
        assert canonical_volume_name(volume) == volume
    assert config.volume_name(long_a) != config.volume_name(long_b)
    assert config.volume_name(long_a) == config.volume_name(long_a)


def test_deploy_mounts_the_box_volume(monkeypatch):
    box = make_box(min_concurrency=0)
    box.stop = MagicMock(return_value=True)
    seen = {}
    box.client.deploy.side_effect = lambda *a, **k: seen.update(
        volume=box_module.os.environ[config.ENV_VOLUME]
    )

    message = box.deploy()

    box.client.volumes.create.assert_called_once_with("devbox-demo")
    assert seen["volume"] == "devbox-demo"
    assert "volume devbox-demo" in message


def test_deploy_with_team_data_skips_the_volume():
    box = make_box(min_concurrency=0)
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

    box = make_box()
    box.client.volumes.create.side_effect = VolumeAlreadyExistsError("exists")

    assert box.ensure_volume() == "devbox-demo"


def test_ensure_volume_points_ineligible_teams_at_team_data():
    from fal.exceptions.volumes import VolumeAccountNotEligibleError

    box = make_box()
    box.client.volumes.create.side_effect = VolumeAccountNotEligibleError("no")

    with pytest.raises(DevError, match="--team-data"):
        box.ensure_volume()


def test_deploy_waits_for_a_runner_still_saving(monkeypatch):
    box = make_box([make_runner("r1", state=RunnerState.DRAINING)], min_concurrency=0)
    monkeypatch.setattr(box_module, "POLL_SECONDS", 0)
    monkeypatch.setattr(config, "LEASE_WAIT_SECONDS", 0)

    with pytest.raises(DevError, match="still saving"):
        box.deploy()

    box.client.deploy.assert_not_called()
    box.client.runners.stop.assert_not_called()


def test_deploy_of_a_new_box_does_not_look_up_its_runners():
    box = make_box(deployed=False)

    def runners(app, since=None):
        if not box.client.deploy.called:
            raise RuntimeError("Application not found")
        return []

    box.client.apps.runners.side_effect = runners

    box.deploy()

    box.client.deploy.assert_called_once()


def test_ssh_proxy_retries_the_runner_and_arming_lookups_together(monkeypatch):
    monkeypatch.setattr(box_module.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        box_module,
        "shell_session",
        lambda *a, **k: k["stdout"].write(box_module.SSH_READY_MARKER) or 0,
    )
    monkeypatch.setattr(box_module.sys, "stdout", MagicMock())
    box = make_box([make_runner("r1")])
    box.client.apps.list.side_effect = [
        RuntimeError("blip"),
        box.client.apps.list.return_value,
    ]

    assert box.ssh_proxy(timeout=1) == 0


def test_save_work_failure_says_the_save_may_still_run():
    box = make_box([make_runner("r1")])
    box.run = MagicMock(side_effect=DevError("timed out"))

    with pytest.raises(DevError, match="may still be in progress.*--no-save"):
        box.save_work()


def test_run_starts_each_attempt_with_an_empty_buffer(monkeypatch):
    import grpc

    class Expired(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNAUTHENTICATED

    box = make_box()
    calls = []

    def session(client, runner_id, command, **kwargs):
        calls.append(1)
        kwargs["stdout"].write(b"partial")
        if len(calls) == 1:
            raise Expired()
        return 0

    monkeypatch.setattr(box_module, "shell_session", session)

    assert box.run("r1", ["ls"]) == (0, "partial")


def test_a_new_box_defaults_to_the_rtx_pro_6000_not_an_h100():
    box = make_box(deployed=False)
    box.ensure_volume = MagicMock(return_value="devbox-demo")
    box.stop = MagicMock(return_value=True)
    seen = {}
    box.client.deploy.side_effect = lambda *a, **k: seen.update(
        machine=box_module.os.environ[config.ENV_MACHINE_TYPE]
    )

    message = box.deploy()

    assert seen["machine"] == "GPU-RTXPRO6000"
    assert "1x GPU-RTXPRO6000" in message


def test_local_public_keys_reads_ssh_dir(monkeypatch, tmp_path):
    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    (ssh / "id_ed25519.pub").write_text("ssh-ed25519 AAAA me@laptop\n")
    (ssh / "notes.pub").write_text("not a key\n")
    (ssh / "id_ed25519-cert.pub").write_text("ssh-ed25519-cert-v01@openssh.com C\n")
    (ssh / "id_ed25519").write_text("PRIVATE")
    monkeypatch.setattr(box_module.Path, "home", lambda: tmp_path)

    assert box_module.local_public_keys() == ["ssh-ed25519 AAAA me@laptop"]


def test_ssh_proxy_authorizes_the_local_keys_before_sshd(monkeypatch):
    monkeypatch.setattr(box_module, "local_public_keys", lambda: ["ssh-ed25519 K"])
    seen = []

    def session(client, runner_id, command, **kwargs):
        seen.append(command)
        kwargs["stdout"].write(box_module.SSH_READY_MARKER)
        return 0

    monkeypatch.setattr(box_module, "shell_session", session)
    monkeypatch.setattr(box_module.sys, "stdout", MagicMock())

    make_box([make_runner("r1")]).ssh_proxy(timeout=1)

    command = seen[0]
    assert command[-1] == "ssh-ed25519 K" and command[-2] == "sh"
    script = command[2]
    assert script.index("authorized_keys") < script.index("sshd -i")


def test_the_authorize_script_is_idempotent(tmp_path):
    import subprocess

    script = box_module.AUTHORIZE_KEYS.replace("/root/.ssh", str(tmp_path / ".ssh"))
    for _ in range(2):
        subprocess.run(
            ["sh", "-c", script, "sh", "ssh-ed25519 K1", "ssh-ed25519 K2"], check=True
        )

    keys = (tmp_path / ".ssh" / "authorized_keys").read_text().splitlines()
    assert keys == ["ssh-ed25519 K1", "ssh-ed25519 K2"]
    assert oct((tmp_path / ".ssh" / "authorized_keys").stat().st_mode)[-3:] == "600"


def test_the_authorize_script_never_fails_when_the_dir_is_unwritable(tmp_path):
    import subprocess

    blocker = tmp_path / "file"
    blocker.write_text("x")
    script = box_module.AUTHORIZE_KEYS.replace("/root/.ssh", str(blocker / ".ssh"))

    result = subprocess.run(
        ["sh", "-c", script + " && echo ready", "sh", "ssh-ed25519 K"],
        capture_output=True,
        check=False,
    )

    assert result.stdout.strip() == b"ready"


def test_the_authorize_script_keeps_a_key_file_without_a_final_newline(tmp_path):
    import subprocess

    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    (ssh / "authorized_keys").write_text("ssh-ed25519 OLD")
    script = box_module.AUTHORIZE_KEYS.replace("/root/.ssh", str(ssh))

    subprocess.run(["sh", "-c", script, "sh", "ssh-ed25519 NEW"], check=True)

    keys = (ssh / "authorized_keys").read_text().splitlines()
    assert keys == ["ssh-ed25519 OLD", "ssh-ed25519 NEW"]
