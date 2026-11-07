import argparse
from unittest.mock import MagicMock

import pytest

from dev.fakes import make_box
from fal.dev import config, guard, state
from fal.dev.account import Account
from fal.dev.errors import DevError


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    for var in (config.ENV_OWNER, config.ENV_ALLOWED_MACHINES, config.ENV_MAX_GPUS):
        monkeypatch.delenv(var, raising=False)


def _box(stored, mine="me"):
    box = make_box(min_concurrency=0)
    box.stop = MagicMock(return_value=True)
    box.owner_identity = lambda: mine
    box.stored_owner = lambda attempts=3: stored
    return box


def test_identity_prefers_the_override(monkeypatch):
    monkeypatch.setenv(config.ENV_OWNER, " alex ")
    assert guard.identity() == "alex"


def test_identity_is_the_personal_nickname(monkeypatch):
    monkeypatch.setattr("fal.auth.key_credentials", lambda: None)
    monkeypatch.setattr(
        "fal.dev.team.accounts",
        lambda: [
            {"nickname": "me", "is_personal": True},
            {"nickname": "red", "is_personal": False},
        ],
    )
    assert guard.identity() == "me"


def test_identity_falls_back_to_os_user_and_host_under_a_key(monkeypatch):
    monkeypatch.setattr("fal.auth.key_credentials", lambda: ("id", "secret"))
    monkeypatch.setattr(guard.getpass, "getuser", lambda: "troy")
    monkeypatch.setattr(guard.socket, "gethostname", lambda: "laptop")
    assert guard.identity() == "troy@laptop"


def test_identity_failure_is_an_error_not_a_stranger(monkeypatch):
    monkeypatch.setattr("fal.auth.key_credentials", lambda: None)

    def boom():
        raise OSError("offline")

    monkeypatch.setattr("fal.dev.team.accounts", boom)
    with pytest.raises(DevError, match="FAL_DEV_OWNER"):
        guard.identity()


def test_owner_strings_are_made_printable_and_bounded():
    assert guard.clean("al\x1b]0;x\x07ex") == "al]0;xex"
    assert len(guard.clean("a" * 500)) == guard.OWNER_MAX


def test_an_unnamed_account_is_not_checked_but_says_so():
    box = _box("alex")
    box.account = Account(None, False, via_key=True)
    box._say = MagicMock()
    guard.check_owner(box, take_over=False)
    assert "not checked" in box._say.call_args.args[0]


def test_the_warning_lookup_makes_one_attempt():
    box = _box("alex")
    seen = []
    box.stored_owner = lambda attempts=3: seen.append(attempts) or "alex"
    box._say = MagicMock()
    guard.warn_if_foreign(box, argparse.Namespace(host=None))
    assert seen == [1]


@pytest.mark.parametrize("command", ["cmd_shell", "cmd_exec", "cmd_ssh_proxy"])
def test_commands_into_a_box_warn_about_a_foreign_owner(monkeypatch, command):
    from fal.dev.cli import box as box_commands

    box = MagicMock()
    warned = []
    monkeypatch.setattr(box_commands, "get_box", lambda args, **kw: box)
    monkeypatch.setattr(box_commands, "warn_if_foreign", lambda b, a: warned.append(b))
    monkeypatch.setattr(box_commands, "_cleanup_on_hangup", lambda: None)
    monkeypatch.setattr(box_commands, "shell_session", lambda *a, **k: 0)
    monkeypatch.setattr(box_commands.execute, "run", lambda *a, **k: 0)
    args = argparse.Namespace(command=["--", "ls"], timeout=1, console=None)
    getattr(box_commands, command)(args)
    assert warned == [box]


def test_stored_owner_other_status_is_an_error(monkeypatch):
    import openapi_fal_rest.api.applications.app_metadata as app_metadata

    from fal.dev.box import Box

    box = make_box()
    box.client._create_rest_client = MagicMock()
    monkeypatch.setattr(
        app_metadata,
        "sync_detailed",
        lambda **kw: MagicMock(status_code=500, parsed=None),
    )
    with pytest.raises(DevError, match="HTTP 500"):
        Box.stored_owner(box)


def test_suggested_name_is_a_valid_box_name():
    name = guard.suggested_name("Troy.C@laptop", "dev")
    assert config.NAME_PATTERN.fullmatch(name)
    assert name == "troy-c-dev"
    assert config.NAME_PATTERN.fullmatch(guard.suggested_name("x" * 60, "dev"))


@pytest.mark.parametrize("stored", [None, "", "me"])
def test_owner_check_passes_for_new_legacy_or_own_boxes(stored):
    guard.check_owner(_box(stored), take_over=False)


def test_owner_check_refuses_a_teammates_box_and_suggests_a_name():
    with pytest.raises(DevError) as exc:
        guard.check_owner(_box("alex"), take_over=False)
    assert "belongs to alex" in str(exc.value)
    commands = [command for command, _ in exc.value.fixes]
    assert "fal dev deploy --name me-demo" in commands
    assert "fal dev deploy --name demo --take-over" in commands


def test_take_over_overrides_and_says_so():
    box = _box("alex")
    box._say = MagicMock()
    guard.check_owner(box, take_over=True)
    assert "taking over" in box._say.call_args.args[0]


def test_deploy_refuses_before_touching_anything():
    box = _box("alex")
    with pytest.raises(DevError):
        box.deploy()
    box.client.volumes.create.assert_not_called()
    box.client.deploy.assert_not_called()


def test_deploy_fails_on_an_unreadable_identity_before_stopping_anything():
    box = _box(None)

    def boom():
        raise DevError("no login")

    box.owner_identity = boom
    with pytest.raises(DevError, match="no login"):
        box.deploy()
    box.stop.assert_not_called()
    box.client.volumes.create.assert_not_called()


def test_deploy_stamps_the_owner_and_take_over_proceeds():
    box = _box("alex", mine="me")
    box.deploy(take_over=True)
    kwargs = box.client.deploy.call_args.kwargs
    assert kwargs["annotations"] == {config.OWNER_ANNOTATION: "me"}
    assert box.known_owner == "me"


def test_warning_for_a_teammates_box_appears_once():
    box = _box("alex")
    box._say = MagicMock()
    args = argparse.Namespace(host="api.example")
    guard.warn_if_foreign(box, args)
    guard.warn_if_foreign(box, args)
    assert box._say.call_count == 1
    assert "alex" in box._say.call_args.args[0]


def test_take_over_proceeds_when_the_owner_cannot_be_read():
    box = _box("alex")
    box.stored_owner = MagicMock(side_effect=DevError("endpoint down"))
    box._say = MagicMock()
    guard.check_owner(box, take_over=True)
    assert "could not read" in box._say.call_args.args[0]
    with pytest.raises(DevError, match="endpoint down"):
        guard.check_owner(box, take_over=False)


def test_a_recent_lookup_is_not_repeated():
    box = _box("me")
    args = argparse.Namespace(host=None)
    guard.warn_if_foreign(box, args)
    box.stored_owner = MagicMock()
    guard.warn_if_foreign(box, args)
    box.stored_owner.assert_not_called()


def test_a_warned_box_skips_the_lookup():
    box = _box("alex")
    box._say = MagicMock()
    args = argparse.Namespace(host=None)
    guard.warn_if_foreign(box, args)
    box.stored_owner = MagicMock()
    guard.warn_if_foreign(box, args)
    box.stored_owner.assert_not_called()


def test_a_failed_display_read_is_retried_not_remembered():
    box = _box("alex")
    box.stored_owner = MagicMock(side_effect=[DevError("down"), "alex"])
    assert box.owner() is None
    assert box.known_owner is None
    assert box.owner() is None  # backed off
    box._owner_retry_at = 0.0
    assert box.owner() == "alex"
    assert box.owner() == "alex"
    assert box.stored_owner.call_count == 2


def test_no_warning_for_your_own_box_or_a_failed_lookup():
    box = _box("me")
    box._say = MagicMock()
    guard.warn_if_foreign(box, argparse.Namespace(host=None))
    box.stored_owner = MagicMock(side_effect=DevError("down"))
    guard.warn_if_foreign(box, argparse.Namespace(host=None))
    box._say.assert_not_called()


def test_limits_are_off_by_default():
    guard.check_limits("GPU-B200", 64)


def test_allowed_machines(monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOWED_MACHINES, "GPU-A100, gpu-h100")
    guard.check_limits("GPU-H100")
    with pytest.raises(DevError, match="GPU-B200 is not allowed"):
        guard.check_limits("GPU-B200")


def test_max_gpus(monkeypatch):
    monkeypatch.setenv(config.ENV_MAX_GPUS, "2")
    guard.check_limits(None, 2)
    with pytest.raises(DevError, match="4 GPUs is over the limit of 2"):
        guard.check_limits(None, 4)


def test_max_gpus_must_be_a_number(monkeypatch):
    monkeypatch.setenv(config.ENV_MAX_GPUS, "lots")
    with pytest.raises(DevError, match="whole number"):
        guard.check_limits(None, 1)


def test_deploy_checks_limits_before_creating_anything(monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOWED_MACHINES, "GPU-H100")
    box = _box(None)
    with pytest.raises(DevError):
        box.deploy(machine_type="GPU-B200")
    box.client.volumes.create.assert_not_called()


def test_deploy_checks_the_current_machine_when_none_is_given(monkeypatch):
    monkeypatch.setenv(config.ENV_ALLOWED_MACHINES, "GPU-A100")
    box = _box(None)  # the existing app is on GPU-H100
    with pytest.raises(DevError, match="GPU-H100 is not allowed"):
        box.deploy()


def test_stored_owner_reads_the_annotation(monkeypatch):
    import openapi_fal_rest.api.applications.app_metadata as app_metadata

    box = make_box()
    box.client._create_rest_client = MagicMock()
    parsed = MagicMock(
        additional_properties={"annotations": {config.OWNER_ANNOTATION: "alex"}}
    )
    response = MagicMock(status_code=200, parsed=parsed)
    seen = {}
    monkeypatch.setattr(
        app_metadata,
        "sync_detailed",
        lambda **kw: seen.update(kw) or response,
    )
    from fal.dev.box import Box

    assert Box.stored_owner(box) == "alex"
    assert seen["app_user_id"] == "my-team"
    assert seen["app_alias_or_id"] == "devbox-demo"


def test_stored_owner_is_none_without_annotations_or_app(monkeypatch):
    import openapi_fal_rest.api.applications.app_metadata as app_metadata

    from fal.dev.box import Box

    box = make_box()
    box.client._create_rest_client = MagicMock()
    for status, parsed in ((200, MagicMock(additional_properties={})), (404, None)):
        monkeypatch.setattr(
            app_metadata,
            "sync_detailed",
            lambda parsed=parsed, status=status, **kw: MagicMock(
                status_code=status, parsed=parsed
            ),
        )
        assert Box.stored_owner(box) is None


def test_stored_owner_failure_is_an_error_not_a_pass(monkeypatch):
    import openapi_fal_rest.api.applications.app_metadata as app_metadata

    from fal.dev import box as box_module
    from fal.dev.box import Box

    monkeypatch.setattr(box_module.time, "sleep", lambda s: None)
    box = make_box()
    box.client._create_rest_client = MagicMock()
    monkeypatch.setattr(
        app_metadata, "sync_detailed", MagicMock(side_effect=OSError("flaky"))
    )
    with pytest.raises(DevError, match="could not read who owns"):
        Box.stored_owner(box)
