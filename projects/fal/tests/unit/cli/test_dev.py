import argparse

import pytest

from fal.dev.cli.box import (
    HOT_ACTIONS,
    SUBCOMMANDS,
    _deploy,
    _down,
    _exec,
    _hot,
    _ssh_proxy,
    _up,
    default_name,
    resolve_name,
)
from fal.cli.main import parse_args
from fal.cli.parser import FalParserExit
from fal.dev import config, state


@pytest.fixture(autouse=True)
def state_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(config.ENV_DEFAULT_NAME, raising=False)


def test_dev_deploy_defaults():
    args = parse_args(["dev", "deploy"])

    assert args.func == _deploy
    assert args.name is None
    assert args.gpu is None
    assert args.num_gpus == config.DEFAULT_NUM_GPUS
    assert args.idle_minutes == config.DEFAULT_IDLE_MINUTES
    assert not args.claude
    assert not args.no_save
    assert args.dockerfile is None
    assert not args.team_data
    assert parse_args(["dev", "deploy", "--team-data"]).team_data


def test_dev_box_commands_registered():
    assert parse_args(["dev", "up", "--gpu", "GPU-B200"]).func == _up
    assert parse_args(["dev", "down", "--no-save"]).no_save
    assert parse_args(["dev", "down"]).func == _down
    assert parse_args(["dev", "ssh-proxy", "--name", "klein"]).func == _ssh_proxy
    assert parse_args(["dev", "hold", "2.5"]).hours == 2.5


def test_dev_exec_keeps_command_after_separator():
    args = parse_args(["dev", "exec", "--", "nvidia-smi", "-L"])

    assert args.func == _exec
    assert args.command[-2:] == ["nvidia-smi", "-L"]


def test_dev_accepts_team():
    args = parse_args(["dev", "status", "--team", "my-team"])

    assert args.team == "my-team"


@pytest.mark.parametrize("name", ["Bad", "-x", "a" * 41, "has_underscore"])
def test_dev_rejects_invalid_box_names(name):
    with pytest.raises(FalParserExit):
        parse_args(["dev", "status", "--name", name])


def test_dev_hot_parser():
    args = parse_args(
        ["dev", "hot", "sync", "./proj", "--watch", "--run", "-p", "steps=4"]
    )

    assert args.func == _hot
    assert args.action == "sync"
    assert args.local == "./proj"
    assert args.watch and args.run
    assert args.param == ["steps=4"]


def test_dev_hot_rejects_unknown_action():
    assert "start" in HOT_ACTIONS
    with pytest.raises(FalParserExit):
        parse_args(["dev", "hot", "reload"])


def test_subcommands_hook_registers_more_commands(monkeypatch):
    def add_extra(subparsers, parents):
        parser = subparsers.add_parser("extra", parents=parents)
        parser.set_defaults(func="extra")

    monkeypatch.setattr("fal.dev.cli.box.SUBCOMMANDS", [*SUBCOMMANDS, add_extra])

    assert parse_args(["dev", "extra"]).func == "extra"


def test_default_name_from_env(monkeypatch):
    monkeypatch.setenv(config.ENV_DEFAULT_NAME, "klein")

    assert default_name() == "klein"


def test_default_name_from_user(monkeypatch):
    monkeypatch.setattr("getpass.getuser", lambda: "Jane.Doe")

    assert default_name() == "jane-doe"


def _ns(name=None, host="api.example", team=None):
    return argparse.Namespace(name=name, host=host, team=team)


def test_resolve_name_remembers_explicit_name_per_folder(monkeypatch, tmp_path):
    monkeypatch.setattr("fal.dev.cli.box.default_name", lambda: "someone")

    assert resolve_name(_ns()) == "someone"
    assert resolve_name(_ns(name="klein")) == "klein"
    assert resolve_name(_ns()) == "klein"

    # Another folder, team or host does not see it.
    assert resolve_name(_ns(team="other")) == "someone"
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    assert resolve_name(_ns()) == "someone"


def test_dev_group_includes_app_and_dash():
    assert parse_args(["dev", "dash", "--once"]).once
    assert parse_args(["dev", "app", "status"]).cmd == "status"
    assert parse_args(["dev", "up", "--dash"]).dash


def test_dev_dash_rejects_invalid_box_names():
    with pytest.raises(FalParserExit):
        parse_args(["dev", "dash", "My_Box", "--once"])
