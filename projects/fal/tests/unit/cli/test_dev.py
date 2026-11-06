import argparse
import importlib
import subprocess
import sys
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from fal.cli.main import _get_main_parser, parse_args
from fal.cli.parser import FalParserExit
from fal.dev import config, state
from fal.dev.cli import box as box_commands
from fal.dev.cli.common import default_name, resolve_name
from fal.sdk import RunnerState

pytestmark = pytest.mark.usefixtures("dev_account")


@pytest.fixture(autouse=True)
def state_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(config.ENV_DEFAULT_NAME, raising=False)


def _run(monkeypatch, module, handler, argv):
    """Parse argv and run the command, with its handler replaced by a recorder."""
    seen = []
    monkeypatch.setattr(f"fal.dev.cli.{module}.cmd_{handler}", seen.append)
    args = parse_args(argv)
    args.func(args)
    return seen[0]


def test_dev_deploy_defaults(monkeypatch):
    args = _run(monkeypatch, "box", "deploy", ["dev", "deploy"])

    assert args.name is None
    assert args.machine_type is None
    assert args.num_gpus == config.DEFAULT_NUM_GPUS
    assert args.idle_minutes == config.DEFAULT_IDLE_MINUTES
    assert not args.claude
    assert not args.no_save
    assert args.dockerfile is None
    assert not args.team_data
    assert parse_args(["dev", "deploy", "--team-data"]).team_data


def test_dev_box_commands_run_their_handlers(monkeypatch):
    up = _run(monkeypatch, "box", "up", ["dev", "up", "--machine-type", "GPU-B200"])
    assert up.machine_type == "GPU-B200"
    assert _run(monkeypatch, "box", "down", ["dev", "down", "--no-save"]).no_save
    proxy = _run(monkeypatch, "box", "ssh_proxy", ["dev", "ssh-proxy", "--name", "k"])
    assert proxy.name == "k"
    assert _run(monkeypatch, "box", "hold", ["dev", "hold", "2.5"]).hours == 2.5


def test_dev_gpu_flag_is_gone():
    with pytest.raises(FalParserExit):
        parse_args(["dev", "up", "--gpu", "GPU-B200"])


def test_dev_exec_keeps_command_after_separator(monkeypatch):
    args = _run(monkeypatch, "box", "exec", ["dev", "exec", "--", "nvidia-smi", "-L"])

    assert args.command[-2:] == ["nvidia-smi", "-L"]


def test_dev_accepts_team():
    args = parse_args(["dev", "status", "--team", "my-team"])

    assert args.team == "my-team"


@pytest.mark.parametrize("name", ["Bad", "-x", "a" * 41, "has_underscore"])
@pytest.mark.parametrize("command", ["status", "dash", "hot status"])
def test_dev_rejects_invalid_box_names(command, name):
    with pytest.raises(FalParserExit):
        parse_args(["dev", *command.split(), "--name", name])


def test_dev_hot_subcommands_take_only_their_flags(monkeypatch):
    args = _run(
        monkeypatch,
        "hot",
        "sync",
        ["dev", "hot", "sync", "./proj", "--watch", "--run", "-p", "steps=4"],
    )
    assert args.local == "./proj"
    assert args.watch and args.run
    assert args.param == ["steps=4"]

    args = _run(monkeypatch, "hot", "run", ["dev", "hot", "run", "--prompt", "a fox"])
    assert args.prompt == "a fox"

    for argv in (
        ["dev", "hot", "run", "--watch"],
        ["dev", "hot", "status", "./proj"],
        ["dev", "hot", "logs", "-p", "a=1"],
        ["dev", "hot", "reload"],
    ):
        with pytest.raises(FalParserExit):
            parse_args(argv)


@pytest.mark.parametrize(
    "action", ["start", "sync", "run", "status", "logs", "restart", "stop"]
)
def test_dev_hot_actions_run_their_handlers(monkeypatch, action):
    _run(monkeypatch, "hot", action, ["dev", "hot", action])


def test_dev_group_includes_app_and_dash(monkeypatch):
    assert _run(monkeypatch, "dash", "dash", ["dev", "dash", "--once"]).once
    assert _run(monkeypatch, "app", "status", ["dev", "app", "status"]).name is None
    assert _run(monkeypatch, "box", "up", ["dev", "up", "--dash"]).dash


def test_dev_names_are_flags_everywhere(monkeypatch):
    assert _run(monkeypatch, "dash", "dash", ["dev", "dash", "--name", "k"]).name == "k"
    assert _run(monkeypatch, "app", "logs", ["dev", "app", "logs", "--name", "x"]).name
    save = _run(
        monkeypatch, "app", "request", ["dev", "app", "request", "/", "--save", "base"]
    )
    assert (save.name, save.save) == (None, "base")
    for argv in (["dev", "dash", "k"], ["dev", "app", "logs", "-s", "x"]):
        with pytest.raises(FalParserExit):
            parse_args(argv)


def test_default_name_from_env(monkeypatch):
    monkeypatch.setenv(config.ENV_DEFAULT_NAME, "klein")

    assert default_name() == "klein"


def test_default_name_from_user(monkeypatch):
    monkeypatch.setattr("getpass.getuser", lambda: "Jane.Doe")

    assert default_name() == "jane-doe"


def _ns(name=None, host="api.example", team=None):
    return argparse.Namespace(name=name, host=host, team=team)


def test_resolve_name_remembers_explicit_name_per_folder(monkeypatch, tmp_path):
    monkeypatch.setattr("fal.dev.cli.common.default_name", lambda: "someone")

    assert resolve_name(_ns()) == "someone"
    assert resolve_name(_ns(name="klein")) == "klein"
    assert resolve_name(_ns()) == "klein"

    # Another folder or host does not see it.
    assert resolve_name(_ns(host="other.example")) == "someone"
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    assert resolve_name(_ns()) == "someone"


def test_building_the_parsers_imports_no_handler_module():
    """`fal` builds every parser on each run, so `fal dev` must stay cheap."""
    script = (
        "import sys\n"
        "from fal.cli.main import parse_args\n"
        "commands = [\n"
        "    'deploy', 'up', 'down', 'status', 'shell', 'exec', 'logs', 'hold',\n"
        "    'ssh-config', 'ssh-proxy', 'hot start', 'hot sync', 'hot run',\n"
        "    'hot status', 'hot logs', 'hot restart', 'hot stop', 'dash',\n"
        "    'app start', 'app sync', 'app request', 'app benchmark compare',\n"
        "    'app benchmark list', 'app profile', 'app deps install',\n"
        "    'app deps uninstall', 'app deps reset', 'app deps status',\n"
        "    'app restart', 'app logs', 'app status', 'app stop',\n"
        "]\n"
        "for command in commands:\n"
        "    try:\n"
        "        parse_args(['dev', *command.split(), '--help'])\n"
        "    except BaseException as exc:\n"
        "        assert type(exc).__name__ == 'FalParserExit', (command, exc)\n"
        "    else:\n"
        "        raise AssertionError(command)\n"
        "heavy = [m for m in sys.modules if m.startswith('fal.dev.') and m not in (\n"
        "    'fal.dev.config', 'fal.dev.cli', 'fal.dev.cli.parser', 'fal.dev.errors',\n"
        "    'fal.dev.remote', 'fal.dev.remote.protocol')]\n"
        "assert not heavy, heavy\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


def _leaf_parsers(parser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                yield from _leaf_parsers(child)
    if parser.get_default("func") is not None:
        yield parser


def test_every_dev_command_resolves_its_handler():
    dev = next(
        action.choices["dev"]
        for action in _get_main_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    leaves = list(_leaf_parsers(dev))
    assert len(leaves) > 30
    for leaf in leaves:
        module, handler = leaf.get_default("func").__name__.split(".")
        implementation = importlib.import_module(f"fal.dev.cli.{module}")
        assert callable(getattr(implementation, f"cmd_{handler}")), leaf.prog


def test_dev_dash_takes_no_app_or_idle_override():
    for flag in (["--app", "x"], ["--idle-minutes", "30"]):
        with pytest.raises(FalParserExit):
            parse_args(["dev", "dash", *flag])


def _runner(runner_id, state, uptime):
    runner = MagicMock(runner_id=runner_id, state=state)
    runner.uptime = timedelta(seconds=uptime)
    return runner


@pytest.fixture
def fake_box(monkeypatch):
    box = MagicMock(app="devbox-k", host=None)
    box.require_runner.return_value = "r1"
    monkeypatch.setattr(box_commands, "get_box", lambda args: box)
    return box


def test_dev_shell_and_exec_open_a_session_on_the_boxs_runner(monkeypatch, fake_box):
    sessions = []
    monkeypatch.setattr(
        box_commands,
        "shell_session",
        lambda *args, **kwargs: sessions.append((args, kwargs)) or 0,
    )
    shell = parse_args(["dev", "shell"])
    shell.console = MagicMock()
    exec_ = parse_args(["dev", "exec", "--", "nvidia-smi"])
    exec_.console = MagicMock()

    assert box_commands.cmd_shell(shell) == 0
    assert box_commands.cmd_exec(exec_) == 0

    (args, kwargs), (exec_args, _) = sessions
    assert args == (fake_box.client, "r1", None)
    assert (kwargs["interactive"], kwargs["remote_tty"]) == (True, True)
    assert exec_args == (fake_box.client, "r1", ["nvidia-smi"])


def test_dev_logs_follow_the_live_runner(monkeypatch, fake_box):
    printed = []
    monkeypatch.setattr(
        box_commands, "print_logs", lambda *args, **kw: printed.append((args, kw))
    )
    args = parse_args(["dev", "logs"])
    args.console = MagicMock()

    box_commands.cmd_logs(args)

    assert printed == [((fake_box.client, "r1", args.console), {"follow": True})]


def test_dev_logs_all_reads_the_runner_that_is_still_around(monkeypatch, fake_box):
    printed = []
    monkeypatch.setattr(
        box_commands, "print_logs", lambda *args, **kw: printed.append((args, kw))
    )
    fake_box.runners.return_value = [
        _runner("gone", RunnerState.TERMINATED, 5),
        _runner("backing-off", RunnerState.CRASH_BACKOFF, 900),
        _runner("older", RunnerState.PENDING, 1800),
    ]
    args = parse_args(["dev", "logs", "--all"])
    args.console = MagicMock()

    box_commands.cmd_logs(args)

    ((call_args, kwargs),) = printed
    assert call_args[1] == "backing-off"
    assert kwargs["follow"] is False
    assert kwargs["since"] is not None


def test_dev_ssh_config_proxy_keeps_the_host_and_team(monkeypatch, capsys):
    args = parse_args(
        ["dev", "ssh-config", "--name", "k", "--host", "api.test", "--team", "t"]
    )

    box_commands.cmd_ssh_config(args)

    entry = capsys.readouterr().out
    assert "dev ssh-proxy --name k --host api.test --team t" in entry
