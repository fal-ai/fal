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
from fal.dev.cli.common import default_name, remember_name, resolve_name
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
    assert not args.starship
    assert not args.no_save
    assert args.dockerfile is None
    assert not args.team_data
    assert parse_args(["dev", "deploy", "--team-data"]).team_data


def test_dev_box_commands_run_their_handlers(monkeypatch):
    up = _run(monkeypatch, "box", "up", ["dev", "up", "--machine-type", "GPU-B200"])
    assert up.machine_type == "GPU-B200"
    assert not up.starship
    assert parse_args(["dev", "up", "--starship"]).starship
    assert parse_args(["dev", "deploy", "--starship"]).starship
    assert _run(monkeypatch, "box", "down", ["dev", "down", "--no-save"]).no_save
    proxy = _run(monkeypatch, "box", "ssh_proxy", ["dev", "ssh-proxy", "--name", "k"])
    assert proxy.name == "k"
    assert _run(monkeypatch, "box", "hold", ["dev", "hold", "2.5"]).hours == 2.5
    assert _run(monkeypatch, "box", "hold", ["dev", "hold", "--off"]).off
    assert _run(monkeypatch, "box", "hold", ["dev", "hold", "0"]).hours == 0
    assert _run(monkeypatch, "box", "down", ["dev", "down", "--yes"]).yes


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
    "flags",
    [["--prompt", "a fox"], ["-p", "a=1"], ["-o", "x.png"], ["--watch", "-p", "a=1"]],
)
def test_dev_hot_sync_rejects_request_flags_without_watch_run(flags, capsys):
    from fal.dev.cli import hot as hot_commands

    args = parse_args(["dev", "hot", "sync", "./proj", *flags])
    with pytest.raises(FalParserExit):
        hot_commands.cmd_sync(args)
    assert "only apply with --watch --run" in capsys.readouterr().err


def test_dev_hot_sync_rejects_run_without_watch(capsys):
    from fal.dev.cli import hot as hot_commands

    args = parse_args(["dev", "hot", "sync", "./proj", "--run"])
    with pytest.raises(FalParserExit):
        hot_commands.cmd_sync(args)
    assert "--run needs --watch" in capsys.readouterr().err


def test_dev_exec_help_explains_shell_syntax_and_background_jobs(capsys):
    with pytest.raises(FalParserExit):
        parse_args(["dev", "exec", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "exec -- bash -c 'df -h /work && nvidia-smi'" in out
    assert "not buffered" in out
    assert "nohup python train.py" in out


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
    monkeypatch.setenv(config.ENV_DEFAULT_NAME, "demo")

    assert default_name() == "demo"


def test_default_name_from_user(monkeypatch):
    monkeypatch.setattr("getpass.getuser", lambda: "Jane.Doe")

    assert default_name() == "jane-doe"


def _ns(name=None, host="api.example", team=None):
    return argparse.Namespace(name=name, host=host, team=team)


def test_resolve_name_uses_the_remembered_name_per_folder(monkeypatch, tmp_path):
    monkeypatch.setattr("fal.dev.cli.common.default_name", lambda: "someone")

    assert resolve_name(_ns()) == "someone"
    assert resolve_name(_ns(name="demo")) == "demo"
    # An explicit name alone is not remembered: only a confirmed box is.
    assert resolve_name(_ns()) == "someone"
    remember_name(_ns(), "demo")
    assert resolve_name(_ns()) == "demo"

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
    execs = []
    monkeypatch.setattr(
        box_commands.execute, "run", lambda *args: execs.append(args) or 0
    )
    shell = parse_args(["dev", "shell"])
    shell.console = MagicMock()
    exec_ = parse_args(["dev", "exec", "--", "nvidia-smi"])
    exec_.console = MagicMock()

    assert box_commands.cmd_shell(shell) == 0
    assert box_commands.cmd_exec(exec_) == 0

    ((args, kwargs),) = sessions
    assert args == (fake_box.client, "r1", None)
    assert (kwargs["interactive"], kwargs["remote_tty"]) == (True, True)
    assert execs == [(fake_box.client, "r1", ["nvidia-smi"], exec_.console)]


def test_dev_logs_follow_the_live_runner(monkeypatch, fake_box):
    tail_lines = [{"message": "a"}, {"message": "b"}]
    fake_box.logs.return_value = tail_lines
    shown = MagicMock()
    monkeypatch.setattr(box_commands, "LogPrinter", lambda console: shown)
    printed = []
    monkeypatch.setattr(
        box_commands, "print_logs", lambda *args, **kw: printed.append((args, kw))
    )
    args = parse_args(["dev", "logs"])
    args.console = MagicMock()

    box_commands.cmd_logs(args)

    ((follow_args, follow_kwargs),) = printed
    assert follow_args == (fake_box.client, "r1", args.console)
    assert follow_kwargs["follow"] is True and follow_kwargs["since"] is not None
    # The tail is the newest lines the API returned, oldest first.
    assert [c.args[0] for c in shown.print.call_args_list] == tail_lines
    fake_box.logs.assert_called_once()
    assert fake_box.logs.call_args.args[:2] == ("r1", box_commands.FOLLOW_BACKLOG_LINES)


def test_dev_logs_on_a_stopped_box_points_at_all(fake_box):
    from fal.dev.errors import DevError

    fake_box.require_runner.side_effect = DevError("devbox-k is not running.")
    args = parse_args(["dev", "logs"])
    args.console = MagicMock()

    with pytest.raises(DevError, match="not running.*fal dev logs --all"):
        box_commands.cmd_logs(args)


def test_dev_logs_all_picks_the_latest_started_not_the_shortest_lived(
    monkeypatch, fake_box
):
    from datetime import datetime, timezone

    printed = []
    monkeypatch.setattr(
        box_commands, "print_logs", lambda *args, **kw: printed.append(args)
    )
    # The first runner lived 22 s; the latest lived 13 min: uptime sorts them wrong.
    fake_box.runners.return_value = [
        _runner("first", RunnerState.TERMINATED, 22),
        _runner("latest", RunnerState.TERMINATED, 780),
    ]
    starts = {
        "first": datetime(2026, 11, 6, 0, 13, tzinfo=timezone.utc),
        "latest": datetime(2026, 11, 6, 0, 40, tzinfo=timezone.utc),
    }
    monkeypatch.setattr(
        box_commands,
        "get_runner_info",
        lambda rest, runner_id: MagicMock(started_at=starts[runner_id]),
    )
    args = parse_args(["dev", "logs", "--all"])
    args.console = MagicMock()

    box_commands.cmd_logs(args)

    assert printed[0][1] == "latest"


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


def _named_box(exists):
    box = MagicMock(app="devbox-typo")
    box.name = "typo"
    box.app_info.return_value = MagicMock() if exists else None
    return box


def _patch_get_box(monkeypatch, box):
    from fal.dev.cli import common

    monkeypatch.setattr(common, "resolve_account", lambda *a, **k: MagicMock())
    monkeypatch.setattr(common, "announce", lambda account: None)
    monkeypatch.setattr(common, "Box", lambda *a, **k: box)
    return common


def test_a_name_that_matches_no_box_is_not_remembered(monkeypatch, tmp_path):
    common = _patch_get_box(monkeypatch, _named_box(exists=False))
    monkeypatch.setattr(common, "default_name", lambda: "someone")

    common.get_box(_ns(name="typo"))

    assert resolve_name(_ns()) == "someone"


def test_a_name_that_matches_a_box_is_remembered(monkeypatch):
    common = _patch_get_box(monkeypatch, _named_box(exists=True))

    common.get_box(_ns(name="typo"))

    assert resolve_name(_ns()) == "typo"


def test_a_failed_lookup_does_not_remember_the_name(monkeypatch):
    box = _named_box(exists=True)
    box.app_info.side_effect = RuntimeError("api down")
    common = _patch_get_box(monkeypatch, box)
    monkeypatch.setattr(common, "default_name", lambda: "someone")

    common.get_box(_ns(name="typo"))

    assert resolve_name(_ns()) == "someone"


def test_creating_commands_remember_the_name_only_after_they_succeed(monkeypatch):
    box = _named_box(exists=False)
    box.deploy.side_effect = RuntimeError("deploy failed")
    _patch_get_box(monkeypatch, box)
    monkeypatch.setattr(box_commands, "get_box", lambda args, **kw: box)
    monkeypatch.setattr(box_commands, "_confirm_deploy", lambda *a: None)
    args = parse_args(["dev", "deploy", "--name", "typo"])

    with pytest.raises(RuntimeError):
        box_commands.cmd_deploy(args)
    assert resolve_name(parse_args(["dev", "status"])) != "typo"

    box.deploy.side_effect = None
    box.deploy.return_value = "ok"
    box_commands.cmd_deploy(args)
    assert resolve_name(parse_args(["dev", "status"])) == "typo"


def test_up_says_when_it_creates_a_box_from_a_remembered_name(monkeypatch, capsys):
    remember_name(parse_args(["dev", "up"]), "typo")
    box = _named_box(exists=False)
    monkeypatch.setattr(box_commands, "get_box", lambda args, **kw: box)
    monkeypatch.setattr(box_commands, "_confirm_deploy", lambda *a: None)
    box.up.return_value = MagicMock(runner_id="r1", machine_type="GPU-RTXPRO6000")
    args = parse_args(["dev", "up"])
    args.console = MagicMock()

    box_commands.cmd_up(args)

    err = capsys.readouterr().err
    assert "no box named typo exists yet (remembered for this folder)" in err
    box.up.assert_called_once()


def test_ssh_proxy_reports_the_failure_reason_on_stderr(monkeypatch, capsys):
    box = MagicMock()
    box.ssh_proxy.side_effect = RuntimeError("fal API unreachable")
    monkeypatch.setattr(box_commands, "get_box", lambda *a, **k: box)
    monkeypatch.setattr(box_commands, "_cleanup_on_hangup", lambda: None)

    code = box_commands.cmd_ssh_proxy(parse_args(["dev", "ssh-proxy", "--name", "k"]))

    assert code == 1
    assert "ssh-proxy failed: fal API unreachable" in capsys.readouterr().err


def test_ssh_proxy_leaves_dev_errors_to_the_error_panel(monkeypatch):
    from fal.dev.errors import DevError

    box = MagicMock()
    box.ssh_proxy.side_effect = DevError("devbox-k did not start")
    monkeypatch.setattr(box_commands, "get_box", lambda *a, **k: box)
    monkeypatch.setattr(box_commands, "_cleanup_on_hangup", lambda: None)

    with pytest.raises(DevError, match="did not start"):
        box_commands.cmd_ssh_proxy(parse_args(["dev", "ssh-proxy", "--name", "k"]))


def test_dev_hold_off_rejects_hours(monkeypatch):
    args = parse_args(["dev", "hold", "--off", "2"])
    monkeypatch.setattr(box_commands, "get_box", lambda a: pytest.fail("no box"))
    with pytest.raises(FalParserExit):
        box_commands.cmd_hold(args)


def test_dev_hold_without_hours_or_off_is_a_usage_error(monkeypatch):
    args = parse_args(["dev", "hold"])
    monkeypatch.setattr(box_commands, "get_box", lambda a: pytest.fail("no box"))
    with pytest.raises(FalParserExit):
        box_commands.cmd_hold(args)


def test_dev_list_status_and_destroy_parse(monkeypatch):
    listed = _run(monkeypatch, "box", "list", ["dev", "list", "--team", "t", "--json"])
    assert listed.json
    assert listed.team == "t"
    assert not hasattr(listed, "name")
    assert _run(monkeypatch, "box", "status", ["dev", "status", "--json"]).json
    assert not _run(monkeypatch, "box", "status", ["dev", "status"]).json
    destroy = _run(monkeypatch, "box", "destroy", ["dev", "destroy", "--name", "x"])
    assert destroy.name == "x"
    assert not destroy.yes
    assert parse_args(["dev", "destroy", "--yes"]).yes
    with pytest.raises(FalParserExit):
        parse_args(["dev", "destroy", "--keep-data"])


def _destroy_box(monkeypatch, exists=True, app="devbox-gone"):
    box = MagicMock(app=app, host=None)
    box.name = app.removeprefix("devbox-")
    box.app_info.return_value = MagicMock() if exists else None
    box.destroy.return_value = (["r1", "r2"], True)

    def get_box(args, **kw):
        assert kw == {"remember": False, "remember_team": False}
        return box

    monkeypatch.setattr(box_commands, "get_box", get_box)
    return box


def _destroy_args(**values):
    args = parse_args(["dev", "destroy", "--name", "gone"])
    args.console = MagicMock()
    for key, value in values.items():
        setattr(args, key, value)
    return args


def test_destroy_with_yes_deletes_forgets_and_says_the_volume_is_kept(
    monkeypatch, tmp_path
):
    box = _destroy_box(monkeypatch)
    args = _destroy_args(yes=True)
    state.update_box("devbox-gone", args.host, team="t")
    state.update(tmp_path, args.host, name="gone")

    box_commands.cmd_destroy(args)

    box.destroy.assert_called_once()
    assert state.load_box("devbox-gone", args.host) == {}
    assert "name" not in state.load(tmp_path, args.host)
    said = " ".join(str(c.args[0]) for c in args.console.print.call_args_list)
    assert "killed 2 runner" in said
    assert "volume devbox-gone is kept (no delete API yet)" in said
    assert "fal dev up --name gone" in said


def test_destroy_with_yes_needs_an_explicit_name(monkeypatch):
    from fal.dev.errors import DevError

    box = _destroy_box(monkeypatch)

    with pytest.raises(DevError, match="--yes needs --name"):
        box_commands.cmd_destroy(_destroy_args(yes=True, name=None))
    box.destroy.assert_not_called()


def test_list_does_not_remember_the_team(monkeypatch):
    from fal.dev import inventory

    seen = {}

    def fake_resolve(team, app, host, *, remember=True):
        seen["remember"] = remember
        return MagicMock(fal_team=team, runs_as="team t")

    monkeypatch.setattr(box_commands, "resolve_account", fake_resolve)
    monkeypatch.setattr(inventory, "list_boxes", lambda *a, **k: [])
    monkeypatch.setattr(box_commands, "SyncServerlessClient", MagicMock())
    args = parse_args(["dev", "list", "--team", "t"])
    args.console = MagicMock()

    box_commands.cmd_list(args)

    assert seen["remember"] is False


def test_destroy_without_a_terminal_requires_yes(monkeypatch):
    from fal.dev.errors import DevError

    box = _destroy_box(monkeypatch)
    monkeypatch.setattr(box_commands.sys, "stdin", MagicMock(isatty=lambda: False))

    with pytest.raises(DevError, match="--yes"):
        box_commands.cmd_destroy(_destroy_args())
    box.destroy.assert_not_called()


@pytest.mark.parametrize(
    "typed, deleted", [("gone", True), ("nope", False), ("  gone ", True)]
)
def test_destroy_on_a_terminal_needs_the_box_name_typed(monkeypatch, typed, deleted):
    from fal.dev.errors import DevError

    box = _destroy_box(monkeypatch)
    monkeypatch.setattr(box_commands.sys, "stdin", MagicMock(isatty=lambda: True))
    monkeypatch.setattr(box_commands.Prompt, "ask", lambda *a, **k: typed)

    if deleted:
        box_commands.cmd_destroy(_destroy_args())
        box.destroy.assert_called_once()
    else:
        with pytest.raises(DevError, match="cancelled"):
            box_commands.cmd_destroy(_destroy_args())
        box.destroy.assert_not_called()


def test_destroy_refuses_what_is_not_a_devbox(monkeypatch):
    from fal.dev.errors import DevError

    box = _destroy_box(monkeypatch, app="dev-other")

    with pytest.raises(DevError, match="not a dev box"):
        box_commands.cmd_destroy(_destroy_args(yes=True))
    box.destroy.assert_not_called()


def test_destroy_of_a_missing_box_is_an_error(monkeypatch):
    from fal.dev.errors import DevError

    box = _destroy_box(monkeypatch, exists=False)

    with pytest.raises(DevError, match="no box named"):
        box_commands.cmd_destroy(_destroy_args(yes=True))
    box.destroy.assert_not_called()


def test_box_destroy_refuses_non_devbox_apps_and_keeps_the_volume(monkeypatch):
    from fal.dev.app import deploy
    from fal.dev.errors import DevError
    from tests.unit.dev.fakes import make_box

    box = make_box()
    calls = []
    monkeypatch.setattr(
        deploy, "delete_app", lambda *a, **k: calls.append(a) or (["r"], True)
    )

    assert box.destroy() == (["r"], True)
    assert calls[0][0] == "devbox-demo"
    box.app = "dev-other"
    with pytest.raises(DevError, match="not a dev box"):
        box.destroy()


def test_down_points_at_destroy(monkeypatch):
    box = MagicMock()
    monkeypatch.setattr(box_commands, "get_box", lambda args: box)
    args = _destroy_args(yes=True, no_save=False)

    box_commands.cmd_down(args)

    assert "fal dev destroy" in args.console.print.call_args.args[0]


def test_status_json_prints_the_dash_snapshot_and_exits_zero(monkeypatch):
    from fal.dev.cli import summary

    box = MagicMock()
    monkeypatch.setattr(box_commands, "get_box", lambda args: box)
    snap = MagicMock()
    monkeypatch.setattr(summary, "snapshot", lambda b: snap)
    monkeypatch.setattr(
        "fal.dev.cli.dash.payload", lambda b, s: '{"info": {"state": "stopped"}}'
    )
    args = parse_args(["dev", "status", "--json"])
    args.console = MagicMock()

    assert box_commands.cmd_status(args) is None
    args.console.print_json.assert_called_once_with('{"info": {"state": "stopped"}}')


def test_list_prints_json_with_the_remembered_box(monkeypatch):
    import json

    from fal.dev import inventory

    row = inventory.BoxRow(
        "a", "devbox-a", "STOPPED", "GPU-H100", None, None, True, "devbox-a"
    )
    other = inventory.BoxRow(
        "b", "devbox-b", "STOPPED", None, None, None, True, "devbox-b"
    )
    monkeypatch.setattr(inventory, "list_boxes", lambda *a, **k: [row, other])
    monkeypatch.setattr(box_commands, "SyncServerlessClient", MagicMock())
    monkeypatch.setattr(box_commands, "remembered_name", lambda args: "b")
    args = parse_args(["dev", "list", "--json"])
    args.console = MagicMock()

    box_commands.cmd_list(args)

    out = json.loads(args.console.print_json.call_args.args[0])
    assert [(r["name"], r["remembered"]) for r in out] == [("a", False), ("b", True)]


def test_hold_until_reads_the_hold_and_raises_on_garbage(monkeypatch):
    import time as time_module

    from fal.dev.errors import DevError
    from tests.unit.dev.fakes import make_box, make_runner

    box = make_box([make_runner("r1")])
    outputs = iter(
        [
            (0, f"{time_module.time() + 600}\n"),
            (0, "0\n"),
            (0, "oops"),
            (1, "sh: error"),
        ]
    )
    monkeypatch.setattr(box, "run", lambda *a, **k: next(outputs))

    assert box.hold_until() > time_module.time()
    assert box.hold_until() is None
    for _ in range(2):
        with pytest.raises(DevError, match="could not read the hold"):
            box.hold_until()
