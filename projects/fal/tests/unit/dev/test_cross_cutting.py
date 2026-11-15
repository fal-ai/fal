"""Help, errors, JSON results, streams and exit codes shared by `fal dev` commands."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from rich.console import Console

from dev.fakes import make_box
from fal.cli.main import parse_args
from fal.cli.parser import FalParserExit
from fal.dev import state, team
from fal.dev.account import Account
from fal.dev.cli import box as box_cli
from fal.dev.cli import error_panel, summary
from fal.dev.errors import DevError


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "dev-state")
    monkeypatch.chdir(tmp_path)


def _help(argv, capsys) -> str:
    with pytest.raises(FalParserExit):
        parse_args([*argv, "--help"])
    return capsys.readouterr().out


def _console():
    out = io.StringIO()
    return Console(file=out, width=200, force_terminal=False), out


# --- top-level help ----------------------------------------------------------


def test_top_level_help_has_a_quick_start_and_lists_commands_in_journey_order(capsys):
    text = _help(["dev"], capsys)

    assert "Personal dev boxes (CPU or GPU)" in text
    listed = text.split("Commands:")[1].split("Quick start")[0]
    names = [line.split()[0] for line in listed.splitlines()[2:] if line.strip()]
    assert names == [
        *("up", "shell", "exec", "cp", "status", "list", "logs", "dash", "hold"),
        *("view", "hot", "down", "destroy"),
        *("deploy", "ssh-config", "ssh-proxy"),
    ]
    for line in (
        "fal dev up ",
        "fal dev shell ",
        "fal dev hot start ",
        "fal dev down ",
    ):
        assert line in text.split("Quick start")[1]
    # The proxy note and the animation note come after the commands.
    assert text.index("Corporate networks") > text.index("Quick start")
    assert text.index("FAL_DEV_NO_ANIMATION") > text.index("Quick start")
    assert "ssh-proxy" in listed and "not for direct use" in listed


def test_top_level_help_documents_exit_codes_and_name_and_team_rules(capsys):
    text = _help(["dev"], capsys)

    for code in ("0 ok", "2 usage", "3 box not found", "4 auth or team", "5 timeout"):
        assert code in text
    assert "130" in text
    assert "FAL_DEV_NAME" in text and "OS username" in text
    assert "personal" in text and "remembered for the box" in text


COMMANDS = [
    ["up"],
    ["down"],
    ["destroy"],
    ["logs"],
    ["status"],
    ["list"],
    ["shell"],
    ["exec"],
    ["ssh-config"],
    ["dash"],
    ["deploy"],
    ["hold"],
    ["cp"],
    ["hot"],
    ["hot", "start"],
    ["hot", "sync"],
    ["view"],
    ["ssh-proxy"],
]


@pytest.mark.parametrize("argv", COMMANDS, ids=" ".join)
def test_every_command_uses_one_help_formatter(argv, capsys):
    text = _help(["dev", *argv], capsys)

    assert text.startswith("Usage:")
    assert "usage:" not in text and "\noptions:" not in text


@pytest.mark.parametrize(
    "argv",
    [
        ["up"],
        ["down"],
        ["destroy"],
        ["logs"],
        ["status"],
        ["list"],
        ["shell"],
        ["exec"],
        ["ssh-config"],
        ["dash"],
        ["deploy"],
        ["hold"],
        ["cp"],
    ],
    ids=" ".join,
)
def test_commands_have_an_examples_block(argv, capsys):
    assert "examples:" in _help(["dev", *argv], capsys)


@pytest.mark.parametrize("argv", [["up"], ["down"], ["status"], ["exec"], ["hold"]])
def test_name_and_team_help_are_short(argv, capsys):
    text = " ".join(_help(["dev", *argv], capsys).split())

    assert "--name NAME Box name (default: this folder's box)" in text
    assert "--team TEAM Team the box runs as (remembered per box)" in text
    assert "OS username" not in text


def test_hold_documents_off_not_zero(capsys):
    text = " ".join(_help(["dev", "hold"], capsys).split())

    assert "--off clears the hold" in text
    assert "(or 0)" not in text and "0 clears" not in text


def test_yes_help_says_what_each_command_skips(capsys):
    def yes(command):
        text = " ".join(_help(["dev", command], capsys).split())
        return text.split("--yes ")[1].split(" --")[0]

    assert "production apps" in yes("up") and "production apps" in yes("deploy")
    assert "GPU is busy" in yes("down")
    assert "typing the box name" in yes("destroy")


def test_multi_line_flag_help_moved_to_the_description(capsys):
    text = " ".join(_help(["dev", "deploy"], capsys).split())

    assert "--team-data Keep the box on the team-shared /data (not remembered)." in text
    assert "--team-data keeps the box on the team-shared /data" in text
    assert (
        "--starship Use your local starship prompt on the box (not remembered)." in text
    )
    assert "--secret NAME gives the box a team secret" in text
    assert "--timeout" not in text  # deploy has none


def test_timeouts_are_explained(capsys):
    up = " ".join(_help(["dev", "up"], capsys).split())
    proxy = " ".join(_help(["dev", "ssh-proxy"], capsys).split())

    assert "default: 1800" in up and "builds the image" in up
    assert "default: 900" in proxy and "no image build" in proxy


# --- parse errors ------------------------------------------------------------


def test_a_parse_error_shows_the_commands_own_usage_and_a_fix(capsys):
    with pytest.raises(FalParserExit) as exit_:
        parse_args(["dev", "up", "--nmae", "x"])

    err = capsys.readouterr().err
    assert exit_.value.status == 2
    assert err.startswith("Usage: fal dev up ")
    assert "Usage: fal [-h]" not in err
    assert "error: fal dev up: unrecognized arguments: --nmae x" in err
    assert "try: fal dev up --name  (did you mean this?)" in err
    assert "try: fal dev up --help" in err


def test_a_parse_error_under_json_also_prints_an_error_object(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fal", "dev", "status", "--json", "--bogus"])
    with pytest.raises(FalParserExit) as exit_:
        parse_args(["dev", "status", "--json", "--bogus"])

    out, _ = capsys.readouterr()
    assert exit_.value.status == 2
    error = json.loads(out)["error"]
    assert error["title"] == "fal dev status"
    assert "--bogus" in error["message"]


def test_an_unrelated_command_suggests_nothing(capsys):
    with pytest.raises(FalParserExit):
        parse_args(["dev", "bogus"])

    err = capsys.readouterr().err
    assert "did you mean" not in err
    assert "try: fal dev --help" in err


def test_an_unknown_command_suggests_the_closest_one(capsys):
    with pytest.raises(FalParserExit):
        parse_args(["dev", "upp"])

    err = capsys.readouterr().err
    assert err.startswith("Usage: fal dev [-h] command")
    assert "try: fal dev up  (did you mean this?)" in err


def test_a_missing_command_is_an_error_panel_with_a_fix(capsys):
    with pytest.raises(FalParserExit):
        parse_args(["dev"])

    err = capsys.readouterr().err
    assert "error: fal dev: the following arguments are required: command" in err
    assert "try: fal dev --help" in err


def test_usage_mistakes_the_parser_cannot_see_use_the_commands_usage(capsys):
    args = parse_args(["dev", "hold"])

    assert args.func(args) == 2
    err = capsys.readouterr().err
    assert err.startswith("Usage: fal dev hold ")
    assert "error: fal dev hold: say how many hours to hold, or pass --off" in err
    assert "try: fal dev hold 4  (keep the box up for 4 hours)" in err


def test_exec_without_a_command_is_a_usage_error(capsys):
    args = parse_args(["dev", "exec"])

    assert args.func(args) == 2
    err = capsys.readouterr().err
    assert err.startswith("Usage: fal dev exec ")
    assert "try: fal dev exec -- nvidia-smi" in err


# --- error panel -------------------------------------------------------------


def _exc(**kwargs):
    return DevError("box demo is stopped", **kwargs)


def test_non_tty_errors_are_error_title_message_then_try_lines():
    console, out = _console()
    error_panel.show(
        _exc(title="the box is stopped", fixes=[("fal dev up", "start it")]), console
    )

    assert out.getvalue().splitlines() == [
        "error: the box is stopped: box demo is stopped",
        "try: fal dev up  (start it)",
    ]


def test_the_title_defaults_to_the_command_name():
    console, out = _console()
    error_panel.show(_exc(), console, "fal dev logs")
    assert out.getvalue().splitlines()[0].startswith("error: fal dev logs: ")

    terminal = Console(record=True, width=100, force_terminal=True)
    error_panel.show(_exc(), terminal, "fal dev logs")
    assert "error  fal dev logs" in terminal.export_text()


def test_a_command_failure_exits_with_the_errors_code_and_the_command_title(capsys):
    args = parse_args(["dev", "hold", "500"])

    assert args.func(args) == 2
    err = capsys.readouterr().err
    assert "error: fal dev hold: a hold is more than 0 and at most 168 hours" in err
    assert "try: fal dev hold 168" in err


def test_warnings_use_the_panel_style():
    plain, out = _console()
    error_panel.warn("the GPU is busy.", plain, [("fal dev down --yes", "anyway")])
    assert out.getvalue().splitlines() == [
        "warning: the GPU is busy.",
        "try: fal dev down --yes  (anyway)",
    ]
    terminal = Console(record=True, width=100, force_terminal=True)
    error_panel.warn("the GPU is busy.", terminal)
    assert "warning  warning" in terminal.export_text()


# --- common failures carry fixes and codes ---------------------------------


def test_a_stopped_box_exits_3_with_a_fix():
    with pytest.raises(DevError) as caught:
        make_box([]).require_runner()
    assert caught.value.code == 3
    assert ("fal dev up --name demo", "start it") in caught.value.fixes


def test_a_wrong_team_exits_4_and_names_the_teams_to_try(monkeypatch):
    listed = [
        {"nickname": "me", "is_personal": True},
        {"nickname": "red", "is_personal": False},
    ]
    monkeypatch.setattr(team, "key_account", lambda: None)
    monkeypatch.setattr(team, "accounts", lambda: listed)

    with pytest.raises(DevError, match="no team named 'gone'") as caught:
        team.resolve_account("gone", "devbox-demo", None)
    assert caught.value.code == 4
    assert caught.value.title == "wrong team"
    assert [cmd for cmd, _ in caught.value.fixes] == ["--team me", "--team red"]


def test_not_logged_in_is_an_error_panel_with_login_fixes(monkeypatch, capsys):
    from fal.exceptions.auth import UnauthenticatedException

    def fail(args):
        raise UnauthenticatedException()

    monkeypatch.setattr(box_cli, "cmd_status", fail)
    args = parse_args(["dev", "status"])

    assert args.func(args) == 4
    err = capsys.readouterr().err
    assert "error: not logged in: " in err
    assert "try: fal auth login" in err


def test_a_box_that_was_never_deployed_exits_3_with_up_as_the_fix(monkeypatch, capsys):
    import grpc

    class NotFound(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.NOT_FOUND

        def details(self):
            return "Application {'user': 'u', 'name': 'devbox-ghost'} not found"

    def fail(args):
        raise NotFound()

    monkeypatch.setattr(box_cli, "cmd_logs", fail)
    args = parse_args(["dev", "logs", "--name", "ghost"])

    assert args.func(args) == 3
    err = capsys.readouterr().err
    assert "error: no such box: box ghost is not deployed" in err
    assert "try: fal dev up --name ghost" in err


def _grpc_error(code_name, details):
    import grpc

    class Err(grpc.RpcError):
        def code(self):
            return getattr(grpc.StatusCode, code_name)

        def details(self):
            return details

    return Err()


@pytest.mark.parametrize(
    "code_name, details",
    [
        ("NOT_FOUND", "Runner r-1 not found"),
        ("INTERNAL", "Application {'user': 'u', 'name': 'devbox-ghost'} not found"),
    ],
)
def test_other_grpc_errors_are_not_called_undeployed(code_name, details):
    from fal.dev.cli.parser import _not_deployed

    assert _not_deployed(_grpc_error(code_name, details)) is None


def test_ssh_proxy_maps_an_undeployed_box_to_exit_3(monkeypatch):
    error = _grpc_error(
        "NOT_FOUND", "Application {'user': 'u', 'name': 'devbox-ghost'} not found"
    )
    monkeypatch.setattr(
        box_cli, "get_box", lambda *a, **k: (_ for _ in ()).throw(error)
    )
    monkeypatch.setattr(box_cli, "_cleanup_on_hangup", lambda: None)
    args = parse_args(["dev", "ssh-proxy", "--name", "ghost"])

    assert args.func(args) == 3


def test_ssh_proxy_maps_logged_out_to_exit_4(monkeypatch):
    from fal.exceptions.auth import UnauthenticatedException

    def fail(*a, **k):
        raise UnauthenticatedException()

    monkeypatch.setattr(box_cli, "get_box", fail)
    monkeypatch.setattr(box_cli, "_cleanup_on_hangup", lambda: None)
    args = parse_args(["dev", "ssh-proxy", "--name", "ghost"])

    assert args.func(args) == 4


def test_ssh_proxy_ended_by_ssh_exits_0(monkeypatch):
    box = SimpleNamespace(
        ssh_proxy=lambda timeout: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    monkeypatch.setattr(box_cli, "get_box", lambda *a, **k: box)
    monkeypatch.setattr(box_cli, "warn_if_foreign", lambda *a: None)
    monkeypatch.setattr(box_cli, "_cleanup_on_hangup", lambda: None)
    args = parse_args(["dev", "ssh-proxy", "--name", "demo"])

    assert args.func(args) == 0


def test_a_destroy_of_a_missing_box_exits_3_with_fixes(monkeypatch):
    box = SimpleNamespace(
        name="gone",
        app="devbox-gone",
        app_info=lambda: None,
    )
    monkeypatch.setattr(box_cli, "get_box", lambda *a, **k: box)
    args = SimpleNamespace(yes=False, name=None, parser=None)

    with pytest.raises(DevError) as caught:
        box_cli.cmd_destroy(args)
    assert caught.value.code == 3 and caught.value.title == "no such box"
    assert ("fal dev list", "see your boxes") in caught.value.fixes


def test_ctrl_c_exits_130(monkeypatch, capsys):
    def interrupted(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(box_cli, "cmd_status", interrupted)
    args = parse_args(["dev", "status"])

    assert args.func(args) == 130
    assert "interrupted" in capsys.readouterr().err


# --- --json ------------------------------------------------------------------


def test_a_failure_under_json_prints_an_error_object_and_exits_non_zero(capsys):
    args = parse_args(["dev", "hold", "500", "--json"])

    assert args.func(args) == 2
    out, err = capsys.readouterr()
    error = json.loads(out)["error"]
    assert error["title"] == "fal dev hold"
    assert "at most 168 hours" in error["message"]
    assert error["fixes"] == [
        {"command": "fal dev hold 168", "why": "the longest hold"}
    ]
    assert "error: fal dev hold:" in err


def test_a_failure_without_json_prints_nothing_on_stdout(capsys):
    args = parse_args(["dev", "hold", "500"])

    args.func(args)
    assert capsys.readouterr().out == ""


class _Box:
    name = "demo"
    app = "devbox-demo"
    account = Account("troy", True)
    progress = None
    start_cancelled = None

    def __init__(self, saved=None, hold=True):
        self.saved = saved
        self.holds = hold

    def down(self, no_save):
        return None if no_save else self.saved

    def hold(self, hours):
        return self.holds

    def hold_until(self):
        return 2_000_000_000.0

    def clear_hold(self):
        return True


def _run(monkeypatch, command, box, *argv):
    monkeypatch.setattr(box_cli, "get_box", lambda *a, **k: box)
    monkeypatch.setattr(box_cli, "_confirm_down", lambda box, args: None)
    args = parse_args(["dev", command, *argv])
    console, out = _console()
    err, errors = _console()
    args.console, args.err = console, err
    handler = box_cli.cmd_down if command == "down" else box_cli.cmd_hold
    return handler(args), out.getvalue(), errors.getvalue()


def test_down_json_prints_one_result_object(monkeypatch):
    _, out, err = _run(
        monkeypatch,
        "down",
        _Box("devbox-save: saved /work (4.0K) in 1s"),
        "--json",
    )

    data = json.loads(out)
    assert data["name"] == "demo" and data["state"] == "stopped"
    assert data["saved"] is True and "4.0K" in data["save_report"]
    assert data["kept"] == "/data" and "total_seconds" in data["timings"]
    assert err == ""


def test_down_json_says_when_nothing_was_saved(monkeypatch):
    _, out, _ = _run(monkeypatch, "down", _Box(), "--json", "--no-save")
    assert json.loads(out)["saved"] is False
    _, out, _ = _run(monkeypatch, "down", _Box(), "--json")
    assert json.loads(out)["saved"] is None


def test_hold_json_has_held_until(monkeypatch):
    _, out, err = _run(monkeypatch, "hold", _Box(), "4", "--json")

    data = json.loads(out)
    assert data["name"] == "demo" and data["state"] == "held" and data["hours"] == 4
    assert data["held_until"].endswith("+00:00")
    assert err == ""


def test_hold_json_when_a_longer_hold_is_kept_reads_the_existing_one(monkeypatch):
    _, out, _ = _run(monkeypatch, "hold", _Box(hold=False), "1", "--json")

    data = json.loads(out)
    assert data["state"] == "kept_longer"
    assert data["held_until"] == "2033-05-18T03:33:20+00:00"


def test_hold_off_json(monkeypatch):
    _, out, _ = _run(monkeypatch, "hold", _Box(), "--off", "--json")
    assert json.loads(out) == {"name": "demo", "state": "cleared", "held_until": None}


def test_up_json_prints_the_result_and_the_checklist_stays_on_stderr(
    monkeypatch, capsys
):
    class Box(_Box):
        was_live = False

        def app_info(self):
            return object()

        def owner(self):
            return None

        def up(self, timeout, machine_type, on_wait, **kwargs):
            return None

    runner = SimpleNamespace(machine_type="GPU-H100", runner_id="r1")
    snap = SimpleNamespace(state=SimpleNamespace(runner=runner))
    monkeypatch.setattr(box_cli, "get_box", lambda *a, **k: Box())
    monkeypatch.setattr(box_cli, "_cleanup_on_hangup", lambda: None)
    monkeypatch.setattr(summary, "snapshot", lambda box: snap)
    args = parse_args(["dev", "up", "--json", "--yes"])
    console, out = _console()
    args.console = console

    assert box_cli.cmd_up(args) is None
    data = json.loads(out.getvalue())
    assert data["name"] == "demo" and data["state"] == "running"
    assert data["machine_type"] == "GPU-H100" and data["runner_id"] == "r1"
    assert set(data["timings"]) == {"runner_seconds", "total_seconds"}
    assert capsys.readouterr().out == ""


def test_up_json_with_dash_is_a_usage_error(capsys):
    args = parse_args(["dev", "up", "--json", "--dash"])

    assert args.func(args) == 2
    assert "--dash keeps running" in capsys.readouterr().err


def test_status_json_prints_only_json_on_stdout(monkeypatch, capsys):
    box = SimpleNamespace(name="demo")
    monkeypatch.setattr(box_cli, "get_box", lambda *a, **k: box)
    monkeypatch.setattr(summary, "snapshot", lambda box: object())
    import fal.dev.cli.dash as dash

    monkeypatch.setattr(dash, "payload", lambda box, snap: json.dumps({"a": 1}))
    args = parse_args(["dev", "status", "--json"])

    box_cli.cmd_status(args)
    out, err = capsys.readouterr()
    assert json.loads(out) == {"a": 1}
    assert err == ""


def test_list_json_prints_only_json_on_stdout(monkeypatch, capsys):
    monkeypatch.setattr(box_cli.inventory, "list_boxes", lambda *a, **k: [])
    monkeypatch.setattr(
        box_cli,
        "resolve_account",
        lambda *a, **k: SimpleNamespace(runs_as="troy", fal_team=None),
    )
    monkeypatch.setattr(box_cli, "SyncServerlessClient", lambda **k: None)
    args = parse_args(["dev", "list", "--json"])

    box_cli.cmd_list(args)
    out, err = capsys.readouterr()
    assert json.loads(out) == []
    assert err == ""


# --- stdout and stderr --------------------------------------------------------


def test_down_status_chatter_goes_to_stderr_not_stdout(monkeypatch):
    _, out, err = _run(monkeypatch, "down", _Box("devbox-save: saved /work (4.0K)"))

    assert out == ""
    assert "stopped demo" in err and "/work saved (4.0K)" in err


def test_hold_cleared_goes_to_stderr(monkeypatch):
    _, out, err = _run(monkeypatch, "hold", _Box(), "--off")

    assert out == ""
    assert "hold cleared" in err


def test_the_command_wrapper_gives_handlers_a_stderr_console(monkeypatch):
    seen = {}

    def handler(args):
        seen["err"] = args.err

    monkeypatch.setattr(box_cli, "cmd_status", handler)
    args = parse_args(["dev", "status"])
    args.func(args)

    assert seen["err"].stderr is True


def test_without_an_err_console_chatter_falls_back_to_the_console():
    from fal.dev.cli.common import err_console

    console = object()
    assert err_console(SimpleNamespace(console=console)) is console


def test_the_consoles_honour_no_color_and_non_tty():
    code = (
        "from fal.console import console\n"
        "from fal.dev.cli.parser import _stderr_console\n"
        "err = _stderr_console()\n"
        "print(console.no_color, console.is_terminal, err.no_color, err.is_terminal)\n"
    )
    base = {k: v for k, v in os.environ.items() if k not in ("NO_COLOR", "FORCE_COLOR")}

    def run(extra):
        done = subprocess.run(
            [sys.executable, "-I", "-c", code],
            env={**base, **extra},
            capture_output=True,
            text=True,
            check=True,
        )
        return done.stdout.split()

    # Piped output is never a terminal, so nothing is styled either way.
    assert run({"NO_COLOR": "1"}) == ["True", "False", "True", "False"]
    assert run({}) == ["False", "False", "False", "False"]
