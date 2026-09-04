import json
import os
import struct
from unittest.mock import MagicMock, patch

import pytest

from fal.cli.main import parse_args
from fal.cli.parser import FalParserExit
from fal.cli.runners import _exec, _get_tty_size, _gpus

if os.name != "nt":
    import fcntl
    import termios

_GPUS_PAYLOAD = {
    "gpus": {"H100": 394, "B200": 353, "H200": 334},
    "total": 1120,
}


@pytest.mark.skipif(os.name == "nt", reason="Pseudo-terminals are Unix-only")
def test_get_tty_size():
    master_fd, slave_fd = os.openpty()
    try:
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
        assert _get_tty_size(slave_fd) == (40, 120)
    finally:
        os.close(master_fd)
        os.close(slave_fd)


def test_get_tty_size_fallback():
    assert _get_tty_size(-1) == (24, 80)


def test_gpus_parser_registered():
    args = parse_args(["runners", "gpus"])
    assert args.func == _gpus


@pytest.mark.parametrize(
    ("argv", "interactive"),
    [
        pytest.param(
            ["runners", "exec", "runner-id", "-it", "--", "python"],
            True,
            id="flag_after_id",
        ),
        pytest.param(
            ["runners", "exec", "-it", "runner-id", "--", "python"],
            True,
            id="flag_before_id",
        ),
        pytest.param(
            ["runners", "exec", "runner-id", "--", "python"],
            False,
            id="no_flag",
        ),
    ],
)
def test_exec_parses_fal_flags_on_either_side_of_runner_id(argv, interactive):
    args = parse_args(argv)
    assert args.func == _exec
    assert args.id == "runner-id"
    assert args.interactive is interactive
    assert args.command[-1] == "python"


def test_exec_keeps_command_flags_out_of_fal_parsing():
    args = parse_args(["runners", "exec", "runner-id", "--", "tail", "-f", "/x"])
    assert args.interactive is False
    assert args.command[-3:] == ["tail", "-f", "/x"]


@pytest.mark.parametrize(
    "argv",
    [
        ["runners", "exec", "runner-id"],
        ["runners", "exec", "runner-id", "--"],
        ["runners", "exec", "runner-id", "--bogus", "--", "env"],
    ],
    ids=["no_command", "separator_only", "unknown_flag_before_separator"],
)
def test_exec_rejects_missing_command_and_unknown_flags(argv, capsys):
    with pytest.raises(FalParserExit) as exc:
        parse_args(argv)
    assert exc.value.status == 2
    assert "fal runners exec: error:" in capsys.readouterr().err


def _exec_with_command(mock_client_cls, command):
    import isolate_proto

    sent = []

    def shell_runner(inputs):
        sent.extend(inputs)
        return [isolate_proto.ShellRunnerOutput(exit_code=0)]

    stub = mock_client_cls.return_value._create_host.return_value._connection.stub
    stub.ShellRunner.side_effect = shell_runner

    args = parse_args(["runners", "exec", "runner-id", "--", "python"])
    args.command = command
    args.console = MagicMock()
    return args.func(args), sent, args.console


@pytest.mark.parametrize(
    "command",
    [["python"], ["--", "python"]],
    ids=["bare", "with_separator"],
)
@patch("fal.cli.runners.SyncServerlessClient")
def test_exec_strips_leading_separator_before_sending(mock_client_cls, command):
    exit_code, sent, _ = _exec_with_command(mock_client_cls, command)

    assert exit_code == 0
    assert list(sent[0].command) == ["python"]


@patch("fal.cli.runners.SyncServerlessClient")
def test_exec_rejects_separator_only_command(mock_client_cls):
    exit_code, sent, console = _exec_with_command(mock_client_cls, ["--"])

    assert exit_code == 1
    assert sent == []
    assert "No command specified" in console.print.call_args[0][0]


def _mock_client(payload):
    client = MagicMock()
    client.runners.gpus.return_value = payload
    return client


@patch("fal.cli.runners.SyncServerlessClient")
def test_gpus_json(mock_client_cls):
    mock_client_cls.return_value = _mock_client(_GPUS_PAYLOAD)

    args = parse_args(["runners", "gpus", "--json"])
    args.console = MagicMock()
    args.func(args)

    output = args.console.print.call_args[0][0]
    result = json.loads(output)

    assert result["total"] == 1120
    # Sorted by gpus desc by render_gpus
    assert list(result["gpus"].items()) == [
        ("H100", 394),
        ("B200", 353),
        ("H200", 334),
    ]


@patch("fal.cli.runners.SyncServerlessClient")
def test_gpus_pretty_runs(mock_client_cls):
    mock_client_cls.return_value = _mock_client(_GPUS_PAYLOAD)

    args = parse_args(["runners", "gpus"])
    args.console = MagicMock()
    args.func(args)

    printed = " ".join(str(call.args[0]) for call in args.console.print.call_args_list)
    assert "Total: 1120" in printed


@patch("fal.cli.runners.SyncServerlessClient")
def test_gpus_empty(mock_client_cls):
    mock_client_cls.return_value = _mock_client({"gpus": {}, "total": 0})

    args = parse_args(["runners", "gpus", "--json"])
    args.console = MagicMock()
    args.func(args)

    output = args.console.print.call_args[0][0]
    assert json.loads(output) == {"gpus": {}, "total": 0}


@patch("fal.cli.runners.SyncServerlessClient")
def test_gpus_propagates_api_error(mock_client_cls):
    client = MagicMock()
    client.runners.gpus.side_effect = RuntimeError("Failed to fetch metrics: 500 boom")
    mock_client_cls.return_value = client

    args = parse_args(["runners", "gpus"])
    args.console = MagicMock()

    with pytest.raises(RuntimeError, match="Failed to fetch metrics"):
        args.func(args)


@patch("fal.cli.runners.SyncServerlessClient")
def test_interactive_shell_errors_on_windows(_mock_client_cls):
    args = parse_args(["runners", "shell", "runner-id"])
    args.console = MagicMock()

    with patch("fal.cli.runners.os.name", "nt"):
        assert args.func(args) == 1

    args.console.print.assert_called_once_with(
        "[red]Error:[/] Interactive runner shell is not supported on Windows."
    )
