import io
import json
import os
import struct
from unittest.mock import MagicMock, patch

import grpc
import isolate_proto
import pytest

from fal.cli.main import parse_args
from fal.cli.parser import FalParserExit
from fal.cli.runners import _exec, _get_tty_size, _gpus, shell_session

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

    def shell_runner(inputs, **kwargs):
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


@patch("fal.cli.runners.SyncServerlessClient")
def test_shell_session_separates_output_streams(mock_client_cls):
    stub = mock_client_cls.return_value._create_host.return_value._connection.stub
    stub.ShellRunner.return_value = [
        isolate_proto.ShellRunnerOutput(data=b"out"),
        isolate_proto.ShellRunnerOutput(data=b"err", stream=2),
        isolate_proto.ShellRunnerOutput(exit_code=0),
    ]
    stdout = io.BytesIO()
    stderr = io.BytesIO()

    exit_code = shell_session(
        mock_client_cls.return_value,
        "runner-id",
        ["true"],
        interactive=False,
        remote_tty=False,
        console=None,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stdout.getvalue() == b"out"
    assert stderr.getvalue() == b"err"


@patch("fal.cli.runners.SyncServerlessClient")
def test_shell_session_reads_output_up_to_the_marker(mock_client_cls):
    stub = mock_client_cls.return_value._create_host.return_value._connection.stub
    # The exit code arrives before the output ends; the marker is split in two.
    stub.ShellRunner.return_value = [
        isolate_proto.ShellRunnerOutput(exit_code=0),
        isolate_proto.ShellRunnerOutput(data=b"result\n__en"),
        isolate_proto.ShellRunnerOutput(data=b"d__\n"),
        isolate_proto.ShellRunnerOutput(data=b"never read"),
    ]
    stdout = io.BytesIO()

    shell_session(
        mock_client_cls.return_value,
        "runner-id",
        ["true"],
        interactive=False,
        remote_tty=False,
        console=None,
        stdout=stdout,
        timeout=5,
        until=b"__end__",
    )

    assert stdout.getvalue() == b"result\n__end__\n"
    assert stub.ShellRunner.call_args.kwargs == {"timeout": 5}


@patch("fal.cli.runners.SyncServerlessClient")
def test_shell_session_without_a_console_raises_connection_errors(mock_client_cls):
    stub = mock_client_cls.return_value._create_host.return_value._connection.stub
    stub.ShellRunner.side_effect = RuntimeError("no route")

    with pytest.raises(RuntimeError, match="no route"):
        shell_session(
            mock_client_cls.return_value,
            "runner-id",
            ["true"],
            interactive=False,
            remote_tty=False,
            console=None,
            stdout=io.BytesIO(),
        )


@patch("fal.cli.runners.SyncServerlessClient")
def test_exec_without_terminal_requests_no_tty_and_closes_stdin(mock_client_cls):
    with patch("fal.cli.runners.sys.stdin") as stdin:
        stdin.isatty.return_value = False
        exit_code, sent, _ = _exec_with_command(mock_client_cls, ["python"])

    assert exit_code == 0
    assert sent[0].HasField("tty") and sent[0].tty is False
    assert [msg.close for msg in sent] == [False, True]


@pytest.mark.skipif(os.name == "nt", reason="Interactive shell is Unix-only")
@patch("fal.cli.runners.SyncServerlessClient")
def test_exec_interactive_with_piped_stdin_streams_raw_bytes(mock_client_cls):
    sent = []

    def shell_runner(inputs, **kwargs):
        sent.extend(inputs)
        return [isolate_proto.ShellRunnerOutput(exit_code=0)]

    stub = mock_client_cls.return_value._create_host.return_value._connection.stub
    stub.ShellRunner.side_effect = shell_runner

    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"SSH-2.0-client\r\n")
    os.close(write_fd)
    args = parse_args(["runners", "exec", "-it", "runner-id", "--", "sshd", "-i"])
    args.console = MagicMock()
    try:
        with patch("fal.cli.runners.sys.stdin") as stdin:
            stdin.isatty.return_value = False
            stdin.fileno.return_value = read_fd
            assert args.func(args) == 0
    finally:
        os.close(read_fd)

    assert sent[0].tty is False
    assert not sent[0].HasField("tty_size")
    assert list(sent[0].command) == ["sshd", "-i"]
    assert [msg.data for msg in sent[1:-1]] == [b"SSH-2.0-client\r\n"]
    assert sent[-1].close is True


@pytest.mark.skipif(os.name == "nt", reason="Pseudo-terminals are Unix-only")
@patch("fal.cli.runners.SyncServerlessClient")
def test_exec_interactive_on_terminal_requests_tty_and_sends_size(mock_client_cls):
    sent = []

    def shell_runner(inputs, **kwargs):
        for msg in inputs:
            sent.append(msg)
            if msg.HasField("tty_size"):
                break
        return [isolate_proto.ShellRunnerOutput(exit_code=0)]

    stub = mock_client_cls.return_value._create_host.return_value._connection.stub
    stub.ShellRunner.side_effect = shell_runner

    master_fd, slave_fd = os.openpty()
    args = parse_args(["runners", "exec", "-it", "runner-id", "--", "bash"])
    args.console = MagicMock()
    try:
        with patch("fal.cli.runners.sys.stdin") as stdin:
            stdin.isatty.return_value = True
            stdin.fileno.return_value = slave_fd
            assert args.func(args) == 0
    finally:
        os.close(master_fd)
        os.close(slave_fd)

    # The local `import tty` inside the session must not leak into the message.
    assert sent[0].tty is True
    assert sent[1].HasField("tty_size")


@pytest.mark.parametrize("platform_name", ["posix", "nt"])
@patch("fal.cli.runners.SyncServerlessClient")
def test_shell_with_piped_stdin(mock_client_cls, platform_name):
    sent = []

    def shell_runner(inputs, **kwargs):
        sent.extend(inputs)
        return [isolate_proto.ShellRunnerOutput(exit_code=0)]

    stub = mock_client_cls.return_value._create_host.return_value._connection.stub
    stub.ShellRunner.side_effect = shell_runner

    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    args = parse_args(["runners", "shell", "runner-id"])
    args.console = MagicMock()
    try:
        with patch("fal.cli.runners.sys.stdin") as stdin:
            stdin.isatty.return_value = False
            stdin.fileno.return_value = read_fd
            with patch("fal.cli.runners.os.name", platform_name):
                exit_code = args.func(args)
    finally:
        os.close(read_fd)

    if platform_name == "nt":
        assert exit_code == 1
        args.console.print.assert_called_once_with(
            "[red]Error:[/] Interactive runner shell is not supported on Windows."
        )
        stub.ShellRunner.assert_not_called()
        assert sent == []
    else:
        assert exit_code == 0
        assert sent[0].tty is True
        assert all(not msg.close for msg in sent)


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


def test_print_logs_keeps_the_range_within_the_runners_lifetime():
    from datetime import datetime, timedelta, timezone

    from fal.cli import runners

    started = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    ended = started + timedelta(hours=1)
    info = runners.RestRunnerInfo(started_at=started, ended_at=ended)
    seen = {}

    def fake_get_logs(rest_client, params, since, until, lines, *, oldest):
        seen.update(params=params, since=since, until=until)
        return iter([])

    with (
        patch.object(runners, "_get_runner_info", return_value=info),
        patch.object(runners, "_get_logs", fake_get_logs),
    ):
        # The `fal dev logs --all` path: a start far before the runner existed.
        runners.print_logs(
            MagicMock(),
            "r1",
            MagicMock(),
            follow=False,
            since=started - timedelta(hours=23),
        )

    assert seen["params"] == {"job_id": "r1"}
    assert (seen["since"], seen["until"]) == (started, ended)


def test_interactive_shell_without_a_console_raises_on_windows():
    with patch("fal.cli.runners.os.name", "nt"):
        with pytest.raises(NotImplementedError, match="Windows"):
            shell_session(
                MagicMock(),
                "r1",
                None,
                interactive=True,
                remote_tty=True,
                console=None,
            )


@patch("fal.cli.runners.SyncServerlessClient")
def test_shell_session_stops_waiting_for_a_marker_after_the_exit_code(
    mock_client_cls, monkeypatch
):
    import threading

    from fal.cli import runners

    monkeypatch.setattr(runners, "UNTIL_GRACE_SECONDS", 0.05)
    stub = mock_client_cls.return_value._create_host.return_value._connection.stub
    cancelled = threading.Event()

    class Call:
        def __iter__(self):
            yield isolate_proto.ShellRunnerOutput(exit_code=3)
            # Nothing more arrives, so only the cancel ends the stream.
            assert cancelled.wait(5)
            raise _Cancelled()

        def cancel(self):
            cancelled.set()

    class _Cancelled(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.CANCELLED

    stub.ShellRunner.return_value = Call()

    code = shell_session(
        mock_client_cls.return_value,
        "r1",
        ["python3"],
        interactive=False,
        remote_tty=False,
        console=None,
        stdout=io.BytesIO(),
        until=b"__end__",
    )

    assert code == 3
