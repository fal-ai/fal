import io
import json
import os
import struct
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import isolate_proto
import pytest

from fal.cli.main import parse_args
from fal.cli.parser import FalParserExit
from fal.cli.runners import (
    FINISHED_LOG_GRACE,
    FOLLOW_STREAM_OVERLAP,
    RestRunnerInfo,
    _exec,
    _follow_start,
    _get_logs,
    _get_runner_info,
    _get_tty_size,
    _gpus,
    _resolve_log_window,
    _shell_session,
)

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
    args = parse_args(["runners", "exec", "runner-id", "--", "true"])

    exit_code = _shell_session(
        args,
        command=["true"],
        interactive=False,
        remote_tty=False,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stdout.getvalue() == b"out"
    assert stderr.getvalue() == b"err"


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

    def shell_runner(inputs):
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

    def shell_runner(inputs):
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

    def shell_runner(inputs):
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


_T0 = datetime(2026, 11, 6, 5, 55, 7, tzinfo=timezone.utc)


def test_log_window_ignores_started_at_for_live_runner():
    # started_at lags the setup logs for live runners, so it must not clamp.
    info = RestRunnerInfo(started_at=_T0, ended_at=None)
    since = _T0 - timedelta(hours=1)

    assert _resolve_log_window(None, None, info) == (None, None)
    assert _resolve_log_window(since, None, info) == (since, None)


def test_log_window_bounds_stopped_runner_by_finished_at_with_grace():
    info = RestRunnerInfo(started_at=_T0, ended_at=_T0 + timedelta(minutes=5))

    since, until = _resolve_log_window(None, None, info)

    assert since is None
    assert until == _T0 + timedelta(minutes=5) + FINISHED_LOG_GRACE


def test_log_window_keeps_explicit_until_and_orders_bounds():
    info = RestRunnerInfo(started_at=_T0, ended_at=_T0 + timedelta(minutes=5))
    until = _T0 + timedelta(hours=1)

    assert _resolve_log_window(None, until, info)[1] == until
    assert _resolve_log_window(until, _T0, info) == (_T0, until)


@pytest.mark.parametrize("key", ["finished_at", "ended_at"])
@patch("fal.cli.runners.httpx.Client")
def test_get_runner_info_reads_finished_or_ended_at(mock_httpx_client, key):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {
        "started_at": "2026-11-06T05:55:07.178320Z",
        key: "2026-11-06T06:00:00Z",
    }
    mock_httpx_client.return_value.__enter__.return_value.get.return_value = resp

    info = _get_runner_info(MagicMock(), "rid")

    assert info.started_at == _T0.replace(microsecond=178320)
    assert info.ended_at == datetime(2026, 11, 6, 6, 0, tzinfo=timezone.utc)


def _history_lines(n):
    # History is served newest first.
    return [
        {"timestamp": (_T0 + timedelta(seconds=i)).isoformat(), "message": str(i)}
        for i in reversed(range(n))
    ]


@patch("fal.cli.runners._iter_logs")
def test_get_logs_latest_and_oldest_lines(mock_iter):
    mock_iter.side_effect = lambda *a, **k: iter(_history_lines(10))

    latest = [x["message"] for x in _get_logs(MagicMock(), {}, None, None, 3)]
    oldest = [
        x["message"] for x in _get_logs(MagicMock(), {}, None, None, 3, oldest=True)
    ]

    assert latest == ["9", "8", "7"]
    assert oldest == ["2", "1", "0"]


@patch("fal.cli.runners._stream_logs")
@patch("fal.cli.runners._iter_logs")
@patch("fal.cli.runners._get_runner_info")
@patch("fal.cli.runners.SyncServerlessClient")
def test_follow_replays_latest_lines_then_streams_after_them(
    _client, mock_info, mock_iter, mock_stream
):
    mock_info.return_value = RestRunnerInfo(started_at=_T0, ended_at=None)
    mock_iter.side_effect = lambda *a, **k: iter(_history_lines(150))
    mock_stream.return_value = iter([])

    args = parse_args(["runners", "logs", "rid", "--follow", "--json"])
    args.console = MagicMock()
    args.func(args)

    printed = [json.loads(c[0][0]) for c in args.console.print.call_args_list]
    assert len(printed) == 100
    assert printed[0]["message"] == "50" and printed[-1]["message"] == "149"
    # No clamp to started_at on the history replay; the stream overlaps it.
    assert mock_iter.call_args[0][2] is None
    stream_since = mock_stream.call_args[0][2]
    assert stream_since == _T0 + timedelta(seconds=149) - FOLLOW_STREAM_OVERLAP


@patch("fal.cli.runners._stream_logs")
@patch("fal.cli.runners._iter_logs")
@patch("fal.cli.runners._get_runner_info")
@patch("fal.cli.runners.SyncServerlessClient")
def test_follow_stream_skips_lines_already_replayed(
    _client, mock_info, mock_iter, mock_stream
):
    mock_info.return_value = RestRunnerInfo(started_at=_T0, ended_at=None)
    history = _history_lines(5)
    mock_iter.side_effect = lambda *a, **k: iter(history)
    late = {"timestamp": (_T0 + timedelta(seconds=2.5)).isoformat(), "message": "late"}
    # The overlap window re-sends lines 3 and 4 plus one that arrived late.
    mock_stream.return_value = iter([history[1], late, history[0]])

    args = parse_args(["runners", "logs", "rid", "--follow", "--json"])
    args.console = MagicMock()
    args.func(args)

    printed = [
        json.loads(c[0][0])["message"] for c in args.console.print.call_args_list
    ]
    assert printed == ["0", "1", "2", "3", "4", "late"]


@patch("fal.cli.runners._stream_logs")
@patch("fal.cli.runners._iter_logs")
@patch("fal.cli.runners._get_runner_info")
@patch("fal.cli.runners.SyncServerlessClient")
def test_follow_with_zero_lines_replays_nothing(
    _client, mock_info, mock_iter, mock_stream
):
    mock_info.return_value = RestRunnerInfo(started_at=_T0, ended_at=None)
    mock_iter.side_effect = lambda *a, **k: iter(_history_lines(10))
    mock_stream.return_value = iter([])

    args = parse_args(["runners", "logs", "rid", "--follow", "-n", "0"])
    args.console = MagicMock()
    args.func(args)

    args.console.print.assert_not_called()
    # Only new lines: the stream starts now, with no overlap into history.
    stream_since = mock_stream.call_args[0][2]
    assert stream_since > datetime.now(timezone.utc) - timedelta(seconds=5)


@patch("fal.cli.runners._stream_logs")
@patch("fal.cli.runners._iter_logs")
@patch("fal.cli.runners._get_runner_info")
@patch("fal.cli.runners.SyncServerlessClient")
def test_follow_overlap_never_reaches_before_the_replay(
    _client, mock_info, mock_iter, mock_stream
):
    # 100 replayed lines span 10s, well inside the 30s overlap.
    mock_info.return_value = RestRunnerInfo(started_at=_T0, ended_at=None)
    dense = [
        {
            "timestamp": (_T0 + timedelta(milliseconds=100 * i)).isoformat(),
            "message": str(i),
        }
        for i in reversed(range(300))
    ]
    mock_iter.side_effect = lambda *a, **k: iter(dense)
    mock_stream.return_value = iter([])

    args = parse_args(["runners", "logs", "rid", "--follow", "--json"])
    args.console = MagicMock()
    args.func(args)

    assert mock_stream.call_args[0][2] == _T0 + timedelta(seconds=20)


@patch("fal.cli.runners._stream_logs")
@patch("fal.cli.runners._iter_logs")
@patch("fal.cli.runners._get_runner_info")
@patch("fal.cli.runners.SyncServerlessClient")
def test_follow_dedupe_matches_reformatted_timestamps_and_keeps_repeats(
    _client, mock_info, mock_iter, mock_stream
):
    mock_info.return_value = RestRunnerInfo(started_at=_T0, ended_at=None)
    ts = _T0 + timedelta(seconds=1)
    mock_iter.side_effect = lambda *a, **k: iter(
        [{"timestamp": ts.isoformat(), "message": "dot"}]
    )
    # Same line in another timestamp format, then a genuine repeat.
    zulu = ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    mock_stream.return_value = iter(
        [
            {"timestamp": zulu, "message": "dot"},
            {"timestamp": zulu, "message": "dot"},
        ]
    )

    args = parse_args(["runners", "logs", "rid", "--follow", "--json"])
    args.console = MagicMock()
    args.func(args)

    assert len(args.console.print.call_args_list) == 2


def test_follow_start_survives_unparseable_timestamps():
    replay = [{"timestamp": "not-a-time", "message": "x"}]
    start = _follow_start(replay, 100)
    assert start <= datetime.now(timezone.utc) - FOLLOW_STREAM_OVERLAP


def test_follow_rejects_oldest_lines():
    args = parse_args(["runners", "logs", "rid", "--follow", "-n", "+5"])
    args.console = MagicMock()
    args.parser = MagicMock()
    args.parser.error.side_effect = SystemExit(2)
    with patch("fal.cli.runners.SyncServerlessClient"), patch(
        "fal.cli.runners._get_runner_info",
        return_value=RestRunnerInfo(started_at=_T0, ended_at=None),
    ):
        with pytest.raises(SystemExit):
            args.func(args)
    assert "--follow" in args.parser.error.call_args[0][0]


@patch("fal.cli.runners._stream_logs")
@patch("fal.cli.runners._iter_logs")
@patch("fal.cli.runners._get_runner_info")
@patch("fal.cli.runners.SyncServerlessClient")
def test_follow_with_since_skips_replay(_client, mock_info, mock_iter, mock_stream):
    mock_info.return_value = RestRunnerInfo(started_at=_T0, ended_at=None)
    mock_stream.return_value = iter([])

    args = parse_args(["runners", "logs", "rid", "--follow", "--since", "1h"])
    args.console = MagicMock()
    args.func(args)

    mock_iter.assert_not_called()
    assert mock_stream.call_args[0][2] is not None


@pytest.mark.parametrize(
    ("extra", "expect_since"), [([], False), (["--since", "1h"], True)]
)
@patch("fal.cli.runners._stream_logs")
@patch("fal.cli.runners._iter_logs")
@patch("fal.cli.runners._get_runner_info")
@patch("fal.cli.runners.SyncServerlessClient")
def test_follow_finished_runner_reads_history(
    _client, mock_info, mock_iter, mock_stream, extra, expect_since
):
    past = datetime(2020, 1, 1, tzinfo=timezone.utc)
    mock_info.return_value = RestRunnerInfo(
        started_at=past, ended_at=past + timedelta(minutes=5)
    )
    mock_iter.side_effect = lambda *a, **k: iter(_history_lines(150))

    args = parse_args(["runners", "logs", "rid", "--follow", "--json", *extra])
    args.console = MagicMock()
    args.func(args)

    mock_stream.assert_not_called()
    # One JSON object per line, oldest first, like a live --follow.
    logs = [json.loads(c[0][0]) for c in args.console.print.call_args_list]
    assert len(logs) == (150 if expect_since else 100)
    assert logs[-1]["message"] == "149"
    assert logs[0]["message"] == ("0" if expect_since else "50")
    assert (mock_iter.call_args[0][2] is not None) == expect_since
