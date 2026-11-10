import io
import os
import subprocess
from unittest.mock import MagicMock

import pytest

from fal.dev import execute
from fal.dev.remote import signals

MARKER = b"__fal_dev_exec_abc:"


def _filter():
    out = io.BytesIO()
    return execute.MarkerFilter(out, MARKER), out


def test_filter_passes_output_and_swallows_the_marker_across_chunks():
    f, out = _filter()
    data = b"line 1\nline 2" + MARKER + b"17\n"
    for i in range(0, len(data), 3):  # split inside the marker too
        f.write(data[i : i + 3])
    f.finish()

    assert out.getvalue() == b"line 1\nline 2"
    assert f.complete and f.code == 17


def test_filter_without_a_marker_is_incomplete_and_loses_nothing():
    f, out = _filter()
    f.write(b"partial output")
    f.finish()

    assert out.getvalue() == b"partial output"
    assert not f.complete


def test_filter_marker_without_its_newline_is_not_complete():
    f, _ = _filter()
    f.write(b"x" + MARKER + b"0")

    assert f.found and not f.complete


def test_wrap_runs_the_command_after_a_fresh_marker():
    a, marker_a = execute.wrap(["seq", "3"])
    b, marker_b = execute.wrap(["seq", "3"])

    assert a[:2] == ["sh", "-c"] and a[3] == "sh" and a[6:] == ["seq", "3"]
    assert a[2].startswith(signals.EXEC_TAG)
    assert marker_a != marker_b and marker_a.decode() == a[4] + a[5]
    # Never whole in the process list, so `ps aux` cannot print it.
    assert marker_a.decode() not in " ".join(a)


def _run_wrapped(command):
    wrapped, marker = execute.wrap(command)
    # Skip the lingering sleep: only the drain behaviour matters here.
    wrapped[2] = wrapped[2].replace(f"sleep {execute.LINGER_SECONDS}", "")
    proc = subprocess.run(wrapped, capture_output=True, check=False)
    return proc, marker


def test_the_wrapper_ends_both_streams_with_the_exit_code():
    proc, marker = _run_wrapped(["sh", "-c", "echo out; echo err >&2; exit 7"])

    assert proc.stdout == b"out\n" + marker + b"7\n"
    assert proc.stderr == b"err\n" + marker + b"7\n"
    assert proc.returncode == 7


def _fake_session(frames_out, frames_err=()):
    def session(client, runner_id, command, **kwargs):
        for chunk in frames_out:
            kwargs["stdout"].write(chunk)
            if kwargs["done"]():
                break
        for chunk in frames_err:
            kwargs["stderr"].write(chunk)
        return 1

    return session


@pytest.fixture
def streams(monkeypatch):
    monkeypatch.setattr(execute.secrets, "token_hex", lambda n: "abc")
    return io.BytesIO(), io.BytesIO()


def test_run_returns_the_commands_exit_code(monkeypatch, streams):
    monkeypatch.setattr(
        "fal.cli.runners.shell_session",
        _fake_session([b"hello\n", MARKER + b"3\n"], [MARKER + b"3\n"]),
    )

    assert execute.run(MagicMock(), "r1", ["x"], MagicMock(), *streams) == 3
    assert streams[0].getvalue() == b"hello\n"


def test_run_never_reports_success_for_a_truncated_stream(monkeypatch, streams):
    console = MagicMock()
    # The stream ended after part of the output: no marker on either stream.
    monkeypatch.setattr("fal.cli.runners.shell_session", _fake_session([b"1\n2\n"]))

    code = execute.run(MagicMock(), "r1", ["seq"], console, *streams)

    assert code == execute.TRUNCATED_EXIT
    assert streams[0].getvalue() == b"1\n2\n"
    assert "incomplete" in console.print.call_args.args[0]


def test_run_needs_the_stderr_marker_too(monkeypatch, streams):
    monkeypatch.setattr(
        "fal.cli.runners.shell_session", _fake_session([b"a\n", MARKER + b"0\n"])
    )

    assert (
        execute.run(MagicMock(), "r1", ["x"], MagicMock(), *streams)
        == execute.TRUNCATED_EXIT
    )


def _fake_proc(tmp_path, uptime=100000.0):
    (tmp_path / "uptime").write_text(f"{uptime} 0.0\n")

    def process(pid, *argv, age=10.0):
        (tmp_path / pid).mkdir()
        (tmp_path / pid / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv))
        ticks = int((uptime - age) * os.sysconf("SC_CLK_TCK"))
        fields = ["S"] + ["0"] * 18 + [str(ticks)]  # starttime is field 22
        (tmp_path / pid / "stat").write_text(f"{pid} (sh x) " + " ".join(fields))

    return process


def test_exec_commands_counts_only_tagged_shells(tmp_path):
    process = _fake_proc(tmp_path)
    tagged = signals.EXEC_TAG + "\nsleep 30\n"
    process("1", "sh", "-c", tagged, "sh", "m", "x")
    process("2", "sh", "-c", tagged, "sh", "m", "y")
    process("3", "sh", "-c", "sleep 30", "sh")  # e.g. the dashboard's probe
    process("4", "python3", "-c", signals.EXEC_TAG)

    assert signals.exec_commands(str(tmp_path)) == 2
    assert signals.exec_commands(str(tmp_path / "missing")) == 0


def test_an_abandoned_exec_stops_counting_as_activity(tmp_path):
    process = _fake_proc(tmp_path)
    tagged = signals.EXEC_TAG + "\nsleep 1"
    process("1", "sh", "-c", tagged, "sh", age=60)
    process("2", "sh", "-c", tagged, "sh", age=signals.EXEC_MAX_AGE_SECONDS + 60)

    assert signals.exec_commands(str(tmp_path)) == 1
