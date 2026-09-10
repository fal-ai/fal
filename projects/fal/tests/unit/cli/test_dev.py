from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from fal.cli.dev import (
    _DEFAULT_IDLE_TIMEOUT,
    _DEFAULT_MACHINE_TYPE,
    _dev,
    _DevboxResultHandler,
    _DevboxStartup,
    _project_key,
    _read_runner_id,
    _wait_for_runner_id,
    _write_runner_id,
)
from fal.cli.main import parse_args
from fal.sdk import RunnerState


def _runner(runner_id: str, state: RunnerState = RunnerState.RUNNING):
    runner = MagicMock()
    runner.runner_id = runner_id
    runner.state = state
    runner.alias = ""
    return runner


def test_dev_parser_defaults():
    args = parse_args(["dev"])

    assert args.func == _dev
    assert args.machine_type == _DEFAULT_MACHINE_TYPE
    assert args.idle_timeout == _DEFAULT_IDLE_TIMEOUT


def test_project_key_depends_on_project_and_connection(tmp_path):
    key = _project_key(tmp_path, "host", "team")

    assert key == _project_key(tmp_path, "host", "team")
    assert key != _project_key(tmp_path, "other-host", "team")
    assert key != _project_key(tmp_path, "host", "other-team")


def test_runner_state_round_trip(tmp_path):
    path = tmp_path / "nested" / "state.json"

    _write_runner_id(path, "runner-id")

    assert _read_runner_id(path) == "runner-id"


def test_devbox_result_handler_captures_runner_id_when_runner_starts():
    from types import SimpleNamespace

    runner_ids = Queue()
    handler = _DevboxResultHandler(runner_ids)

    handler.on_log(SimpleNamespace(message="Starting runner runner-id"))
    assert runner_ids.empty()

    handler.on_log(SimpleNamespace(message="Runner started"))
    assert runner_ids.get_nowait() == "runner-id"


def test_wait_for_runner_id_propagates_background_error():
    error = RuntimeError("run failed")
    startup = _DevboxStartup(Queue(), Queue())
    startup.errors.put(error)

    with pytest.raises(RuntimeError, match="run failed"):
        _wait_for_runner_id(startup)


@patch("fal.cli.dev._attach", return_value=0)
@patch("fal.cli.dev._state_path")
@patch("fal.cli.dev.SyncServerlessClient")
def test_dev_reattaches_to_saved_live_runner(
    client_cls, state_path, attach, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "state.json"
    _write_runner_id(path, "runner-id")
    state_path.return_value = path
    client_cls.return_value.runners.list.return_value = [_runner("runner-id")]
    args = parse_args(["dev"])
    args.console = MagicMock()

    assert _dev(args) == 0

    attach.assert_called_once_with(args, "runner-id")
    assert "Reattaching" in args.console.print.call_args.args[0]


@patch("fal.cli.dev._attach", return_value=0)
@patch("fal.cli.dev._write_runner_id")
@patch("fal.cli.dev._wait_for_runner_id", return_value="new-runner")
@patch(
    "fal.cli.dev._start_devbox",
    return_value=_DevboxStartup(Queue(), Queue()),
)
@patch("fal.cli.dev._state_path", return_value=Path("state.json"))
@patch("fal.cli.dev.SyncServerlessClient")
def test_dev_creates_saves_and_attaches(
    client_cls,
    _state_path,
    start_devbox,
    wait_for_runner_id,
    write_runner_id,
    attach,
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    args = parse_args(["dev", "--machine-type", "GPU-H100"])
    args.console = MagicMock()

    assert _dev(args) == 0

    start_devbox.assert_called_once_with(args, client_cls.return_value, tmp_path)
    wait_for_runner_id.assert_called_once_with(start_devbox.return_value)
    client_cls.return_value.runners.list.assert_not_called()
    write_runner_id.assert_called_once_with(Path("state.json"), "new-runner")
    assert [call.args for call in args.console.print.call_args_list] == [
        ("Creating devbox...",),
        ("Runner id: new-runner",),
        ("Attached to devbox new-runner",),
    ]
    attach.assert_called_once_with(args, "new-runner")
