import importlib
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import isolate_proto
from isolate.server.interface import from_grpc
from rich.console import Console

import fal.api.api as api_module
from fal.api.api import UserFunctionException, _handle_grpc_error

cli_main = importlib.import_module("fal.cli.main")


def test_main_shows_synthetic_remote_exception_for_deserialization_error(
    monkeypatch,
) -> None:
    console = Console(record=True, width=120, force_terminal=False, color_system=None)
    stringized_traceback = (
        "Traceback (most recent call last):\n"
        '  File "/app/handler.py", line 13, in boom\n'
        "huggingface_hub.errors.LocalEntryNotFoundError: test error\n"
        "additional remote detail\n"
    )
    result = isolate_proto.HostedRunResult(
        return_value=isolate_proto.SerializedObject(
            method="pickle",
            definition=b"not a pickle",
            was_it_raised=True,
            stringized_traceback=stringized_traceback,
            exception_type_name="LocalEntryNotFoundError",
            exception_message="test error\nadditional remote detail",
        ),
    )

    @_handle_grpc_error()
    def fail(_args):
        from_grpc(result)

    monkeypatch.setattr(cli_main, "console", console)
    monkeypatch.setattr(cli_main, "_check_latest_version", lambda: None)
    monkeypatch.setattr(cli_main, "debugtools", lambda _args: nullcontext())
    monkeypatch.setattr(
        cli_main, "parse_args", lambda _argv: SimpleNamespace(func=fail)
    )
    warning = MagicMock()
    monkeypatch.setattr(api_module, "logger", SimpleNamespace(warning=warning))

    assert cli_main.main([]) == 1
    output = console.export_text()
    remote_exception = "LocalEntryNotFoundError: test error"

    assert output.index("Traceback (most recent call last)") < output.index(
        remote_exception
    )
    assert "additional remote detail" in output
    assert "in boom:13" in output
    assert "Remote exception class was not importable locally" not in output
    assert "Unhandled user exception" in output
    assert "\nException\n" not in output
    assert "ExceptionDeserializationError" not in output
    assert "UnpicklingError" not in output
    assert "invalid load key" not in output
    assert "Error while deserializing the given object" not in output
    warning.assert_called_once_with(
        "Failed to deserialize remote exception",
        exc_info=True,
    )


def test_remote_exception_deserialization_is_user_function_exception() -> None:
    stringized_traceback = (
        "Traceback (most recent call last):\n"
        '  File "/app/handler.py", line 13, in boom\n'
        "dvc.exceptions.DvcException: This is a test error\n"
    )
    result = isolate_proto.HostedRunResult(
        return_value=isolate_proto.SerializedObject(
            method="pickle",
            definition=b"not a pickle",
            was_it_raised=True,
            stringized_traceback=stringized_traceback,
            exception_type_name="DvcException",
            exception_message="This is a test error",
        ),
    )

    @_handle_grpc_error()
    def fail():
        from_grpc(result)

    try:
        fail()
    except UserFunctionException as exc:
        assert str(exc) == "Uncaught user function exception"
        assert type(exc.__cause__).__name__ == "DvcException"
        assert str(exc.__cause__) == "This is a test error"
        assert exc.__cause__.__cause__ is None
    else:
        raise AssertionError("expected UserFunctionException")
