import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import isolate_proto
from isolate.server.interface import from_grpc
from rich.console import Console

from fal.api.api import _handle_grpc_error

cli_main = importlib.import_module("fal.cli.main")


def test_main_shows_remote_exception_without_local_deserialization_traceback(
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

    assert cli_main.main([]) == 1
    output = console.export_text()
    remote_exception = "huggingface_hub.errors.LocalEntryNotFoundError: test error"

    assert "The application raised this error on the runner" in output
    assert output.index("Traceback (most recent call last)") < output.index(
        remote_exception
    )
    assert "additional remote detail" in output
    assert "in boom:13" in output
    assert "\nException\n" not in output
    assert "pickle data was truncated" not in output
