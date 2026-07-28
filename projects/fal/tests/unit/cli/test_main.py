import importlib
import os
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import isolate_proto
from isolate.server.interface import from_grpc
from rich.console import Console

import fal.api.api as api_module
from fal.api.api import UserFunctionException, _handle_grpc_error

cli_main = importlib.import_module("fal.cli.main")
fal_version = importlib.import_module("fal._version")


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


def test_update_check_can_be_disabled(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("check_updates = false\n")
    monkeypatch.setenv("FAL_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("FAL_PROFILE", raising=False)

    get_latest_version = MagicMock(return_value="99.0.0")
    monkeypatch.setattr(fal_version, "get_latest_version", get_latest_version)
    monkeypatch.setattr(fal_version, "version_tuple", (1, 0, 0))

    test_console = SimpleNamespace(is_terminal=True, print=MagicMock())
    monkeypatch.setattr(cli_main, "console", test_console)

    cli_main._check_latest_version()

    get_latest_version.assert_not_called()
    test_console.print.assert_not_called()


def _render_print_error(monkeypatch, msg, *, width=120, styles=False, **console_kwargs):
    console = Console(
        record=True,
        width=width,
        force_terminal=console_kwargs.pop("force_terminal", False),
        color_system=console_kwargs.pop("color_system", None),
        soft_wrap=True,
        **console_kwargs,
    )
    monkeypatch.setattr(cli_main, "console", console)
    cli_main._print_error(msg)
    return console.export_text(styles=styles)


def test_print_error_single_line_is_unchanged(monkeypatch) -> None:
    output = _render_print_error(monkeypatch, "You must run `fal run` first.")
    assert output.strip() == "✘ You must run `fal run` first."


def test_print_error_multiline_hangs_under_the_cross(monkeypatch) -> None:
    output = _render_print_error(
        monkeypatch,
        "New revision did not become ready.\n"
        "Check the logs: https://fal.ai/logs?revisionId=abc-123",
        width=120,
    )
    lines = [line for line in output.split("\n") if line]
    assert lines[0].startswith("✘ New revision did not become ready.")
    assert lines[1] == "  Check the logs: https://fal.ai/logs?revisionId=abc-123"
    # every continuation line hangs under the message instead of at column 0
    assert all(line.startswith("  ") for line in lines[1:])


def test_print_error_wraps_long_line_with_hanging_indent(monkeypatch) -> None:
    output = _render_print_error(monkeypatch, "alpha " * 20, width=24)
    lines = [line for line in output.split("\n") if line]
    assert len(lines) > 1  # soft-wrapped into multiple visual lines
    assert lines[0].startswith("✘ ")
    assert all(line.startswith("  ") for line in lines[1:])


def test_print_error_empty_message_prints_bare_cross(monkeypatch) -> None:
    output = _render_print_error(monkeypatch, "")
    assert output.strip() == "✘"


def test_print_error_preserves_repr_highlighting(monkeypatch) -> None:
    def render(msg):
        return _render_print_error(
            monkeypatch,
            msg,
            width=200,
            force_terminal=True,
            color_system="standard",
            styles=True,
        )

    with_url = render("Check the logs: https://fal.ai/logs?revisionId=abc-123")
    without = render("no highlightable tokens present here")
    # URL / number highlighting adds ANSI styling beyond the red cross glyph;
    # building a Text without re-applying the highlighter would drop it.
    assert with_url.count("\x1b[") > without.count("\x1b[")


def test_print_error_renders_with_cp1252_output() -> None:
    script = "\n".join(
        [
            "from fal.cli.main import _print_error",
            '_print_error("Setup failed within the startup timeout.\\n"',
            '             "Check the logs: https://fal.ai/logs?app=fail-app&revisionId=abc-123")',
        ]
    )
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"startup timeout" in result.stdout
    assert b"Check the logs" in result.stdout
