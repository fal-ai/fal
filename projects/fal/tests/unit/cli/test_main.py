import importlib
import io
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


def test_update_panel_centers_both_lines(monkeypatch, tmp_path) -> None:
    # A long upgrade command (the spelled-out interpreter fallback) is wider
    # than the headline, so the headline has to be centered against it too.
    config_path = tmp_path / "config.toml"
    config_path.write_text("")
    monkeypatch.setenv("FAL_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("FAL_PROFILE", raising=False)

    monkeypatch.setattr(fal_version, "get_latest_version", lambda: "99.0.0")
    monkeypatch.setattr(fal_version, "version_tuple", (1, 0, 0))

    command = (
        "/opt/homebrew/opt/python@3.14/bin/python3.14 -m pip install --upgrade fal"
    )
    monkeypatch.setattr(
        importlib.import_module("fal._installer"),
        "get_upgrade_command",
        lambda *_args, **_kwargs: command,
    )

    console = Console(
        record=True, width=120, force_terminal=True, color_system=None, no_color=True
    )
    monkeypatch.setattr(cli_main, "console", console)

    cli_main._check_latest_version()

    lines = [line for line in console.export_text().splitlines() if "│" in line]
    headline = next(line for line in lines if "A new version" in line)
    command_line = next(line for line in lines if command in line)

    def padding(line: str) -> tuple:
        body = line.strip("│")
        return len(body) - len(body.lstrip()), len(body) - len(body.rstrip())

    # Nothing truncated, and the headline is centered rather than left-flush.
    assert command in command_line
    left, right = padding(headline)
    assert abs(left - right) <= 1
    assert left > padding(command_line)[0]


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


# A logs URL with a UUID revision id: 116 chars once the "  Check the logs: "
# gutter is on it, so it overruns rich's 80-column non-terminal default.
LONG_LOGS_URL = (
    "https://fal.ai/dashboard/logs?app=my-app"
    "&revisionId=01937c8e-6f4b-7c3d-9e2a-1234567890ab"
)


def _render_print_error(
    monkeypatch,
    msg,
    *,
    width=120,
    styles=False,
    force_terminal=True,
    color_system=None,
    soft_wrap=True,
    **console_kwargs,
):
    # file=StringIO keeps rich from deriving ascii_only from pytest's captured
    # stdout, so the exact-"✘" assertions below hold under `pytest -s` too.
    console = Console(
        record=True,
        file=io.StringIO(),
        width=width,
        force_terminal=force_terminal,
        color_system=color_system,
        soft_wrap=soft_wrap,
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


def test_print_error_keeps_a_long_url_whole_when_piped(monkeypatch) -> None:
    """`fal deploy | tee`, CI logs, redirects: not a terminal, so rich falls back
    to 80 columns. Folding there would insert a real newline mid-URL, which no
    terminal rejoins on copy and no single-line `grep revisionId=` matches."""
    output = _render_print_error(
        monkeypatch,
        f"New revision did not become ready.\nCheck the logs: {LONG_LOGS_URL}",
        width=80,
        force_terminal=False,
        soft_wrap=False,
    )
    assert f"  Check the logs: {LONG_LOGS_URL}" in output.split("\n")


def test_print_error_keeps_a_long_url_whole_in_a_narrow_terminal(monkeypatch) -> None:
    """Same guarantee on a terminal too small for the URL: prose still wraps at
    word boundaries, but the URL overflows onto its own line rather than folding
    — the terminal soft-wraps it, and terminals rejoin soft-wraps on copy."""
    output = _render_print_error(
        monkeypatch,
        f"New revision did not become ready.\nCheck the logs: {LONG_LOGS_URL}",
        width=40,
    )
    lines = output.split("\n")
    assert f"  {LONG_LOGS_URL}" in lines
    assert "✘ New revision did not become ready." in lines


def test_print_error_blank_lines_stay_empty(monkeypatch) -> None:
    """A gutter prepended unconditionally would turn blank lines into two spaces,
    which shows up in captured CI logs and exact-match greps."""
    output = _render_print_error(monkeypatch, "reason\n\ndetail\n")
    assert output.split("\n")[:4] == ["✘ reason", "", "  detail", ""]


def test_print_error_leaves_no_trailing_whitespace(monkeypatch) -> None:
    # Text.wrap only strips whitespace beyond the wrap width, so the space that
    # broke "logs: " from the URL survives inside the budget without an rstrip.
    output = _render_print_error(
        monkeypatch, f"Check the logs: {LONG_LOGS_URL}", width=40
    )
    assert not any(line != line.rstrip() for line in output.split("\n"))


def test_print_error_does_not_interpret_markup(monkeypatch) -> None:
    """The main win of rendering via Text() rather than console.print(str):
    brackets in a server-controlled message are literal text. Before, a token
    like `[expected int]` was silently eaten as a style tag, and a stray
    close-tag `[/red]` raised MarkupError from inside main()'s except handler."""
    output = _render_print_error(monkeypatch, "invalid argument [expected int] got str")
    assert output.strip() == "✘ invalid argument [expected int] got str"

    output = _render_print_error(monkeypatch, "KeyError: [/red] unmatched")
    assert output.strip() == "✘ KeyError: [/red] unmatched"


def test_print_error_preserves_repr_highlighting(monkeypatch) -> None:
    def render(msg):
        return _render_print_error(
            monkeypatch,
            msg,
            width=200,
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
    # rich sizes non-terminal consoles from COLUMNS too; a leaked narrow value
    # (tmux, some Docker images, CI wrappers) would split the asserted phrase.
    env.pop("COLUMNS", None)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"startup timeout" in result.stdout
    assert b"Check the logs" in result.stdout


def test_main_ignores_incomplete_argcomplete_environment(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("check_updates = false\n")
    env = os.environ.copy()
    env["FAL_CONFIG_PATH"] = str(config_path)
    env["_ARGCOMPLETE"] = "1"
    env.pop("COMP_LINE", None)
    env.pop("COMP_POINT", None)

    result = subprocess.run(
        [sys.executable, "-m", "fal", "--definitely-invalid"],
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 2
    assert b"error:" in result.stderr


def test_main_prints_deploy_failure_reason_and_logs_link(monkeypatch) -> None:
    """isolate-cloud#8647 (CLI boundary): a failed deploy comes back as a
    grpc.RpcError whose details() carry the categorized reason + logs link; main()
    must surface that to the terminal via _print_error, not swallow it. Guards the
    seam between the gRPC error status and the rendered ``✘`` line."""
    import grpc

    reason = (
        "New revision did not become ready within the startup timeout — the runner "
        "never passed its health check (often a setup() failure or a crash-looping "
        "runner).\n"
        "Check the logs: https://fal.ai/dashboard/logs"
        "?app=fail-app&revisionId=rev-new-123"
    )

    class _DeployRpcError(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.FAILED_PRECONDITION

        def details(self):
            return reason

    def deploy(_args):
        raise _DeployRpcError()

    console = Console(record=True, width=200, force_terminal=False, color_system=None)
    monkeypatch.setattr(cli_main, "console", console)
    monkeypatch.setattr(cli_main, "_check_latest_version", lambda: None)
    monkeypatch.setattr(cli_main, "debugtools", lambda _args: nullcontext())
    monkeypatch.setattr(
        cli_main, "parse_args", lambda _argv: SimpleNamespace(func=deploy)
    )

    assert cli_main.main([]) == 1

    output = console.export_text()
    # The categorized reason and the revision-scoped logs link both reach stdout.
    assert "New revision did not become ready within the startup timeout" in output
    assert "Check the logs:" in output
    assert "revisionId=rev-new-123" in output
    # ...rendered as an error (cross glyph, ascii-fallback tolerant).
    assert output.lstrip().startswith(("✘", "x "))
