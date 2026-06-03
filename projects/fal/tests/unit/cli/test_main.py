from __future__ import annotations

import importlib
from contextlib import nullcontext
from types import SimpleNamespace, TracebackType

from rich.console import Console

from fal.api import FalSerializationError

cli_main = importlib.import_module("fal.cli.main")
REMOTE_ERROR = "huggingface_hub.errors.LocalEntryNotFoundError: test error"


def _remote_traceback() -> TracebackType:
    """Build a traceback that stands in for one reconstructed from a runner."""
    try:
        raise RuntimeError("remote boom")
    except RuntimeError as exc:
        assert exc.__traceback__ is not None
        return exc.__traceback__


def _run_failing_main(monkeypatch, error: FalSerializationError) -> str:
    console = Console(record=True, width=120, force_terminal=False, color_system=None)

    def fail(_args):
        raise error

    monkeypatch.setattr(cli_main, "console", console)
    monkeypatch.setattr(cli_main, "_check_latest_version", lambda: None)
    monkeypatch.setattr(cli_main, "debugtools", lambda _args: nullcontext())
    monkeypatch.setattr(
        cli_main, "parse_args", lambda _argv: SimpleNamespace(func=fail)
    )

    assert cli_main.main([]) == 1
    return console.export_text()


def test_main_labels_remote_traceback_with_remote_exception(monkeypatch) -> None:
    try:
        raise ModuleNotFoundError("No module named 'huggingface_hub'")
    except ModuleNotFoundError as cause:
        error = FalSerializationError(
            "Error while deserializing the given object. Could not find module "
            "'huggingface_hub'.",
            original_traceback=_remote_traceback(),
            remote_error=REMOTE_ERROR,
        )
        error.__cause__ = cause

    output = _run_failing_main(monkeypatch, error)
    flat = " ".join(output.split())

    assert "The application raised this error on the runner" in output
    assert "LocalEntryNotFoundError: test error" in flat
    assert "_remote_traceback" in output

    assert "direct cause" not in output
    assert "ModuleNotFoundError: No module named 'huggingface_hub'" not in output
    assert "Could not find module 'huggingface_hub'" in output


def test_main_shows_remote_error_without_traceback(monkeypatch) -> None:
    output = _run_failing_main(
        monkeypatch,
        FalSerializationError(
            "Error while deserializing the given object",
            remote_error=REMOTE_ERROR,
        ),
    )

    assert "The application raised this error on the runner" in output
    assert REMOTE_ERROR in output
    assert "Error while deserializing the given object" in output


def test_main_falls_back_to_local_traceback_without_remote(monkeypatch) -> None:
    output = _run_failing_main(
        monkeypatch,
        FalSerializationError("Error while deserializing the given object"),
    )

    assert "The application raised this error on the runner" not in output
    assert "FalSerializationError" in output
    assert "Error while deserializing the given object" in output
