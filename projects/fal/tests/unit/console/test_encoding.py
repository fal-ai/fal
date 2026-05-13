from __future__ import annotations

import io
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from enum import IntEnum
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
from rich.console import Console

from fal.cli.runners import LogPrinter
from fal.console.encoding import make_terminal_safe
from fal.logging.isolate import IsolateLogPrinter


class StrictEncodingStream(io.StringIO):
    def __init__(self, encoding: str) -> None:
        super().__init__()
        self._encoding = encoding

    @property
    def encoding(self) -> str:
        return self._encoding

    def write(self, text: str) -> int:
        text.encode(self.encoding)
        return super().write(text)


class NoEncodingStream(io.StringIO):
    @property
    def encoding(self):
        return None


class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    STDERR = 30


class LogSource(IntEnum):
    BUILDER = 1
    BRIDGE = 2
    USER = 3


def test_make_terminal_safe_escapes_unencodable_characters() -> None:
    stream = StrictEncodingStream("cp1252")

    assert make_terminal_safe("hello 😊", stream) == "hello \\U0001f60a"


def test_make_terminal_safe_preserves_encodable_text() -> None:
    stream = StrictEncodingStream("utf-8")

    assert make_terminal_safe("hello 😊", stream) == "hello 😊"


def test_make_terminal_safe_preserves_text_without_stream_encoding() -> None:
    stream = NoEncodingStream()

    assert make_terminal_safe("hello 😊", stream) == "hello 😊"


@pytest.mark.parametrize(
    ("level", "stream_name"),
    [(LogLevel.INFO, "stdout"), (LogLevel.STDERR, "stderr")],
)
def test_isolate_log_printer_escapes_user_log_for_stream_encoding(
    level: LogLevel, stream_name: str
) -> None:
    stream = StrictEncodingStream("cp1252")
    log = SimpleNamespace(
        message="remote 😊",
        level=level,
        source=LogSource.USER,
    )
    printer = IsolateLogPrinter()
    printer._maybe_print_header = lambda source: None

    with _isolate_logs_module(), patch.object(sys, stream_name, stream):
        printer.print(log)

    assert stream.getvalue() == "remote \\U0001f60a\n"


def test_isolate_log_printer_escapes_rendered_system_log_for_stdout_encoding() -> None:
    stream = StrictEncodingStream("cp1252")
    log = SimpleNamespace(
        message="remote 😊",
        level=LogLevel.INFO,
        source=LogSource.BUILDER,
        bound_env=None,
    )
    printer = IsolateLogPrinter()
    printer._maybe_print_header = lambda source: None

    with _isolate_logs_module(), patch.object(sys, "stdout", stream):
        printer.print(log)

    assert "\\U0001f60a" in stream.getvalue()


def test_runner_log_printer_escapes_rendered_log_for_console_encoding() -> None:
    stream = StrictEncodingStream("cp1252")
    console = Console(file=stream, force_terminal=False, color_system=None, width=120)
    printer = LogPrinter(console)

    printer.print(
        {
            "timestamp": "2026-05-19T12:34:56Z",
            "level": "info",
            "message": "remote 😊",
        }
    )

    assert "\\U0001f60a" in stream.getvalue()


@contextmanager
def _isolate_logs_module() -> Iterator[None]:
    isolate_module = ModuleType("isolate")
    logs_module = ModuleType("isolate.logs")
    logs_module.LogLevel = LogLevel
    logs_module.LogSource = LogSource
    isolate_module.logs = logs_module

    with patch.dict(
        sys.modules,
        {
            "isolate": isolate_module,
            "isolate.logs": logs_module,
        },
    ):
        yield
