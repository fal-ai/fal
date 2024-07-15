from __future__ import annotations

import sys
from datetime import datetime, timezone

from isolate.logs import Log, LogLevel, LogSource
from structlog.dev import ConsoleRenderer
from structlog.typing import EventDict

from .style import LEVEL_STYLES

_renderer = ConsoleRenderer(level_styles=LEVEL_STYLES)


class IsolateLogPrinter:
    debug: bool

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self._current_source: LogSource | None = None

    def _maybe_print_header(self, source: LogSource):
        from fal.console import console

        if source == self._current_source:
            return

        msg = {
            LogSource.BUILDER: "Building the environment",
            LogSource.BRIDGE: "Unpacking user code",
            LogSource.USER: "Running",
        }.get(source)

        if msg:
            console.print(f"==> {msg}", style="bold green")

        self._current_source = source

    def print(self, log: Log):
        if log.level < LogLevel.INFO and not self.debug:
            return

        self._maybe_print_header(log.source)

        if log.source == LogSource.USER:
            stream = sys.stderr if log.level == LogLevel.STDERR else sys.stdout
            print(log.message, file=stream)
            return

        level = str(log.level)

        if hasattr(log, "timestamp"):
            timestamp = log.timestamp
        else:
            # Default value for timestamp if user has old `isolate` version.
            # Even if the controller version is controller by us, which means that
            # the timestamp is being sent in the gRPC message.
            # The `isolate` version users interpret that message with is out of our
            # control. So we need to handle this case.
            timestamp = datetime.now(timezone.utc)

        event: EventDict = {
            "event": log.message,
            "level": level,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        }
        if self.debug and log.bound_env and log.bound_env.key != "global":
            event["bound_env"] = log.bound_env.key

        # Use structlog processors to get consistent output with local logs
        message = _renderer.__call__(logger={}, name=level, event_dict=event)
        print(message)
