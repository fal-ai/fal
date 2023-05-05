from __future__ import annotations

from datetime import datetime, timezone

from isolate.logs import Log, LogLevel
from structlog.dev import ConsoleRenderer
from structlog.typing import EventDict

from .style import LEVEL_STYLES

_renderer = ConsoleRenderer(level_styles=LEVEL_STYLES)


class IsolateLogPrinter:

    debug: bool

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    def print(self, log: Log):
        if log.level < LogLevel.INFO and not self.debug:
            return
        level = str(log.level)

        if hasattr(log, "timestamp"):
            timestamp = log.timestamp
        else:
            # Default value for timestamp if user has old `isolate` version.
            # Even if the controller version is controller by us, which means that the timestamp
            # is being sent in the gRPC message.
            # The `isolate` version users interpret that message with is out of our control.
            # So we need to handle this case.
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

    def print_dict(self, log: dict):
        level = LogLevel[log["level"]]
        if level < LogLevel.INFO and not self.debug:
            return
        if "timestamp" in log.keys():
            timestamp = log["timestamp"]
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

        event: EventDict = {
            "event": log["message"],
            "level": log["level"],
            "timestamp": timestamp[:-3],
        }

        message = _renderer.__call__(logger={}, name=log["level"], event_dict=event)
        print(message)
