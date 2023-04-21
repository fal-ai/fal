from __future__ import annotations

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
        event: EventDict = {
            "event": log.message,
            "level": level,
            "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        }
        if self.debug and log.bound_env and log.bound_env.key != "global":
            event["bound_env"] = log.bound_env.key

        # Use structlog processors to get consistent output with local logs
        message = _renderer.__call__(logger={}, name=level, event_dict=event)
        print(message)
