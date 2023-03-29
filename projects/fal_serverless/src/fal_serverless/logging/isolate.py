from __future__ import annotations

from isolate.logs import Log, LogLevel
from structlog.dev import ConsoleRenderer
from structlog.processors import TimeStamper
from structlog.typing import EventDict

from .style import LEVEL_STYLES

_renderer = ConsoleRenderer(level_styles=LEVEL_STYLES)

_timestamper = TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False)


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
        }
        if self.debug:
            bound_env = log.bound_env.key if log.bound_env is not None else "global"
            event["bound_env"] = bound_env

        # Use structlog processors to get consistent output with local logs
        event = _timestamper.__call__(logger={}, name=level, event_dict=event)
        message = _renderer.__call__(logger={}, name=level, event_dict=event)
        print(message)
