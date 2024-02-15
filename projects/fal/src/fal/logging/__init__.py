from __future__ import annotations

from typing import Any

import structlog
from structlog.typing import EventDict, WrappedLogger

from .style import LEVEL_STYLES
from .user import add_user_id

# Unfortunately structlog console processor does not support
# more general theming as a public API. Consider a PR on the
# structlog repo to add better support for it.
structlog.dev._ColorfulStyles.bright = ""


class DebugConsoleLogProcessor:
    """
    Log processor that prints to the console the well-formatted
    messages only when in debug mode (enabled by the `--debug` flag).
    """

    debug: bool = False
    renderer = structlog.dev.ConsoleRenderer(level_styles=LEVEL_STYLES)

    def __call__(self, logger: WrappedLogger, method_name: str, event_dict: EventDict):
        if self.debug:
            return self.renderer.__call__(logger, method_name, event_dict)
        raise structlog.DropEvent


_console_log_output = DebugConsoleLogProcessor()


def set_debug_logging(debug: bool):
    """Enable/disable console log output."""
    _console_log_output.debug = debug


structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
        add_user_id,
        _console_log_output,
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
)


def get_logger(*args: Any, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(*args, **initial_values)
