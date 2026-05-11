from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from structlog.dev import ConsoleRenderer
from structlog.typing import EventDict

from .style import LEVEL_STYLES

if TYPE_CHECKING:
    from isolate.logs import Log, LogSource
    from rich.console import Console

_renderer = ConsoleRenderer(level_styles=LEVEL_STYLES)


class IsolateLogPrinter:
    """Print streamed isolate logs with phase-header transitions.

    When ``console`` is omitted, phase headers go through the
    ``fal.console`` module-level console and raw log messages go
    through ``sys.stdout``/``sys.stderr`` (preserving the source's
    stdout/stderr split). When a ``console`` is provided, all output
    is routed through it — useful for callers (e.g. the CLI's
    metadata-probe phase) that wrap output with an indented or otherwise
    decorated console so the whole stream stays inside the wrap.
    """

    debug: bool

    def __init__(
        self,
        debug: bool = False,
        *,
        console: Console | None = None,
    ) -> None:
        self.debug = debug
        self._current_source: LogSource | None = None
        self._console_override = console

    def _resolve_console(self) -> Console:
        if self._console_override is not None:
            return self._console_override
        from fal.console import console as _global_console

        return _global_console

    def _emit_message(self, message: str, *, stderr: bool = False) -> None:
        if self._console_override is None:
            stream = sys.stderr if stderr else sys.stdout
            print(message, file=stream)
            return
        # Write to the console's underlying file so any wrapper around
        # it (e.g. an indented stream) sees the bytes — but skip rich's
        # rendering pipeline since the structlog renderer already baked
        # in ANSI escapes for the destination terminal.
        self._console_override.file.write(message + "\n")
        self._console_override.file.flush()

    def _maybe_print_header(self, source: LogSource):
        from isolate.logs import LogSource
        from rich.rule import Rule

        from fal.console.icons import CHECK_ICON

        if source == self._current_source:
            return

        console = self._resolve_console()

        # Print build completion when transitioning out of BUILDER phase
        if self._current_source == LogSource.BUILDER:
            console.print(Rule(style="dim"))
            console.print(f"{CHECK_ICON} Build complete", style="bold green")
            console.print("")

        # Print phase header when entering a new phase
        if source == LogSource.BUILDER:
            console.print("Building environment...", style="bold")
            console.print(Rule(style="dim"))
        elif source == LogSource.BRIDGE:
            console.print("Setting up runtime...", style="bold")
        elif source == LogSource.USER:
            console.print("Running...", style="bold")

        self._current_source = source

    def print(self, log: Log):
        from isolate.logs import LogLevel, LogSource

        # Skip depot build summary links (users can't access them)
        if "https://depot.dev" in log.message:
            return

        if log.level < LogLevel.INFO and not self.debug:
            return

        self._maybe_print_header(log.source)

        if log.source == LogSource.USER:
            self._emit_message(
                log.message,
                stderr=log.level == LogLevel.STDERR,
            )
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
        self._emit_message(message)
