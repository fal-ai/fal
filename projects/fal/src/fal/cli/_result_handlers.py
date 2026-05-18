"""CLI-specific ResultHandler subclasses.

Built directly on the ``ResultHandler`` base so the CLI's presentation evolves
independently of whatever default handlers the ``fal.api`` layer happens to ship.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fal import flags
from fal.api.api import ResultHandler
from fal.logging.isolate import IsolateLogPrinter

if TYPE_CHECKING:
    from rich.console import Console


class CliRunResultHandler(ResultHandler):
    """Renders an ephemeral-app banner + streams logs for ``fal run``."""

    def __init__(
        self,
        *,
        console: Console,
        auth_mode: str,
        endpoints: list[str],
    ) -> None:
        self.console = console
        self.auth_mode = auth_mode
        self.endpoints = endpoints
        self.log_printer = IsolateLogPrinter(debug=flags.DEBUG)

    def on_service_urls(self, urls: Any) -> None:
        from rich.text import Text  # noqa: PLC0415

        from fal.console.icons import get_section_icon  # noqa: PLC0415
        from fal.console.rules import print_rule  # noqa: PLC0415
        from fal.flags import URL_OUTPUT  # noqa: PLC0415

        section_icon = get_section_icon(self.console)
        self.console.print("")

        lines = Text()
        AUTH_EXPLANATIONS = {
            "public": "no authentication required",
            "private": "only you/team can access",
            "shared": "any authenticated user can access",
        }
        auth_desc = AUTH_EXPLANATIONS.get(self.auth_mode, self.auth_mode)
        lines.append(f"{section_icon} Auth: {self.auth_mode} ", style="bold")
        lines.append(f"({auth_desc})\n\n", style="dim")

        if URL_OUTPUT != "none":
            lines.append(f"{section_icon} Playground ", style="bold")
            lines.append("(open in browser)\n", style="dim")
            for endpoint in self.endpoints:
                lines.append(f"  {urls.playground}{endpoint}\n", style="cyan")

        if URL_OUTPUT == "all":
            lines.append("\n")
            lines.append(f"{section_icon} API Endpoints ", style="bold")
            lines.append("(use in code)\n", style="dim")
            for endpoint in self.endpoints:
                lines.append(f"  Sync   {urls.run}{endpoint}\n", style="cyan")
                lines.append(f"  Async  {urls.queue}{endpoint}\n", style="cyan")

        title = Text(f"Ephemeral App ({self.auth_mode})", style="bold")
        subtitle = Text("Deleted when process exits", style="dim")
        print_rule(self.console, title, style="green")
        self.console.print(lines)
        print_rule(self.console, subtitle, style="green")

    def on_log(self, log: Any) -> None:
        from isolate.logs import LogSource  # noqa: PLC0415

        # Obsolete messages from before service_urls were added.
        if (
            "Access the playground at" in log.message
            or "And API access through" in log.message
        ):
            return
        # The CLI orchestrates an explicit BuildEnvironment RPC before Run,
        # so any BUILDER-source log that arrives here is either a cache-hit
        # confirmation or a redundant rebuild we don't want to re-frame as
        # its own "Building environment..." phase.
        if log.source == LogSource.BUILDER:
            return
        self.log_printer.print(log)


class CliRegisterResultHandler(ResultHandler):
    """Streams build/deploy logs for ``fal deploy``."""

    def __init__(self, *, console: Console) -> None:
        self.console = console
        self.log_printer = IsolateLogPrinter(debug=flags.DEBUG)

    def on_log(self, log: Any) -> None:
        from isolate.logs import LogSource  # noqa: PLC0415

        # Build phase is rendered by CliBuildEnvironmentResultHandler in the
        # explicit pre-build step; the cache-hit confirmation the server
        # streams here would be a redundant second header.
        if log.source == LogSource.BUILDER:
            return
        self.log_printer.print(log)


class CliBuildEnvironmentResultHandler(ResultHandler):
    """Renders the build phase as a top-level CLI step, surrounded by a
    ``Building environment...`` header and a ``✓ Build complete`` footer.

    Used to drive an explicit ``BuildEnvironment`` RPC before
    ``Run`` / ``RegisterApplication`` so the CLI doesn't have to infer the
    build phase from the log stream's source field.
    """

    def __init__(self, *, console: Console) -> None:
        from isolate.logs import LogSource  # noqa: PLC0415

        self.console = console
        self.log_printer = IsolateLogPrinter(debug=flags.DEBUG)
        # IsolateLogPrinter tracks the current phase in _current_source. The CLI
        # renders the build-phase header / footer itself, so keep the printer in
        # BUILDER mode to avoid its own "Building environment..." transition.
        self.log_printer._current_source = LogSource.BUILDER
        self._header_printed = False

    def __call__(self, partial_result: Any) -> None:
        from fal.sdk import HostedRunState  # noqa: PLC0415

        super().__call__(partial_result)
        status = getattr(partial_result, "status", None)
        if status is not None and status.state is HostedRunState.SUCCESS:
            self._print_footer()

    def _print_header_once(self) -> None:
        if self._header_printed:
            return

        from fal.console.rules import print_rule  # noqa: PLC0415

        self.console.print("Building environment...", style="bold")
        print_rule(self.console, style="dim")
        self._header_printed = True

    def _print_footer(self) -> None:
        from fal.console.icons import CHECK_ICON  # noqa: PLC0415
        from fal.console.rules import print_rule  # noqa: PLC0415

        if not self._header_printed:
            self.console.print(f"{CHECK_ICON} Build complete", style="bold green")
            self.console.print("")
            return

        print_rule(self.console, style="dim")
        self.console.print(f"{CHECK_ICON} Build complete", style="bold green")
        self.console.print("")

    def on_log(self, log: Any) -> None:
        self._print_header_once()
        self.log_printer.print(log)
