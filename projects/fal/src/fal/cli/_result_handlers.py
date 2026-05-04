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
        from rich.rule import Rule  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

        from fal.flags import URL_OUTPUT  # noqa: PLC0415

        self.console.print("")

        lines = Text()
        AUTH_EXPLANATIONS = {
            "public": "no authentication required",
            "private": "only you/team can access",
            "shared": "any authenticated user can access",
        }
        auth_desc = AUTH_EXPLANATIONS.get(self.auth_mode, self.auth_mode)
        lines.append(f"▸ Auth: {self.auth_mode} ", style="bold")
        lines.append(f"({auth_desc})\n\n", style="dim")

        if URL_OUTPUT != "none":
            lines.append("▸ Playground ", style="bold")
            lines.append("(open in browser)\n", style="dim")
            for endpoint in self.endpoints:
                lines.append(f"  {urls.playground}{endpoint}\n", style="cyan")

        if URL_OUTPUT == "all":
            lines.append("\n")
            lines.append("▸ API Endpoints ", style="bold")
            lines.append("(use in code)\n", style="dim")
            for endpoint in self.endpoints:
                lines.append(f"  Sync   {urls.run}{endpoint}\n", style="cyan")
                lines.append(f"  Async  {urls.queue}{endpoint}\n", style="cyan")

        title = Text(f"Ephemeral App ({self.auth_mode})", style="bold")
        subtitle = Text("Deleted when process exits", style="dim")
        self.console.print(Rule(title, style="green"))
        self.console.print(lines)
        self.console.print(Rule(subtitle, style="green"))

    def on_log(self, log: Any) -> None:
        # Obsolete messages from before service_urls were added.
        if (
            "Access the playground at" in log.message
            or "And API access through" in log.message
        ):
            return
        self.log_printer.print(log)


class CliRegisterResultHandler(ResultHandler):
    """Streams build/deploy logs for ``fal deploy``."""

    def __init__(self, *, console: Console) -> None:
        self.console = console
        self.log_printer = IsolateLogPrinter(debug=flags.DEBUG)

    def on_log(self, log: Any) -> None:
        self.log_printer.print(log)
