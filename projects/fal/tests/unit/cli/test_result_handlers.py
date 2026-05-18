from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock

from isolate.logs import LogSource
from rich.console import Console

from fal.cli._result_handlers import (
    CliBuildEnvironmentResultHandler,
    CliRegisterResultHandler,
    CliRunResultHandler,
)
from fal.sdk import BuildEnvironmentResult, HostedRunState, HostedRunStatus


def test_cli_run_result_handler_skips_builder_logs():
    handler = CliRunResultHandler(
        console=MagicMock(),
        auth_mode="public",
        endpoints=["/"],
    )
    handler.log_printer = MagicMock()

    handler.on_log(SimpleNamespace(source=LogSource.BUILDER, message="cache hit"))

    handler.log_printer.print.assert_not_called()


def test_cli_register_result_handler_skips_builder_logs():
    handler = CliRegisterResultHandler(console=MagicMock())
    handler.log_printer = MagicMock()

    handler.on_log(SimpleNamespace(source=LogSource.BUILDER, message="cache hit"))

    handler.log_printer.print.assert_not_called()


def test_cli_build_environment_result_handler_uses_minimal_cache_hit_footer():
    output = StringIO()
    console = Console(
        file=output,
        force_terminal=False,
        color_system=None,
        width=80,
    )
    handler = CliBuildEnvironmentResultHandler(console=console)

    handler(BuildEnvironmentResult(status=HostedRunStatus(HostedRunState.SUCCESS)))

    rendered = output.getvalue()
    assert "Build complete" in rendered
    assert "Building environment" not in rendered
    assert "─" not in rendered
