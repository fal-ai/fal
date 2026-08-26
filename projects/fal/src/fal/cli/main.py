# PYTHON_ARGCOMPLETE_OK
import argparse

import rich

from fal._version import __version__
from fal.console import console
from fal.console.icons import get_cross_icon

from . import (
    api,
    apps,
    auth,
    completion,
    create,
    deploy,
    doctor,
    environments,
    files,
    keys,
    profile,
    queue,
    run,
    runners,
    secrets,
    teams,
)
from .debug import debugtools, get_debug_parser
from .parser import FalParser, FalParserExit

_CHECK_UPDATES_CONFIG_KEY = "check_updates"


def _get_main_parser() -> argparse.ArgumentParser:
    parents = [get_debug_parser()]
    parser = FalParser(
        prog="fal",
        parents=parents,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Show fal version.",
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        required=True,
    )

    for cmd in [
        api,
        auth,
        apps,
        environments,
        queue,
        deploy,
        run,
        keys,
        profile,
        secrets,
        doctor,
        create,
        runners,
        teams,
        files,
        completion,
    ]:
        cmd.add_parser(subparsers, parents)

    return parser


def parse_args(argv=None):
    parser = _get_main_parser()
    args = parser.parse_args(argv)
    args.console = console
    args.parser = parser
    return args


def _print_error(msg):
    # Print with a hanging indent: the cross leads the first line, continuation
    # lines are indented 2 columns. Multi-line deploy failures (a reason + a
    # "Check the logs: <url>" pointer) then read as one block instead of
    # collapsing to column 0 and blending into surrounding terminal output.
    from rich.text import Text

    # overflow="ignore" keeps a token longer than the wrap width — a logs URL
    # with a UUID revision id is ~116 chars — on one line instead of folding it
    # mid-token, so it stays clickable and copy-pasteable. (Text.wrap only
    # short-circuits wrapping entirely when "ignore" arrives as its *argument*,
    # not as the Text's own attribute, so word wrapping still happens here.)
    text = Text(str(msg), overflow="ignore")
    # Building a Text bypasses the ReprHighlighter that console.print(str) applies,
    # so re-apply it to keep URLs / numbers / paths styled (e.g. the logs link).
    # `_highlight` is rich's record of the `highlight=` constructor flag; checking
    # `console.highlighter` instead would be a no-op guard, since rich coerces a
    # `highlighter=None` into a NullHighlighter instance.
    if console._highlight:
        text = console.highlighter(text)

    if console.is_terminal:
        # The terminal soft-wraps continuation lines back to column 0, so wrap
        # here instead to keep them under the hanging indent.
        lines = text.wrap(console, max(console.width - 2, 1))
    else:
        # Redirected output (pipes, CI logs, files) has no visual width to wrap
        # to, and rich would fall back to 80 columns. Indent at the explicit
        # newlines only, so a single-line grep over a CI log still matches.
        lines = text.split(allow_blank=True)

    for i, line in enumerate(lines):
        gutter = f"{get_cross_icon(console)} " if i == 0 else "  "
        rendered = Text.from_markup(gutter) + line
        # Blank lines in the message would otherwise render as two spaces of
        # gutter, and `Text.wrap` only strips whitespace *beyond* the wrap width,
        # so a word-break space can survive at the end of a wrapped line.
        rendered.rstrip()
        # Already wrapped above; stop the console from wrapping or cropping again.
        console.print(rendered, soft_wrap=True)


def _get_check_updates_config() -> bool:
    from fal.config import Config

    try:
        return Config().get_global(_CHECK_UPDATES_CONFIG_KEY) is not False
    except (OSError, ValueError):
        return True


def _check_latest_version():
    from packaging.version import parse
    from rich.panel import Panel
    from rich.text import Text

    from fal._installer import get_upgrade_command
    from fal._version import get_latest_version, version_tuple

    # If we have a dev version, we don't want to check for updates
    if len(version_tuple) >= 4:
        if "dev" in str(version_tuple[3]):
            return

    if not console.is_terminal:
        return

    if not _get_check_updates_config():
        return

    latest_version = get_latest_version()
    parsed = parse(latest_version)
    latest_version_tuple = (parsed.major, parsed.minor, parsed.micro)

    if latest_version_tuple <= version_tuple:
        return

    line1 = Text.assemble(
        ("A new version of fal is available: ", "bold white"),
        (latest_version, "bold green"),
    )
    line2 = Text.assemble((get_upgrade_command(), "bold cyan"))
    # Center both against the longest line: the command can now be longer than
    # the headline, and `align` truncates to `width` rather than padding.
    width = max(len(line1), len(line2))
    line1.align("center", width=width)
    line2.align("center", width=width)

    panel = Panel(
        line1 + "\n\n" + line2,
        border_style="yellow",
        padding=(1, 2),
        highlight=True,
        expand=False,
    )
    console.print(panel)


def main(argv=None) -> int:
    import grpc

    from fal.api import FalSerializationError, UserFunctionException

    _check_latest_version()

    ret = 1
    try:
        args = parse_args(argv)

        with debugtools(args):
            ret = args.func(args)
    except (UserFunctionException, FalSerializationError) as _exc:
        cause = _exc.__cause__
        exc: BaseException = cause or _exc
        tb = rich.traceback.Traceback.from_exception(
            type(exc),
            exc,
            exc.__traceback__,
        )
        console.print(tb)

        if isinstance(_exc, UserFunctionException):
            msg = "Unhandled user exception"
        else:
            msg = str(_exc)
        _print_error(msg)
    except KeyboardInterrupt:
        _print_error("Aborted.")
    except grpc.RpcError as exc:
        if exc.code() == grpc.StatusCode.UNAVAILABLE:
            from fal.api.api import _format_unavailable_error

            _print_error(_format_unavailable_error(exc))
        else:
            _print_error(exc.details())
    except FalParserExit as exc:
        ret = exc.status
    except Exception as exc:
        msg = str(exc)
        cause = exc.__cause__
        if cause is not None:
            msg += f": {str(cause)}"
        _print_error(msg)

    return ret
