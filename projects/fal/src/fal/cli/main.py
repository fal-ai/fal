import argparse
import traceback
from types import TracebackType

import rich

from fal._version import __version__
from fal.console import console
from fal.console.icons import get_cross_icon

from . import (
    api,
    apps,
    auth,
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
    console.print(f"{get_cross_icon(console)} {msg}")


def _format_remote_traceback(remote_traceback: TracebackType) -> str:
    """Format the runner's traceback frames.

    isolate reconstructs the remote frames but not the remote exception's type
    (it isn't importable here), so we render the frames alone rather than
    labeling them with the local deserialization error, which would be
    misleading.
    """
    return "".join(traceback.format_tb(remote_traceback)).rstrip()


def _print_local_traceback(error: BaseException) -> None:
    target = error.__cause__ or error
    console.print(
        rich.traceback.Traceback.from_exception(
            type(target),
            target,
            target.__traceback__,
        )
    )


def _check_latest_version():
    from packaging.version import parse
    from rich.panel import Panel
    from rich.text import Text

    from fal._version import get_latest_version, version_tuple

    latest_version = get_latest_version()
    parsed = parse(latest_version)
    latest_version_tuple = (parsed.major, parsed.minor, parsed.micro)

    # If we have a dev version, we don't want to check for updates
    if len(version_tuple) >= 4:
        if "dev" in str(version_tuple[3]):
            return

    if latest_version_tuple <= version_tuple:
        return

    if not console.is_terminal:
        return

    line1 = Text.assemble(
        ("A new version of fal is available: ", "bold white"),
        (latest_version, "bold green"),
    )
    line2 = Text.assemble(("pip install --upgrade fal", "bold cyan"))
    line2.align("center", width=len(line1))

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
    except FalSerializationError as _exc:
        if _exc.original_traceback is not None:
            # The app raised an error on the runner that we couldn't
            # reconstruct locally. Show where it failed on the runner, then
            # explain the local deserialization failure separately below —
            # they are two distinct errors and conflating them is misleading.
            console.print(
                "[bold]Traceback from the runner (most recent call last):[/bold]"
            )
            console.print(_format_remote_traceback(_exc.original_traceback))
        else:
            _print_local_traceback(_exc)
        _print_error(str(_exc))
    except UserFunctionException as _exc:
        _print_local_traceback(_exc)
        _print_error("Unhandled user exception")
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
