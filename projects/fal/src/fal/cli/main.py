import argparse

import rich

from fal._version import __version__
from fal.console import console
from fal.console.icons import CROSS_ICON

from . import (
    api,
    apps,
    auth,
    create,
    deploy,
    doctor,
    files,
    keys,
    profile,
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
    console.print(f"{CROSS_ICON} {msg}")


def _check_latest_version():
    from packaging.version import parse
    from rich.panel import Panel
    from rich.text import Text

    from fal._version import get_latest_version, version_tuple

    latest_version = get_latest_version()
    parsed = parse(latest_version)
    latest_version_tuple = (parsed.major, parsed.minor, parsed.micro)
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

    from fal.api import UserFunctionException

    _check_latest_version()

    ret = 1
    try:
        args = parse_args(argv)

        with debugtools(args):
            ret = args.func(args)
    except UserFunctionException as _exc:
        cause = _exc.__cause__
        exc: BaseException = cause or _exc
        tb = rich.traceback.Traceback.from_exception(
            type(exc),
            exc,
            exc.__traceback__,
        )
        console.print(tb)
        _print_error("Unhandled user exception")
    except KeyboardInterrupt:
        _print_error("Aborted.")
    except grpc.RpcError as exc:
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
