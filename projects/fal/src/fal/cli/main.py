import argparse

import rich

from fal import __version__
from fal.console import console
from fal.console.icons import CROSS_ICON

from . import apps, auth, deploy, keys, run, secrets
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

    for cmd in [auth, apps, deploy, run, keys, secrets]:
        cmd.add_parser(subparsers, parents)

    return parser


def parse_args(argv=None):
    parser = _get_main_parser()
    args = parser.parse_args(argv)
    args.console = console
    args.parser = parser
    return args


def main(argv=None) -> int:
    import grpc

    from fal.api import UserFunctionException

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
        console.print("Unhandled user exception")
    except KeyboardInterrupt:
        console.print("Aborted.")
    except grpc.RpcError as exc:
        console.print(exc.details())
    except FalParserExit as exc:
        ret = exc.status
    except Exception as exc:
        msg = f"{CROSS_ICON} {str(exc)}"
        cause = exc.__cause__
        if cause is not None:
            msg += f": {str(cause)}"
        console.print(msg)

    return ret
