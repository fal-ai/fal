import argparse
from typing import List

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


def _remote_exception_summary(stringized_traceback):
    if not stringized_traceback:
        return None

    in_frames = False
    summary_lines: List[str] = []
    for line in stringized_traceback.splitlines():
        if line.startswith("Traceback (most recent call last):"):
            if summary_lines:
                break
            in_frames = True
            continue
        if summary_lines:
            if line.startswith(
                (
                    "During handling of the above exception",
                    "The above exception was the direct cause",
                )
            ):
                break
            summary_lines.append(line.rstrip())
            continue
        if not in_frames or not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            summary_lines.append(line.strip())

    if summary_lines:
        return "\n".join(summary_lines).strip()
    return None


def _render_remote_traceback(remote_exception, remote_traceback):
    if remote_exception:
        qualname, _, message = remote_exception.partition(": ")
        exc_type = type(qualname, (Exception,), {})
    else:
        exc_type, message = Exception, ""

    return rich.traceback.Traceback.from_exception(
        exc_type,
        exc_type(message),
        remote_traceback,
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
    from isolate.connections.common import ExceptionDeserializationError

    from fal.api import FalSerializationError, UserFunctionException
    from fal.sdk import RemoteExceptionDeserializationError

    _check_latest_version()

    ret = 1
    try:
        args = parse_args(argv)

        with debugtools(args):
            ret = args.func(args)
    except (UserFunctionException, FalSerializationError) as _exc:
        cause = _exc.__cause__

        if (
            isinstance(_exc, FalSerializationError)
            and isinstance(cause, ExceptionDeserializationError)
            and cause.original_traceback is not None
        ):
            # Keep the runner error separate from local deserialization failure.
            stringized_traceback = None
            if isinstance(cause, RemoteExceptionDeserializationError):
                stringized_traceback = cause.stringized_traceback
            remote_exception = _remote_exception_summary(stringized_traceback)
            console.print(
                "[bold]The application raised this error on the runner:[/bold]"
            )
            console.print(
                _render_remote_traceback(remote_exception, cause.original_traceback)
            )
        else:
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
