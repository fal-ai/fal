from __future__ import annotations

import json

from fal.api.client import SyncServerlessClient

from .deploy import _resolve_team_and_app_ref
from .parser import FalClientParser, RefAction, add_env_argument, get_output_parser


def _build(args):
    from fal.api.deploy import build as build_api

    from ._result_handlers import (
        CliBuildEnvironmentResultHandler,
        CliRegisterResultHandler,
    )

    team, app_ref = _resolve_team_and_app_ref(args)

    client = SyncServerlessClient(host=args.host, team=team)
    res = build_api(
        client,
        app_ref,
        app_name=args.app_name,
        force_env_build=args.no_cache,
        environment_name=args.env,
        result_handler=CliRegisterResultHandler(console=args.console),
        build_result_handler=CliBuildEnvironmentResultHandler(console=args.console),
    )

    _render_build_result(args, res)


def _render_build_result(args, res) -> None:
    if args.output == "json":
        args.console.print(
            json.dumps({"revision": res.revision, "app_name": res.app_name})
        )
    elif args.output == "pretty":
        from fal.console.icons import get_check_icon

        args.console.print(
            f"{get_check_icon(args.console)} Built successfully",
            style="bold green",
        )
        args.console.print("")
        args.console.print(f"Revision: {res.revision}")
        args.console.print("")
        args.console.print(
            "[dim]This revision is not serving traffic. "
            "Deploy it with `fal deploy` to point an alias at it.[/dim]"
        )
    else:
        raise AssertionError(f"Invalid output format: {args.output}")


def add_parser(main_subparsers, parents):
    build_help = (
        "Build a fal application into a new revision without deploying it. "
        "No alias is pointed at the revision, so it does not serve traffic."
    )

    epilog = (
        "Examples:\n"
        "  fal build\n"
        "  fal build path/to/myfile.py\n"
        "  fal build path/to/myfile.py::MyApp\n"
        "  fal build my-app\n"
    )

    parser = main_subparsers.add_parser(
        "build",
        parents=[
            *parents,
            get_output_parser(),
            FalClientParser(add_help=False),
        ],
        description=build_help,
        help=build_help,
        epilog=epilog,
    )

    parser.add_argument(
        "app_ref",
        nargs="?",
        action=RefAction,
        help=(
            "Application reference. Either a file path or a file path and a "
            "function name separated by '::'. If no reference is provided, the "
            "command will look for a pyproject.toml file with a [tool.fal.apps] "
            "section and build the application specified with the provided app name.\n"
            "File path example: path/to/myfile.py::MyApp\n"
            "App name example: my-app (configure team in pyproject.toml)\n"
        ),
    )

    parser.add_argument(
        "--app-name",
        help="Application name to build with.",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not use the cache for the environment build.",
    )

    add_env_argument(parser)

    parser.set_defaults(func=_build)
