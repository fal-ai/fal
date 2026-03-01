import argparse
from dataclasses import replace
from pathlib import Path

from ._utils import AppData, get_app_data_from_toml, is_app_name
from .parser import FalClientParser, RefAction


def _run(args):
    from fal.api.client import SyncServerlessClient
    from fal.utils import load_function_from

    # Handle deprecated --force-env-build flag
    if args.force_env_build:
        args.console.print(
            "[bold yellow]Warning:[/bold yellow] --force-env-build is deprecated, "
            "use --no-cache instead"
        )

    team = args.team
    func_ref = args.func_ref

    app_data = AppData(auth=args.auth, name=args.app_name)
    args.console.print(
        "[bold yellow]Warning:[/bold yellow] "
        "`fal run` ignores fal.App auth and pyproject.toml auth and "
        "defaults to public. "
        "In the next major release, the default will be private and "
        "fal.App auth and pyproject.toml auth will be respected. "
        "Use --auth to set the authentication mode."
    )
    if is_app_name(func_ref):
        app_name = func_ref[0]
        toml_data = get_app_data_from_toml(app_name)
        app_data = replace(
            toml_data,
            auth=app_data.auth,
            name=app_data.name,
        )
        team = team or app_data.team
        file_path, func_name = RefAction.split_ref(app_data.ref)
    else:
        file_path, func_name = func_ref
        # Turn relative path into absolute path for files
        file_path = str(Path(file_path).absolute())
        ref = f"{file_path}::{func_name}" if func_name else file_path
        app_data = replace(app_data, ref=ref)

    no_cache = args.no_cache or args.force_env_build
    client = SyncServerlessClient(host=args.host, team=team)
    host = client._create_host(local_file_path=file_path, environment_name=args.env)

    loaded = load_function_from(
        host,
        file_path,
        func_name,
        force_env_build=no_cache,
        options=app_data.options,
        app_name=app_data.name,
        app_auth=app_data.auth,
    )

    isolated_function = loaded.function
    # let our exc handlers handle UserFunctionException
    isolated_function.reraise = False
    if args.local:
        isolated_function.run_local()
    else:
        isolated_function()


def add_parser(main_subparsers, parents):
    from fal.sdk import ALIAS_AUTH_MODES

    def valid_auth_option(option):
        if option not in ALIAS_AUTH_MODES:
            raise argparse.ArgumentTypeError(f"{option} is not a auth option")
        return option

    run_help = "Run fal function."
    epilog = "Examples:\n  fal run path/to/myfile.py::myfunc\n  fal run my-app\n"
    parser = main_subparsers.add_parser(
        "run",
        description=run_help,
        parents=[*parents, FalClientParser(add_help=False)],
        help=run_help,
        epilog=epilog,
    )
    parser.add_argument(
        "func_ref",
        action=RefAction,
        help="Function reference. Configure team in pyproject.toml for app names.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not use the cache for the environment build.",
    )
    parser.add_argument(
        "--app-name",
        help="Application name to run with.",
    )
    parser.add_argument(
        "--auth",
        type=valid_auth_option,
        default="public",
        help=(
            "Application authentication mode (private, public, shared), "
            "defaults to public. "
        ),
    )
    parser.add_argument(
        "--force-env-build",
        action="store_true",
        help=(
            "[DEPRECATED: Use --no-cache instead] "
            "Ignore the environment build cache and force rebuild."
        ),
    )
    parser.add_argument(
        "--env",
        dest="env",
        help="Target environment (defaults to main).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally without serverless.",
    )
    parser.set_defaults(func=_run)
