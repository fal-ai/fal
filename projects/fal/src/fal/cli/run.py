from pathlib import Path

from ._utils import get_app_data_from_toml, is_app_name
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

    if is_app_name(func_ref):
        app_name = func_ref[0]
        app_ref, *_, toml_team = get_app_data_from_toml(app_name)
        team = team or toml_team
        file_path, func_name = RefAction.split_ref(app_ref)
    else:
        file_path, func_name = func_ref
        # Turn relative path into absolute path for files
        file_path = str(Path(file_path).absolute())

    no_cache = args.no_cache or args.force_env_build
    client = SyncServerlessClient(host=args.host, team=team)
    host = client._create_host(local_file_path=file_path, environment_name=args.env)

    loaded = load_function_from(host, file_path, func_name, force_env_build=no_cache)

    isolated_function = loaded.function
    # let our exc handlers handle UserFunctionException
    isolated_function.reraise = False
    isolated_function()


def add_parser(main_subparsers, parents):
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
    parser.set_defaults(func=_run)
