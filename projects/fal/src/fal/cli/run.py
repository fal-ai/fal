from pathlib import Path

from ._utils import get_app_data_from_toml, is_app_name
from .parser import FalClientParser, RefAction


def _run(args):
    from fal.api.client import SyncServerlessClient
    from fal.utils import load_function_from

    team = None
    func_ref = args.func_ref

    # Check if it contains a slash and might be team/app-name format
    if is_app_name(func_ref) and "/" in func_ref[0]:
        # Try to interpret as team/app-name format
        team, app_name = func_ref[0].split("/", 1)

    if is_app_name(func_ref):
        app_name = func_ref[0]
        app_ref, *_ = get_app_data_from_toml(app_name)
        file_path, func_name = RefAction.split_ref(app_ref)
    else:
        file_path, func_name = func_ref
        # Turn relative path into absolute path for files
        file_path = str(Path(file_path).absolute())

    client = SyncServerlessClient(host=args.host, team=team)
    host = client._create_host(local_file_path=file_path)

    loaded = load_function_from(host, file_path, func_name)

    isolated_function = loaded.function
    # let our exc handlers handle UserFunctionException
    isolated_function.reraise = False
    isolated_function()


def add_parser(main_subparsers, parents):
    run_help = "Run fal function."
    epilog = (
        "Examples:\n"
        "  fal run path/to/myfile.py::myfunc\n"
        "  fal run my-app\n"
        "  fal run my-team/my-app"
    )
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
        help="Function reference. Use <team-name>/<app-name> to run with a specific team.",
    )
    parser.set_defaults(func=_run)
