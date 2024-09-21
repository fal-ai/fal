from ._utils import get_app_data_from_toml, is_app_name
from .parser import FalClientParser, RefAction


def _run(args):
    from fal.api import FalServerlessHost
    from fal.utils import load_function_from

    host = FalServerlessHost(args.host)

    if is_app_name(args.func_ref):
        app_name = args.func_ref[0]
        app_ref, _, _ = get_app_data_from_toml(app_name)
        file_path, func_name = RefAction.split_ref(app_ref)
    else:
        file_path, func_name = args.func_ref

    loaded = load_function_from(host, file_path, func_name)

    isolated_function = loaded.function
    # let our exc handlers handle UserFunctionException
    isolated_function.reraise = False
    isolated_function()


def add_parser(main_subparsers, parents):
    run_help = "Run fal function."
    epilog = "Examples:\n" "  fal run path/to/myfile.py::myfunc"
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
        help="Function reference.",
    )
    parser.set_defaults(func=_run)
