from .parser import FalClientParser, RefAction


def _run(args):
    from fal.api import FalServerlessHost
    from fal.utils import load_function_from

    host = FalServerlessHost(args.host)
    isolated_function, _ = load_function_from(host, *args.func_ref)
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
