import argparse
from dataclasses import replace
from pathlib import Path

from ._utils import AppData, _validate_port, get_app_data_from_toml, is_app_name
from .parser import FalClientParser, RefAction, add_env_argument


def _run(args):
    from fal.api.client import SyncServerlessClient
    from fal.utils import load_function_from

    exposed_port = args.exposed_port
    exposed_metrics_port = args.exposed_metrics_port
    if not args.local and (
        exposed_port is not None or exposed_metrics_port is not None
    ):
        raise ValueError(
            "--exposed-port and --exposed-metrics-port can only be used with --local."
        )

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
        if app_data.ref is not None:
            file_path, func_name = RefAction.split_ref(app_data.ref)
        else:
            file_path, func_name = None, None
    else:
        file_path, func_name = func_ref
        # Turn relative path into absolute path for files
        file_path = str(Path(file_path).absolute())
        ref = f"{file_path}::{func_name}" if func_name else file_path
        app_data = replace(app_data, ref=ref)

    no_cache = args.no_cache or args.force_env_build
    client = SyncServerlessClient(host=args.host, team=team)
    host = client._create_host(
        local_file_path=file_path or "",
        environment_name=args.env,
    )

    loaded = load_function_from(
        host,
        file_path,
        func_name,
        force_env_build=no_cache,
        options=app_data.options,
        app_name=app_data.name,
        app_auth=app_data.auth,
        limit_max_requests=args.limit_max_requests,
        python_entry_point=app_data.python_entry_point,
    )

    isolated_function = loaded.function
    if args.machine_type is not None:
        isolated_function.options.host["machine_type"] = args.machine_type

    if not args.local:
        from ._result_handlers import PrepareRequirementsCallback

        isolated_function.options = host.prepare_options(
            isolated_function.options,
            func=isolated_function.func,
            on_progress=PrepareRequirementsCallback(console=args.console),
        )

        # Explicit build phase so the CLI gets a clean "build → run" split
        # instead of inferring it from the log stream's source field.
        from ._result_handlers import CliBuildEnvironmentResultHandler

        host.build_environment(
            isolated_function.options,
            application_name=loaded.app_name,
            environment_name=args.env,
            result_handler=CliBuildEnvironmentResultHandler(console=args.console),
        )
        # Endpoints/openapi for the result handler aren't available locally
        # in entrypoint mode; this fetches them from the worker.
        isolated_function.fetch_metadata(build_environment=False)

    from fal.api.run import run as run_api

    from ._result_handlers import CliRunResultHandler

    run_api(
        isolated_function,
        local=args.local,
        exposed_port=exposed_port,
        exposed_metrics_port=exposed_metrics_port,
        result_handler=CliRunResultHandler(
            console=args.console,
            auth_mode=loaded.app_auth or "public",
            endpoints=isolated_function.endpoints,
        ),
        reraise=False,
        build_environment=None if args.local else False,
    )


def add_parser(main_subparsers, parents):
    from fal.sdk import ALIAS_AUTH_MODES

    def valid_auth_option(option):
        if option not in ALIAS_AUTH_MODES:
            raise argparse.ArgumentTypeError(f"{option} is not a auth option")
        return option

    def valid_port_option(option):
        try:
            port = int(option)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{option} is not a valid port") from None

        try:
            _validate_port("port", port)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from None
        return port

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
    add_env_argument(parser)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally without serverless.",
    )
    parser.add_argument(
        "--exposed-port",
        type=valid_port_option,
        metavar="PORT",
        help="Port for the --local app server (default: 8080).",
    )
    parser.add_argument(
        "--exposed-metrics-port",
        type=valid_port_option,
        metavar="PORT",
        help="Port for the --local metrics server (default: 9090).",
    )
    parser.add_argument(
        "--machine-type",
        type=str,
        help="Machine type to use for this run.",
    )
    parser.add_argument(
        "--limit-max-requests",
        type=int,
        default=None,
        help="For fal.App runs, gracefully stop the server after serving N requests.",
    )
    parser.set_defaults(func=_run)
