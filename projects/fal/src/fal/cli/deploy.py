import argparse
import json

from fal.api.client import SyncServerlessClient

from .parser import FalClientParser, RefAction, get_output_parser


def _deploy(args):
    from ._utils import get_app_data_from_toml, is_app_name

    team = None
    app_ref = args.app_ref

    # If the app_ref is an app name, get team from pyproject.toml
    if app_ref and is_app_name(app_ref):
        try:
            _, _, _, _, team = get_app_data_from_toml(app_ref[0])
        except (ValueError, FileNotFoundError):
            # If we can't find the app in pyproject.toml, team remains None
            pass

    client = SyncServerlessClient(host=args.host, team=team)
    res = client.deploy(
        app_ref,
        app_name=args.app_name,
        auth=args.auth,
        strategy=args.strategy,
        reset_scale=args.app_scale_settings,
    )
    app_id = res.revision
    resolved_app_name = res.app_name

    if args.output == "json":
        args.console.print(
            json.dumps({"revision": app_id, "app_name": resolved_app_name})
        )
    elif args.output == "pretty":
        args.console.print(
            "Registered a new revision for function "
            f"'{resolved_app_name}' (revision='{app_id}')."
        )
        args.console.print("Playground:")
        for url in res.urls.get("playground", {}).values():
            args.console.print(f"\t{url}")
        args.console.print("Synchronous Endpoints:")
        for url in res.urls.get("sync", {}).values():
            args.console.print(f"\t{url}")
        args.console.print("Asynchronous Endpoints (Recommended):")
        for url in res.urls.get("async", {}).values():
            args.console.print(f"\t{url}")
    else:
        raise AssertionError(f"Invalid output format: {args.output}")


def add_parser(main_subparsers, parents):
    from fal.sdk import ALIAS_AUTH_MODES

    def valid_auth_option(option):
        if option not in ALIAS_AUTH_MODES:
            raise argparse.ArgumentTypeError(f"{option} is not a auth option")
        return option

    deploy_help = (
        "Deploy a fal application. "
        "If no app reference is provided, the command will look for a "
        "pyproject.toml file with a [tool.fal.apps] section and deploy the "
        "application specified with the provided app name."
    )

    epilog = (
        "Examples:\n"
        "  fal deploy\n"
        "  fal deploy path/to/myfile.py\n"
        "  fal deploy path/to/myfile.py::MyApp\n"
        "  fal deploy path/to/myfile.py::MyApp --app-name myapp --auth public\n"
        "  fal deploy my-app\n"
    )

    parser = main_subparsers.add_parser(
        "deploy",
        parents=[
            *parents,
            get_output_parser(),
            FalClientParser(add_help=False),
        ],
        description=deploy_help,
        help=deploy_help,
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
            "section and deploy the application specified with the provided app name.\n"
            "File path example: path/to/myfile.py::MyApp\n"
            "App name example: my-app (configure team in pyproject.toml)\n"
        ),
    )

    parser.add_argument(
        "--app-name",
        help="Application name to deploy with.",
    )

    parser.add_argument(
        "--auth",
        type=valid_auth_option,
        help="Application authentication mode (private, public).",
    )
    parser.add_argument(
        "--strategy",
        choices=["recreate", "rolling"],
        help="Deployment strategy.",
        default="rolling",
    )
    parser.add_argument(
        "--no-scale",
        action="store_false",
        dest="app_scale_settings",
        default=False,
        help="Use the previous deployment of the application for scale settings. "
        "This is the default behavior.",
    )
    parser.add_argument(
        "--reset-scale",
        action="store_true",
        dest="app_scale_settings",
        help="Use the application code for scale settings.",
    )

    parser.set_defaults(func=_deploy)
