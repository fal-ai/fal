import argparse
from collections import namedtuple
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

from ._utils import get_app_data_from_toml, is_app_name
from .parser import FalClientParser, RefAction

User = namedtuple("User", ["user_id", "username"])


def _remove_http_and_port_from_url(url):
    # Remove http://
    if "http://" in url:
        url = url.replace("http://", "")

    # Remove https://
    if "https://" in url:
        url = url.replace("https://", "")

    # Remove port information
    url_parts = url.split(":")
    if len(url_parts) > 1:
        url = url_parts[0]

    return url


def _get_user() -> User:
    import json
    from http import HTTPStatus

    import openapi_fal_rest.api.users.get_current_user as get_current_user

    from fal.api import FalServerlessError
    from fal.rest_client import REST_CLIENT

    try:
        user_details_response = get_current_user.sync_detailed(
            client=REST_CLIENT,
        )
    except Exception as e:
        raise FalServerlessError(f"Error fetching user details: {str(e)}")

    if user_details_response.status_code != HTTPStatus.OK:
        try:
            content = json.loads(user_details_response.content.decode("utf8"))
        except Exception:
            raise FalServerlessError(
                f"Error fetching user details: {user_details_response}"
            )
        else:
            raise FalServerlessError(content["detail"])
    try:
        full_user_id = user_details_response.parsed.user_id
        _provider, _, user_id = full_user_id.partition("|")
        if not user_id:
            user_id = full_user_id

        return User(user_id=user_id, username=user_details_response.parsed.nickname)
    except Exception as e:
        raise FalServerlessError(f"Could not parse the user data: {e}")


def _deploy_from_reference(
    app_ref: Tuple[Optional[Union[Path, str]], ...],
    app_name: str,
    args,
    auth: Optional[Literal["public", "shared", "private"]] = None,
    deployment_strategy: Optional[Literal["recreate", "rolling"]] = None,
    scale: bool = True,
):
    from fal.api import FalServerlessError, FalServerlessHost
    from fal.utils import load_function_from

    file_path, func_name = app_ref
    if file_path is None:
        # Try to find a python file in the current directory
        options = list(Path(".").glob("*.py"))
        if len(options) == 0:
            raise FalServerlessError("No python files found in the current directory")
        elif len(options) > 1:
            raise FalServerlessError(
                "Multiple python files found in the current directory. "
                "Please specify the file path of the app you want to deploy."
            )

        [file_path] = options
        file_path = str(file_path)  # type: ignore

    user = _get_user()
    host = FalServerlessHost(args.host)
    loaded = load_function_from(
        host,
        file_path,  # type: ignore
        func_name,  # type: ignore
    )
    isolated_function = loaded.function
    app_name = app_name or loaded.app_name  # type: ignore
    app_auth = auth or loaded.app_auth
    deployment_strategy = deployment_strategy or "recreate"

    app_id = host.register(
        func=isolated_function.func,
        options=isolated_function.options,
        application_name=app_name,
        application_auth_mode=app_auth,  # type: ignore
        metadata=isolated_function.options.host.get("metadata", {}),
        deployment_strategy=deployment_strategy,
        scale=scale,
    )

    if app_id:
        env_host = _remove_http_and_port_from_url(host.url)
        env_host = env_host.replace("api.", "").replace("alpha.", "")

        env_host_parts = env_host.split(".")

        # keep the last 3 parts
        playground_host = ".".join(env_host_parts[-3:])

        # just replace .ai for .run
        endpoint_host = env_host.replace(".ai", ".run")

        args.console.print(
            "Registered a new revision for function "
            f"'{app_name}' (revision='{app_id}')."
        )
        args.console.print("Playground:")
        for endpoint in loaded.endpoints:
            args.console.print(
                f"\thttps://{playground_host}/models/{user.username}/{app_name}{endpoint}"
            )
        args.console.print("Endpoints:")
        for endpoint in loaded.endpoints:
            args.console.print(
                f"\thttps://{endpoint_host}/{user.username}/{app_name}{endpoint}"
            )


def _deploy(args):
    # my-app
    if is_app_name(args.app_ref):
        # we do not allow --app-name and --auth to be used with app name
        if args.app_name or args.auth:
            raise ValueError("Cannot use --app-name or --auth with app name reference.")

        app_name = args.app_ref[0]
        app_ref, app_auth, app_deployment_strategy, app_no_scale = (
            get_app_data_from_toml(app_name)
        )
        file_path, func_name = RefAction.split_ref(app_ref)

    # path/to/myfile.py::MyApp
    else:
        file_path, func_name = args.app_ref
        app_name = args.app_name
        app_auth = args.auth
        app_deployment_strategy = args.strategy
        app_no_scale = args.no_scale

    _deploy_from_reference(
        (file_path, func_name),
        app_name,
        args,
        app_auth,
        app_deployment_strategy,
        scale=not app_no_scale,
    )


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
        parents=[*parents, FalClientParser(add_help=False)],
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
            "App name example: my-app\n"
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
        default="recreate",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help=(
            "Use min_concurrency/max_concurrency/concurrency_buffer/max_multiplexing "
            "from previous deployment of application with this name, if exists. "
            "Otherwise will use the values from the application code."
        ),
    )

    parser.set_defaults(func=_deploy)
