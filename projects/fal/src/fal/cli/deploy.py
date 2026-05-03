from __future__ import annotations

import argparse
import json

from fal.api.client import SyncServerlessClient

from .deploy_check import _resolve_deploy_check_source, deploy_with_check
from .parser import FalClientParser, RefAction, add_env_argument, get_output_parser


def _deploy(args):
    from ._result_handlers import CliRegisterResultHandler

    team, app_ref = _resolve_team_and_app_ref(args)

    # Handle deprecated --force-env-build flag
    if args.force_env_build:
        args.console.print(
            "[bold yellow]Warning:[/bold yellow] --force-env-build is deprecated, "
            "use --no-cache instead"
        )

    no_cache = args.no_cache or args.force_env_build
    client = SyncServerlessClient(host=args.host, team=team)
    result_handler = CliRegisterResultHandler(console=args.console)

    deploy_check_source = _resolve_deploy_check_source(args, client)
    if deploy_check_source is not None:
        # The pre-deploy summary already shows the resolved app name.
        res = deploy_with_check(
            args,
            client,
            app_ref,
            force_env_build=no_cache,
            source=deploy_check_source,
            result_handler=result_handler,
        )
    else:
        from fal.api import deploy as deploy_api

        prepared = deploy_api.prepare_deployment(
            client,
            app_ref,
            app_name=args.app_name,
            auth=args.auth,
            strategy=args.strategy,
            reset_scale=args.app_scale_settings,
            force_env_build=no_cache,
            environment_name=args.env,
        )
        args.console.print(
            f"Deploying '{prepared.display_name}' as app '{prepared.loaded.app_name}'",
            style="bold",
        )
        args.console.print("")
        res = deploy_api.execute_prepared_deployment(
            prepared, result_handler=result_handler
        )

    _render_deploy_result(args, res)


def _resolve_team_and_app_ref(args) -> tuple[str | None, tuple[str | None, str | None]]:
    from ._utils import get_app_data_from_toml, is_app_name

    team = args.team
    app_ref = args.app_ref

    # If the app_ref is an app name, get team from pyproject.toml
    if app_ref and is_app_name(app_ref):
        try:
            toml_data = get_app_data_from_toml(
                app_ref[0], emit_deprecation_warnings=False
            )
            team = team or toml_data.team
        except (ValueError, FileNotFoundError):
            # If we can't find the app in pyproject.toml, team remains None
            pass

    return team, app_ref


def _render_deploy_result(args, res) -> None:
    app_id = res.revision
    resolved_app_name = res.app_name

    if args.output == "json":
        args.console.print(
            json.dumps({"revision": app_id, "app_name": resolved_app_name})
        )
    elif args.output == "pretty":
        from rich.rule import Rule
        from rich.text import Text

        from fal.console.icons import CHECK_ICON
        from fal.flags import URL_OUTPUT

        args.console.print(
            f"{CHECK_ICON} Deployed successfully",
            style="bold green",
        )
        args.console.print("")

        # Build panel content with grouped sections
        lines = Text()

        # Auth mode section
        AUTH_EXPLANATIONS = {
            "public": "no authentication required",
            "private": "only you/team can access",
            "shared": "any authenticated user can access",
        }
        auth_desc = AUTH_EXPLANATIONS.get(res.auth_mode, res.auth_mode)
        lines.append(f"▸ Auth: {res.auth_mode} ", style="bold")
        lines.append(f"({auth_desc})\n\n", style="dim")

        # Playground section
        if URL_OUTPUT != "none":
            lines.append("▸ Playground ", style="bold")
            lines.append("(open in browser)\n", style="dim")
            for url in res.urls.get("playground", {}).values():
                lines.append(f"  {url}\n", style="cyan")

        # API Endpoints section
        if URL_OUTPUT == "all":
            lines.append("\n")
            lines.append("▸ API Endpoints ", style="bold")
            lines.append("(use in code)\n", style="dim")
            sync_urls = list(res.urls.get("sync", {}).values())
            async_urls = list(res.urls.get("async", {}).values())
            for sync_url, async_url in zip(sync_urls, async_urls):
                lines.append(f"  Sync   {sync_url}\n", style="cyan")
                lines.append(f"  Async  {async_url}\n", style="cyan")

            # Logs section
            lines.append("\n")
            lines.append("▸ Logs\n", style="bold")
            lines.append(f"  {res.log_url}", style="cyan")

        title = Text(resolved_app_name, style="bold")
        args.console.print(Rule(title, style="green"))
        args.console.print(lines)
        args.console.print(Rule("", style="green"))

        # Reminder about scaling parameter inheritance
        if not args.app_scale_settings:
            args.console.print("")
            note = (
                "[yellow]Note: Scaling parameters (keep_alive, min_concurrency, etc.) "
                "are inherited from the previous deployment. "
                "Use --reset-scale to apply code changes.[/yellow]"
            )
            args.console.print(note)
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
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Show a pre-deployment summary before deploying. "
            "Prompts for confirmation unless --yes is also set."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Skip interactive deploy confirmation prompts. "
            "When combined with --check, the summary is still shown."
        ),
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
    add_env_argument(parser)

    parser.set_defaults(func=_deploy)
