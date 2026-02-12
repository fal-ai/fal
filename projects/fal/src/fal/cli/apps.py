from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import fal.cli.runners as runners
from fal.api.client import SyncServerlessClient
from fal.sdk import RunnerState, deconstruct_alias

from ._utils import get_client
from .parser import FalClientParser, SinceAction, get_output_parser

if TYPE_CHECKING:
    from fal.sdk import AliasInfo, ApplicationInfo

CODE_SPECIFIC_SCALING_PARAMS = [
    "max_multiplexing",
    "startup_timeout",
    "machine_types",
]


def _apps_table(apps: list[AliasInfo]):
    from rich.table import Table

    table = Table()
    table.add_column("Name", no_wrap=True)
    table.add_column("Env")
    table.add_column("Revision")
    table.add_column("Auth")
    table.add_column("Min Concurrency")
    table.add_column("Max Concurrency")
    table.add_column("Concurrency Buffer")
    table.add_column("Scaling Delay")
    table.add_column("Max Multiplexing")
    table.add_column("Keep Alive")
    table.add_column("Request Timeout")
    table.add_column("Startup Timeout")
    table.add_column("Machine Type")
    table.add_column("Runners")
    table.add_column("Regions")

    for app in apps:
        if app.concurrency_buffer_perc > 0:
            concurrency_buffer_str = (
                f"{app.concurrency_buffer_perc}%, min {app.concurrency_buffer}"
            )
        else:
            concurrency_buffer_str = str(app.concurrency_buffer)

        display_name = deconstruct_alias(app.alias, app.environment_name)

        table.add_row(
            display_name,
            app.environment_name or "main",
            app.revision,
            app.auth_mode,
            str(app.min_concurrency),
            str(app.max_concurrency),
            concurrency_buffer_str,
            str(app.scaling_delay),
            str(app.max_multiplexing),
            str(app.keep_alive),
            str(app.request_timeout),
            str(app.startup_timeout),
            " ".join(app.machine_types),
            str(app.active_runners),
            " ".join(app.valid_regions),
        )

    return table


def _list(args):
    client = SyncServerlessClient(host=args.host, team=args.team)
    apps = client.apps.list(filter=args.filter, environment_name=args.env)

    if args.sort_by_runners:
        apps.sort(key=lambda x: x.active_runners)
    else:
        apps.sort(key=lambda x: x.alias)

    if args.output == "pretty":
        table = _apps_table(apps)
        args.console.print(table)
    elif args.output == "json":
        apps_as_dicts = [asdict(a) for a in apps]
        json_res = json.dumps({"apps": apps_as_dicts})
        args.console.print(json_res)
    else:
        raise AssertionError(f"Invalid output format: {args.output}")


def _add_list_parser(subparsers, parents):
    list_help = "List applications."
    parser = subparsers.add_parser(
        "list",
        description=list_help,
        help=list_help,
        parents=[*parents, get_output_parser()],
    )
    parser.add_argument(
        "--sort-by-runners",
        action="store_true",
        help="Sort by number of runners ascending",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Filter applications by alias contents",
    )
    parser.add_argument(
        "--env",
        dest="env",
        help="Target environment (defaults to main).",
    )
    parser.set_defaults(func=_list)


def _app_rev_table(revs: list[ApplicationInfo]):
    from rich.table import Table

    table = Table()
    table.add_column("Revision", no_wrap=True)
    table.add_column("Env")
    table.add_column("Min Concurrency")
    table.add_column("Max Concurrency")
    table.add_column("Max Multiplexing")
    table.add_column("Keep Alive")
    table.add_column("Request Timeout")
    table.add_column("Startup Timeout")
    table.add_column("Machine Type")
    table.add_column("Runners")
    table.add_column("Regions")
    table.add_column("Created")

    for rev in revs:
        table.add_row(
            rev.application_id,
            rev.environment_name or "main",
            str(rev.min_concurrency),
            str(rev.max_concurrency),
            str(rev.max_multiplexing),
            str(rev.keep_alive),
            str(rev.request_timeout),
            str(rev.startup_timeout),
            " ".join(rev.machine_types),
            str(rev.active_runners),
            " ".join(rev.valid_regions),
            str(rev.created_at),
        )

    return table


def _list_rev(args):
    client = get_client(args.host, args.team)
    with client.connect() as connection:
        revs = connection.list_applications(args.app_name, environment_name=args.env)
        table = _app_rev_table(revs)

    args.console.print(table)


def _add_list_rev_parser(subparsers, parents):
    list_help = "List application revisions."
    parser = subparsers.add_parser(
        "list-rev",
        description=list_help,
        help=list_help,
        parents=parents,
    )
    parser.add_argument(
        "app_name",
        nargs="?",
        help="Application name.",
    )
    parser.add_argument(
        "--env",
        dest="env",
        help="Target environment (defaults to main).",
    )
    parser.set_defaults(func=_list_rev)


def _scale(args):
    client = SyncServerlessClient(host=args.host, team=args.team)
    if (
        args.keep_alive is None
        and args.max_multiplexing is None
        and args.max_concurrency is None
        and args.min_concurrency is None
        and args.concurrency_buffer is None
        and args.concurrency_buffer_perc is None
        and args.scaling_delay is None
        and args.request_timeout is None
        and args.startup_timeout is None
        and args.machine_types is None
        and args.regions is None
    ):
        args.console.log("No parameters for update were provided, ignoring.")
        return

    app_info = client.apps.scale(
        args.app_name,
        keep_alive=args.keep_alive,
        max_multiplexing=args.max_multiplexing,
        max_concurrency=args.max_concurrency,
        min_concurrency=args.min_concurrency,
        concurrency_buffer=args.concurrency_buffer,
        concurrency_buffer_perc=args.concurrency_buffer_perc,
        scaling_delay=args.scaling_delay,
        request_timeout=args.request_timeout,
        startup_timeout=args.startup_timeout,
        machine_types=args.machine_types,
        regions=args.regions,
        environment_name=args.env,
    )
    table = _apps_table([app_info])

    args.console.print(table)

    code_specific_changes = set()
    for param in CODE_SPECIFIC_SCALING_PARAMS:
        if getattr(args, param) is not None:
            code_specific_changes.add(f"[bold]{param}[/bold]")

    if len(code_specific_changes) > 0:
        args.console.print(
            "[bold yellow]Note:[/bold yellow] Please be aware that "
            f"{', '.join(code_specific_changes)} will be reset on the next deployment. "
            "See https://docs.fal.ai/serverless/deployment-operations/scale-your-application#code-specific-settings-reset-on-deploy for details."  # noqa: E501
        )


def _add_scale_parser(subparsers, parents):
    scale_help = "Scale application."
    parser = subparsers.add_parser(
        "scale",
        description=scale_help,
        help=scale_help,
        parents=parents,
    )
    parser.add_argument(
        "app_name",
        help="Application name.",
    )
    parser.add_argument(
        "--keep-alive",
        type=int,
        help="Keep alive (seconds).",
    )
    parser.add_argument(
        "--max-multiplexing",
        type=int,
        help="Maximum multiplexing",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        help="Maximum concurrency.",
    )
    parser.add_argument(
        "--min-concurrency",
        type=int,
        help="Minimum concurrency",
    )
    parser.add_argument(
        "--concurrency-buffer",
        type=int,
        help="Concurrency buffer (min)",
    )
    parser.add_argument(
        "--concurrency-buffer-perc",
        type=int,
        help="Concurrency buffer %%",
    )
    parser.add_argument(
        "--scaling-delay",
        type=int,
        help="Scaling delay (seconds).",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        help="Request timeout (seconds). If a request takes longer, it is aborted and "
        "the runner gracefully stopped as it could be in a bad state.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        help="Startup timeout (seconds).",
    )
    parser.add_argument(
        "--machine-types",
        type=str,
        nargs="+",
        dest="machine_types",
        help="Machine types (pass several items to set multiple).",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        help="Valid regions (pass several items to set multiple).",
    )
    parser.add_argument(
        "--env",
        dest="env",
        help="Target environment (defaults to main).",
    )
    parser.set_defaults(func=_scale)


def _rollout(args):
    client = SyncServerlessClient(host=args.host, team=args.team)
    client.apps.rollout(args.app_name, force=args.force, environment_name=args.env)
    args.console.log(f"Rolled out application {args.app_name}")


def _add_rollout_parser(subparsers, parents):
    rollout_help = "Rollout application."
    parser = subparsers.add_parser(
        "rollout",
        description=rollout_help,
        help=rollout_help,
        parents=parents,
    )
    parser.add_argument(
        "app_name",
        help="Application name.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rollout.",
    )
    parser.add_argument(
        "--env",
        dest="env",
        help="Target environment (defaults to main).",
    )
    parser.set_defaults(func=_rollout)


def _set_rev(args):
    client = get_client(args.host, args.team)
    with client.connect() as connection:
        alias_info = connection.create_alias(
            args.app_name, args.app_rev, args.auth, environment_name=args.env
        )
        table = _apps_table([alias_info])

    args.console.print(table)


def _add_set_rev_parser(subparsers, parents):
    from fal.sdk import ALIAS_AUTH_MODES

    set_help = "Set application to a particular revision."
    parser = subparsers.add_parser(
        "set-rev",
        description=set_help,
        help=set_help,
        parents=parents,
    )
    parser.add_argument(
        "app_name",
        help="Application name.",
    )
    parser.add_argument(
        "app_rev",
        help="Application revision.",
    )
    parser.add_argument(
        "--auth",
        choices=ALIAS_AUTH_MODES,
        default=None,
        help="Application authentication mode.",
    )
    parser.add_argument(
        "--env",
        dest="env",
        help="Target environment (defaults to main).",
    )
    parser.set_defaults(func=_set_rev)


def _runners(args):
    client = SyncServerlessClient(host=args.host, team=args.team)
    start_time = args.since
    alias_runners = client.apps.runners(
        args.app_name, since=start_time, state=args.state, environment_name=args.env
    )
    if args.output == "pretty":
        runners_table = runners.runners_table(alias_runners)
        pending_runners = [
            runner for runner in alias_runners if runner.state == RunnerState.PENDING
        ]
        setup_runners = [
            runner for runner in alias_runners if runner.state == RunnerState.SETUP
        ]
        failing_runners = [
            runner
            for runner in alias_runners
            if runner.state == RunnerState.FAILURE_DELAY
        ]
        args.console.print(
            f"Runners: {len(alias_runners) - len(pending_runners) - len(setup_runners)}"
        )
        args.console.print(f"Runners Pending: {len(pending_runners)}")
        args.console.print(f"Runners Setting Up: {len(setup_runners)}")
        if len(failing_runners) > 0:
            args.console.print(
                f"[red]Runners Failing to start:[/] {len(failing_runners)}"
            )
        # Drop the alias column, which is the first column
        runners_table.columns.pop(0)
        args.console.print(runners_table)

        requests_table = runners.runners_requests_table(alias_runners)
        args.console.print(f"Requests: {len(requests_table.rows)}")
        args.console.print(requests_table)
    elif args.output == "json":
        runners._list_json(args, alias_runners)
    else:
        raise AssertionError(f"Invalid output format: {args.output}")


def _add_runners_parser(subparsers, parents):
    runners_help = "List application runners."
    parser = subparsers.add_parser(
        "runners",
        description=runners_help,
        help=runners_help,
        parents=[*parents, get_output_parser()],
    )
    parser.add_argument(
        "app_name",
        help="Application name.",
    )
    parser.add_argument(
        "--since",
        default=None,
        action=SinceAction,
        limit="1 day",
        help=(
            "Show terminated runners since the given time. "
            "Accepts 'now', relative like '30m', '1h', '1d', "
            "or an ISO timestamp. Max 24 hours."
        ),
    )
    parser.add_argument(
        "--state",
        choices=[
            "all",
            "idle",
            "running",
            "pending",
            "setup",
            "failure_delay",
            "terminated",
        ],
        nargs="+",
        default=None,
        help=("Filter by runner state(s). Choose one or more, or 'all'(default)."),
    )
    parser.add_argument(
        "--env",
        dest="env",
        help="Target environment (defaults to main).",
    )
    parser.set_defaults(func=_runners)


def _delete(args):
    client = get_client(args.host, args.team)
    with client.connect() as connection:
        res = connection.delete_alias(args.app_name, environment_name=args.env)
        if res is None:
            args.console.print(f"Application {args.app_name!r} not found.")
        else:
            args.console.print(f"Application {args.app_name!r} deleted ({res})")


def _add_delete_parser(subparsers, parents):
    delete_help = "Delete application."
    parser = subparsers.add_parser(
        "delete",
        description=delete_help,
        help=delete_help,
        parents=parents,
    )
    parser.add_argument(
        "app_name",
        help="Application name.",
    )
    parser.add_argument(
        "--env",
        dest="env",
        help="Target environment (defaults to main).",
    )
    parser.set_defaults(func=_delete)


def _delete_rev(args):
    client = get_client(args.host, args.team)
    with client.connect() as connection:
        connection.delete_application(args.app_rev)


def _add_delete_rev_parser(subparsers, parents):
    delete_help = "Delete application revision."
    parser = subparsers.add_parser(
        "delete-rev",
        description=delete_help,
        help=delete_help,
        parents=parents,
    )
    parser.add_argument(
        "app_rev",
        help="Application revision.",
    )
    parser.set_defaults(func=_delete_rev)


def add_parser(main_subparsers, parents):
    apps_help = "Manage fal applications."
    parser = main_subparsers.add_parser(
        "apps",
        aliases=["app"],
        description=apps_help,
        help=apps_help,
        parents=parents,
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        required=True,
    )

    parents = [*parents, FalClientParser(add_help=False)]

    _add_list_parser(subparsers, parents)
    _add_list_rev_parser(subparsers, parents)
    _add_set_rev_parser(subparsers, parents)
    _add_scale_parser(subparsers, parents)
    _add_rollout_parser(subparsers, parents)
    _add_runners_parser(subparsers, parents)
    _add_delete_parser(subparsers, parents)
    _add_delete_rev_parser(subparsers, parents)
