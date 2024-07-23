from __future__ import annotations

from typing import TYPE_CHECKING

from .parser import FalClientParser

if TYPE_CHECKING:
    from fal.sdk import AliasInfo, ApplicationInfo


def _apps_table(apps: list[AliasInfo]):
    from rich.table import Table

    table = Table()
    table.add_column("Name", no_wrap=True)
    table.add_column("Revision")
    table.add_column("Auth")
    table.add_column("Min Concurrency")
    table.add_column("Max Concurrency")
    table.add_column("Max Multiplexing")
    table.add_column("Keep Alive")
    table.add_column("Active Workers")

    for app in apps:
        table.add_row(
            app.alias,
            app.revision,
            app.auth_mode,
            str(app.min_concurrency),
            str(app.max_concurrency),
            str(app.max_multiplexing),
            str(app.keep_alive),
            str(app.active_runners),
        )

    return table


def _list(args):
    from fal.sdk import FalServerlessClient

    client = FalServerlessClient(args.host)
    with client.connect() as connection:
        apps = connection.list_aliases()
        table = _apps_table(apps)

    args.console.print(table)


def _add_list_parser(subparsers, parents):
    list_help = "List applications."
    parser = subparsers.add_parser(
        "list",
        description=list_help,
        help=list_help,
        parents=parents,
    )
    parser.set_defaults(func=_list)


def _app_rev_table(revs: list[ApplicationInfo]):
    from rich.table import Table

    table = Table()
    table.add_column("Revision", no_wrap=True)
    table.add_column("Min Concurrency")
    table.add_column("Max Concurrency")
    table.add_column("Max Multiplexing")
    table.add_column("Keep Alive")
    table.add_column("Active Workers")

    for rev in revs:
        table.add_row(
            rev.application_id,
            str(rev.min_concurrency),
            str(rev.max_concurrency),
            str(rev.max_multiplexing),
            str(rev.keep_alive),
            str(rev.active_runners),
        )

    return table


def _list_rev(args):
    from fal.sdk import FalServerlessClient

    client = FalServerlessClient(args.host)
    with client.connect() as connection:
        revs = connection.list_applications()
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
    parser.set_defaults(func=_list_rev)


def _scale(args):
    from fal.sdk import FalServerlessClient

    client = FalServerlessClient(args.host)
    with client.connect() as connection:
        if (
            args.keep_alive is None
            and args.max_multiplexing is None
            and args.max_concurrency is None
            and args.min_concurrency is None
        ):
            args.console.log("No parameters for update were provided, ignoring.")
            return

        alias_info = connection.update_application(
            application_name=args.app_name,
            keep_alive=args.keep_alive,
            max_multiplexing=args.max_multiplexing,
            max_concurrency=args.max_concurrency,
            min_concurrency=args.min_concurrency,
        )
        table = _apps_table([alias_info])

    args.console.print(table)


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
    parser.set_defaults(func=_scale)


def _set_rev(args):
    from fal.sdk import FalServerlessClient

    client = FalServerlessClient(args.host)
    with client.connect() as connection:
        connection.create_alias(args.app_name, args.app_rev, args.auth)


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
        default="private",
        help="Application authentication mode.",
    )
    parser.set_defaults(func=_set_rev)


def _runners(args):
    from rich.table import Table

    from fal.sdk import FalServerlessClient

    client = FalServerlessClient(args.host)
    with client.connect() as connection:
        runners = connection.list_alias_runners(alias=args.app_name)

    table = Table()
    table.add_column("Runner ID")
    table.add_column("In Flight Requests")
    table.add_column("Expires in")
    table.add_column("Uptime")

    for runner in runners:
        table.add_row(
            runner.runner_id,
            str(runner.in_flight_requests),
            (
                "N/A (active)"
                if not runner.expiration_countdown
                else f"{runner.expiration_countdown}s"
            ),
            f"{runner.uptime} ({runner.uptime.total_seconds()}s)",
        )

    args.console.print(table)


def _add_runners_parser(subparsers, parents):
    runners_help = "List application runners."
    parser = subparsers.add_parser(
        "runners",
        description=runners_help,
        help=runners_help,
        parents=parents,
    )
    parser.add_argument(
        "app_name",
        help="Application name.",
    )
    parser.set_defaults(func=_runners)


def _delete(args):
    from fal.sdk import FalServerlessClient

    client = FalServerlessClient(args.host)
    with client.connect() as connection:
        connection.delete_alias(args.app_name)


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
    parser.set_defaults(func=_delete)


def _delete_rev(args):
    from fal.sdk import FalServerlessClient

    client = FalServerlessClient(args.host)
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
    _add_runners_parser(subparsers, parents)
    _add_delete_parser(subparsers, parents)
    _add_delete_rev_parser(subparsers, parents)
