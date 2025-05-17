from __future__ import annotations

from typing import List

from fal.sdk import RunnerInfo

from ._utils import get_client
from .parser import FalClientParser


def runners_table(runners: List[RunnerInfo]):
    from rich.table import Table

    table = Table()
    table.add_column("Alias")
    table.add_column("Runner ID")
    table.add_column("In Flight Requests")
    table.add_column("Missing Leases")
    table.add_column("Expires In")
    table.add_column("Uptime")
    table.add_column("Revision")

    for runner in runners:
        external_metadata = runner.external_metadata
        present = external_metadata.get("present_in_group", True)

        num_leases_with_request = len(
            [
                lease
                for lease in external_metadata.get("leases", [])
                if lease.get("request_id") is not None
            ]
        )

        table.add_row(
            runner.alias,
            # Mark lost runners in red
            runner.runner_id if present else f"[red]{runner.runner_id}[/]",
            str(runner.in_flight_requests),
            str(runner.in_flight_requests - num_leases_with_request),
            (
                "N/A (active)"
                if runner.expiration_countdown is None
                else f"{runner.expiration_countdown}s"
            ),
            f"{runner.uptime} ({runner.uptime.total_seconds()}s)",
            runner.revision,
        )

    return table


def runners_requests_table(runners: list[RunnerInfo]):
    from rich.table import Table

    table = Table()
    table.add_column("Runner ID")
    table.add_column("Request ID")
    table.add_column("Caller ID")

    for runner in runners:
        for lease in runner.external_metadata.get("leases", []):
            if not (req_id := lease.get("request_id")):
                continue

            table.add_row(
                runner.runner_id,
                req_id,
                lease.get("caller_user_id") or "",
            )

    return table


def _kill(args):
    client = get_client(args.host, args.team)
    with client.connect() as connection:
        connection.kill_runner(args.id)


def _list(args):
    client = get_client(args.host, args.team)
    with client.connect() as connection:
        runners = connection.list_runners()
        args.console.print(f"Runners: {len(runners)}")
        args.console.print(runners_table(runners))

        requests_table = runners_requests_table(runners)
        args.console.print(f"Requests: {len(requests_table.rows)}")
        args.console.print(requests_table)


def _add_kill_parser(subparsers, parents):
    kill_help = "Kill a runner."
    parser = subparsers.add_parser(
        "kill",
        description=kill_help,
        help=kill_help,
        parents=parents,
    )
    parser.add_argument(
        "id",
        help="Runner ID.",
    )
    parser.set_defaults(func=_kill)


def _add_list_parser(subparsers, parents):
    list_help = "List runners."
    parser = subparsers.add_parser(
        "list",
        description=list_help,
        help=list_help,
        parents=parents,
    )
    parser.set_defaults(func=_list)


def add_parser(main_subparsers, parents):
    runners_help = "Manage fal runners."
    parser = main_subparsers.add_parser(
        "runners",
        description=runners_help,
        help=runners_help,
        parents=parents,
        aliases=["machine"],  # backwards compatibility
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        required=True,
        parser_class=FalClientParser,
    )

    _add_kill_parser(subparsers, parents)
    _add_list_parser(subparsers, parents)
