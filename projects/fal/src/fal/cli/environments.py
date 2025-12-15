from ._utils import get_client
from .parser import FalClientParser, get_output_parser


def _list(args):
    import json

    client = get_client(args.host, args.team)
    with client.connect() as connection:
        environments = connection.list_environments()

    if args.output == "json":
        json_envs = [
            {
                "name": env.name,
                "description": env.description,
                "is_default": env.is_default,
                "created_at": str(env.created_at),
            }
            for env in environments
        ]
        args.console.print(json.dumps({"environments": json_envs}))
    elif args.output == "pretty":
        from rich.table import Table

        table = Table()
        table.add_column("Name")
        table.add_column("Description")
        table.add_column("Default")
        table.add_column("Created At")

        for env in environments:
            table.add_row(
                env.name,
                env.description or "",
                "Yes" if env.is_default else "",
                str(env.created_at),
            )

        args.console.print(table)
    else:
        raise AssertionError(f"Invalid output format: {args.output}")


def _add_list_parser(subparsers, parents):
    list_help = "List environments."
    parser = subparsers.add_parser(
        "list",
        description=list_help,
        help=list_help,
        parents=[*parents, get_output_parser()],
    )
    parser.set_defaults(func=_list)


def _create(args):
    client = get_client(args.host, args.team)
    with client.connect() as connection:
        env = connection.create_environment(args.name, description=args.description)
        args.console.print(f"Created environment '{env.name}'")


def _add_create_parser(subparsers, parents):
    create_help = "Create an environment."
    parser = subparsers.add_parser(
        "create",
        description=create_help,
        help=create_help,
        parents=parents,
    )
    parser.add_argument(
        "name",
        help="Environment name.",
    )
    parser.add_argument(
        "--description",
        help="Environment description.",
    )
    parser.set_defaults(func=_create)


def _delete(args):
    if not args.yes:
        args.console.print(
            f"[bold yellow]Warning:[/bold yellow] Deleting environment "
            f"'{args.name}' will permanently delete:\n"
            "  • All secrets in this environment\n"
            "  • All apps deployed to this environment\n"
        )
        confirmation = input(
            f"Type the environment name '{args.name}' to confirm deletion: "
        ).strip()

        if confirmation != args.name:
            args.console.print(
                "[red]Deletion cancelled.[/red] Environment name did not match."
            )
            return

    client = get_client(args.host, args.team)
    with client.connect() as connection:
        connection.delete_environment(args.name)
        args.console.print(f"[green]Deleted environment '{args.name}'[/green]")


def _add_delete_parser(subparsers, parents):
    delete_help = "Delete an environment."
    parser = subparsers.add_parser(
        "delete",
        description=delete_help,
        help=delete_help,
        parents=parents,
    )
    parser.add_argument(
        "name",
        help="Environment name.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    parser.set_defaults(func=_delete)


def add_parser(main_subparsers, parents):
    envs_help = "Manage fal environments."
    parser = main_subparsers.add_parser(
        "environments",
        aliases=["env"],
        parents=parents,
        description=envs_help,
        help=envs_help,
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        required=True,
        parser_class=FalClientParser,
    )

    _add_list_parser(subparsers, parents)
    _add_create_parser(subparsers, parents)
    _add_delete_parser(subparsers, parents)
