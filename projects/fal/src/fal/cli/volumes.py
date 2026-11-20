from fal.api.client import SyncServerlessClient

from .parser import FalClientParser


def _create(args):
    client = SyncServerlessClient(host=args.host, team=args.team)
    name = client.volumes.create(args.name)
    args.console.print(name)


def _add_create_parser(subparsers, parents):
    create_help = "Create a volume and print its canonical name."
    parser = subparsers.add_parser(
        "create",
        description=create_help,
        help=create_help,
        parents=parents,
    )
    parser.add_argument(
        "name",
        help="Volume name (letters, digits, '.', '_', '-'; stored lowercase).",
    )
    parser.set_defaults(func=_create)


def _list(args):
    import json

    client = SyncServerlessClient(host=args.host, team=args.team)
    volumes = client.volumes.list()

    if args.output == "json":
        json_volumes = [
            {"name": volume.name, "created_at": str(volume.created_at)}
            for volume in volumes
        ]
        args.console.print(json.dumps({"volumes": json_volumes}))
    elif args.output == "pretty":
        from rich.table import Table

        table = Table()
        table.add_column("Name")
        table.add_column("Created At")
        for volume in volumes:
            table.add_row(volume.name, str(volume.created_at))
        args.console.print(table)
    else:
        raise AssertionError(f"Invalid output format: {args.output}")


def _add_list_parser(subparsers, parents):
    from .parser import get_output_parser

    list_help = "List volumes."
    parser = subparsers.add_parser(
        "list",
        description=list_help,
        help=list_help,
        parents=[*parents, get_output_parser()],
    )
    parser.set_defaults(func=_list)


def _delete(args):
    client = SyncServerlessClient(host=args.host, team=args.team)
    client.volumes.delete(args.name)


def _add_delete_parser(subparsers, parents):
    delete_help = "Delete a volume. Fails while a deployed app still mounts it."
    parser = subparsers.add_parser(
        "delete",
        description=delete_help,
        help=delete_help,
        parents=parents,
    )
    parser.add_argument("name", help="Volume name.")
    parser.set_defaults(func=_delete)


def add_parser(main_subparsers, parents):
    volumes_help = "Manage fal volumes."
    parser = main_subparsers.add_parser(
        "volumes",
        aliases=["volume"],
        parents=parents,
        description=volumes_help,
        help=volumes_help,
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        required=True,
        parser_class=FalClientParser,
    )

    _add_create_parser(subparsers, parents)
    _add_list_parser(subparsers, parents)
    _add_delete_parser(subparsers, parents)
