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
