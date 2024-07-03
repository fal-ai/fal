from .parser import DictAction, FalClientParser


def _set(args):
    from fal.sdk import FalServerlessClient

    client = FalServerlessClient(args.host)
    with client.connect() as connection:
        for name, value in args.secrets.items():
            connection.set_secret(name, value)


def _add_set_parser(subparsers, parents):
    set_help = "Set a secret."
    epilog = "Examples:\n" "  fal secrets set HF_TOKEN=hf_***"

    parser = subparsers.add_parser(
        "set",
        description=set_help,
        help=set_help,
        parents=parents,
        epilog=epilog,
    )
    parser.add_argument(
        "secrets",
        metavar="NAME=VALUE",
        nargs="+",
        action=DictAction,
        help="Secret NAME=VALUE pairs.",
    )
    parser.set_defaults(func=_set)


def _list(args):
    from rich.table import Table

    from fal.sdk import FalServerlessClient

    table = Table()
    table.add_column("Secret Name")
    table.add_column("Created At")

    client = FalServerlessClient(args.host)
    with client.connect() as connection:
        for secret in connection.list_secrets():
            table.add_row(secret.name, str(secret.created_at))

    args.console.print(table)


def _add_list_parser(subparsers, parents):
    list_help = "List secrets."
    parser = subparsers.add_parser(
        "list",
        description=list_help,
        help=list_help,
        parents=parents,
    )
    parser.set_defaults(func=_list)


def _unset(args):
    from fal.sdk import FalServerlessClient

    client = FalServerlessClient(args.host)
    with client.connect() as connection:
        connection.delete_secret(args.secret)


def _add_unset_parser(subparsers, parents):
    unset_help = "Unset a secret."
    parser = subparsers.add_parser(
        "unset",
        description=unset_help,
        help=unset_help,
        parents=parents,
    )
    parser.add_argument(
        "secret",
        metavar="NAME",
        help="Secret's name.",
    )
    parser.set_defaults(func=_unset)


def add_parser(main_subparsers, parents):
    secrets_help = "Manage fal secrets."
    parser = main_subparsers.add_parser(
        "secrets",
        aliases=["secret"],
        parents=parents,
        description=secrets_help,
        help=secrets_help,
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        required=True,
        parser_class=FalClientParser,
    )

    _add_set_parser(subparsers, parents)
    _add_list_parser(subparsers, parents)
    _add_unset_parser(subparsers, parents)
