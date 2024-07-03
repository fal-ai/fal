from fal.sdk import KeyScope

from .parser import FalClientParser


def _create(args):
    from fal.sdk import FalServerlessClient

    client = FalServerlessClient(args.host)
    with client.connect() as connection:
        parsed_scope = KeyScope(args.scope)
        result = connection.create_user_key(parsed_scope, args.desc)
        args.console.print(
            f"Generated key id and key secret, with the scope `{args.scope}`.\n"
            "This is the only time the secret will be visible.\n"
            "You will need to generate a new key pair if you lose access to this "
            "secret."
        )
        args.console.print(f"FAL_KEY='{result[1]}:{result[0]}'")


def _add_create_parser(subparsers, parents):
    create_help = "Create a key."
    parser = subparsers.add_parser(
        "create",
        description=create_help,
        help=create_help,
        parents=parents,
    )
    parser.add_argument(
        "--scope",
        required=True,
        choices=[KeyScope.ADMIN.value, KeyScope.API.value],
        help="The privilage scope of the key.",
    )
    parser.add_argument(
        "--desc",
        help='Key description (e.g. "My Test Key")',
    )
    parser.set_defaults(func=_create)


def _list(args):
    from rich.table import Table

    from fal.sdk import FalServerlessClient

    client = FalServerlessClient(args.host)
    table = Table()
    table.add_column("Key ID")
    table.add_column("Created At")
    table.add_column("Scope")
    table.add_column("Description")

    with client.connect() as connection:
        keys = connection.list_user_keys()
        for key in keys:
            table.add_row(
                key.key_id,
                str(key.created_at),
                str(key.scope.value),
                key.alias,
            )

    args.console.print(table)


def _add_list_parser(subparsers, parents):
    list_help = "List keys."
    parser = subparsers.add_parser(
        "list",
        description=list_help,
        help=list_help,
        parents=parents,
    )
    parser.set_defaults(func=_list)


def _revoke(args):
    from fal.sdk import FalServerlessClient

    client = FalServerlessClient(args.host)
    with client.connect() as connection:
        connection.revoke_user_key(args.key_id)


def _add_revoke_parser(subparsers, parents):
    revoke_help = "Revoke key."
    parser = subparsers.add_parser(
        "revoke",
        description=revoke_help,
        help=revoke_help,
        parents=parents,
    )
    parser.add_argument(
        "key_id",
        help="Key ID.",
    )
    parser.set_defaults(func=_revoke)


def add_parser(main_subparsers, parents):
    keys_help = "Manage fal keys."
    parser = main_subparsers.add_parser(
        "keys",
        aliases=["key"],
        description=keys_help,
        help=keys_help,
        parents=parents,
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        required=True,
        parser_class=FalClientParser,
    )

    _add_create_parser(subparsers, parents)
    _add_list_parser(subparsers, parents)
    _add_revoke_parser(subparsers, parents)
