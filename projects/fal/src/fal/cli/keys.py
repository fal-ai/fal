from fal.api.client import SyncServerlessClient
from fal.sdk import KeyScope

from .parser import FalClientParser


def _create(args):
    client = SyncServerlessClient(host=args.host, team=args.team)
    parsed_scope = KeyScope(args.scope)
    key_id, key_secret = client.keys.create(scope=parsed_scope, description=args.desc)
    args.console.print(
        f"Generated key id and key secret, with the scope `{args.scope}`.\n"
        "This is the only time the secret will be visible.\n"
        "You will need to generate a new key pair if you lose access to this "
        "secret."
    )
    args.console.print(f"FAL_KEY='{key_id}:{key_secret}'")


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
    import json

    client = SyncServerlessClient(host=args.host, team=args.team)
    keys = client.keys.list()

    if args.output == "json":
        json_keys = [
            {
                "key_id": key.key_id,
                "created_at": str(key.created_at),
                "scope": str(key.scope.value),
                "description": key.alias,
            }
            for key in keys
        ]
        args.console.print(json.dumps({"keys": json_keys}))
    elif args.output == "pretty":
        from rich.table import Table

        table = Table()
        table.add_column("Key ID")
        table.add_column("Created At")
        table.add_column("Scope")
        table.add_column("Description")

        for key in keys:
            table.add_row(
                key.key_id,
                str(key.created_at),
                str(key.scope.value),
                key.alias,
            )

        args.console.print(table)
    else:
        raise AssertionError(f"Invalid output format: {args.output}")


def _add_list_parser(subparsers, parents):
    from .parser import get_output_parser

    list_help = "List keys."
    parser = subparsers.add_parser(
        "list",
        description=list_help,
        help=list_help,
        parents=[*parents, get_output_parser()],
    )
    parser.set_defaults(func=_list)


def _revoke(args):
    client = SyncServerlessClient(host=args.host, team=args.team)
    client.keys.revoke(args.key_id)


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
