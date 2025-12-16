from fal.api.client import SyncServerlessClient

from .parser import DictAction, FalClientParser


def _set(args):
    client = SyncServerlessClient(host=args.host, team=args.team)
    for name, value in args.secrets.items():
        client.secrets.set(name, value, environment_name=args.env)


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
    parser.add_argument(
        "--env",
        dest="env",
        help="Target environment (defaults to main).",
    )
    parser.set_defaults(func=_set)


def _list(args):
    import json

    client = SyncServerlessClient(host=args.host, team=args.team)
    secrets = client.secrets.list(environment_name=args.env)

    if args.output == "json":
        json_secrets = [
            {
                "name": secret.name,
                "environment": secret.environment_name,
                "created_at": str(secret.created_at),
            }
            for secret in secrets
        ]
        args.console.print(json.dumps({"secrets": json_secrets}))
    elif args.output == "pretty":
        from rich.table import Table

        table = Table()
        table.add_column("Secret Name")
        table.add_column("Environment")
        table.add_column("Created At")

        for secret in secrets:
            table.add_row(
                secret.name,
                secret.environment_name or "main",
                str(secret.created_at),
            )

        args.console.print(table)
    else:
        raise AssertionError(f"Invalid output format: {args.output}")


def _add_list_parser(subparsers, parents):
    from .parser import get_output_parser

    list_help = "List secrets."
    parser = subparsers.add_parser(
        "list",
        description=list_help,
        help=list_help,
        parents=[*parents, get_output_parser()],
    )
    parser.add_argument(
        "--env",
        dest="env",
        help="Target environment (defaults to main).",
    )
    parser.set_defaults(func=_list)


def _unset(args):
    client = SyncServerlessClient(host=args.host, team=args.team)
    client.secrets.unset(args.secret, environment_name=args.env)


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
    parser.add_argument(
        "--env",
        dest="env",
        help="Target environment (defaults to main).",
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
