from fal.cli.auth import _list_accounts, _set_account, _unset_account


def _unset_account_cmd(args):
    _unset_account(args, notify=True)


def add_parser(main_subparsers, parents):
    account_help = "Manage accounts."
    parser = main_subparsers.add_parser(
        "account",
        aliases=["accounts", "team", "teams"],
        description=account_help,
        help=account_help,
        parents=parents,
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        dest="cmd",
        required=True,
    )

    from .parser import get_output_parser

    list_help = "List accounts."
    list_parser = subparsers.add_parser(
        "list",
        description=list_help,
        help=list_help,
        parents=[*parents, get_output_parser()],
    )
    list_parser.set_defaults(func=_list_accounts)

    set_help = "Set the current account."
    set_parser = subparsers.add_parser(
        "set",
        description=set_help,
        help=set_help,
        parents=parents,
    )
    set_parser.add_argument("account", help="The account to set.")
    set_parser.set_defaults(func=_set_account)

    unset_help = "Unset the current account."
    unset_parser = subparsers.add_parser(
        "unset",
        description=unset_help,
        help=unset_help,
        parents=parents,
    )
    unset_parser.set_defaults(func=_unset_account_cmd)
