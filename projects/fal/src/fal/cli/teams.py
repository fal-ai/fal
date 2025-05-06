from fal.cli.auth import _list_accounts, _set_account, _unset_account


def add_parser(main_subparsers, parents):
    teams_help = "Manage teams."
    parser = main_subparsers.add_parser(
        "teams",
        aliases=["team"],
        description=teams_help,
        help=teams_help,
        parents=parents,
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        dest="cmd",
        required=True,
    )

    list_help = "List teams."
    list_parser = subparsers.add_parser(
        "list",
        description=list_help,
        help=list_help,
        parents=parents,
    )
    list_parser.set_defaults(func=_list_accounts)

    set_help = "Set the current team."
    set_parser = subparsers.add_parser(
        "set",
        description=set_help,
        help=set_help,
        parents=parents,
    )
    set_parser.add_argument("account", help="The team to set.")
    set_parser.set_defaults(func=_set_account)

    unset_help = "Unset the current team."
    unset_parser = subparsers.add_parser(
        "unset",
        description=unset_help,
        help=unset_help,
        parents=parents,
    )
    unset_parser.set_defaults(func=_unset_account)
