def _list(args):
    from rich.table import Table

    from fal.auth import USER
    from fal.config import Config

    table = Table()
    table.add_column("Default")
    table.add_column("Team")
    table.add_column("Full Name")
    table.add_column("ID")

    default_team = Config().get("team")

    for team in USER.teams:
        default = default_team and default_team.lower() == team["nickname"].lower()
        table.add_row(
            "*" if default else "", team["nickname"], team["full_name"], team["user_id"]
        )

    args.console.print(table)


def _set(args):
    from fal.config import Config
    from fal.sdk import USER

    team = args.team.lower()
    for team_info in USER.teams:
        if team_info["nickname"].lower() == team:
            break
    else:
        raise ValueError(f"Team {args.team} not found")

    with Config().edit() as config:
        config.set("team", team)


def _unset(args):
    from fal.config import Config

    with Config().edit() as config:
        config.unset("team")


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
    list_parser.set_defaults(func=_list)

    set_help = "Set the current team."
    set_parser = subparsers.add_parser(
        "set",
        description=set_help,
        help=set_help,
        parents=parents,
    )
    set_parser.add_argument("team", help="The team to set.")
    set_parser.set_defaults(func=_set)

    unset_help = "Unset the current team."
    unset_parser = subparsers.add_parser(
        "unset",
        description=unset_help,
        help=unset_help,
        parents=parents,
    )
    unset_parser.set_defaults(func=_unset)
