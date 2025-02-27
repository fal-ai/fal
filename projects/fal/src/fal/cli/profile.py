from rich.table import Table

from fal.config import Config


def _list(args):
    config = Config()

    table = Table()
    table.add_column("Default")
    table.add_column("Profile")
    table.add_column("Settings")

    for profile in config.profiles():
        table.add_row(
            "*" if profile == config._profile else "",
            profile,
            ", ".join(key for key in config._config[profile]),
        )

    args.console.print(table)


def _set(args):
    config = Config()
    config.set_internal("profile", args.profile)
    args.console.print(f"Default profile set to [cyan]{args.profile}[/].")
    config.save()


def add_parser(main_subparsers, parents):
    auth_help = "Profile management."
    parser = main_subparsers.add_parser(
        "profile",
        description=auth_help,
        help=auth_help,
        parents=parents,
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        dest="cmd",
        required=True,
    )

    list_help = "List all profiles."
    list_parser = subparsers.add_parser(
        "list",
        description=list_help,
        help=list_help,
        parents=parents,
    )
    list_parser.set_defaults(func=_list)

    set_help = "Set default profile."
    set_parser = subparsers.add_parser(
        "set",
        description=set_help,
        help=set_help,
        parents=parents,
    )
    set_parser.add_argument(
        "profile",
        help="Profile name.",
    )
    set_parser.set_defaults(func=_set)
