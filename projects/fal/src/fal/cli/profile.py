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
    config.set_internal("profile", args.PROFILE)
    args.console.print(f"Default profile set to [cyan]{args.PROFILE}[/].")
    config.profile = args.PROFILE
    config.save()


def _unset(args):
    config = Config()
    config.set_internal("profile", None)
    args.console.print("Default profile unset.")
    config.profile = None
    config.save()


def _key_set(args):
    config = Config()
    key_id, key_secret = args.KEY.split(":", 1)
    config.set("key", f"{key_id}:{key_secret}")
    args.console.print(f"Key set for profile [cyan]{config.profile}[/].")
    config.save()


def _delete(args):
    config = Config()
    if config.profile == args.PROFILE:
        config.set_internal("profile", None)

    config.delete(args.PROFILE)
    args.console.print(f"Profile [cyan]{args.PROFILE}[/] deleted.")
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
        "PROFILE",
        help="Profile name.",
    )
    set_parser.set_defaults(func=_set)

    unset_help = "Unset default profile."
    unset_parser = subparsers.add_parser(
        "unset",
        description=unset_help,
        help=unset_help,
        parents=parents,
    )
    unset_parser.set_defaults(func=_unset)

    key_set_help = "Set key for profile."
    key_set_parser = subparsers.add_parser(
        "key",
        description=key_set_help,
        help=key_set_help,
        parents=parents,
    )
    key_set_parser.add_argument(
        "KEY",
        help="Key ID and secret separated by a colon.",
    )
    key_set_parser.set_defaults(func=_key_set)

    delete_help = "Delete profile."
    delete_parser = subparsers.add_parser(
        "delete",
        description=delete_help,
        help=delete_help,
        parents=parents,
    )
    delete_parser.add_argument(
        "PROFILE",
        help="Profile name.",
    )
    delete_parser.set_defaults(func=_delete)
