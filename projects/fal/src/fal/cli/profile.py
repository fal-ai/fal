from __future__ import annotations

from rich.table import Table

from fal.config import Config


def _list(args):
    import json

    config = Config()

    if args.output == "json":
        json_profiles = []
        for profile in config.profiles():
            json_profiles.append(
                {
                    "profile": profile,
                    "is_set": profile == config._profile,
                    "settings": list(config._config[profile].keys()),
                }
            )
        args.console.print(json.dumps({"profiles": json_profiles}))
    elif args.output == "pretty":
        table = Table()
        table.add_column("Set")
        table.add_column("Profile")
        table.add_column("Settings")

        for profile in config.profiles():
            table.add_row(
                "*" if profile == config._profile else "",
                profile,
                ", ".join(key for key in config._config[profile]),
            )

        args.console.print(table)
    else:
        raise AssertionError(f"Invalid output format: {args.output}")


def _set(args):
    config = Config()

    # Check if the profile exists
    if args.PROFILE not in config._config:
        # Profile doesn't exist, offer to create it
        args.console.print(f"Profile [cyan]{args.PROFILE}[/] does not exist.")
        create_profile = input("Would you like to create it? (y/N): ").strip().lower()

        if create_profile in ["y", "yes"]:
            # Create the profile by setting it
            with config.edit() as config:
                config.set_internal("profile", args.PROFILE)
                config._config[args.PROFILE] = {}
                config._profile = args.PROFILE
            args.console.print(
                f"Profile [cyan]{args.PROFILE}[/] created and set as default."
            )
        else:
            args.console.print("Profile creation cancelled.")
            return
    else:
        # Profile exists, just set it as default
        with config.edit() as config:
            config.set_internal("profile", args.PROFILE)
            config._profile = args.PROFILE
        args.console.print(f"Default profile set to [cyan]{args.PROFILE}[/].")

    # Check if key is set for the profile
    if not config.get("key"):
        args.console.print(
            "No key set for profile. Use [bold]fal profile key[/] to set a key."
        )


def _unset(args, config: Config | None = None):
    config = config or Config()

    with config.edit() as config:
        config.profile = None
        args.console.print("Default profile unset.")


def _key_set(args):
    config = Config(validate_profile=True)

    args.console.print("Go to https://fal.ai/dashboard/keys to create a key.")
    args.console.print(
        "Enter the key in the format [cyan][bold]id:secret[/][/]. You can copy and paste it from the browser."  # noqa: E501
    )

    while True:
        key = input("Enter the key: ")
        if ":" in key:
            break
        args.console.print(
            "[yellow]Invalid key. The key must be in the format:[/] [cyan][bold]id:secret[/][/]."  # noqa: E501
        )

    with config.edit():
        config.set("key", key)
        args.console.print(f"Key set for profile [cyan]{config.profile}[/].")


def _host_set(args):
    with Config().edit() as config:
        config.set("host", args.HOST)
        args.console.print(f"Fal host set to [cyan]{args.HOST}[/].")


def _create(args):
    config = Config()

    # Check if the profile already exists
    if args.PROFILE in config._config:
        args.console.print(f"Profile [cyan]{args.PROFILE}[/] already exists.")
        return

    # Create the profile
    with config.edit() as config:
        config._config[args.PROFILE] = {}

    args.console.print(
        f"Profile [cyan]{args.PROFILE}[/] created. Setting it as default profile..."
    )
    _set(args)


def _delete(args):
    with Config().edit() as config:
        if config.profile == args.PROFILE:
            config.set_internal("profile", None)

        config.delete_profile(args.PROFILE)
        args.console.print(f"Profile [cyan]{args.PROFILE}[/] deleted.")


def add_parser(main_subparsers, parents):
    auth_help = "Profile management."
    parser = main_subparsers.add_parser(
        "profile",
        aliases=["profiles"],
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

    from .parser import get_output_parser

    list_help = "List all profiles."
    list_parser = subparsers.add_parser(
        "list",
        description=list_help,
        help=list_help,
        parents=[*parents, get_output_parser()],
    )
    list_parser.set_defaults(func=_list)

    set_help = "Set default profile. If the profile doesn't exist, you'll be prompted to create it."  # noqa: E501
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
    key_set_parser.set_defaults(func=_key_set)

    host_set_help = "Set fal host."
    host_set_parser = subparsers.add_parser(
        "host",
        description=host_set_help,
        help=host_set_help,
        parents=parents,
    )
    host_set_parser.add_argument(
        "HOST",
        help="Fal host.",
    )
    host_set_parser.set_defaults(func=_host_set)

    create_help = "Create a new profile."
    create_parser = subparsers.add_parser(
        "create",
        description=create_help,
        help=create_help,
        parents=parents,
    )
    create_parser.add_argument(
        "PROFILE",
        help="Profile name.",
    )
    create_parser.set_defaults(func=_create)

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
