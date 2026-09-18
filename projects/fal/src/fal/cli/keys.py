import sys

from fal.api.client import SyncServerlessClient
from fal.sdk import KeyPreset, KeyScope

from .parser import FalClientParser

PRESET_DESCRIPTIONS = {
    KeyPreset.FULL: "Full access to everything in your account.",
    KeyPreset.API: "Run models and upload files.",
    KeyPreset.DEPLOY: "Deploy apps and upload files.",
    KeyPreset.READONLY: "Read models, apps and logs.",
}


def _prompt_preset(args) -> KeyPreset:
    """Prompt the user to select a key preset. Returns the chosen preset."""
    from rich.prompt import Prompt
    from rich.style import Style
    from rich.table import Table

    if not sys.stdin.isatty() or not args.console.is_terminal:
        raise ValueError(
            "Picking a key preset requires interactive input. "
            "Re-run with --preset to pick one non-interactively."
        )

    args.console.print("Create a key\n")

    presets = list(KeyPreset)
    table = Table(border_style=Style(frame=False), show_header=False)
    table.add_column("#")
    table.add_column("Preset")
    table.add_column("Description")

    for idx, preset in enumerate(presets, 1):
        table.add_row(f"  {idx}", preset.value, PRESET_DESCRIPTIONS[preset])

    args.console.print(table)

    indices = [str(i) for i in range(1, len(presets) + 1)]
    preset_names = [preset.value for preset in presets]
    choice = Prompt.ask(
        "Select a preset",
        choices=indices + preset_names,
        default=KeyPreset.API.value,
        show_choices=False,
        case_sensitive=False,
    )

    if choice.upper() in preset_names:
        return KeyPreset(choice.upper())
    else:
        return presets[int(choice) - 1]


def _resolve_preset(args) -> KeyPreset:
    if args.preset:
        return KeyPreset(args.preset)
    elif args.scope:
        return KeyPreset.from_scope(KeyScope(args.scope))
    else:
        return _prompt_preset(args)


def _create(args):
    preset = _resolve_preset(args)
    client = SyncServerlessClient(host=args.host, team=args.team)
    key_id, key_secret = client.keys.create(preset=preset, description=args.desc)
    args.console.print(
        f"Generated key id and key secret, with the preset `{preset.value}`.\n"
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
    # Without either flag the preset is picked interactively.
    permissions = parser.add_mutually_exclusive_group()
    permissions.add_argument(
        "--preset",
        choices=[preset.value for preset in KeyPreset],
        help="The permission preset of the key.",
    )
    permissions.add_argument(
        "--scope",
        choices=[KeyScope.ADMIN.value, KeyScope.API.value],
        help="Deprecated, use --preset. The privilage scope of the key.",
    )
    parser.add_argument(
        "--desc",
        "--alias",
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
                "scope": key.scope.value if key.scope else None,
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
                key.scope.value if key.scope else "-",
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
