from __future__ import annotations

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


CUSTOM_CHOICE = "CUSTOM"


def _prompt_grant(args) -> tuple[KeyPreset | None, list[str] | None]:
    """Prompt for a preset or a custom permission list. Exactly one is set."""
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

    table.add_row(
        f"  {len(presets) + 1}", CUSTOM_CHOICE, "Pick the permissions yourself."
    )
    args.console.print(table)

    indices = [str(i) for i in range(1, len(presets) + 2)]
    preset_names = [preset.value for preset in presets]
    choice = Prompt.ask(
        "Select permissions — pick a preset, or CUSTOM to choose your own",
        choices=indices + preset_names + [CUSTOM_CHOICE],
        default=KeyPreset.API.value,
        show_choices=False,
        case_sensitive=False,
    )

    if choice.isdigit():
        index = int(choice) - 1
        if index == len(presets):
            return None, _prompt_permissions(args)
        return presets[index], None
    elif choice.upper() == CUSTOM_CHOICE:
        return None, _prompt_permissions(args)
    else:
        return KeyPreset(choice.upper()), None


def _prompt_permissions(args) -> list[str]:
    """Prompt for a permission list. The server is what validates the names."""
    from rich.prompt import Prompt

    while True:
        answer = Prompt.ask(
            "Permissions (comma separated, e.g. models:requests:submit)"
        )
        permissions = _split_permissions([answer])
        if permissions:
            return permissions

        args.console.print("[red]Enter at least one permission.[/]")


def _split_permissions(values: list[str]) -> list[str]:
    """Flatten repeated --permission flags, each of which may be a list."""
    return [
        permission.strip()
        for value in values
        for permission in value.split(",")
        if permission.strip()
    ]


def _resolve_grant(args) -> tuple[KeyPreset | None, list[str] | None]:
    if args.preset:
        return KeyPreset(args.preset), None
    elif args.permission:
        permissions = _split_permissions(args.permission)
        if not permissions:
            raise ValueError("--permission needs at least one permission name.")
        return None, permissions
    elif args.scope:
        return KeyPreset.from_scope(KeyScope(args.scope)), None
    else:
        return _prompt_grant(args)


def _create(args):
    preset, permissions = _resolve_grant(args)
    client = SyncServerlessClient(host=args.host, team=args.team)
    key_id, key_secret = client.keys.create(
        preset=preset, permissions=permissions, description=args.desc
    )
    granted = preset.value if preset else ", ".join(permissions or [])
    args.console.print(
        f"Generated key id and key secret, with `{granted}`.\n"
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
        "--permission",
        action="append",
        metavar="PERMISSION",
        help="Grant an explicit permission instead of a preset. Repeatable, "
        "and accepts a comma separated list.",
    )
    permissions.add_argument(
        "--scope",
        choices=[KeyScope.ADMIN.value, KeyScope.API.value],
        help="Deprecated, use --preset. The privilege scope of the key.",
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
