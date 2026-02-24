from __future__ import annotations

from fal.auth import current_user_info
from fal.auth.local import load_preference, save_preference
from fal.cli import profile
from fal.sdk import get_default_credentials

AUTH_CONNECTIONS = [
    {"name": "github", "label": "Continue with GitHub"},
    {"name": "google", "label": "Continue with Google"},
    {"name": "sso", "label": "Continue with SSO"},
]

_LAST_CONNECTION_KEY = "last_auth_connection"


def _load_last_connection() -> str | None:
    return load_preference(_LAST_CONNECTION_KEY)


def _save_last_connection(connection: str) -> None:
    save_preference(_LAST_CONNECTION_KEY, connection)


def _prompt_connection(args) -> str:
    """Prompt the user to select an auth connection. Returns the connection name."""
    from rich.prompt import Prompt
    from rich.style import Style
    from rich.table import Table

    last_connection = _load_last_connection()

    args.console.print("Login or sign up\n")

    table = Table(border_style=Style(frame=False), show_header=False)
    table.add_column("#")
    table.add_column("Connection")

    connection_names = [c["name"] for c in AUTH_CONNECTIONS]
    # SSO saves the actual domain, not "sso"
    is_last_sso = last_connection and last_connection not in connection_names

    for idx, conn in enumerate(AUTH_CONNECTIONS, 1):
        if conn["name"] == last_connection:
            suffix = " [dim](Last used)[/]"
        elif conn["name"] == "sso" and is_last_sso:
            suffix = f" [dim](Last used: {last_connection})[/]"
        else:
            suffix = ""
        table.add_row(f"  {idx}", f"{conn['label']}{suffix}")

    args.console.print(table)

    default = None
    if last_connection:
        for idx, conn in enumerate(AUTH_CONNECTIONS, 1):
            if conn["name"] == last_connection:
                default = str(idx)
                break
        if default is None and is_last_sso:
            default = str(connection_names.index("sso") + 1)

    indices = [str(i) for i in range(1, len(AUTH_CONNECTIONS) + 1)]
    choice = Prompt.ask(
        "Select a connection",
        choices=indices + connection_names,
        default=default,
        show_choices=False,
        show_default=bool(default),
    )

    if choice in connection_names:
        connection = choice
    else:
        connection = AUTH_CONNECTIONS[int(choice) - 1]["name"]

    if connection == "sso":
        sso_default = last_connection if is_last_sso else None
        while True:
            connection = Prompt.ask(
                "Enter your enterprise single sign-on domain",
                default=sso_default,
                show_default=bool(sso_default),
            ).strip()
            if connection:
                break
            args.console.print("[red]Domain cannot be empty.[/]")

    _save_last_connection(connection)
    return connection


def _login(args):
    from fal.auth import UserAccess, login
    from fal.config import Config
    from fal.console.icons import CHECK_ICON, CROSS_ICON
    from fal.exceptions import FalServerlessException

    if args.connection:
        connection = args.connection.strip()
        _save_last_connection(connection)
    else:
        connection = _prompt_connection(args)

    try:
        login(args.console, connection=connection)
        args.console.print(f"{CHECK_ICON} Authenticated successfully, welcome!")
    except FalServerlessException as e:
        args.console.print(f"{CROSS_ICON} {e}")
        return

    current_team = Config().get_internal("team")
    if current_team and not any(
        a["nickname"] == current_team for a in UserAccess().accounts
    ):
        _unset_account(args)

    _set_account(args)


def _logout(args):
    from fal.auth import logout
    from fal.console.icons import CHECK_ICON, CROSS_ICON
    from fal.exceptions import FalServerlessException

    try:
        logout(args.console)
        args.console.print(f"{CHECK_ICON} Logged out of [cyan bold]fal[/]. Bye!")
    except FalServerlessException as e:
        args.console.print(f"{CROSS_ICON} {e}")
        return

    _unset_account(args)


def _list_accounts(args):
    import json

    from fal.auth import UserAccess
    from fal.config import Config

    user_access = UserAccess()
    config = Config()
    personal = next(a for a in user_access.accounts if a["is_personal"])
    current_account_name = config.get_internal("team") or personal["nickname"]

    # NOTE: might be used by other commands that don't have the --output/--json flag
    output = getattr(args, "output", "pretty")
    if output == "json":
        json_accounts = []
        for account in user_access.accounts:
            selected = account["nickname"] == current_account_name
            json_accounts.append(
                {
                    "nickname": account["nickname"],
                    "full_name": account["full_name"],
                    "type": "personal"
                    if account["is_personal"]
                    else "org"
                    if account["is_org"]
                    else "team",
                    "is_selected": selected,
                }
            )
        args.console.print(json.dumps({"teams": json_accounts}))
    elif output == "pretty":
        from rich.style import Style
        from rich.table import Table

        table = Table(border_style=Style(frame=False), show_header=False)
        table.add_column("#")
        table.add_column("Account")
        table.add_column("Type")

        for idx, account in enumerate(user_access.accounts):
            selected = account["nickname"] == current_account_name
            color = "bold yellow" if selected else None
            account_type = (
                "Personal"
                if account["is_personal"]
                else "Org"
                if account["is_org"]
                else "Team"
            )

            table.add_row(
                f"* {idx + 1}" if selected else f"  {idx + 1}",
                f"{account['full_name']}\n[dim]{account['nickname']}[/]",
                account_type,
                style=color,
            )

        args.console.print(table)
    else:
        raise AssertionError(f"Invalid output format: {output}")


def _unset_account(args, *, notify: bool = False) -> str | None:
    from fal.config import Config

    with Config().edit() as config:
        previous = config.get_internal("team")
        config.unset_internal("team")

    if notify:
        from fal.auth import UserAccess

        user_access = UserAccess()
        personal = next(a for a in user_access.accounts if a["is_personal"])
        args.console.print(
            f"Using personal account [cyan]{personal['full_name']}[/] "
            f"([cyan]{personal['nickname']}[/]). "
            "You can change this with [bold]fal account set[/]"
        )

    return previous


def _set_account(args):
    from rich.prompt import Prompt

    from fal.auth import UserAccess
    from fal.config import Config

    user_access = UserAccess()

    if hasattr(args, "account") and args.account:
        if args.account.isdigit():
            acc_index = int(args.account) - 1
            account = user_access.accounts[acc_index]
        else:
            account = user_access.get_account(args.account)
    else:
        config = Config()
        explicit_team = config.get_internal("team")
        _list_accounts(args)

        indices = list(map(str, range(1, len(user_access.accounts) + 1)))
        team_names = [account["nickname"] for account in user_access.accounts]

        personal = next(a for a in user_access.accounts if a["is_personal"])
        default_name = explicit_team or personal["nickname"]
        default = next(
            (
                str(i + 1)
                for i, a in enumerate(user_access.accounts)
                if a["nickname"] == default_name
            ),
            None,
        )

        acc_choice = Prompt.ask(
            "Select an account by number",
            choices=indices + team_names,
            show_choices=False,
            default=default,
            show_default=bool(default),
        )
        if acc_choice in indices:
            acc_index = int(acc_choice) - 1
            account = user_access.accounts[acc_index]
        else:
            account = user_access.get_account(acc_choice)

    account_type = (
        "personal" if account["is_personal"] else "org" if account["is_org"] else "team"
    )
    args.console.print(
        f"Using {account_type} account [cyan]{account['full_name']}[/] "
        f"([cyan]{account['nickname']}[/]). "
        "You can change this later with [bold]fal account set[/]"
    )

    with Config().edit() as config:
        config.set_internal("team", account["nickname"])

        # Unset the profile if set
        if current_profile := config.get_internal("profile"):
            args.console.print(
                f"\n[yellow]Unsetting profile [cyan]{current_profile}[/] "
                "to make team selection effective.[/]"
            )
            profile._unset(args, config=config)


def _whoami(args):
    creds = get_default_credentials()
    user_info = current_user_info(creds.to_headers())

    full_name = user_info["full_name"]
    nickname = user_info["nickname"]
    user_id = user_info["user_id"]

    args.console.print(f"Hello, {full_name}: {nickname!r} - {user_id!r}")


def add_parser(main_subparsers, parents):
    auth_help = "Authenticate with fal."
    parser = main_subparsers.add_parser(
        "auth",
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

    login_help = "Log in a user."
    login_parser = subparsers.add_parser(
        "login",
        description=login_help,
        help=login_help,
        parents=parents,
    )
    login_parser.add_argument(
        "--connection",
        help="Auth connection (e.g. github, google, or an SSO domain)."
        " Skips the interactive prompt.",
        default=None,
    )
    login_parser.set_defaults(func=_login)

    logout_help = "Log out the currently logged-in user."
    logout_parser = subparsers.add_parser(
        "logout",
        description=logout_help,
        help=logout_help,
        parents=parents,
    )
    logout_parser.set_defaults(func=_logout)

    whoami_help = "Show the currently authenticated user."
    whoami_parser = subparsers.add_parser(
        "whoami",
        description=whoami_help,
        help=whoami_help,
        parents=parents,
    )
    whoami_parser.set_defaults(func=_whoami)
