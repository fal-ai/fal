from fal.auth import current_user_info
from fal.cli import profile
from fal.sdk import get_default_credentials


def _login(args):
    from fal.auth import login
    from fal.console.icons import CHECK_ICON, CROSS_ICON
    from fal.exceptions import FalServerlessException

    try:
        login(args.console)
        args.console.print(f"{CHECK_ICON} Authenticated successfully, welcome!")
    except FalServerlessException as e:
        args.console.print(f"{CROSS_ICON} {e}")
        return

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
    from rich.style import Style
    from rich.table import Table

    from fal.auth import UserAccess
    from fal.config import Config

    user_access = UserAccess()
    config = Config()
    current_account_name = config.get_internal("team") or user_access.info["nickname"]

    table = Table(border_style=Style(frame=False), show_header=False)
    table.add_column("#")
    table.add_column("Nickname")
    table.add_column("Type")

    for idx, account in enumerate(user_access.accounts):
        selected = account["nickname"] == current_account_name
        color = "bold yellow" if selected else None

        table.add_row(
            f"* {idx + 1}" if selected else f"  {idx + 1}",
            account["nickname"],
            "Personal" if account["is_personal"] else "Team",
            style=color,
        )

    args.console.print(table)


def _unset_account(args):
    from fal.config import Config

    with Config().edit() as config:
        config.unset_internal("team")


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
        _list_accounts(args)
        indices = list(map(str, range(1, len(user_access.accounts) + 1)))
        team_names = [account["nickname"] for account in user_access.accounts]
        acc_choice = Prompt.ask(
            "Select an account by number",
            choices=indices + team_names,
            show_choices=False,
        )
        if acc_choice in indices:
            acc_index = int(acc_choice) - 1
            account = user_access.accounts[acc_index]
        else:
            account = user_access.get_account(acc_choice)

    if account["is_personal"]:
        args.console.print(
            f"Using personal account [cyan]{account['nickname']}[/]. "
            "You can change this later with [bold]fal team set[/]"
        )
    else:
        args.console.print(
            f"Using team account [cyan]{account['nickname']}[/]. "
            "You can change this later with [bold]fal team set[/]"
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
