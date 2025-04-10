def _login(args):
    from fal.auth import login
    from fal.config import Config

    login()

    with Config().edit() as config:
        config.unset("team")

    _set_account(args)


def _logout(args):
    from fal.auth import logout
    from fal.config import Config

    logout()

    with Config().edit() as config:
        config.unset("team")


def _list_accounts(args):
    from rich.style import Style
    from rich.table import Table

    from fal.auth import UserAccess
    from fal.config import Config

    user_access = UserAccess()
    config = Config()
    current_account = config.get("team") or user_access.info["nickname"]

    table = Table(border_style=Style(frame=False), show_header=False)
    table.add_column("#")
    table.add_column("Nickname")
    table.add_column("Type")

    for idx, account in enumerate(user_access.accounts):
        selected = account["nickname"] == current_account
        color = "bold yellow" if selected else None

        table.add_row(
            f"* {idx + 1}" if selected else f"  {idx + 1}",
            account["nickname"],
            "Personal" if account["is_personal"] else "Team",
            style=color,
        )

    args.console.print(table)


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
        config.set("team", account["nickname"])


def _whoami(args):
    from fal.auth import UserAccess
    from fal.config import Config

    user_access = UserAccess()
    config = Config()

    team = config.get("team")
    if team:
        account = user_access.get_account(team)
    else:
        account = user_access.get_account(user_access.info["nickname"])

    nickname = account["nickname"]
    full_name = account["full_name"]
    user_id = account["user_id"]

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
