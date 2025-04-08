from fal.auth import USER, login, logout


def _login(args):
    from fal.config import Config
    from fal.console.prompt import prompt

    login()

    teams = [team["nickname"].lower() for team in USER.teams]
    if not teams:
        return

    args.console.print("")
    args.console.print(
        f"You ({USER.info['name']}) are a member of the following teams:\n",
    )
    for idx, team in enumerate(USER.teams):
        args.console.print(f"  {idx + 1}. {team['nickname']}")
    args.console.print("")

    team_choice = prompt(
        args.console,
        "Pick a team account to use (leave blank for personal account)",
        choices=teams,
        show_choices=False,
        default=None,
    )
    args.console.print("")

    with Config().edit() as config:
        if team_choice:
            args.console.print(
                f"Setting team to [cyan]{team}[/]. "
                "You can change this later with [bold]fal team set[/]."
            )
            config.set("team", team)
        else:
            args.console.print(
                "Using your personal account. "
                "You can change this later with [bold]fal team set[/]."
            )
            config.unset("team")


def _logout(args):
    from fal.config import Config

    logout()
    with Config().edit() as config:
        config.unset("team")


def _whoami(args):
    user_name = USER.info["name"]
    sub = USER.info["sub"]
    args.console.print(f"Hello, {user_name} - '{sub}'")


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
