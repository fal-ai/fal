from .parser import FalClientParser


def _kill(args):
    from fal.sdk import FalServerlessClient

    client = FalServerlessClient(args.host)
    with client.connect() as connection:
        connection.kill_runner(args.id)


def _add_kill_parser(subparsers, parents):
    kill_help = "Kill a runner."
    parser = subparsers.add_parser(
        "kill",
        description=kill_help,
        help=kill_help,
        parents=parents,
    )
    parser.add_argument(
        "id",
        help="Runner ID.",
    )
    parser.set_defaults(func=_kill)


def add_parser(main_subparsers, parents):
    runners_help = "Manage fal runners."
    parser = main_subparsers.add_parser(
        "runners",
        description=runners_help,
        help=runners_help,
        parents=parents,
        aliases=["machine"],  # backwards compatibility
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        required=True,
        parser_class=FalClientParser,
    )

    _add_kill_parser(subparsers, parents)
