from .parser import FalClientParser


def _kill(args):
    from fal.sdk import FalServerlessClient

    client = FalServerlessClient(args.host)
    with client.connect() as connection:
        connection.kill_runner(args.id)


def _add_kill_parser(subparsers, parents):
    kill_help = "Kill a machine."
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
    machine_help = "Manage fal machines."
    parser = main_subparsers.add_parser(
        "machine",
        description=machine_help,
        help=machine_help,
        parents=parents,
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        required=True,
        parser_class=FalClientParser,
    )

    _add_kill_parser(subparsers, parents)
