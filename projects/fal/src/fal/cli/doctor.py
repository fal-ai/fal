import os
import platform


def _doctor(args):
    import isolate
    from rich.table import Table

    import fal

    table = Table(show_header=False, show_lines=False, box=None)
    table.add_column("name", no_wrap=True, style="bold")
    table.add_column("value", no_wrap=True)

    table.add_row("fal", fal.__version__)
    table.add_row("isolate", isolate.__version__)

    table.add_row("", "")
    table.add_row("python", platform.python_version())
    table.add_row("platform", platform.platform())

    table.add_row("", "")
    table.add_row("FAL_HOST", fal.flags.GRPC_HOST)
    table.add_row("FAL_KEY", os.getenv("FAL_KEY", "").split(":")[0])

    args.console.print(table)


def add_parser(main_subparsers, parents):
    doctor_help = "fal version and misc environment information."
    parser = main_subparsers.add_parser(
        "doctor",
        description=doctor_help,
        help=doctor_help,
        parents=parents,
    )
    parser.set_defaults(func=_doctor)
