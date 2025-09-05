import json
import os
import platform


def _doctor(args):
    import isolate

    import fal

    if args.output == "json":
        data = {
            "fal": fal.__version__,
            "isolate": isolate.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "FAL_HOST": fal.flags.GRPC_HOST,
            "FAL_KEY": os.getenv("FAL_KEY", "").split(":")[0],
        }
        args.console.print(json.dumps(data))
    elif args.output == "pretty":
        from rich.table import Table

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
    else:
        raise AssertionError(f"Invalid output format: {args.output}")


def add_parser(main_subparsers, parents):
    from .parser import get_output_parser

    doctor_help = "fal version and misc environment information."
    parser = main_subparsers.add_parser(
        "doctor",
        description=doctor_help,
        help=doctor_help,
        parents=[*parents, get_output_parser()],
    )
    parser.set_defaults(func=_doctor)
