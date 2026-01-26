import re

import rich

from fal import flags

# = or := only
KV_SPLIT_RE = re.compile(r"(=|:=)")


def _api(args):
    """Handle the api command execution."""
    from . import cli_nested_json

    params_split = [KV_SPLIT_RE.split(param) for param in args.params]
    params = cli_nested_json.interpret_nested_json(  # type: ignore
        [(key, value) for key, _, value in params_split]
    )

    if args.model_id.endswith("/stream"):
        stream_run(args.model_id, params)
    else:
        queue_run(args.model_id, params)


def stream_run(model_id: str, params: dict):
    import fal.apps

    res = fal.apps.stream(model_id, params)  # type: ignore
    for line in res:
        if isinstance(line, str):
            rich.print(line)
        else:
            if isinstance(line, memoryview):
                rich.print(line.tobytes().decode())
            else:
                rich.print(line.decode())


def queue_run(model_id: str, params: dict):
    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    import fal.apps

    handle = fal.apps.submit(model_id, params)  # type: ignore
    logs = []  # type: ignore

    with Live(auto_refresh=False) as live:
        for event in handle.iter_events(logs=True):
            if isinstance(event, fal.apps.Queued):
                status = Text(f"⏳ Queued (position: {event.position})", style="yellow")
            elif isinstance(event, fal.apps.InProgress):
                status = Text("🔄 In Progress", style="blue")
                if event.logs:
                    logs.extend(log.get("message", str(log)) for log in event.logs)
                    logs = logs[-10:]  # Keep only last 10 logs
            else:
                status = Text("✅ Done", style="green")

            status_panel = Panel(status, title="Status")
            logs_panel = Panel("\n".join(logs), title="Logs")

            live.update(Group(status_panel, logs_panel))
            live.refresh()

        if flags.DEBUG:
            response = handle.fetch_raw_response()
            # Print headers and body
            headers = "\n".join(
                f"{header}: {value}" for header, value in response.headers.multi_items()
            )
            headers_panel = Panel(headers, title="Headers")

            body = rich.pretty.Pretty(response.json())
            live.update(Group(headers_panel, body))
            live.refresh()
        else:
            result = handle.fetch_result()
            live.update(rich.pretty.Pretty(result))


def add_parser(main_subparsers, parents):
    """Add the api command to the main parser."""
    api_help = "Call a fal API endpoint directly"
    parser = main_subparsers.add_parser(
        "api",
        description=api_help,
        help=api_help,
        parents=parents,
    )

    parser.add_argument(
        "model_id",
        help="Name of the Model ID to call",
    )

    parser.add_argument(
        "params",
        nargs="*",
        help="Key-value pairs (e.g. key=value or nested[a][b]=value)",
    )

    parser.set_defaults(func=_api)
