import json
import re

import httpx
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
        return stream_run(args.model_id, params)
    else:
        return queue_run(args.model_id, params)


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


def _format_log(log: dict) -> str:
    """Render a log entry.

    An app that lets an exception escape an endpoint logs it as a one-line
    ``{"traceback": ...}`` envelope (see ``fal.api.api``); unwrap it so the
    traceback reads as a traceback instead of an escaped JSON blob.
    """
    message = log.get("message", str(log))

    try:
        payload = json.loads(message)
    except (TypeError, ValueError):
        return message

    if isinstance(payload, dict) and "traceback" in payload:
        return str(payload["traceback"]).rstrip()

    return message


def _response_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()

    if isinstance(body, dict) and "detail" in body:
        detail = body["detail"]
        return detail if isinstance(detail, str) else json.dumps(detail)

    return response.text.strip()


def queue_run(model_id: str, params: dict):
    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    import fal.apps
    from fal.console.icons import (  # noqa: PLC0415
        get_cross_icon,
        get_status_done_icon,
        get_status_progress_icon,
        get_status_queued_icon,
    )

    handle = fal.apps.submit(model_id, params)  # type: ignore
    logs = []  # type: ignore
    target_console = rich.get_console()
    status_queued_icon = get_status_queued_icon(target_console)
    status_progress_icon = get_status_progress_icon(target_console)
    status_done_icon = get_status_done_icon(target_console)

    # A status response returns the entries inside a moving time window, not a
    # growing prefix, so identity decides what is new -- a positional cursor
    # stops advancing once the window slides past what we already hold.
    seen = set()

    def consume_logs(entries) -> list:
        new_lines = []
        for entry in entries or []:
            key = (entry.get("timestamp"), entry.get("message"))
            if key in seen:
                continue
            seen.add(key)
            new_lines.append(_format_log(entry))

        logs.extend(new_lines)
        return new_lines

    try:
        with Live(auto_refresh=False, console=target_console) as live:
            for event in handle.iter_events(logs=True):
                if isinstance(event, fal.apps.Queued):
                    status = Text(
                        f"{status_queued_icon} Queued (position: {event.position})",
                        style="yellow",
                    )
                elif isinstance(event, fal.apps.InProgress):
                    status = Text(f"{status_progress_icon} In Progress", style="blue")
                    consume_logs(event.logs)
                else:
                    status = Text(f"{status_done_icon} Done", style="green")

                request_id = handle.request_id
                status_panel = Panel(
                    status,
                    title="Status",
                    subtitle=request_id,
                    subtitle_align="right",
                )
                # Text(), not a markup string: a log line holding "[/]" or
                # "[gw0]" would otherwise raise MarkupError or be swallowed.
                logs_panel = Panel(Text("\n".join(logs[-10:])), title="Logs")

                live.update(Group(status_panel, logs_panel))
                live.refresh()

            if flags.DEBUG:
                response = handle.fetch_raw_response()
                headers = "\n".join(
                    f"{header}: {value}"
                    for header, value in response.headers.multi_items()
                )
                headers_panel = Panel(Text(headers), title="Headers")

                body = rich.pretty.Pretty(response.json())
                live.update(Group(headers_panel, body))
                live.refresh()
            else:
                result = handle.fetch_result()
                live.update(rich.pretty.Pretty(result))
    except KeyboardInterrupt:
        rich.print("[yellow]Cancelling request...[/yellow]")
        handle.cancel()
        rich.print("[green]Request cancelled.[/green]")
    except httpx.HTTPStatusError as exc:
        # This arm also sees a status poll fail mid-flight. The result endpoint
        # answering at all means the request reached a terminal state; a failed
        # poll says nothing about the request, so fall back to re-reading the
        # status there. That re-read also carries the logs explaining an app
        # failure, which iter_events returns too early to have seen.
        from_poll = exc.request.url.path.rstrip("/").endswith("/status")

        final_status = None
        try:
            final_status = handle.status(logs=True)
        except httpx.HTTPError:
            pass

        completed = isinstance(final_status, fal.apps.Completed)
        if completed:
            new_lines = consume_logs(final_status.logs)
            if new_lines:
                target_console.print(
                    Panel(Text("\n".join(new_lines)), title="Logs", border_style="red")
                )

        detail = _response_detail(exc.response)
        summary = Text.from_markup(f"{get_cross_icon(target_console)} ")

        if completed or not from_poll:
            error = final_status.error or final_status.error_type if completed else None
            summary.append(
                f"Request {handle.request_id} failed "
                f"with HTTP {exc.response.status_code}: {error or detail}"
            )
        else:
            summary.append(
                f"Could not read request {handle.request_id} "
                f"(HTTP {exc.response.status_code}: {detail}). "
                "It may still be running -- check its status before resubmitting."
            )

        target_console.print(summary)
        return 1


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
