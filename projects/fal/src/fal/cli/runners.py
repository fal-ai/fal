from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import struct
import sys
import termios
import tty
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from queue import Empty, Queue
from threading import Thread
from typing import Iterator, List

import grpc
import httpx
from httpx_sse import connect_sse
from rich.console import Console
from structlog.typing import EventDict

from fal.api.client import SyncServerlessClient
from fal.rest_client import REST_CLIENT
from fal.sdk import RunnerInfo, RunnerState

from .parser import FalClientParser, SinceAction, get_output_parser


def runners_table(runners: List[RunnerInfo]):
    from rich.table import Table

    table = Table()
    table.add_column("Alias")
    table.add_column("Runner ID")
    table.add_column("In Flight\nRequests")
    table.add_column("Expires In")
    table.add_column("Uptime")
    table.add_column("Revision")
    table.add_column("State")

    for runner in runners:
        external_metadata = runner.external_metadata
        present = external_metadata.get("present_in_group", True)

        num_leases_with_request = len(
            [
                lease
                for lease in external_metadata.get("leases", [])
                if lease.get("request_id") is not None
            ]
        )

        in_flight = str(runner.in_flight_requests)
        missing_leases = runner.in_flight_requests - num_leases_with_request
        if missing_leases > 0:
            # Show a small indicator of in flight requests that are not visible in the
            # leases lists
            # This can be due to race conditions, so only important to report if it's
            # consistent
            in_flight = f"{in_flight} [dim]({missing_leases})[/]"

        uptime = timedelta(
            seconds=int(runner.uptime.total_seconds()),
        )
        table.add_row(
            runner.alias,
            # Mark lost runners in red
            runner.runner_id if present else f"[red]{runner.runner_id}[/]",
            in_flight,
            (
                "N/A"
                if runner.expiration_countdown is None
                else f"{runner.expiration_countdown}s"
            ),
            f"{uptime} ({uptime.total_seconds():.0f}s)",
            runner.revision,
            runner.state.value,
        )

    return table


def runners_requests_table(runners: list[RunnerInfo]):
    from rich.table import Table

    table = Table()
    table.add_column("Runner ID")
    table.add_column("Request ID")
    table.add_column("Caller ID")

    for runner in runners:
        for lease in runner.external_metadata.get("leases", []):
            if not (req_id := lease.get("request_id")):
                continue

            table.add_row(
                runner.runner_id,
                req_id,
                lease.get("caller_user_id") or "",
            )

    return table


def _get_tty_size():
    """Get current terminal dimensions."""
    try:
        h, w = struct.unpack("HH", fcntl.ioctl(0, termios.TIOCGWINSZ, b"\0" * 4))[:2]
        return h, w
    except (OSError, ValueError):
        return 24, 80  # Fallback to standard size


def _shell(args):
    """Execute interactive shell in runner."""
    import isolate_proto

    client = SyncServerlessClient(host=args.host, team=args.team)
    stub = client._create_host()._connection.stub
    runner_id = args.id

    # Setup terminal for raw mode
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(fd)

    # Message queue for stdin data and resize events
    messages = Queue()  # type: ignore
    stop_flag = False

    def handle_resize(*_):
        messages.put(("resize", None))

    signal.signal(signal.SIGWINCH, handle_resize)

    def read_stdin():
        """Read stdin in a background thread."""
        nonlocal stop_flag
        while not stop_flag:
            try:
                data = os.read(fd, 4096)
                if not data:
                    break
                messages.put(("data", data))
            except OSError:
                break

    reader = Thread(target=read_stdin, daemon=True)
    reader.start()

    def stream_inputs():
        """Generate input stream for gRPC."""
        # Send initial message with runner_id
        yield isolate_proto.ShellRunnerInput(runner_id=runner_id)

        # Send terminal size
        msg = isolate_proto.ShellRunnerInput()
        h, w = _get_tty_size()
        msg.tty_size.height = h
        msg.tty_size.width = w
        yield msg

        # Stream stdin data and resize events
        while True:
            try:
                msg_type, data = messages.get(timeout=0.1)
            except Empty:
                continue

            if msg_type == "data":
                yield isolate_proto.ShellRunnerInput(data=data)
            elif msg_type == "resize":
                msg = isolate_proto.ShellRunnerInput()
                h, w = _get_tty_size()
                msg.tty_size.height = h
                msg.tty_size.width = w
                yield msg

    exit_code = 1
    try:
        for output in stub.ShellRunner(stream_inputs()):
            if output.HasField("exit_code"):
                exit_code = output.exit_code
                break
            if output.data:
                sys.stdout.buffer.write(output.data)
                sys.stdout.buffer.flush()
            if output.close:
                break
        exit_code = exit_code or 0
    except grpc.RpcError as exc:
        args.console.print(f"\n[red]Connection error:[/] {exc.details()}")
    except Exception as exc:
        args.console.print(f"\n[red]Error:[/] {exc}")
    finally:
        stop_flag = True
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return exit_code


def _stop(args):
    client = SyncServerlessClient(host=args.host, team=args.team)
    client.runners.stop(args.id)


def _kill(args):
    client = SyncServerlessClient(host=args.host, team=args.team)
    client.runners.kill(args.id)


def _list_json(args, runners: list[RunnerInfo]):
    json_runners = [
        {
            "alias": r.alias,
            "runner_id": r.runner_id,
            "in_flight_requests": r.in_flight_requests,
            "expiration_countdown": r.expiration_countdown,
            "uptime_seconds": int(r.uptime.total_seconds()),
            "revision": r.revision,
            "state": r.state.value,
        }
        for r in runners
    ]

    res = {
        "runners": json_runners,
    }
    args.console.print(json.dumps(res))


def _list(args):
    client = SyncServerlessClient(host=args.host, team=args.team)
    start_time = args.since
    runners = client.runners.list(since=start_time)

    if args.state:
        states = set(args.state)
        if "all" not in states:
            runners = [
                r
                for r in runners
                if r.state.value.lower() in states
                or (
                    "terminated" in states and r.state.value.lower() == "dead"
                )  # TODO for backwards compatibility. remove later
            ]

    pending_runners = [
        runner for runner in runners if runner.state == RunnerState.PENDING
    ]
    setup_runners = [runner for runner in runners if runner.state == RunnerState.SETUP]
    terminated_runners = [
        runner
        for runner in runners
        if runner.state == RunnerState.DEAD or runner.state == RunnerState.TERMINATED
    ]
    if args.output == "pretty":
        args.console.print(
            "Runners: "
            + str(
                len(runners)
                - len(pending_runners)
                - len(setup_runners)
                - len(terminated_runners)
            )
        )
        args.console.print(f"Runners Pending: {len(pending_runners)}")
        args.console.print(f"Runners Setting Up: {len(setup_runners)}")
        args.console.print(runners_table(runners))

        requests_table = runners_requests_table(runners)
        args.console.print(f"Requests: {len(requests_table.rows)}")
        args.console.print(requests_table)
    elif args.output == "json":
        _list_json(args, runners)
    else:
        raise AssertionError(f"Invalid output format: {args.output}")


def _add_stop_parser(subparsers, parents):
    stop_help = "Stop a runner gracefully."
    parser = subparsers.add_parser(
        "stop",
        description=stop_help,
        help=stop_help,
        parents=parents,
    )
    parser.add_argument(
        "id",
        help="Runner ID.",
    )
    parser.set_defaults(func=_stop)


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


def _add_list_parser(subparsers, parents):
    list_help = "List runners."
    parser = subparsers.add_parser(
        "list",
        description=list_help,
        help=list_help,
        parents=[*parents, get_output_parser()],
    )
    parser.add_argument(
        "--since",
        default=None,
        action=SinceAction,
        limit="1 day",
        help=(
            "Show terminated runners since the given time. "
            "Accepts 'now', relative like '30m', '1h', '1d', "
            "or an ISO timestamp. Max 24 hours."
        ),
    )
    parser.add_argument(
        "--state",
        choices=["all", "running", "pending", "setup", "terminated"],
        nargs="+",
        default=None,
        help=("Filter by runner state(s). Choose one or more, or 'all'(default)."),
    )
    parser.set_defaults(func=_list)


def _to_iso_naive(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_ts(ts: str) -> datetime:
    # Support both 'Z' and offset formats
    ts_norm = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts_norm)


def _to_aware_utc(dt: datetime) -> datetime:
    # Treat naive datetimes as UTC
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _post_history(
    client: httpx.Client,
    base_params: dict[str, str],
    since: datetime | None,
    until: datetime | None,
    page_size: int,
) -> tuple[list, str | None]:
    params: dict[str, str] = dict(base_params)
    if since is not None:
        params["since"] = _to_iso_naive(since)
    if until is not None:
        params["until"] = _to_iso_naive(until)
    params["page_size"] = str(page_size)
    resp = client.post("/logs/history", params=params)
    if resp.status_code != HTTPStatus.OK:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"Failed to fetch logs history: {detail}")
    data = resp.json()
    items = data.get("items", []) if isinstance(data, dict) else []
    next_until = data.get("next_until") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise RuntimeError("Unexpected logs history response format")
    return items, next_until


@dataclass
class RestRunnerInfo:
    started_at: datetime | None
    ended_at: datetime | None


def _get_runner_info(runner_id: str) -> RestRunnerInfo:
    headers = REST_CLIENT.get_headers()
    with httpx.Client(
        base_url=REST_CLIENT.base_url, headers=headers, timeout=30
    ) as client:
        resp = client.get(f"/runners/{runner_id}")
        if resp.status_code == HTTPStatus.NOT_FOUND:
            raise RuntimeError(f"Runner {runner_id} not found")
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(
                f"Failed to fetch runner info: {resp.status_code} {resp.text}"
            )
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected runner info response format: {resp.text}")

        start: datetime | None = None
        end: datetime | None = None

        started_at = data.get("started_at")
        if started_at is not None:
            try:
                start = _to_aware_utc(_parse_ts(started_at))
            except Exception:
                start = None

        ended_at = data.get("ended_at")
        if ended_at is not None:
            try:
                end = _to_aware_utc(_parse_ts(ended_at))
            except Exception:
                end = None

        return RestRunnerInfo(started_at=start, ended_at=end)


def _stream_logs(
    base_params: dict[str, str], since: datetime | None, until: datetime | None
) -> Iterator[dict]:
    headers = REST_CLIENT.get_headers()
    params: dict[str, str] = base_params.copy()
    if since is not None:
        params["since"] = _to_iso_naive(since)
    if until is not None:
        params["until"] = _to_iso_naive(until)
    with httpx.Client(
        base_url=REST_CLIENT.base_url,
        headers=headers,
        timeout=None,
        follow_redirects=True,
    ) as client:
        with connect_sse(
            client,
            method="POST",
            url="/logs/stream",
            params=params,
            headers={"Accept": "text/event-stream"},
        ) as event_source:
            for sse in event_source.iter_sse():
                if not sse.data:
                    continue
                if sse.event == "error":
                    raise RuntimeError(f"Error streaming logs: {sse.data}")
                try:
                    yield json.loads(sse.data)
                except Exception:
                    continue


DEFAULT_PAGE_SIZE = 1000


def _iter_logs(
    base_params: dict[str, str], start: datetime | None, end: datetime | None
) -> Iterator[dict]:
    headers = REST_CLIENT.get_headers()
    with httpx.Client(
        base_url=REST_CLIENT.base_url,
        headers=headers,
        timeout=300,
        follow_redirects=True,
    ) as client:
        cursor_until = end
        while True:
            items, next_until = _post_history(
                client, base_params, start, cursor_until, DEFAULT_PAGE_SIZE
            )

            yield from items

            if not next_until:
                break

            new_until_dt = _to_aware_utc(_parse_ts(next_until))
            if start is not None and new_until_dt <= start:
                break
            cursor_until = new_until_dt


def _get_logs(
    params: dict[str, str],
    since: datetime | None,
    until: datetime | None,
    lines_count: int | None,
    *,
    oldest: bool = False,
) -> Iterator[dict]:
    if lines_count is None:
        yield from _iter_logs(params, since, until)
        return

    if oldest:
        produced = 0
        for log in _iter_logs(params, since, until):
            if produced >= lines_count:
                break
            produced += 1
            yield log
        return

    # newest tail: collect into a fixed-size deque, then yield
    tail: deque[dict] = deque(maxlen=lines_count)
    for log in _iter_logs(params, since, until):
        tail.append(log)
    for log in tail:
        yield log


class LogPrinter:
    def __init__(self, console: Console) -> None:
        from structlog.dev import ConsoleRenderer

        from fal.logging.style import LEVEL_STYLES

        self._console = console
        self._renderer = ConsoleRenderer(level_styles=LEVEL_STYLES)

    def _render_log(self, log: dict) -> str:
        ts_str: str = log["timestamp"]
        timestamp = _to_aware_utc(_parse_ts(ts_str))
        local_ts = timestamp.astimezone()
        tz_offset = local_ts.strftime("%z")
        # Insert ':' into offset for readability, e.g. +0300 -> +03:00
        if tz_offset and len(tz_offset) == 5:
            tz_offset = tz_offset[:3] + ":" + tz_offset[3:]

        event: EventDict = {
            "event": log.get("message", ""),
            "level": str(log.get("level", "")).upper(),
            "timestamp": f"{local_ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}{tz_offset}",
        }
        return self._renderer(logger={}, name=event["level"], event_dict=event)

    def print(self, log: dict) -> None:
        self._console.print(self._render_log(log), highlight=False)


DEFAULT_STREAM_SINCE = timedelta(minutes=1)


def _logs(args):
    params: dict[str, str] = {"job_id": args.id}
    if args.search is not None:
        params["search"] = args.search

    runner_info = _get_runner_info(args.id)
    follow: bool = args.follow
    since = args.since
    if follow:
        since = since or (datetime.now(timezone.utc) - DEFAULT_STREAM_SINCE)
    else:
        since = since or runner_info.started_at
    until = args.until or runner_info.ended_at

    # Normalize to aware UTC for comparisons
    if since is not None:
        since = _to_aware_utc(since)
    if until is not None:
        until = _to_aware_utc(until)

    # Sanity limiters: clamp within runner lifetime when known
    if runner_info.started_at is not None:
        if since is not None and since < runner_info.started_at:
            since = runner_info.started_at
        if until is not None and until < runner_info.started_at:
            until = runner_info.started_at
    if runner_info.ended_at is not None:
        if since is not None and since > runner_info.ended_at:
            since = runner_info.ended_at
        if until is not None and until > runner_info.ended_at:
            until = runner_info.ended_at

    # Ensure ordering if both are present
    if since is not None and until is not None and until < since:
        since, until = until, since

    lines_arg = args.lines
    lines_count: int | None = None
    lines_oldest = False
    if lines_arg is not None:
        if lines_arg.startswith("+"):
            lines_str = lines_arg[1:]
            lines_oldest = True
        else:
            lines_str = lines_arg
        try:
            lines_count = int(lines_str)
        except ValueError:
            args.parser.error("Invalid -n|--lines value. Use an integer or +integer.")

    if follow:
        logs_gen = _stream_logs(params, since, until)
    else:
        logs_gen = _get_logs(params, since, until, lines_count, oldest=lines_oldest)

    printer = LogPrinter(args.console)

    if follow:
        for log in logs_gen:
            if args.output == "json":
                args.console.print(json.dumps(log))
            else:
                printer.print(log)
        return

    if args.output == "json":
        args.console.print(json.dumps({"logs": list(logs_gen)}))
    else:
        for log in reversed(list(logs_gen)):
            printer.print(log)


def _add_logs_parser(subparsers, parents):
    logs_help = "Show logs for a runner."
    parser = subparsers.add_parser(
        "logs",
        aliases=["log"],
        description=logs_help,
        help=logs_help,
        parents=[*parents, get_output_parser()],
    )
    parser.add_argument(
        "id",
        help="Runner ID.",
    )
    parser.add_argument(
        "--search",
        default=None,
        help="Search for string in logs.",
    )
    parser.add_argument(
        "--since",
        default=None,
        action=SinceAction,
        help=(
            "Show logs since the given time. "
            "Accepts 'now', relative like '30m', '1h', or an ISO timestamp. "
            "Defaults to runner start time or to '1m ago' in --follow mode."
        ),
    )
    parser.add_argument(
        "--until",
        default=None,
        action=SinceAction,
        help=(
            "Show logs until the given time. "
            "Accepts 'now', relative like '30m', '1h', or an ISO timestamp. "
            "Defaults to runner finish time or 'now' if it is still running."
        ),
    )
    parser.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Follow logs live. If --since is not specified, implies '--since 1m ago'.",
    )
    parser.add_argument(
        "--lines",
        "-n",
        default=None,
        type=str,
        help=(
            "Only show latest N log lines. "
            "If '+' prefix is used, show oldest N log lines. "
            "Ignored if --follow is used."
        ),
    )
    parser.set_defaults(func=_logs)


def _add_shell_parser(subparsers, parents):
    """Add hidden shell command parser."""
    parser = subparsers.add_parser(
        "shell",
        help=argparse.SUPPRESS,
        parents=parents,
    )
    parser.add_argument("id", help="Runner ID.")
    parser.set_defaults(func=_shell)


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

    _add_stop_parser(subparsers, parents)
    _add_kill_parser(subparsers, parents)
    _add_list_parser(subparsers, parents)
    _add_logs_parser(subparsers, parents)
    _add_shell_parser(subparsers, parents)
