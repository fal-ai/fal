from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from queue import Empty, Queue
from threading import Event, Thread, Timer
from typing import TYPE_CHECKING, Iterator, List

if TYPE_CHECKING:
    from openapi_fal_rest.client import Client

import grpc
import httpx
import isolate_proto
from httpx_sse import connect_sse
from rich.console import Console
from structlog.typing import EventDict

from fal.api.client import SyncServerlessClient
from fal.console.encoding import make_terminal_safe
from fal.sdk import ReplaceState, RunnerInfo, RunnerState

from .parser import FalClientParser, SinceAction, get_output_parser


def runners_table(runners: List[RunnerInfo]):
    from rich.table import Table

    table = Table()
    table.add_column("Alias")
    table.add_column("Machine Type")
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
        state = runner.state.value
        if runner.replacement == ReplaceState.WILL_REPLACE:
            state = f"{state} (*)"
        table.add_row(
            runner.alias,
            runner.machine_type,
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
            state,
        )

    return table


def runners_requests_table(runners: list[RunnerInfo]):
    from rich.table import Table

    table = Table()
    table.add_column("Runner ID")
    table.add_column("Request ID")
    table.add_column("Caller ID")
    table.add_column("Time Running")

    for runner in runners:
        for lease in runner.external_metadata.get("leases", []):
            if not (req_id := lease.get("request_id")):
                continue

            running_for = None
            try:
                if leased_at_epoch := lease.get("leased_at_epoch"):
                    started_at = datetime.fromtimestamp(
                        float(leased_at_epoch), timezone.utc
                    )
                    running_for = datetime.now(timezone.utc) - started_at
            except Exception:
                pass

            table.add_row(
                runner.runner_id,
                req_id,
                lease.get("caller_user_id") or "",
                str(running_for or ""),
            )

    return table


def _get_tty_size(fd: int):
    """Get current terminal dimensions."""
    try:
        size = os.get_terminal_size(fd)
        return size.lines, size.columns
    except OSError:
        return 24, 80  # Fallback to standard size


def _shell(args):
    """Open an interactive shell on a runner."""
    # Always a PTY: the login shell expects one even when local stdin is piped.
    client = SyncServerlessClient(host=args.host, team=args.team)
    return shell_session(
        client, args.id, None, interactive=True, remote_tty=True, console=args.console
    )


def _exec(args):
    """Execute a command on a runner."""
    command = args.command
    # argparse may leave the -- separator as the first token (Python < 3.12).
    if command[0] == "--":
        command = command[1:]

    if not command:
        args.console.print("[red]Error:[/] No command specified.")
        return 1

    # A PTY mangles bytes (echo, CR/NL translation, signal characters), so only
    # ask for one when a real terminal is attached.
    remote_tty = args.interactive and sys.stdin.isatty()
    client = SyncServerlessClient(host=args.host, team=args.team)
    return shell_session(
        client,
        args.id,
        command,
        interactive=args.interactive,
        remote_tty=remote_tty,
        console=args.console,
    )


# How long output may keep arriving after the exit code, when reading up to a marker.
UNTIL_GRACE_SECONDS = 2.0


def _give_up(call, gave_up: Event) -> None:
    gave_up.set()
    call.cancel()


def shell_session(
    client: SyncServerlessClient,
    runner_id: str,
    command: list[str] | None,
    *,
    interactive: bool,
    remote_tty: bool,
    console: Console | None,
    stdout=None,
    stderr=None,
    timeout: float | None = None,
    until: bytes | None = None,
    done=None,
    stub=None,
):
    """Stream a shell session on a runner; command=None opens a login shell.

    `remote_tty` controls whether the remote command runs under a pseudo-terminal.
    Without one, remote stdin is closed once local input ends so pipe-reading
    commands see EOF. Remote output goes to `stdout` and `stderr` (default: this
    process's corresponding streams). Servers predating stream identification
    produce output with no stream set, which is treated as stdout.

    Connection errors are printed to `console`, or raised if there is none.
    `timeout` limits the whole call in seconds. The exit code can arrive before
    the last of the output: with `until`, stdout is read up to that marker.
    With `done`, a callable checked after each output frame, the stream is read
    until it returns True or ends, ignoring the exit code and close frames; the
    exit code is then not meaningful (the caller carries its own).
    A `stub` from an earlier call can be passed to reuse its connection.
    """
    if stdout is None:
        stdout = sys.stdout.buffer
    if stderr is None:
        stderr = sys.stderr.buffer

    stub = stub or client._create_host()._connection.stub

    if interactive and os.name == "nt":
        message = "Interactive runner shell is not supported on Windows."
        if console is None:
            raise NotImplementedError(message)
        console.print(f"[red]Error:[/] {message}")
        return 1

    is_tty = interactive and sys.stdin.isatty()
    fd = sys.stdin.fileno() if interactive else -1

    # Message queue for stdin data and resize events
    messages = Queue()  # type: ignore
    stop_flag = False

    if is_tty:
        import termios
        import tty

        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)

        def handle_resize(*_):
            messages.put(("resize", None))

        signal.signal(signal.SIGWINCH, handle_resize)

    if interactive:

        def read_stdin():
            """Read stdin in a background thread."""
            nonlocal stop_flag
            while not stop_flag:
                try:
                    data = os.read(fd, 4096)
                    if not data:
                        messages.put(("eof", None))
                        break
                    messages.put(("data", data))
                except OSError:
                    break

        reader = Thread(target=read_stdin, daemon=True)
        reader.start()

    def stream_inputs():
        """Generate input stream for gRPC."""
        # Send initial message with runner_id
        yield isolate_proto.ShellRunnerInput(
            runner_id=runner_id, command=command, tty=remote_tty
        )

        if not interactive:
            if not remote_tty:
                yield isolate_proto.ShellRunnerInput(close=True)
            return

        # Send terminal size
        if is_tty:
            msg = isolate_proto.ShellRunnerInput()
            h, w = _get_tty_size(fd)
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
                h, w = _get_tty_size(fd)
                msg.tty_size.height = h
                msg.tty_size.width = w
                yield msg
            elif msg_type == "eof":
                if not remote_tty:
                    yield isolate_proto.ShellRunnerInput(close=True)
                return

    exit_code = 1

    def restore_tty() -> None:
        if is_tty:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    marker_seen = False
    tail = b""
    call = None
    grace: Timer | None = None
    gave_up = Event()
    try:
        call = stub.ShellRunner(stream_inputs(), timeout=timeout)
        for output in call:
            if output.data:
                to_stderr = output.HasField("stream") and output.stream == 2
                stream = stderr if to_stderr else stdout
                stream.write(output.data)
                stream.flush()
                if until and not to_stderr:
                    window = tail + output.data
                    marker_seen = marker_seen or until in window
                    tail = window[-len(until) :]
            if done is not None:
                if done():
                    break
                if (output.HasField("exit_code") or output.close) and grace is None:
                    # The stream should end now; if it does not, stop waiting.
                    grace = Timer(UNTIL_GRACE_SECONDS, _give_up, (call, gave_up))
                    grace.start()
                continue
            if output.HasField("exit_code"):
                exit_code = output.exit_code
                if until and not marker_seen:
                    # Late output follows the exit code at once or not at all.
                    grace = Timer(UNTIL_GRACE_SECONDS, _give_up, (call, gave_up))
                    grace.start()
            if output.close or (marker_seen if until else output.HasField("exit_code")):
                break
        exit_code = exit_code or 0
    except grpc.RpcError as exc:
        restore_tty()
        if gave_up.is_set():
            pass  # cancelled once the grace period ran out; the exit code stands
        elif console is None:
            raise
        elif exc.code() == grpc.StatusCode.UNAVAILABLE:
            from fal.api.api import _format_unavailable_error

            console.print(
                f"\n[red]Connection error:[/] {_format_unavailable_error(exc)}"
            )
        else:
            console.print(f"\n[red]Connection error:[/] {exc.details()}")
    except Exception as exc:
        restore_tty()
        if console is None:
            raise
        console.print(f"\n[red]Error:[/] {exc}")
    finally:
        if grace is not None:
            grace.cancel()
        stop_flag = True
        restore_tty()
        if call is not None and hasattr(call, "cancel"):
            call.cancel()  # ends the stream when the loop stopped early

    return exit_code


def _stop(args):
    client = SyncServerlessClient(host=args.host, team=args.team)
    client.runners.stop(args.id, replace_first=args.replace)


def _kill(args):
    client = SyncServerlessClient(host=args.host, team=args.team)
    client.runners.kill(args.id)


def render_gpus(args, gpu_by_type: dict[str, int], total: int):
    sorted_types = sorted(gpu_by_type.items(), key=lambda kv: -kv[1])
    if args.output == "pretty":
        from rich.table import Table

        table = Table()
        table.add_column("Type")
        table.add_column("Count", justify="right")
        for gpu_type, gpus in sorted_types:
            table.add_row(gpu_type, str(gpus))
        args.console.print(table)
        args.console.print(f"Total: {total}")
    elif args.output == "json":
        result = {"gpus": dict(sorted_types), "total": total}
        args.console.print(json.dumps(result))
    else:
        raise AssertionError(f"Invalid output format: {args.output}")


def _gpus(args):
    client = SyncServerlessClient(host=args.host, team=args.team)
    usage = client.runners.gpus()
    render_gpus(args, usage["gpus"], usage["total"])


def _list_json(args, runners: list[RunnerInfo]):
    json_runners = [
        {
            "alias": r.alias,
            "machine_type": r.machine_type,
            "runner_id": r.runner_id,
            "in_flight_requests": r.in_flight_requests,
            "expiration_countdown": r.expiration_countdown,
            "uptime_seconds": int(r.uptime.total_seconds()),
            "revision": r.revision,
            "state": r.state.value,
            "replacement": r.replacement.value,
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
    failing_runners = [
        runner for runner in runners if runner.state == RunnerState.CRASH_BACKOFF
    ]
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
        if len(failing_runners) > 0:
            args.console.print(
                f"[red]Runners being delayed after startup failure:[/] {len(failing_runners)}"  # noqa: E501
            )
            args.console.print(
                "[dim]  Check setup() and logs for errors during startup.[/]"
            )
        args.console.print(runners_table(runners))

        if any(runner.replacement == ReplaceState.WILL_REPLACE for runner in runners):
            args.console.print("[dim](*) Runner has been scheduled for replacement[/]")

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
    parser.add_argument(
        "-r",
        "--replace",
        action="store_true",
        default=False,
        help="Stop the runner after first spawning a replacement.",
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
        choices=[
            "all",
            "idle",
            "running",
            "pending",
            "setup",
            "crash_backoff",
            "terminated",
        ],
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


def _get_runner_info(rest_client: Client, runner_id: str) -> RestRunnerInfo:
    headers = rest_client.get_headers()
    with httpx.Client(
        base_url=rest_client.base_url, headers=headers, timeout=30
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
    rest_client: Client,
    base_params: dict[str, str],
    since: datetime | None,
    until: datetime | None,
) -> Iterator[dict]:
    headers = rest_client.get_headers()
    params: dict[str, str] = base_params.copy()
    if since is not None:
        params["since"] = _to_iso_naive(since)
    if until is not None:
        params["until"] = _to_iso_naive(until)
    with httpx.Client(
        base_url=rest_client.base_url,
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
    rest_client: Client,
    base_params: dict[str, str],
    start: datetime | None,
    end: datetime | None,
) -> Iterator[dict]:
    headers = rest_client.get_headers()
    with httpx.Client(
        base_url=rest_client.base_url,
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
    rest_client: Client,
    params: dict[str, str],
    since: datetime | None,
    until: datetime | None,
    lines_count: int | None,
    *,
    oldest: bool = False,
) -> Iterator[dict]:
    if lines_count is None:
        yield from _iter_logs(rest_client, params, since, until)
        return

    if oldest:
        produced = 0
        for log in _iter_logs(rest_client, params, since, until):
            if produced >= lines_count:
                break
            produced += 1
            yield log
        return

    # newest tail: collect into a fixed-size deque, then yield
    tail: deque[dict] = deque(maxlen=lines_count)
    for log in _iter_logs(rest_client, params, since, until):
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
        self._console.print(
            make_terminal_safe(self._render_log(log), self._console.file),
            highlight=False,
        )


DEFAULT_STREAM_SINCE = timedelta(minutes=1)


def _logs(args):
    lines_count: int | None = None
    lines_oldest = False
    if args.lines is not None:
        lines_str = args.lines
        if lines_str.startswith("+"):
            lines_str = lines_str[1:]
            lines_oldest = True
        try:
            lines_count = int(lines_str)
        except ValueError:
            args.parser.error("Invalid -n|--lines value. Use an integer or +integer.")

    client = SyncServerlessClient(host=args.host, team=args.team)
    print_logs(
        client,
        args.id,
        args.console,
        follow=args.follow,
        since=args.since,
        until=args.until,
        search=args.search,
        lines=lines_count,
        oldest=lines_oldest,
        output=args.output,
    )


def recent_logs(
    client: SyncServerlessClient, runner_id: str, since: datetime, count: int
) -> list[dict]:
    """The newest `count` log lines of a runner since `since`, oldest first."""
    rest_client = client._create_rest_client()
    with httpx.Client(
        base_url=rest_client.base_url, headers=rest_client.get_headers(), timeout=15
    ) as http:
        items, _ = _post_history(http, {"job_id": runner_id}, since, None, count)
    return list(reversed(items))  # the API returns newest first


def print_logs(
    client: SyncServerlessClient,
    runner_id: str,
    console: Console,
    *,
    follow: bool,
    since: datetime | None = None,
    until: datetime | None = None,
    search: str | None = None,
    lines: int | None = None,
    oldest: bool = False,
    output: str = "pretty",
) -> None:
    """Print a runner's logs, as `fal runners logs` does.

    A followed runner's logs start a minute ago unless `since` says otherwise;
    otherwise they start when the runner did. Both ends are kept within the
    runner's lifetime.
    """
    params: dict[str, str] = {"job_id": runner_id}
    if search is not None:
        params["search"] = search

    rest_client = client._create_rest_client()
    runner_info = _get_runner_info(rest_client, runner_id)
    if follow:
        since = since or (datetime.now(timezone.utc) - DEFAULT_STREAM_SINCE)
    else:
        since = since or runner_info.started_at
    until = until or runner_info.ended_at

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

    if follow:
        logs_gen = _stream_logs(rest_client, params, since, until)
    else:
        logs_gen = _get_logs(rest_client, params, since, until, lines, oldest=oldest)

    printer = LogPrinter(console)

    if follow:
        for log in logs_gen:
            if output == "json":
                console.print(json.dumps(log))
            else:
                printer.print(log)
        return

    if output == "json":
        console.print(json.dumps({"logs": list(logs_gen)}))
    else:
        for log in reversed(list(logs_gen)):
            printer.print(log)


def _add_gpus_parser(subparsers, parents):
    gpus_help = "Show GPU usage summary across runners."
    parser = subparsers.add_parser(
        "gpus",
        description=gpus_help,
        help=gpus_help,
        parents=[*parents, get_output_parser()],
    )
    parser.set_defaults(func=_gpus)


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
        help="Open a shell on a runner.",
        parents=parents,
    )
    parser.add_argument("id", help="Runner ID.")
    parser.set_defaults(func=_shell)


def _add_exec_parser(subparsers, parents):
    """Add hidden exec command parser."""
    parser = subparsers.add_parser(
        "exec",
        help="Execute a command on a runner.",
        parents=parents,
    )
    parser.add_argument("id", help="Runner ID.")
    parser.add_argument(
        "-it",
        "--interactive",
        action="store_true",
        help=(
            "Attach stdin. A TTY is allocated only when stdin is a terminal; "
            "with piped stdin the command gets a raw byte stream."
        ),
    )
    # PARSER keeps fal's own flags parseable between the runner id and the
    # command; REMAINDER would swallow them into the command.
    parser.add_argument(
        "command",
        nargs=argparse.PARSER,
        help="Command to execute. Prefix with -- if it starts with a dash.",
    )
    parser.set_defaults(func=_exec)


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
    _add_gpus_parser(subparsers, parents)
    _add_logs_parser(subparsers, parents)
    _add_shell_parser(subparsers, parents)
    _add_exec_parser(subparsers, parents)
