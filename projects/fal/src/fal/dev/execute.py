"""`fal dev exec`: run one command on the box without ever losing its output.

The runner exec stream can end right after the command exits and drop output
still in flight, with an exit code of 0. So the command runs under a wrapper that
prints an end marker, carrying the exit code, on both stdout and stderr once the
command is done, then lingers a moment so the marker itself can drain. The client
reads until it has seen both markers; a stream that ends without them is reported
as truncated instead of as a success.

The wrapper's script starts with EXEC_TAG, which the box's idle watchdog counts
as activity while the command runs (see remote/signals.py).
"""

from __future__ import annotations

import secrets
import sys

from fal.dev.remote.signals import EXEC_TAG

# How long the wrapper outlives the command so its markers can drain; the client
# cancels as soon as it has both.
LINGER_SECONDS = 15
TRUNCATED_EXIT = 75

_SCRIPT = (
    EXEC_TAG
    + """
marker=$1$2; shift 2
"$@"
status=$?
printf '%s%s\\n' "$marker" "$status"
printf '%s%s\\n' "$marker" "$status" >&2
sleep {linger}
exit "$status"
"""
)


def wrap(command: list[str]) -> tuple[list[str], bytes]:
    """The wrapped command to run on the box, and the marker it will print."""
    marker = f"__fal_dev_exec_{secrets.token_hex(8)}:"
    script = _SCRIPT.format(linger=LINGER_SECONDS)
    # The marker is passed in two halves so that it never appears whole in the
    # box's process list, where a command like `ps aux` would print it.
    half = len(marker) // 2
    wrapped = ["sh", "-c", script, "sh", marker[:half], marker[half:], *command]
    return wrapped, marker.encode()


def _partial_marker(data: bytes, marker: bytes) -> int:
    """Length of the longest tail of `data` that is a proper prefix of `marker`."""
    for size in range(min(len(data), len(marker) - 1), 0, -1):
        if marker.startswith(data[-size:]):
            return size
    return 0


class MarkerFilter:
    """Output stream that passes data through and swallows the end marker."""

    def __init__(self, out, marker: bytes):
        self.out = out
        self.marker = marker
        self.buffer = b""
        self.found = False
        self.complete = False
        self.code: int | None = None

    def write(self, data: bytes) -> None:
        if self.complete:
            return
        self.buffer += data
        if not self.found:
            at = self.buffer.find(self.marker)
            if at < 0:
                # Hold back only a tail that could be the start of the marker.
                keep = _partial_marker(self.buffer, self.marker)
                self.out.write(self.buffer[: len(self.buffer) - keep])
                self.buffer = self.buffer[len(self.buffer) - keep :]
                return
            self.out.write(self.buffer[:at])
            self.buffer = self.buffer[at + len(self.marker) :]
            self.found = True
        line, newline, _ = self.buffer.partition(b"\n")
        if newline:
            self.complete = True
            self.code = int(line) if line.strip().isdigit() else 1

    def flush(self) -> None:
        self.out.flush()

    def finish(self) -> None:
        """Pass on what was held back once the stream is over."""
        if not self.found:
            self.out.write(self.buffer)
            self.buffer = b""
        self.out.flush()


def run(
    client, runner_id: str, command: list[str], console, stdout=None, stderr=None
) -> int:
    """Run `command` on the runner; its exit code, or TRUNCATED_EXIT if output was lost."""
    from fal.cli.runners import shell_session

    wrapped, marker = wrap(command)
    out = MarkerFilter(stdout or sys.stdout.buffer, marker)
    err = MarkerFilter(stderr or sys.stderr.buffer, marker)
    shell_session(
        client,
        runner_id,
        wrapped,
        interactive=False,
        remote_tty=False,
        console=console,
        stdout=out,
        stderr=err,
        done=lambda: out.complete and err.complete,
    )
    out.finish()
    err.finish()
    if out.complete and err.complete:
        return out.code or 0
    console.print(
        "[red]Error:[/] the output may be incomplete: the connection ended before "
        "the command's end marker arrived. Run it again, or use `fal dev ssh-config` "
        "and ssh."
    )
    return TRUNCATED_EXIT
