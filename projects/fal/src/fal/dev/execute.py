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

import contextlib
import os
import secrets
import sys
import threading

from fal.dev import config
from fal.dev.errors import DevError
from fal.dev.remote.signals import EXEC_TAG

# How long the wrapper outlives the command so its markers can drain; the client
# cancels as soon as it has both.
LINGER_SECONDS = 15
TRUNCATED_EXIT = 75
# The shell's "command not found".
NOT_FOUND_EXIT = 127
# Piped input is read whole before the command starts (see `run`).
MAX_STDIN_BYTES = 64 * 1024 * 1024

_SCRIPT = (
    EXEC_TAG
    + """
marker=$1$2; shift 2
# Your secrets and variables; the watchdog's key never reaches your command.
[ -r {env_file} ] && . {env_file}
unset {admin_key}
{run}
status=$?
printf '%s%s\\n' "$marker" "$status"
printf '%s%s\\n' "$marker" "$status" >&2
sleep {linger}
exit "$status"
"""
)


def wrap(command: list[str], stdin_bytes: int | None = None) -> tuple[list[str], bytes]:
    """The wrapped command to run on the box, and the marker it will print.

    With `stdin_bytes`, the command reads exactly that many bytes of input and
    then sees end of input.
    """
    marker = f"__fal_dev_exec_{secrets.token_hex(8)}:"
    run_line = '"$@"' if stdin_bytes is None else f'head -c {stdin_bytes} | "$@"'
    script = _SCRIPT.format(
        run=run_line,
        linger=LINGER_SECONDS,
        env_file=config.USER_ENV_FILE,
        admin_key=config.ADMIN_KEY_SECRET,
    )
    # The marker is passed in two halves so that it never appears whole in the
    # box's process list, where a command like `ps aux` would print it.
    half = len(marker) // 2
    wrapped = ["sh", "-c", script, "sh", marker[:half], marker[half:], *command]
    return wrapped, marker.encode()


def not_found_note(machine_type: str | None) -> str:
    """The line after exit 127: the command is missing on the box, not on the laptop."""
    if machine_type and not config.is_gpu(machine_type):
        return "(command not found on the box; this is a CPU box)"
    return "(command not found on the box)"


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
    client,
    runner_id: str,
    command: list[str],
    console,
    stdout=None,
    stderr=None,
    stdin: bool = False,
) -> int:
    """Run `command` on the runner.

    Returns its exit code, or TRUNCATED_EXIT if output was lost. With `stdin`,
    piped local input is read whole and forwarded. The runner stream does not
    pass on the end of input, so the box side reads exactly that many bytes and
    then closes the command's input itself.
    """
    if stdin:
        data = sys.stdin.buffer.read(MAX_STDIN_BYTES + 1)
        if len(data) > MAX_STDIN_BYTES:
            raise DevError(
                f"-i forwards at most {MAX_STDIN_BYTES // (1024 * 1024)} MiB of input",
                fixes=[("fal dev cp FILE :/work/", "copy a large file instead")],
            )
        with _stdin_from(data):
            return _run(client, runner_id, command, console, stdout, stderr, data)
    return _run(client, runner_id, command, console, stdout, stderr, None)


@contextlib.contextmanager
def _stdin_from(data: bytes):
    """Stand `data` in for sys.stdin, as a pipe the shell session can read."""
    read_fd, write_fd = os.pipe()

    def feed() -> None:
        with os.fdopen(write_fd, "wb") as pipe:
            with contextlib.suppress(BrokenPipeError):
                pipe.write(data)

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    original = sys.stdin
    sys.stdin = os.fdopen(read_fd, "r")
    try:
        yield
    finally:
        sys.stdin.close()
        sys.stdin = original


def _run(client, runner_id, command, console, stdout, stderr, data) -> int:
    from fal.cli.runners import shell_session

    stdin = data is not None
    wrapped, marker = wrap(command, len(data) if stdin else None)
    out = MarkerFilter(stdout or sys.stdout.buffer, marker)
    err = MarkerFilter(stderr or sys.stderr.buffer, marker)
    shell_session(
        client,
        runner_id,
        wrapped,
        interactive=stdin,
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
