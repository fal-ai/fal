"""A personal, persistent dev box: a private fal app kept at one runner while in use.

`fal dev deploy` deploys remote/box_app.py as `devbox-<name>`; `up` and `down`
scale it between one runner and none. Its files live on /data, so they survive
restarts: the box's own team-owned volume (`devbox-<name>`) mounted at /data, or
with --team-data the team-shared /data.
"""

from __future__ import annotations

import contextlib
import io
import os
import posixpath
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterator

import grpc

from fal.api.client import SyncServerlessClient
from fal.cli.runners import recent_logs, shell_session
from fal.dev import config
from fal.dev.account import Account
from fal.dev.errors import DevError
from fal.sdk import AliasInfo, RunnerInfo, RunnerState

REMOTE = Path(__file__).parent / "remote"
BOX_APP = REMOTE / "box_app.py"
LIVE_STATES = {RunnerState.RUNNING, RunnerState.IDLE}
# Runner states that will not come back; anything else may still bill.
FINAL_STATES = {RunnerState.DEAD, RunnerState.TERMINATED, RunnerState.TERMINATING}
STOPPING_STATES = {RunnerState.DRAINING}
POLL_SECONDS = 10
SSH_READY_MARKER = b"__devbox_sshd_ready__\n"
# Pythons the default image's torch pin has wheels for.
SUPPORTED_PYTHON = ((3, 10), (3, 13))
# Commands on the runner that take longer than this are given up on.
RUN_TIMEOUT_SECONDS = 60
SAVE_TIMEOUT_SECONDS = 1800
# Credentials are rebuilt this often; building them can cost an API call.
AUTH_TTL_SECONDS = 300


def log(message: str) -> None:
    # stderr: under ssh-proxy, stdout is the SSH connection.
    print(message, file=sys.stderr, flush=True)


def fal_spec(version: str) -> str:
    """The fal requirement for the box image: this release, or the latest one."""
    from packaging.version import InvalidVersion, Version

    try:
        parsed = Version(version)
    except InvalidVersion:
        return "fal"
    # A development build is not on PyPI; the box's fal code ships with the app anyway.
    if parsed.is_devrelease or parsed.is_prerelease or parsed.local:
        return "fal"
    return f"fal=={parsed}"


@contextlib.contextmanager
def _environ(values: dict[str, str]) -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def newest_first(runners: list[RunnerInfo]) -> list[RunnerInfo]:
    return sorted(runners, key=lambda r: r.uptime.total_seconds())


# Appends each public key argument ($@) to root's authorized_keys unless present.
# Never fails: a key that cannot be written must not stop sshd from starting.
AUTHORIZE_KEYS = (
    "{ mkdir -p /root/.ssh && chmod 700 /root/.ssh && "
    "touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys && "
    # A hand-edited file may lack its final newline, which would glue two keys.
    '{ [ -n "$(tail -c1 /root/.ssh/authorized_keys)" ] '
    "&& echo >> /root/.ssh/authorized_keys; true; } && "
    'for k in "$@"; do grep -qxF "$k" /root/.ssh/authorized_keys '
    "|| printf '%s\\n' \"$k\" >> /root/.ssh/authorized_keys; done; true; }"
)
MAX_PUBLIC_KEYS = 10


def local_public_keys() -> list[str]:
    """The public keys of ~/.ssh/id_*.pub (not certificates), at most a few."""
    keys = []
    for path in sorted((Path.home() / ".ssh").glob("id_*.pub")):
        if path.name.endswith("-cert.pub"):
            continue
        try:
            lines = path.read_text().splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        keys += [
            line.strip()
            for line in lines
            if line.startswith(("ssh-", "ecdsa-", "sk-")) and "\0" not in line
        ]
    return keys[:MAX_PUBLIC_KEYS]


def newest_live(runners: list[RunnerInfo]) -> list[RunnerInfo]:
    """Ready runners, newest first."""
    return newest_first([r for r in runners if r.state in LIVE_STATES])


class _HoldUntilMarker:
    """Output stream that drops everything up to and including the SSH marker."""

    def __init__(self, out: BinaryIO) -> None:
        self.out = out
        self.seen = b""
        self.ready = False

    def write(self, data: bytes) -> None:
        if self.ready:
            self.out.write(data)
            return
        self.seen += data
        if SSH_READY_MARKER in self.seen:
            self.ready = True
            rest = self.seen.split(SSH_READY_MARKER, 1)[1]
            if rest:
                self.out.write(rest)

    def flush(self) -> None:
        self.out.flush()


@dataclass(frozen=True)
class BoxState:
    """What the platform says about the box (not the box itself)."""

    # running, stopping (being shut down), starting, stopped or failed (to start)
    code: str
    app: AliasInfo | None
    # The newest runner that is up or on its way, if any.
    runner: RunnerInfo | None

    @property
    def machine_type(self) -> str | None:
        if self.runner:
            return self.runner.machine_type
        return (
            self.app.machine_types[0] if self.app and self.app.machine_types else None
        )


class Box:
    def __init__(self, name: str, *, host: str | None = None, account: Account) -> None:
        self.name = name
        self.app = config.app_name(name)
        self.host = host
        self.account = account
        self.client = SyncServerlessClient(host=host, team=account.fal_team)
        self._stub = None
        self._auth: dict[str, str] = {}
        self._auth_at = 0.0

    def auth_headers(self) -> dict[str, str]:
        """Headers that sign calls to the platform as you, rebuilt now and then."""
        from fal.sdk import get_credentials

        if time.time() - self._auth_at > AUTH_TTL_SECONDS:
            self._auth = get_credentials(team=self.account.fal_team).to_headers()
            self._auth_at = time.time()
        return self._auth

    def runners(self, since: datetime | None = None) -> list[RunnerInfo]:
        return self.client.apps.runners(self.app, since=since)

    def poll_runners(self, attempts: int = 3) -> list[RunnerInfo]:
        """List runners, retrying briefly so one network blip does not abort a start."""
        for _ in range(attempts - 1):
            try:
                return self.runners()
            except Exception:
                time.sleep(POLL_SECONDS)
        return self.runners()

    def app_info(self) -> AliasInfo | None:
        apps = self.client.apps.list(filter=self.app)
        return next((a for a in apps if a.alias == self.app), None)

    def armed(self) -> bool:
        info = self.app_info()
        return bool(info and info.min_concurrency)

    def state(self) -> BoxState:
        runners = self.runners()
        app = self.app_info()
        live = newest_live(runners)
        if live:
            armed = bool(app and app.min_concurrency)
            return BoxState("running" if armed else "stopping", app, live[0])
        pending = newest_first([r for r in runners if r.state not in FINAL_STATES])
        if not pending:
            return BoxState("stopped", app, None)
        states = {r.state for r in pending}
        if RunnerState.CRASH_BACKOFF in states:
            return BoxState("failed", app, pending[0])
        if states <= STOPPING_STATES:
            return BoxState("stopping", app, pending[0])
        return BoxState("starting", app, pending[0])

    def require_runner(self) -> str:
        live = newest_live(self.runners())
        if not live:
            raise DevError(f"{self.app} is not running. Start it with: fal dev up")
        if not self.armed():
            raise DevError(
                f"{self.app} is shutting down. Start it again with: fal dev up"
            )
        return live[0].runner_id

    def run(
        self,
        runner_id: str,
        command: list[str],
        *,
        timeout: float = RUN_TIMEOUT_SECONDS,
        until: bytes | None = None,
    ) -> tuple[int, str]:
        """Run a command on the runner; its exit code and combined output.

        With `until`, output is read up to that marker rather than up to the exit
        code, which can arrive before the last of the output; the exit code is
        then not meaningful. A command still running after `timeout` seconds is
        given up on.
        """
        out = io.BytesIO()
        for retry in (True, False):
            out.seek(0)
            out.truncate()
            if self._stub is None:
                self._stub = self.client._create_host()._connection.stub
            try:
                code = shell_session(
                    self.client,
                    runner_id,
                    command,
                    interactive=False,
                    remote_tty=False,
                    console=None,
                    stdout=out,
                    stderr=out,
                    timeout=timeout,
                    until=until,
                    stub=self._stub,
                )
                break
            except Exception as exc:
                self._stub = None
                # The cached channel carries the access token it was built with.
                expired = (
                    isinstance(exc, grpc.RpcError)
                    and exc.code() == grpc.StatusCode.UNAUTHENTICATED
                )
                if not (expired and retry):
                    reason = exc.details() if isinstance(exc, grpc.RpcError) else exc
                    raise DevError(
                        f"could not run a command on the box: {reason}"
                    ) from exc
        return code, out.getvalue().decode(errors="replace")

    def logs(self, runner_id: str, count: int, since: datetime) -> list[dict]:
        """The newest `count` log lines of a runner since `since`, oldest first."""
        return recent_logs(self.client, runner_id, since, count)

    def scale(self, min_concurrency: int, machine_type: str | None = None) -> None:
        self.client.apps.scale(
            self.app,
            min_concurrency=min_concurrency,
            machine_types=[machine_type] if machine_type else None,
        )

    def stop(self) -> bool:
        """Scale to zero and stop every runner. True once none are left running."""
        self.scale(0)
        for runner in self.runners():
            if runner.state not in FINAL_STATES:
                # Best effort: a runner still starting may refuse; scale 0 releases it.
                with contextlib.suppress(Exception):
                    self.client.runners.stop(runner.runner_id)
        for _ in range(6):
            if all(r.state in FINAL_STATES for r in self.runners()):
                return True
            time.sleep(POLL_SECONDS)
        return False

    def wait_until_stopped(self, timeout: int) -> bool:
        """Wait, without stopping anything, until no runner is left; True if so."""
        deadline = time.time() + timeout
        while any(r.state not in FINAL_STATES for r in self.runners()):
            if time.time() > deadline:
                return False
            log("waiting for the last runner to finish saving ...")
            time.sleep(POLL_SECONDS)
        return True

    def save_work(self) -> None:
        """Save /work on a running box; raise, leaving the box up, if that fails."""
        live = newest_live(self.runners())
        if not live:
            return
        log(f"saving {config.LOCAL_WORK} ...")
        guidance = (
            "the box is still running and the save may still be in progress. Retry, "
            "or to stop anyway and lose changes since the last save, add --no-save"
        )
        try:
            code, output = self.run(
                live[0].runner_id, [config.SAVE_COMMAND], timeout=SAVE_TIMEOUT_SECONDS
            )
        except DevError as exc:
            raise DevError(
                f"could not save {config.LOCAL_WORK} ({exc}); {guidance}"
            ) from exc
        output = " ".join(output.split())
        if code != 0:
            raise DevError(
                f"could not save {config.LOCAL_WORK} "
                f"({output or f'exit {code}'}); {guidance}"
            )
        log(output)

    def check_admin_key(self) -> None:
        """Warn when the idle watchdog's key is missing or exposed to every app."""
        try:
            secrets = self.client.secrets.list()
        except Exception:
            return  # a warning must never block the command
        entry = next((s for s in secrets if s.name == config.ADMIN_KEY_SECRET), None)
        if entry is None:
            log(
                f"warning: the {config.ADMIN_KEY_SECRET} secret is not set, "
                "so this box will not shut itself down when idle. "
                "Remember to run: fal dev down"
            )
        elif entry.default_exposed is not False:
            log(
                f"warning: {config.ADMIN_KEY_SECRET} is exposed to every app on the "
                "team. Set it again with: fal secrets set --not-exposed-by-default "
                f"{config.ADMIN_KEY_SECRET}=<admin key>"
            )

    def deploy(
        self,
        *,
        machine_type: str | None = None,
        num_gpus: int = config.DEFAULT_NUM_GPUS,
        idle_minutes: int = config.DEFAULT_IDLE_MINUTES,
        claude: bool = False,
        no_save: bool = False,
        dockerfile: str | None = None,
        team_data: bool = False,
        hint: bool = True,
    ) -> str:
        from fal._version import __version__

        python = sys.version_info[:2]
        if not dockerfile and not SUPPORTED_PYTHON[0] <= python <= SUPPORTED_PYTHON[1]:
            low, high = (".".join(map(str, v)) for v in SUPPORTED_PYTHON)
            raise DevError(
                "the box runs your Python version, and the default image supports "
                f"Python {low} to {high}; run fal with one of those, or pass "
                "--dockerfile"
            )
        if dockerfile and not Path(dockerfile).is_file():
            raise DevError(f"no such Dockerfile: {dockerfile}")
        volume = None if team_data else self.ensure_volume()
        app = self.app_info()
        was_stopped = not (app and app.min_concurrency)
        if not was_stopped:
            # Stop first: a rolling deploy keeps the old runner, and its /work, alive
            # until the new one is ready, so both would write the snapshot.
            if not no_save:
                self.save_work()
            try:
                stopped = self.stop()
            except BaseException:
                log("the box may be stopping; start it again with: fal dev up")
                raise
            if not stopped:
                raise DevError(
                    "the box is still shutting down; run `fal dev deploy` again, "
                    "then `fal dev up`"
                )
        elif app and not self.wait_until_stopped(config.LEASE_WAIT_SECONDS):
            # A runner stopped moments ago may still be saving /work; the new
            # revision must start from that save.
            raise DevError(
                "the box is still saving and shutting down; run `fal dev deploy` "
                "again in a few minutes"
            )
        current = app.machine_types[0] if app and app.machine_types else None
        machine_type = machine_type or current or config.DEFAULT_GPU
        settings = {
            config.ENV_NAME: self.name,
            config.ENV_MACHINE_TYPE: machine_type,
            config.ENV_NUM_GPUS: str(num_gpus),
            config.ENV_IDLE_MINUTES: str(idle_minutes),
            config.ENV_CLAUDE: "1" if claude else "",
            config.ENV_DOCKERFILE: str(Path(dockerfile).resolve())
            if dockerfile
            else "",
            config.ENV_PYTHON_VERSION: f"{sys.version_info[0]}.{sys.version_info[1]}",
            config.ENV_FAL_SPEC: fal_spec(__version__),
            config.ENV_VOLUME: volume or "",
        }
        # box_app.py reads these when the deploy loads it.
        try:
            with _environ(settings):
                self.client.deploy(
                    (str(BOX_APP), "DevBox"),
                    app_name=self.app,
                    auth="private",
                    # Without this the app's termination grace period, which gives
                    # teardown() time to save /work, is never set.
                    reset_scale=True,
                )
            if not was_stopped:
                # reset_scale applies the app's min_concurrency of 0; start it again.
                self.scale(1)
        except BaseException:
            if not was_stopped:
                log("the box was stopped for the deploy; start it with: fal dev up")
            raise
        if was_stopped and not self.stop():
            # The deploy starts a runner to check the new image; a stopped box should
            # not keep it.
            log(
                "the deploy's check runner is still shutting down; "
                "if it lingers, run: fal dev down"
            )
        next_step = (
            "Start it with: fal dev up"
            if was_stopped
            else "The box is restarting on the new image; check with: fal dev status"
        )
        storage = f"volume {volume}" if volume else "the team-shared /data"
        summary = (
            f"deployed {self.app} on {num_gpus}x {machine_type} with {storage} at "
            f"/data, idle shutdown after {idle_minutes} min."
        )
        return f"{summary} {next_step}" if hint else summary

    def ensure_volume(self) -> str:
        """Create the box's volume if needed; its name."""
        from fal.exceptions.volumes import (
            VolumeAccountNotEligibleError,
            VolumeAlreadyExistsError,
        )

        volume = config.volume_name(self.name)
        try:
            self.client.volumes.create(volume)
        except VolumeAlreadyExistsError:
            return volume
        except VolumeAccountNotEligibleError as exc:
            raise DevError(
                f"this team cannot use volumes ({exc}); deploy with --team-data to "
                "keep the box on the team-shared /data"
            ) from exc
        log(
            f"created volume {volume} for the box's /data. It starts empty: to keep "
            "an existing box's files on the team /data, deploy with --team-data"
        )
        return volume

    def wait_until_ready(self, timeout: int) -> RunnerInfo | None:
        """Poll until a runner is ready; None on timeout. Raises if the start fails."""
        deadline = time.time() + timeout
        last_state = None
        while time.time() < deadline:
            runners = self.poll_runners()
            ready = newest_live(runners)
            if ready:
                return ready[0]
            states = {r.state for r in runners}
            # setup() or the image failed and the platform is backing off retries.
            if RunnerState.CRASH_BACKOFF in states:
                raise DevError(
                    "the box failed to start. See why with: fal dev logs --all"
                )
            pending = sorted(s.value.lower() for s in states - FINAL_STATES)
            state = ", ".join(pending) or "requested"
            if state != last_state:
                log(f"waiting: {state}")
                last_state = state
            time.sleep(POLL_SECONDS)
        return None

    def start(self, timeout: int, machine_type: str | None = None) -> RunnerInfo:
        """Scale to 1 and wait for a ready runner; cancel the start if none comes up."""
        # Only cancel a start this call made: never stop a box that was already up.
        was_live = bool(newest_live(self.runners()))
        runner = None
        try:
            self.scale(1, machine_type)
            runner = self.wait_until_ready(timeout)
        finally:
            # Any way out without a ready box (timeout, failed start, Ctrl-C) cancels
            # the start.
            if runner is None and not was_live:
                try:
                    stopped = self.stop()
                except Exception:
                    stopped = False
                if not stopped:
                    log("the box may still be starting; run: fal dev down")
        if runner is None:
            raise DevError(
                f"no machine became ready within {timeout}s; the start was cancelled. "
                "Try again later."
            )
        return runner

    def up(self, timeout: int, machine_type: str | None = None) -> RunnerInfo:
        if self.app_info() is None:
            log(
                f"{self.app} does not exist yet; "
                "deploying it with the default image ..."
            )
            log(self.deploy(machine_type=machine_type, hint=False))
        live = newest_live(self.runners())
        if live and machine_type and live[0].machine_type != machine_type:
            raise DevError(
                f"the box is running on {live[0].machine_type}. "
                f"Run `fal dev down` first to switch to {machine_type}."
            )
        self.check_admin_key()
        return self.start(timeout, machine_type)

    def down(self, no_save: bool = False) -> None:
        if not no_save:
            self.save_work()
        if not self.stop():
            raise DevError(
                "a runner is still shutting down; check again with `fal dev status` "
                "or rerun `fal dev down`"
            )

    def hold(self, hours: float) -> bool:
        """Keep the box up past the idle timeout; False if a longer hold was kept."""
        if not 0 < hours <= config.MAX_HOLD_HOURS:
            raise DevError(
                f"hours must be more than 0 and at most {config.MAX_HOLD_HOURS}"
            )
        live = newest_live(self.runners())
        if not live:
            raise DevError(f"{self.app} is not running. Start it with: fal dev up")
        # Re-arm the box in case the watchdog has just started shutting it down.
        self.scale(1)
        until = int(time.time() + hours * 3600)
        script = (REMOTE / "scripts" / "hold.sh").read_text()
        kept = str(config.HOLD_KEPT_EXIT)
        code, output = self.run(
            live[0].runner_id,
            ["sh", "-c", script, "sh", config.HOLD_FILE, str(until), kept],
        )
        if code not in (0, config.HOLD_KEPT_EXIT):
            raise DevError(
                f"could not set the hold: {output.strip() or f'exit {code}'}"
            )
        # An idle shutdown that was already under way may have landed after the re-arm.
        if not self.armed():
            self.scale(1)
        return code == 0

    def status(self) -> list[str]:
        state = self.state()
        runner = state.runner
        if runner and runner.state in LIVE_STATES:
            uptime_min = int(runner.uptime.total_seconds()) // 60
            running = (
                "running"
                if state.code == "running"
                else "shutting down (run `fal dev up` to keep it)"
            )
            return [
                f"{self.app}: {running} on {runner.machine_type}",
                f"runner: {runner.runner_id}  uptime: {uptime_min} min",
            ]
        if runner and state.code == "starting":
            return [
                f"{self.app}: starting ({runner.state.value.lower()}). "
                "Cancel with: fal dev down"
            ]
        if runner and state.code == "failed":
            return [f"{self.app}: failed to start. See why with: fal dev logs --all"]
        if state.code == "stopping":
            return [f"{self.app}: stopping"]
        if state.app is None:
            return [f"{self.app}: not deployed. Create it with: fal dev up"]
        return [f"{self.app}: stopped"]

    def ssh_proxy(self, timeout: int) -> int:
        """SSH ProxyCommand: carry the SSH connection over runner exec.

        The exec stream is a plain pipe, which carries SSH's binary protocol as
        is. The box prints a marker once sshd is about to start, so setup errors
        can be reported instead of handed to the SSH client. A stopped (or
        stopping) box is started first, so connecting wakes it.
        """
        from rich.console import Console

        # Editors give up on a failed connect, so ride out brief fal API errors.
        for attempt in range(3):
            try:
                state = self.state()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
        if state.runner and state.runner.state in LIVE_STATES:
            runner_id = state.runner.runner_id
            if state.code == "stopping":
                # Re-arm a box that is shutting down; its runner is still there.
                self.scale(1)
        else:
            log(f"{self.app} is stopped, starting it ...")
            runner_id = self.start(timeout).runner_id
        sshd_log = posixpath.join(posixpath.dirname(config.SSHD_CONFIG), "sshd.log")
        # Authorize this laptop's keys on every connect: nothing else installs them,
        # and the file lives on /data, so duplicates are skipped.
        remote = (
            f"test -f {config.SSHD_CONFIG} "
            f"&& {AUTHORIZE_KEYS} "
            f"&& printf '%s\\n' {SSH_READY_MARKER.decode().strip()} "
            f"&& exec /usr/sbin/sshd -i -f {config.SSHD_CONFIG} -E {sshd_log}"
        )
        out = _HoldUntilMarker(sys.stdout.buffer)
        code = shell_session(
            self.client,
            runner_id,
            ["sh", "-c", remote, "sh", *local_public_keys()],
            interactive=True,
            remote_tty=False,
            # Errors go to stderr: stdout is the SSH connection.
            console=Console(stderr=True),
            stdout=out,
        )
        if not out.ready:
            sys.stderr.write(out.seen.decode(errors="replace"))
            raise DevError(
                "could not start SSH on the box (redeploy it with `fal dev deploy`): "
                f"exit {code}"
            )
        return code


def ssh_config(
    name: str, fal_command: str, host: str | None = None, team: str | None = None
) -> str:
    """An ~/.ssh/config entry for Cursor / VS Code Remote-SSH."""
    alias = config.app_name(name)
    proxy = f"{shlex.quote(fal_command)} dev ssh-proxy --name {name}"
    if host:
        proxy += f" --host {shlex.quote(host)}"
    if team:
        proxy += f" --team {shlex.quote(team)}"
    # Leading blank line: appending to a config without a trailing newline stays valid.
    return (
        f"\nHost {alias}\n"
        f"    HostName {alias}\n"
        "    User root\n"
        f"    ProxyCommand {proxy}\n"
        "    ServerAliveInterval 30\n"
        "    # Connecting authorizes the public keys in ~/.ssh on the box.\n"
        "    # The box keeps its host key on /data, so it only changes if the\n"
        "    # workspace is reset.\n"
        "    StrictHostKeyChecking accept-new"
    )
