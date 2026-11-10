"""A personal, persistent dev box: a private fal app kept at one runner while in use.

`fal dev deploy` deploys remote/box_app.py as `devbox-<name>`; `up` and `down`
scale it between one runner and none. Its files live on /data, so they survive
restarts: the box's own team-owned volume (`devbox-<name>`) mounted at /data, or
with --team-data the team-shared /data.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import posixpath
import shlex
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Callable, Iterator

import grpc
import httpx

from fal.api.client import SyncServerlessClient
from fal.cli.runners import recent_logs, shell_session
from fal.dev import config, state
from fal.dev.account import Account
from fal.dev.errors import DevError
from fal.dev.transport import CONNECT_TIMEOUT, host_of, retry_connect
from fal.sdk import AliasInfo, RunnerInfo, RunnerState

if TYPE_CHECKING:
    from fal.dev.steps import Steps

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
# A failed owner lookup (for display) is tried again after this long.
OWNER_RETRY_SECONDS = 30


def is_network_error(exc: BaseException) -> bool:
    """A failure to reach fal at all. Only for idempotent reads, like polling."""
    if isinstance(exc, grpc.RpcError):
        code = getattr(exc, "code", None)
        return callable(code) and code() in (
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DEADLINE_EXCEEDED,
        )
    return isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
            ConnectionError,
            TimeoutError,
            socket.gaierror,
        ),
    )


# What ssh-proxy tells an editor while a stopped box wakes (a cold start is ~4 min).
COLD_START_MINUTES = 4


def log(message: str) -> None:
    # stderr: under ssh-proxy, stdout is the SSH connection.
    print(message, file=sys.stderr, flush=True)


def local_starship_config() -> str:
    """Your starship.toml, the only file `--starship` sends to the box."""
    path = Path(os.environ.get("STARSHIP_CONFIG") or "~/.config/starship.toml")
    path = path.expanduser()
    if not path.is_file():
        raise DevError(
            f"no starship config at {path}; create one or set $STARSHIP_CONFIG"
        )
    return path.read_text()


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


def merge_user_env(
    remembered: dict,
    secrets: list[str] | None = None,
    set_env: dict[str, str] | None = None,
    unset_secrets: list[str] | None = None,
    unset_env: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """The box's secret names and variables: those remembered, plus --secret and
    --set-env, minus --unset-secret and --unset-env."""
    names = remembered.get("secrets")
    env = remembered.get("env")
    kept = [n for n in names if isinstance(n, str)] if isinstance(names, list) else []
    values = (
        {k: v for k, v in env.items() if isinstance(k, str) and isinstance(v, str)}
        if isinstance(env, dict)
        else {}
    )
    try:
        for name in secrets or []:
            config.check_env_name(name, "secret name")
        for name in set_env or {}:
            config.check_env_name(name, "variable name")
    except ValueError as exc:
        raise DevError(str(exc)) from None
    names = list(dict.fromkeys([*kept, *(secrets or [])]))
    names = [n for n in names if n not in (unset_secrets or [])]
    values = {**values, **(set_env or {})}
    values = {k: v for k, v in values.items() if k not in (unset_env or [])}
    clash = sorted(set(names) & set(values))
    if clash:
        raise DevError(
            f"{', '.join(clash)} is both a secret and an environment variable; "
            "keep one (--unset-secret or --unset-env removes the other)"
        )
    return names, values


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


def live_code(app: AliasInfo | None) -> str:
    """A box with a live runner is stopping once it is no longer armed."""
    return "running" if app and app.min_concurrency else "stopping"


def derive_state(
    app: AliasInfo | None, runners: list[RunnerInfo], live: str | None = None
) -> tuple[str, RunnerInfo | None]:
    """The box's state code and its newest runner that is up or on its way.

    One rule for `Box.state()` and `fal dev list`, so they cannot disagree.
    `live` is the code for a box with a live runner (default: `live_code(app)`).
    """
    if app is None:
        return "stopped", None  # never deployed: no runners
    up = newest_live(runners)
    if up:
        return (live or live_code(app)), up[0]
    pending = newest_first([r for r in runners if r.state not in FINAL_STATES])
    if not pending:
        return "stopped", None
    states = {r.state for r in pending}
    if RunnerState.CRASH_BACKOFF in states:
        return "failed", pending[0]
    if states <= STOPPING_STATES:
        return "stopping", pending[0]
    return "starting", pending[0]


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
        self.start_cancelled: bool | None = None
        # `fal dev up` sets this to show its steps: messages and deploy logs then
        # go to it instead of stderr.
        self.progress: Steps | None = None
        self.volume_created = False
        # Readiness polls that could not connect, and the last such error.
        self.poll_failures = 0
        self.poll_error: BaseException | None = None
        self._stub = None
        self._auth: dict[str, str] = {}
        self._auth_at = 0.0
        self._identity: str | None = None
        self._owner: str | None = None
        self._owner_read = False
        self._owner_retry_at = 0.0

    def _say(self, message: str) -> None:
        if self.progress is not None:
            self.progress.say(message)
        else:
            log(message)

    def _begin(self, key: str, detail: str = "") -> None:
        if self.progress is not None:
            self.progress.begin(key, detail)

    def _finish(self, key: str, detail: str | None = None) -> None:
        if self.progress is not None:
            self.progress.finish(key, detail)

    def _deploy_handlers(self) -> dict:
        """Result handlers that turn the deploy's logs into step changes."""
        progress = self.progress
        if progress is None:
            return {}
        from fal.api.api import ResultHandler

        class Build(ResultHandler):
            def on_log(self, entry) -> None:
                progress.begin("build")
                progress.capture(entry.message)

        class Register(ResultHandler):
            def __call__(self, partial_result) -> None:
                # The build is over once the platform starts registering.
                progress.finish("build")
                progress.begin("deploy")
                super().__call__(partial_result)

            def on_log(self, entry) -> None:
                progress.capture(entry.message)

        return {"build_result_handler": Build(), "result_handler": Register()}

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

    def owner_identity(self) -> str:
        """Who you are, as stamped on boxes you deploy."""
        from fal.dev.guard import identity

        if self._identity is None:
            self._identity = identity()
        return self._identity

    def stored_owner(self, attempts: int = 3) -> str | None:
        """The owner stamped on the box's current deployment; None if there is none.

        Read from the app's metadata over REST, so any machine sees it. Raises
        DevError when the lookup fails: a guard that cannot look must not pass.
        """
        import openapi_fal_rest.api.applications.app_metadata as app_metadata

        from fal.dev.guard import clean

        namespace = self.account.team
        if not namespace:
            return None
        for attempt in range(attempts):
            try:
                response = app_metadata.sync_detailed(
                    app_user_id=namespace,
                    app_alias_or_id=self.app,
                    client=self.client._create_rest_client(),
                )
                break
            except Exception as exc:
                if attempt == attempts - 1:
                    raise DevError(
                        f"could not read who owns {self.app}: {exc}"
                    ) from exc
                time.sleep(2)
        if response.status_code == 404:
            return None
        if response.status_code != 200 or response.parsed is None:
            raise DevError(
                f"could not read who owns {self.app} (HTTP {int(response.status_code)})"
            )
        annotations = response.parsed.additional_properties.get("annotations")
        owner = annotations.get(config.OWNER_ANNOTATION) if annotations else None
        return (clean(owner) or None) if owner else None

    @property
    def known_owner(self) -> str | None:
        """The owner if `owner()` already read it; never touches the network."""
        return self._owner

    def owner(self) -> str | None:
        """The stored owner for display, read once per session; None if unknown."""
        if not self._owner_read:
            if time.time() < self._owner_retry_at:
                return None
            try:
                self._owner = self.stored_owner(attempts=1)
            except Exception:
                # Not remembered, but not retried on every poll either.
                self._owner_retry_at = time.time() + OWNER_RETRY_SECONDS
                return None
            self._owner_read = True
        return self._owner

    def armed(self) -> bool:
        info = self.app_info()
        return bool(info and info.min_concurrency)

    def state(self) -> BoxState:
        app = self.app_info()
        if app is None:
            return BoxState("stopped", None, None)
        code, runner = derive_state(app, self.runners(), self._live_code(app))
        return BoxState(code, app, runner)

    def _live_code(self, app: AliasInfo | None) -> str:
        return live_code(app)

    def require_runner(self) -> str:
        live = newest_live(self.runners())
        if not live:
            raise DevError(
                f"{self.app} is not running",
                title="the box is not running",
                fixes=[("fal dev up", "start it")],
            )
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
            self._say(
                f"warning: the {config.ADMIN_KEY_SECRET} secret is not set, "
                "so this box will not shut itself down when idle. "
                "Remember to run: fal dev down"
            )
        elif entry.default_exposed is not False:
            self._say(
                f"warning: {config.ADMIN_KEY_SECRET} is exposed to every app on the "
                "team. Set it again with: fal secrets set --not-exposed-by-default "
                f"{config.ADMIN_KEY_SECRET}=<admin key>"
            )

    def check_secrets_exist(self, names: list[str]) -> None:
        """Refuse names that are not secrets of this team: the box would not start."""
        if not names:
            return
        try:
            known = {s.name for s in self.client.secrets.list()}
        except Exception:
            return  # the deploy itself reports a missing secret
        missing = [n for n in names if n not in known]
        if missing:
            raise DevError(
                f"no such secret: {', '.join(missing)}. Secrets belong to the team; "
                f"create one with: fal secrets set {missing[0]}=<value>, or stop "
                f"asking for it with: fal dev deploy --unset-secret {missing[0]}",
                title="unknown secret",
            )

    def deploy(
        self,
        *,
        machine_type: str | None = None,
        num_gpus: int = config.DEFAULT_NUM_GPUS,
        idle_minutes: int = config.DEFAULT_IDLE_MINUTES,
        claude: bool = False,
        starship: bool = False,
        no_save: bool = False,
        dockerfile: str | None = None,
        team_data: bool = False,
        hint: bool = True,
        take_over: bool = False,
        secrets: list[str] | None = None,
        set_env: dict[str, str] | None = None,
        unset_secrets: list[str] | None = None,
        unset_env: list[str] | None = None,
    ) -> str:
        from fal._version import __version__
        from fal.dev.guard import check_limits, check_owner

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
        # Before anything is created or stopped: a lookup that fails fails here.
        self.owner_identity()
        check_owner(self, take_over=take_over)
        existing = self.app_info()
        current_type = (
            existing.machine_types[0] if existing and existing.machine_types else None
        )
        check_limits(machine_type or current_type or config.DEFAULT_GPU, num_gpus)
        # Read first: a missing config must not fail after the box was stopped.
        starship_toml = local_starship_config() if starship else ""
        secret_names, user_env = merge_user_env(
            state.load_box(self.app, self.host),
            secrets,
            set_env,
            unset_secrets,
            unset_env,
        )
        self.check_secrets_exist(secret_names)
        volume = None
        if not team_data:
            self._begin("volume")
            volume = self.ensure_volume()
            self._finish(
                "volume", "created, starts empty" if self.volume_created else "exists"
            )
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
                self._say("the box may be stopping; start it again with: fal dev up")
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
            config.ENV_STARSHIP: starship_toml,
            config.ENV_SECRETS: ",".join(secret_names),
            config.ENV_USER_ENV: json.dumps(user_env) if user_env else "",
        }
        # box_app.py reads these when the deploy loads it.
        self._begin("build")
        try:
            with _environ(settings):
                self.client.deploy(
                    (str(BOX_APP), "DevBox"),
                    app_name=self.app,
                    auth="private",
                    annotations={config.OWNER_ANNOTATION: self.owner_identity()},
                    # Without this the app's termination grace period, which gives
                    # teardown() time to save /work, is never set.
                    reset_scale=True,
                    **self._deploy_handlers(),
                )
            if not was_stopped:
                # reset_scale applies the app's min_concurrency of 0; start it again.
                self.scale(1)
        except BaseException:
            if not was_stopped:
                self._say(
                    "the box was stopped for the deploy; start it with: fal dev up"
                )
            raise
        if was_stopped and not self.stop():
            # The deploy starts a runner to check the new image; a stopped box should
            # not keep it.
            self._say(
                "the deploy's check runner is still shutting down; "
                "if it lingers, run: fal dev down"
            )
        self._owner, self._owner_read = self.owner_identity(), True
        self._finish("build")  # a deploy that logged nothing still built
        self._finish("deploy")
        # Remembered, so that later deploys keep them. Best effort: a read-only
        # home must not fail a deploy that worked.
        with contextlib.suppress(OSError):
            state.update_box(self.app, self.host, secrets=secret_names, env=user_env)
        next_step = (
            "Start it with: fal dev up"
            if was_stopped
            else "The box is restarting on the new image; check with: fal dev status"
        )
        storage = f"volume {volume}" if volume else "the team-shared /data"
        machine = (
            f"{num_gpus}x {machine_type}"
            if config.is_gpu(machine_type)
            else machine_type
        )
        summary = (
            f"deployed {self.app} on {machine} with {storage} at "
            f"/data, idle shutdown after {idle_minutes} min, "
            f"secrets: {', '.join(secret_names) or 'none'}, "
            f"env vars: {', '.join(user_env) or 'none'}."
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
            retry_connect(
                lambda: self.client.volumes.create(
                    volume, connect_timeout=CONNECT_TIMEOUT
                ),
                host=host_of(str(self.client._rest_url)),
                step="creating the volume",
            )
        except VolumeAlreadyExistsError:
            return volume
        except VolumeAccountNotEligibleError as exc:
            raise DevError(
                f"this team cannot use volumes ({exc}); deploy with --team-data to "
                "keep the box on the team-shared /data"
            ) from exc
        self.volume_created = True
        self._say(
            f"created volume {volume} for the box's /data. It starts empty: to "
            "keep an existing box's files on the team /data, deploy with "
            "--team-data"
        )
        return volume

    def wait_until_ready(
        self, timeout: int, on_wait: Callable[[str], None] | None = None
    ) -> RunnerInfo | None:
        """Poll until a runner is ready; None on timeout. Raises if the start fails.

        `on_wait` gets the runner state on each change (default: a log line).
        """
        deadline = time.time() + timeout
        last_state = None
        while time.time() < deadline:
            try:
                runners = self.poll_runners()
            except Exception as exc:
                # The deploy is done and the runner is coming up: a poll that
                # cannot connect is skipped, not a reason to give up the box.
                if not is_network_error(exc):
                    raise
                self.poll_failures += 1
                self.poll_error = exc
                time.sleep(POLL_SECONDS)
                continue
            self.poll_error = None
            ready = newest_live(runners)
            if ready:
                return ready[0]
            states = {r.state for r in runners}
            # setup() or the image failed and the platform is backing off retries.
            if RunnerState.CRASH_BACKOFF in states:
                raise DevError(
                    "setup() or the image failed, and the platform is retrying it",
                    title="the box failed to start",
                    fixes=[
                        ("fal dev logs --all", "see why"),
                        ("fal dev down", "if it is still retrying"),
                    ],
                )
            pending = sorted(s.value.lower() for s in states - FINAL_STATES)
            state = ", ".join(pending) or "requested"
            if state != last_state:
                if on_wait:
                    on_wait(state)
                else:
                    log(f"waiting: {state}")
                last_state = state
            time.sleep(POLL_SECONDS)
        return None

    def start(
        self,
        timeout: int,
        machine_type: str | None = None,
        on_wait: Callable[[str], None] | None = None,
    ) -> RunnerInfo:
        """Scale to 1 and wait for a ready runner; cancel the start if none comes up."""
        # Only cancel a start this call made: never stop a box that was already up.
        was_live = bool(newest_live(self.runners()))
        self._begin("runner")
        # After a start that did not come up: True if it was scaled back to 0,
        # False if that failed, None if the box was already up (left running).
        self.start_cancelled = None
        self.poll_failures = 0
        self.poll_error = None
        runner = None
        timed_out = False
        try:
            self.scale(1, machine_type)
            runner = self.wait_until_ready(timeout, on_wait)
            timed_out = runner is None
        finally:
            # Any way out without a ready box (timeout, failed start, Ctrl-C) cancels
            # the start.
            if runner is None and not was_live:
                self.start_cancelled = False  # until the stop is confirmed
                try:
                    stopped = self.stop()
                except Exception:
                    stopped = False
                self.start_cancelled = bool(stopped)
                if not stopped:
                    self._say("the box may still be starting; run: fal dev down")
                elif timed_out and self.poll_error is not None:
                    self._say(
                        f"stopped box {self.app}: could not reach it after "
                        f"{self.poll_failures} failed polls"
                    )
        if runner is not None:
            self._finish("runner")
        if runner is None:
            fixes = [(f"fal dev up --timeout {timeout * 2}", "wait longer")]
            if machine_type and machine_type != "GPU-RTXPRO6000":
                fixes.append(
                    ("fal dev up --machine-type GPU-RTXPRO6000", "another GPU type")
                )
            if self.start_cancelled:
                outcome = "the start was cancelled and the box scaled back to 0"
            elif was_live:
                outcome = "the box was already up and is still running"
            else:
                outcome = "the box may still be starting, and billing"
                fixes.insert(0, ("fal dev down", "make sure it stops"))
            unreachable = (
                f"; the last readiness poll could not connect ({self.poll_error}), "
                f"{self.poll_failures} failed in all"
                if self.poll_error is not None
                else ""
            )
            raise DevError(
                f"no machine became ready within {timeout}s{unreachable}; {outcome}",
                title="no GPU came free",
                fixes=fixes,
            )
        return runner

    def up(
        self,
        timeout: int,
        machine_type: str | None = None,
        on_wait: Callable[[str], None] | None = None,
        starship: bool = False,
        take_over: bool = False,
        secrets: list[str] | None = None,
        set_env: dict[str, str] | None = None,
    ) -> RunnerInfo:
        from fal.dev.guard import check_limits

        check_limits(machine_type)
        exists = self.app_info() is not None
        if starship and exists:
            self._say(
                "the box already exists; use `fal dev deploy --starship` to change it"
            )
        if (secrets or set_env) and exists:
            self._say(
                "the box already exists; use `fal dev deploy --secret` or "
                "`--set-env` to change what it receives"
            )
        if not exists:
            if self.progress is None:
                log(
                    f"{self.app} does not exist yet; "
                    "deploying it with the default image ..."
                )
            summary = self.deploy(
                machine_type=machine_type,
                starship=starship,
                hint=False,
                take_over=take_over,
                secrets=secrets,
                set_env=set_env,
            )
            if self.progress is None:
                log(summary)
        live = newest_live(self.runners())
        if live and machine_type and live[0].machine_type != machine_type:
            raise DevError(
                f"the box is running on {live[0].machine_type}. "
                f"Run `fal dev down` first to switch to {machine_type}."
            )
        self.check_admin_key()
        return self.start(timeout, machine_type, on_wait)

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
            raise DevError(
                f"{self.app} is not running",
                title="the box is not running",
                fixes=[("fal dev up", "start it")],
            )
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

    def clear_hold(self) -> bool:
        """Drop an active hold so idle shutdown applies again; False if none was set."""
        live = newest_live(self.runners())
        if not live:
            raise DevError(
                f"{self.app} is not running",
                title="the box is not running",
                fixes=[("fal dev up", "start it")],
            )
        # Exits HOLD_KEPT_EXIT when no hold was in effect (missing or expired file).
        script = (
            'old=$(cat "$0" 2>/dev/null || echo 0); rm -f "$0"; '
            f'[ "${{old%.*}}" -gt "$(date +%s)" ] 2>/dev/null || exit {config.HOLD_KEPT_EXIT}'
        )
        code, output = self.run(
            live[0].runner_id, ["sh", "-c", script, config.HOLD_FILE]
        )
        if code not in (0, config.HOLD_KEPT_EXIT):
            raise DevError(
                f"could not clear the hold: {output.strip() or f'exit {code}'}"
            )
        return code == 0

    def hold_until(self, timeout: float = 10) -> float | None:
        """When the active hold ends (epoch seconds); None if there is none.

        Reads the hold file on the live runner, so a box that is not up has none.
        """
        live = newest_live(self.runners())
        if not live:
            return None
        script = 'cat "$0" 2>/dev/null || echo 0'
        code, output = self.run(
            live[0].runner_id,
            ["sh", "-c", script, config.HOLD_FILE],
            timeout=timeout,
        )
        try:
            if code != 0:
                raise ValueError(output)
            until = float(output.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise DevError(
                f"could not read the hold: {output.strip()[:100]!r}"
            ) from None
        return until if until > time.time() else None

    def destroy(self) -> tuple[list[str], bool]:
        """Kill the runners and delete the app. Its volume is kept: there is no API.

        Returns the killed runner ids and whether the app still existed.
        """
        from fal.dev.app.deploy import delete_app

        if not self.app.startswith(config.APP_PREFIX):
            raise DevError(f"refusing to delete {self.app!r}: not a dev box")
        return delete_app(self.app, self.host, self.account.fal_team)

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
            log(f"fal dev: starting box {self.app} (about {COLD_START_MINUTES} min)")
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
        "    # A stopped box starts on connect, which takes about 3 minutes: wait for it\n"
        "    # (or run `fal dev up` first). Longer than ssh-proxy's own 900 s wait, so its messages come first.\n"
        "    ConnectTimeout 960\n"
        "    ServerAliveInterval 30\n"
        "    # One connection serves the next ones: each new connection costs ~6 s.\n"
        "    ControlMaster auto\n"
        "    ControlPath ~/.ssh/fal-dev-%C\n"
        "    ControlPersist 2m\n"
        "    # Connecting authorizes the public keys in ~/.ssh on the box.\n"
        "    # The box keeps its host key on /data, so it only changes if the\n"
        "    # workspace is reset.\n"
        "    StrictHostKeyChecking accept-new"
    )
