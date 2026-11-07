"""A persistent GPU dev box running as a fal serverless app.

One app per developer (`devbox-<name>`), deployed and driven by `fal dev`. Its
endpoints are only a wake-up call and the routes of `fal dev hot`; the app
exists so that `min_concurrency=1` keeps one GPU runner alive that you shell into.

Everything under /data survives a restart: the box's own volume, or with
--team-data the team-shared /data. The container's local disk is fresh on every
start, so the workspace, dotfiles and shell history live in
/data/devboxes/<name>. /work is on the fast local disk and is kept in the
workspace as one snapshot file.
"""

import asyncio
import contextlib
import hmac
import json
import os
import re
import secrets as token_secrets
import shlex
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from string import Template
from typing import Iterator

from fastapi import Request, Response

import fal
from fal.api.client import SyncServerlessClient
from fal.container import ContainerImage
from fal.dev import config
from fal.dev.remote import signals

# Evaluated on your laptop during `fal dev deploy` (see config.py) and shipped
# with the app (fal itself ships by value). Always deploy through `fal dev deploy`,
# not `fal deploy`.
BOX_NAME = os.environ.get(config.ENV_NAME, "default")
IDLE_MINUTES = int(os.environ.get(config.ENV_IDLE_MINUTES, config.DEFAULT_IDLE_MINUTES))
CLAUDE_ENABLED = os.environ.get(config.ENV_CLAUDE) == "1"
# The box's volume, mounted at /data; None for the team-shared /data.
VOLUME = os.environ.get(config.ENV_VOLUME) or None
# Your starship.toml, when deployed with --starship.
STARSHIP_TOML = os.environ.get(config.ENV_STARSHIP) or None

APP_NAME = config.app_name(BOX_NAME)
WORKSPACE = Path(config.WORKSPACE_ROOT) / BOX_NAME
LOCAL_WORK = Path(config.LOCAL_WORK)
SNAPSHOT = WORKSPACE / config.SNAPSHOT_FILE
LEASE = WORKSPACE / config.LEASE_FILE
# Serializes this runner's lease writes (heartbeat vs. release).
LEASE_LOCK = Path("/tmp/devbox-lease.lock")
SAVE_STAMP = Path(config.SAVE_STAMP)
WARNING_FILE = Path(config.WARNING_FILE)
# Dotfiles and config directories that live in the workspace and get
# symlinked into $HOME on start.
PERSISTED_HOME_FILES = [".gitconfig"]
PERSISTED_HOME_DIRS = [
    ".ssh",
    ".config/gh",
    ".cursor-server/extensions",
    ".vscode-server/extensions",
]
SHUTDOWN_RETRY_SECONDS = 600
GPU_FAILURES_BEFORE_WARNING = 3
CLAUDE_CONFIG_DIR = WORKSPACE / config.CLAUDE_CONFIG_SUBDIR
# The last screen of `claude remote-control` when it exits, for the box's logs.
CLAUDE_EXIT_LOG = Path("/tmp/devbox-claude-exit.log")
# A server that exits sooner than this (not logged in, setup prompt refused) is
# retried slowly.
CLAUDE_QUICK_EXIT_SECONDS = 60
CLAUDE_RETRY_SECONDS = 300
JSON_TYPE = "application/json"

DOCKERFILE = Path(
    os.environ.get(config.ENV_DOCKERFILE) or Path(__file__).with_name("Dockerfile")
)
# cloudpickle needs the box's Python to match the one that deploys.
PYTHON_VERSION = os.environ.get(config.ENV_PYTHON_VERSION, "3.11")
FAL_SPEC = os.environ.get(config.ENV_FAL_SPEC, "fal")

# Read on your laptop at deploy time and written to the box by setup().
HOT_SERVER_SOURCE = Path(__file__).with_name("hot_server.py").read_text()
PROBE_FILES = {
    "probe.py": Path(__file__).with_name("probe.py").read_text(),
    "signals.py": Path(signals.__file__).read_text(),
    "config.py": Path(config.__file__).read_text(),
}
SCRIPTS = Path(__file__).with_name("scripts")
SAVE_SCRIPT = (SCRIPTS / "devbox-save.sh").read_text()
HOT_SCRIPT = (SCRIPTS / "devbox-hot.sh").read_text()
BASHRC_BLOCK = (SCRIPTS / "bashrc.sh").read_text()
BASHRC_ENV_BLOCK = (SCRIPTS / "bashrc_env.sh").read_text()
SSHD_TEMPLATE = (SCRIPTS / "sshd_config").read_text()
# Last request to a /hot route; counts as activity for idle shutdown.
_last_hot_request = 0.0


def _fill(template: str, **values: object) -> str:
    """The template with its `$NAME` placeholders filled in; other `$` stay."""
    return Template(template).safe_substitute(values)


def _write_atomic(path: Path, text: str) -> None:
    """Replace a file so that no reader sees it half written."""
    tmp = path.with_name(f".{path.name}.{socket.gethostname()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _install_script(path: str, text: str) -> None:
    script = Path(path)
    script.write_text(text)
    script.chmod(0o755)


def _user_env() -> dict[str, str]:
    """The environment for what runs your code, without the watchdog's ADMIN key.

    Only a speed bump: everything on the box runs as root and can read the app's
    environment from /proc, so the box is one trust domain.
    """
    return {k: v for k, v in os.environ.items() if k != config.ADMIN_KEY_SECRET}


def _kill_tmux(session: str) -> None:
    subprocess.run(
        ["tmux", "kill-session", "-t", session], capture_output=True, check=False
    )


def _held() -> bool:
    """True while a `fal dev hold` is in effect."""
    return (signals.hold_until(config.HOLD_FILE) or 0) > time.time()


_gpu_failures = 0


def _gpu_activity() -> str | None:
    """Why the GPUs count as activity, if they do.

    A busy GPU is an unattended training run. A failed query counts too: it must
    never be what shuts down a busy box. Failing for a while is warned about once,
    since the box then never idles out.
    """
    global _gpu_failures  # noqa: PLW0603
    readings, error = signals.gpus()
    if error:
        _gpu_failures += 1
        if _gpu_failures == GPU_FAILURES_BEFORE_WARNING:
            _warn(
                f"GPU status is unknown ({error}), so idle shutdown is off until "
                "it can be read. Run `fal dev down` when done."
            )
        return "GPU status unknown"
    if _gpu_failures >= GPU_FAILURES_BEFORE_WARNING:
        _clear_warning("GPU status is unknown")
    _gpu_failures = 0
    return "GPU busy" if signals.gpu_busy(readings, config.GPU_BUSY_PCT) else None


def _claude_server() -> None:
    """Keep `claude remote-control` running in tmux, listing the box in the Claude app.

    It runs in server mode, so new sessions can be started from the app. It exits
    on its own after a long network outage, and right away until Claude Code is
    logged in, so it is restarted, slowly after a quick exit.
    """
    session = config.CLAUDE_TMUX_SESSION
    # No pre-created session: after a restart it tries to reconnect the previous
    # runner's session, which the server side has already dropped, and fails.
    # Sessions are started from the Claude app instead.
    server = (
        f"claude remote-control --name {shlex.quote(f'devbox {BOX_NAME}')} "
        "--spawn same-dir --no-create-session-in-dir"
    )
    # The pane saves its own last screen when the server exits, to explain why.
    # The server is restarted by stopping claude, never by killing the tmux
    # session: that orphans claude, and nothing in the container reaps it.
    command = (
        f"{server}; code=$?; tmux capture-pane -p -S -40 > {CLAUDE_EXIT_LOG}; "
        f"echo exit $code >> {CLAUDE_EXIT_LOG}"
    )
    started = None
    while True:
        try:
            if not signals.tmux_session(session):
                if started is not None:
                    log = (
                        CLAUDE_EXIT_LOG.read_text().strip()
                        if CLAUDE_EXIT_LOG.exists()
                        else ""
                    )
                    print(
                        f"[devbox] claude remote-control stopped:\n{log[-1500:]}",
                        flush=True,
                    )
                    # Only the server's own exit writes the log; a killed tmux
                    # session restarts at once.
                    exited_itself = (
                        CLAUDE_EXIT_LOG.exists()
                        and CLAUDE_EXIT_LOG.stat().st_mtime >= started
                    )
                    if (
                        exited_itself
                        and time.time() - started < CLAUDE_QUICK_EXIT_SECONDS
                    ):
                        _warn(
                            "claude remote-control exits right away. Log in with "
                            "`claude auth login`, then "
                            f"`tmux attach -t {session}` to answer its first-run "
                            "questions"
                        )
                        time.sleep(CLAUDE_RETRY_SECONDS)
                Path(config.CLAUDE_WORKDIR).mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        "tmux",
                        "new-session",
                        "-d",
                        "-s",
                        session,
                        "-c",
                        config.CLAUDE_WORKDIR,
                        command,
                    ],
                    env=_user_env(),
                    check=True,
                )
                started = time.time()
            elif (
                started is not None
                and time.time() - started > CLAUDE_QUICK_EXIT_SECONDS
            ):
                # Up for a while: drop a stale "exits right away" warning.
                _clear_warning("claude remote-control exits right away")
        except Exception as exc:  # never let the supervisor die silently
            print(f"[devbox] claude supervisor error: {exc!r}", flush=True)
        time.sleep(10)


def _admin_client() -> SyncServerlessClient | None:
    """A fal client signed in with the ADMIN key; None if the key is not set.

    The key the runtime gives the app is not allowed to scale apps.
    """
    admin_key = os.environ.get(config.ADMIN_KEY_SECRET)
    if not admin_key:
        return None
    return SyncServerlessClient(api_key=admin_key)


def _scale_self(min_concurrency: int) -> str | None:
    """Set this box's min_concurrency with the ADMIN key; the error, if any."""
    client = _admin_client()
    if client is None:
        return f"the {config.ADMIN_KEY_SECRET} secret is not set"
    try:
        client.apps.scale(APP_NAME, min_concurrency=min_concurrency)
    except Exception as exc:
        return str(exc)[-500:] or repr(exc)
    return None


def _warn(message: str) -> None:
    print(f"[devbox] {message}", flush=True)
    line = f"devbox: {message}\n"
    # Repeated warnings (shutdown retries) are shown once.
    if line not in (WARNING_FILE.read_text() if WARNING_FILE.exists() else ""):
        with WARNING_FILE.open("a") as f:
            f.write(line)


def _clear_warning(prefix: str) -> None:
    """Remove the warnings starting with `prefix` once their cause is gone."""
    if not WARNING_FILE.exists():
        return
    lines = WARNING_FILE.read_text().splitlines(keepends=True)
    kept = [line for line in lines if not line.startswith(f"devbox: {prefix}")]
    if kept != lines:
        WARNING_FILE.write_text("".join(kept))


def _check_admin_key() -> None:
    """At start, confirm the watchdog's key can see this app; warn shells if not."""
    client = _admin_client()
    if client is None:
        _warn(
            f"{config.ADMIN_KEY_SECRET} is not set, so idle shutdown is off. "
            "Run `fal dev down` when done."
        )
        return
    try:
        aliases = {app.alias for app in client.apps.list()}
    except Exception as exc:
        _warn(
            f"{config.ADMIN_KEY_SECRET} is invalid or could not be checked ({exc}); "
            "idle shutdown may not work."
        )
        return
    if APP_NAME not in aliases:
        _warn(
            f"{config.ADMIN_KEY_SECRET} belongs to another team, "
            "so idle shutdown is off."
        )


def _scale_to_zero() -> bool:
    """Drop min_concurrency to 0; keep_alive then reaps this idle runner."""
    error = _scale_self(0)
    if error:
        print(f"[devbox] idle, but could not stop itself: {error}", flush=True)
    return error is None


def _publish_idle_state(last_active: float, reasons: list[str]) -> None:
    """Write the watchdog's view for `fal dev dash`, which cannot see all of it."""
    state = {
        "last_active": last_active,
        "idle_minutes": IDLE_MINUTES,
        "reasons": reasons,
    }
    _write_atomic(Path(config.IDLE_STATE_FILE), json.dumps(state))


def _idle_watchdog() -> None:
    """Scale the box to zero after IDLE_MINUTES without terminal or GPU activity.

    Keeps running after a shutdown so that a box restarted before this runner
    is reaped is still watched.
    """
    last_active = time.time()
    while True:
        time.sleep(config.WATCHDOG_INTERVAL_SECONDS)
        try:
            now = time.time()
            checks = (
                ("hold", _held),
                ("SSH session", lambda: signals.ssh_sessions() > 0),
                ("exec command", lambda: signals.exec_commands() > 0),
            )
            active = [name for name, check in checks if check()]
            if gpu := _gpu_activity():
                active.insert(0, gpu)
            seen = {
                "terminal input": signals.last_tty_input(),
                "hot request": _last_hot_request,
            }
            if CLAUDE_ENABLED:
                # A message from the Claude app or a running turn: Remote Control
                # sessions take no terminal input, so they need their own signal.
                seen["Claude session"] = signals.claude_activity(str(CLAUDE_CONFIG_DIR))
            latest = max(seen, key=lambda name: seen[name])
            if active:
                last_active = now
            elif seen[latest] > last_active:
                active = [latest]
            last_active = max(last_active, seen[latest])
            # Only for the dashboard: never let it get in the way of shutting down.
            with contextlib.suppress(OSError):
                _publish_idle_state(last_active, active)
            idle_for = now - last_active
            if idle_for < IDLE_MINUTES * 60:
                continue
            print(
                f"[devbox] idle for {int(idle_for // 60)} min, shutting down",
                flush=True,
            )
            # Never stop with unsaved work: a failed save is retried like a failed
            # shutdown.
            if _save_work() and _scale_to_zero():
                last_active = now
            else:
                retry_min = SHUTDOWN_RETRY_SECONDS // 60
                _warn(
                    f"idle shutdown failed, retrying every {retry_min} min. "
                    "Run `fal dev down` when done"
                )
                # Count as idle again SHUTDOWN_RETRY_SECONDS from now.
                last_active = now - IDLE_MINUTES * 60 + SHUTDOWN_RETRY_SECONDS
        except Exception as exc:  # never let the watchdog die silently
            print(f"[devbox] watchdog error: {exc!r}", flush=True)


def _lease_owner() -> str:
    """This runner's lease identity: the container hostname, unique per runner."""
    return socket.gethostname()


def _write_lease(state: str) -> None:
    _write_atomic(LEASE, f"{_lease_owner()} {state} {time.time():.0f}\n")


def _claim_lease() -> None:
    """Wait for the previous runner to release the lease (or go stale), then take it.

    Without this, a runner that starts while the previous one is still saving
    restores the older snapshot and later overwrites the newer one.
    """
    deadline = time.time() + config.LEASE_WAIT_SECONDS
    announced = False
    while True:
        lease = signals.read_lease(LEASE)
        if lease is None:
            break
        owner, state, at = lease
        if (
            owner == _lease_owner()
            or state == "released"
            or time.time() - at > config.LEASE_STALE_SECONDS
        ):
            break
        if time.time() > deadline:
            _warn(
                f"runner {owner} still holds {LEASE}; taking over, "
                "its later saves are refused"
            )
            break
        if not announced:
            print(
                f"[devbox] waiting for runner {owner} to finish saving {LOCAL_WORK}",
                flush=True,
            )
            announced = True
        time.sleep(5)
    _write_lease("active")


@contextlib.contextmanager
def _lease_lock() -> Iterator[None]:
    # Imported here: the laptop loads this module to deploy it, and fcntl is POSIX only.
    import fcntl

    with LEASE_LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _lease_heartbeat() -> None:
    """Keep the lease fresh while this runner owns it; stop once released or lost."""
    while True:
        time.sleep(config.LEASE_HEARTBEAT_SECONDS)
        try:
            with _lease_lock():
                lease = signals.read_lease(LEASE)
                if lease is None or lease[0] != _lease_owner():
                    _warn(
                        f"another runner took over {LOCAL_WORK}; "
                        "this runner no longer saves it"
                    )
                    return
                if lease[1] != "active":
                    return
                _write_lease("active")
        except OSError as exc:
            print(f"[devbox] lease heartbeat failed: {exc}", flush=True)


def _release_lease() -> None:
    """Let a waiting runner start: called after the final save."""
    try:
        with _lease_lock():
            lease = signals.read_lease(LEASE)
            if lease and lease[0] == _lease_owner():
                _write_lease("released")
    except OSError as exc:
        print(f"[devbox] could not release {LEASE}: {exc}", flush=True)


def _restore_work() -> None:
    """Restore /work from the snapshot, and install `devbox-save`.

    An unreadable snapshot is moved aside, never overwritten by the next save.
    """
    LOCAL_WORK.mkdir(parents=True, exist_ok=True)
    if SNAPSHOT.exists():
        start = time.time()
        extract = (
            f"zstd -dc {shlex.quote(str(SNAPSHOT))} | tar -C {LOCAL_WORK.parent} -xpf -"
        )
        result = subprocess.run(
            ["bash", "-o", "pipefail", "-c", extract],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print(
                f"[devbox] restored {LOCAL_WORK} in {time.time() - start:.0f}s",
                flush=True,
            )
        else:
            kept = SNAPSHOT.with_name(f"{SNAPSHOT.name}.unreadable-{int(time.time())}")
            SNAPSHOT.rename(kept)
            _warn(
                f"could not restore {LOCAL_WORK} ({result.stderr.strip()[-200:]}); "
                f"the snapshot is kept at {kept}"
            )
    SAVE_STAMP.touch()
    _install_script(
        config.SAVE_COMMAND,
        _fill(
            SAVE_SCRIPT,
            WORK=LOCAL_WORK,
            WORK_PARENT=LOCAL_WORK.parent,
            WORK_NAME=LOCAL_WORK.name,
            SNAPSHOT=shlex.quote(str(SNAPSHOT)),
            LEASE=shlex.quote(str(LEASE)),
            STAMP=SAVE_STAMP,
            INTERVAL=config.SAVE_INTERVAL_MINUTES,
        ),
    )


def _save_work() -> bool:
    """Run `devbox-save`; True once /work is saved (or unchanged)."""
    try:
        result = subprocess.run(
            [config.SAVE_COMMAND],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _warn(f"could not save {LOCAL_WORK} ({exc})")
        return False
    output = f"{result.stdout} {result.stderr}".strip()
    if result.returncode != 0:
        _warn(f"could not save {LOCAL_WORK}: {output[-300:]}")
        return False
    print(f"[devbox] {output}", flush=True)
    return True


def _save_loop() -> None:
    """Save /work every few minutes, so a crash loses at most that much work."""
    while True:
        time.sleep(config.SAVE_INTERVAL_MINUTES * 60)
        _save_work()


def _prepare_home() -> None:
    """Point dotfiles and shell history at the workspace, and start shells there."""
    home = Path.home()
    persisted_home = WORKSPACE / ".home"
    # Set here so every shell, SSH session and the Claude server share one login
    # and history.
    CLAUDE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["CLAUDE_CONFIG_DIR"] = str(CLAUDE_CONFIG_DIR)
    for rel in PERSISTED_HOME_DIRS:
        (persisted_home / rel).mkdir(parents=True, exist_ok=True)
    (persisted_home / ".ssh").chmod(0o700)
    for rel in PERSISTED_HOME_FILES:
        # Else the link dangles until something writes the file, and git cannot
        # tell a missing config from a broken one.
        (persisted_home / rel).touch()
    for rel in PERSISTED_HOME_FILES + PERSISTED_HOME_DIRS:
        target = persisted_home / rel
        link = home / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        # Leave anything the image itself put there alone.
        if link.is_symlink() or not link.exists():
            link.unlink(missing_ok=True)
            link.symlink_to(target)

    # Easy to find: the real file lives with the box on /data.
    (WORKSPACE / ".devboxrc").touch()
    rc_link = home / ".devboxrc"
    if rc_link.is_symlink() or not rc_link.exists():
        rc_link.unlink(missing_ok=True)
        rc_link.symlink_to(WORKSPACE / ".devboxrc")

    bashrc = home / ".bashrc"
    existing = bashrc.read_text() if bashrc.exists() else ""
    if "# >>> devbox-env >>>" not in existing:
        # At the top: stock .bashrc files return early for non-interactive shells.
        env_block = _fill(
            BASHRC_ENV_BLOCK,
            WORKSPACE=WORKSPACE,
            WORK=LOCAL_WORK,
            CLAUDE_DIR=CLAUDE_CONFIG_DIR,
        )
        existing = f"{env_block}\n{existing}"
        bashrc.write_text(existing)
    if "# >>> devbox >>>" not in existing:
        block = _fill(
            BASHRC_BLOCK,
            WORKSPACE=WORKSPACE,
            WORK=LOCAL_WORK,
            CLAUDE_DIR=CLAUDE_CONFIG_DIR,
            WARNING_FILE=WARNING_FILE,
            BOX_NAME=BOX_NAME,
        )
        bashrc.write_text(f"{existing}\n{block}")
    if STARSHIP_TOML:
        # Off the startup path: a slow download must not delay the box. Shells that
        # open before it finishes get the default prompt.
        threading.Thread(
            target=_setup_starship, args=(home, STARSHIP_TOML), daemon=True
        ).start()


def _setup_starship(home: Path, toml: str) -> None:
    """Install starship and your config; the default prompt stays if that fails."""
    try:
        subprocess.run(
            [
                "sh",
                "-c",
                "command -v starship >/dev/null 2>&1 || "
                f"curl -fsSL {config.STARSHIP_INSTALL_URL} | sh -s -- -y",
            ],
            check=True,
            capture_output=True,
            # Not the watchdog's ADMIN key; the installer is an unpinned script.
            env=_user_env(),
            timeout=config.STARSHIP_INSTALL_TIMEOUT_SECONDS,
        )
        (home / ".config").mkdir(parents=True, exist_ok=True)
        _write_atomic(home / ".config" / "starship.toml", toml)
    except Exception as exc:
        _warn(f"starship is unavailable, using the default prompt ({exc})")


def _sshd_config(host_key: Path) -> str:
    """sshd settings for the `sshd -i` sessions of `fal dev ssh-proxy`."""
    # SSH sessions start with a clean environment; carry over the image's
    # (PATH, CUDA, ...) so they match `fal dev shell`. Never the watchdog's key.
    skip = {config.ADMIN_KEY_SECRET, "HOME", "PWD", "OLDPWD", "SHLVL", "_", "TERM"}
    env = [
        f'"{k}={v}"'
        for k, v in sorted(os.environ.items())
        if k not in skip
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)
        and not re.search(r'["\\\n]', v)
    ]
    return _fill(SSHD_TEMPLATE, HOST_KEY=host_key, ENV=" ".join(env))


def _prepare_ssh() -> None:
    """Write the sshd config that `fal dev ssh-proxy` starts sshd with.

    The host key is kept with the persisted ~/.ssh so it stays the same across
    restarts, outside the workspace folder people open and commit from.
    """
    try:
        # sshd's privilege separation dir
        Path("/run/sshd").mkdir(parents=True, exist_ok=True)
        host_key = WORKSPACE / ".home" / ".ssh" / "devbox_host_ed25519"
        # Also redoes a key cut off mid-write.
        if not host_key.with_suffix(".pub").exists():
            host_key.unlink(missing_ok=True)
            subprocess.run(
                ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(host_key)],
                check=True,
            )
        host_key.chmod(0o600)
        sshd_config = Path(config.SSHD_CONFIG)
        sshd_config.parent.mkdir(parents=True, exist_ok=True)
        sshd_config.touch(mode=0o600)
        sshd_config.chmod(0o600)  # it holds the box's environment
        sshd_config.write_text(_sshd_config(host_key))
    except Exception as exc:
        _warn(f"SSH access is unavailable ({exc})")


def _prepare_tools() -> None:
    """Install the hot server and the dashboard probe; write the runner's /hot token."""
    server = Path(config.HOT_SERVER)
    server.parent.mkdir(parents=True, exist_ok=True)
    server.write_text(HOT_SERVER_SOURCE)
    probe_dir = Path(config.PROBE_DIR)
    probe_dir.mkdir(parents=True, exist_ok=True)
    for name, source in PROBE_FILES.items():
        (probe_dir / name).write_text(source)
    _install_script(
        config.HOT_COMMAND,
        _fill(
            HOT_SCRIPT,
            SERVER=config.HOT_SERVER,
            PORT=config.HOT_PORT,
            RESTART_CODE=config.HOT_RESTART_EXIT_CODE,
        ),
    )
    token = Path(config.HOT_TOKEN_FILE)
    token.touch(mode=0o600)
    token.write_text(token_secrets.token_urlsafe(32))


def _hot_dir(value) -> Path:
    """A project folder under /work; anything else is refused."""
    path = Path(str(value or "")).resolve()
    if path != LOCAL_WORK and LOCAL_WORK not in path.parents:
        raise ValueError(f"the project folder must be under {LOCAL_WORK}")
    return path


def _hot_log(lines: int = 60) -> list[str]:
    try:
        return Path(config.HOT_LOG).read_text(errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def _hot_start(body: dict) -> dict:
    project = _hot_dir(body.get("dir"))
    if not (project / "load.py").exists() or not (project / "infer.py").exists():
        raise ValueError(
            f"{project} needs a load.py and an infer.py (see `fal dev hot --help`)"
        )
    session = config.HOT_TMUX_SESSION
    _kill_tmux(session)
    Path(config.HOT_LOG).write_text("")
    project_arg = shlex.quote(str(project))
    command = f"{config.HOT_COMMAND} {project_arg} 2>&1 | tee -a {config.HOT_LOG}"
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "-c", str(project), command],
        env=_user_env(),
        check=True,
    )
    return {"started": str(project), "tmux_session": session}


def _hot_stop(body: dict) -> dict:
    _kill_tmux(config.HOT_TMUX_SESSION)
    return {"stopped": True}


def _hot_write_files(body: dict) -> dict:
    """Write text files sent from your laptop into a project folder.

    Unchanged files are left alone: rewriting load.py would restart the server.
    """
    project = _hot_dir(body.get("dir"))
    written = []
    for rel, content in (body.get("files") or {}).items():
        target = (project / rel).resolve()
        if project not in target.parents:
            raise ValueError(f"{rel} is outside {project}")
        try:
            if target.read_text() == content:
                continue
        except (OSError, UnicodeDecodeError):
            pass
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(target, content)  # the server never sees a half-written file
        written.append(rel)
    return {"written": written}


# The /hot actions that run on the box itself; any other goes to the hot server.
HOT_ACTIONS = {
    "start": _hot_start,
    "stop": _hot_stop,
    "files": _hot_write_files,
    "log": lambda body: {"log": _hot_log(200)},
}


def _hot_forward(action: str, body: bytes) -> tuple[int, dict, bytes]:
    """Send a request to the hot server on this box."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{config.HOT_PORT}/{action}",
        data=body or b"{}",
        headers={"Content-Type": JSON_TYPE},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()
    except OSError:
        detail = {
            "error": (
                "the hot server is not running (fal dev hot start), or is restarting"
            ),
            "log": _hot_log(200),
        }
        # Not 503: fal takes a runner that answers 503 out of service and replaces it.
        return 409, {"Content-Type": JSON_TYPE}, json.dumps(detail).encode()


class DevBox(fal.App):
    image = ContainerImage.from_dockerfile(
        str(DOCKERFILE),
        build_args={"PYTHON_VERSION": PYTHON_VERSION, "FAL_SPEC": FAL_SPEC},
    )
    machine_type = os.environ.get(config.ENV_MACHINE_TYPE, config.DEFAULT_GPU)
    num_gpus = int(os.environ.get(config.ENV_NUM_GPUS, config.DEFAULT_NUM_GPUS))
    # Opt in to the watchdog's key only. Set it with --not-exposed-by-default
    # so that only apps listing it receive it.
    secrets = [config.ADMIN_KEY_SECRET]
    # The box's own volume replaces the team-shared /data.
    volumes = {config.VOLUME_MOUNT: fal.Volume(VOLUME)} if VOLUME else None

    # Scaled to 1 by `fal dev up`, to 0 by `fal dev down` or the idle watchdog.
    min_concurrency = 0
    max_concurrency = 1
    # How long the runner lingers after min_concurrency drops to 0.
    keep_alive = 120
    # Room for teardown() to save /work when the runner is stopped.
    termination_grace_period_seconds = 300

    def setup(self) -> None:
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        _claim_lease()
        threading.Thread(target=_lease_heartbeat, daemon=True).start()
        _restore_work()
        _prepare_home()
        _check_admin_key()
        _prepare_ssh()
        _prepare_tools()
        threading.Thread(target=_idle_watchdog, daemon=True).start()
        threading.Thread(target=_save_loop, daemon=True).start()
        if CLAUDE_ENABLED:
            threading.Thread(target=_claude_server, daemon=True).start()
        print(
            f"[devbox] ready: workspace={WORKSPACE} "
            f"storage={f'volume {VOLUME}' if VOLUME else 'team /data'} "
            f"idle_minutes={IDLE_MINUTES} "
            f"claude={CLAUDE_ENABLED}",
            flush=True,
        )

    def teardown(self) -> None:
        # Last chance on a graceful stop; `fal dev down` and the watchdog save first.
        _save_work()
        _release_lease()

    # fal apps need at least one endpoint.
    @fal.endpoint("/")
    def status(self) -> dict:
        return {"name": BOX_NAME}

    @fal.endpoint("/wake")
    def wake(self) -> dict:
        """Start the box from anywhere, e.g. a phone shortcut.

        A request to a stopped box starts a runner to serve it; this keeps that
        runner as the box. Send it through queue.fal.run, since a cold start can
        take longer than a direct request waits.
        """
        error = _scale_self(1)
        if error:
            return {"awake": False, "error": error}
        return {"awake": True, "name": BOX_NAME}

    @fal.endpoint("/hot/{action}")
    async def hot(self, action: str, request: Request) -> Response:
        """`fal dev hot`: start the hot server, sync files to it, send it requests."""
        global _last_hot_request  # noqa: PLW0603
        try:
            token = Path(config.HOT_TOKEN_FILE).read_text().strip()
        except OSError:
            token = ""
        given = request.headers.get(config.HOT_TOKEN_HEADER, "").encode()
        if not token or not hmac.compare_digest(given, token.encode()):
            return Response(
                b'{"error": "bad or missing hot token"}', 403, media_type=JSON_TYPE
            )
        _last_hot_request = time.time()
        body = await request.body()
        handler = HOT_ACTIONS.get(action)
        if handler is None:
            status, headers, content = await asyncio.to_thread(
                _hot_forward, action, body
            )
            keep = {k: v for k, v in headers.items() if k.lower().startswith("x-hot-")}
            return Response(
                content,
                status,
                headers=keep,
                media_type=headers.get("Content-Type", JSON_TYPE),
            )
        try:
            result = await asyncio.to_thread(handler, json.loads(body or b"{}"))
        except (
            ValueError,
            TypeError,
            AttributeError,
            OSError,
            subprocess.CalledProcessError,
        ) as exc:
            return Response(
                json.dumps({"error": str(exc)}).encode(), 400, media_type=JSON_TYPE
            )
        return Response(json.dumps(result).encode(), media_type=JSON_TYPE)
