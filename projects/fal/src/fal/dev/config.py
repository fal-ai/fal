"""Names and settings shared by `fal dev` on your laptop and on the box.

Standard library only: the dashboard's probe imports this file on the box.
"""

from __future__ import annotations

import hashlib
import re
import shlex

# Box apps are named `devbox-<name>` and keep their workspace in
# /data/devboxes/<name>. /data is the box's own volume (also `devbox-<name>`);
# with --team-data it is the team-shared /data instead.
APP_PREFIX = "devbox-"
DEFAULT_GPU = "GPU-RTXPRO6000"
DEFAULT_NUM_GPUS = 1
DEFAULT_IDLE_MINUTES = 60
# A GPU above this utilization counts as activity (an unattended job).
GPU_BUSY_PCT = 10

# Team secret holding an ADMIN key, used only by the idle watchdog and /wake:
# the key the runtime gives an app is not allowed to scale apps.
ADMIN_KEY_SECRET = "DEVBOX_ADMIN_KEY"

# Default box name when --name is not given (else your OS username).
ENV_DEFAULT_NAME = "FAL_DEV_NAME"

# Guardrails for teams sharing one account. The owner is stamped on each deploy
# as a revision annotation, which the next deployer reads back from the app's
# metadata. FAL_DEV_OWNER overrides who you are (for CI, or to test the guard).
OWNER_ANNOTATION = "devbox_owner"
ENV_OWNER = "FAL_DEV_OWNER"
# Client-side limits on deploy: a comma list of machine types, and a GPU count.
ENV_ALLOWED_MACHINES = "FAL_DEV_ALLOWED_MACHINES"
ENV_MAX_GPUS = "FAL_DEV_MAX_GPUS"

# Written by the box at start; `fal dev ssh-proxy` runs `sshd -i` with it.
SSHD_CONFIG = "/tmp/devbox-ssh/sshd_config"

# `fal dev deploy` passes its settings to box_app.py through these environment
# variables, because fal evaluates the app's module and class attributes on the
# machine that runs the deploy.
ENV_NAME = "DEVBOX_NAME"
ENV_MACHINE_TYPE = "DEVBOX_MACHINE_TYPE"
ENV_NUM_GPUS = "DEVBOX_NUM_GPUS"
ENV_IDLE_MINUTES = "DEVBOX_IDLE_MINUTES"
ENV_CLAUDE = "DEVBOX_CLAUDE"
ENV_DOCKERFILE = "DEVBOX_DOCKERFILE"
ENV_PYTHON_VERSION = "DEVBOX_PYTHON_VERSION"
ENV_FAL_SPEC = "DEVBOX_FAL_SPEC"
# The contents of your starship.toml, set by --starship; empty for the default prompt.
ENV_STARSHIP = "DEVBOX_STARSHIP"
STARSHIP_INSTALL_URL = "https://starship.rs/install.sh"
STARSHIP_INSTALL_TIMEOUT_SECONDS = 120
# The box's volume name; empty for the team-shared /data.
ENV_VOLUME = "DEVBOX_VOLUME"
# The fal secrets the box receives (comma-separated names, never values) and the
# plain environment variables it sets (a JSON object).
ENV_SECRETS = "DEVBOX_SECRETS"
ENV_USER_ENV = "DEVBOX_USER_ENV"
# Written by the box at start (root only): `export NAME='value'` for every secret
# and variable above. New shells, `fal dev exec` and ssh sessions source it.
USER_ENV_FILE = "/tmp/devbox-user-env"
# The names in that file, as JSON, for `fal dev status` and `fal dev dash`.
USER_ENV_NAMES_FILE = "/tmp/devbox-user-env.json"
ENV_VAR_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Claude Code's config, login and transcripts, under the workspace's persisted home.
CLAUDE_CONFIG_SUBDIR = ".home/.claude"
# tmux session running `claude remote-control` on a box deployed with --claude.
CLAUDE_TMUX_SESSION = "claude"
# Where its sessions start; the Claude app names the box after this folder.
CLAUDE_WORKDIR = "/work/fal-devbox"

WORKSPACE_ROOT = "/data/devboxes"
VOLUME_MOUNT = "/data"
# The controller's limit on volume names.
VOLUME_NAME_MAX = 32
HOLD_FILE = "/tmp/devbox-hold-until"
# Touched after each restore and save; a file under /work newer than it (by mtime
# or ctime) is a change.
SAVE_STAMP = "/tmp/devbox-save-stamp"
# Shown by every new shell when idle shutdown cannot work.
WARNING_FILE = "/tmp/devbox-idle-shutdown-warning"
# The idle watchdog's state each tick, for `fal dev dash`.
IDLE_STATE_FILE = "/tmp/fal-dev-idle.json"
WATCHDOG_INTERVAL_SECONDS = 60
MAX_HOLD_HOURS = 168
# Exit status of the hold script when a longer hold is already in place.
HOLD_KEPT_EXIT = 3

# Fast working directory on the container's local disk. /data is slow for
# thousands of small files (git repos), so /work is saved to the workspace as a
# single snapshot file and restored on start.
LOCAL_WORK = "/work"
SNAPSHOT_FILE = "work.tar.zst"
SAVE_INTERVAL_MINUTES = 15
# Written by the box at start; `fal dev down` and `fal dev deploy` run it first.
SAVE_COMMAND = "/usr/local/bin/devbox-save"
# Names the one runner that may write the snapshot. A runner that starts while
# the previous one is still saving (a quick restart, a preempted runner's
# replacement) waits for it to release the lease or stop heartbeating.
LEASE_FILE = "work.lease"
LEASE_HEARTBEAT_SECONDS = 15
LEASE_STALE_SECONDS = 60
# Longer than the app's termination grace period, so a stopping runner can save.
LEASE_WAIT_SECONDS = 360

# `fal dev hot`: a server on the box keeps a model loaded and reloads the code
# around it. The box's /hot routes need a per-runner token, read over runner
# exec (an admin-only path), so a plain API key that can call the app cannot
# run code on the box.
HOT_PORT = 7860
HOT_TOKEN_FILE = "/tmp/devbox-hot-token"
HOT_TOKEN_HEADER = "X-Devbox-Hot-Token"
HOT_TMUX_SESSION = "hot"
HOT_LOG = "/tmp/devbox-hot.log"
HOT_COMMAND = "/usr/local/bin/devbox-hot"
# The hot server's exit status when it must restart, e.g. to reload the weights.
HOT_RESTART_EXIT_CODE = 3
HOT_SERVER = "/usr/local/lib/devbox-hot/hot_server.py"

# What `fal dev dash` runs on the box: probe.py, with the files it imports.
PROBE_DIR = "/usr/local/lib/devbox-probe"
PROBE_END = "__fal_dev_dash_end__"
# What the dashboard's probe command prints when probe.py is not installed.
PROBE_MISSING = "__fal_dev_probe_missing__"

NAME_MAX = 40
NAME_PATTERN = re.compile(rf"^[a-z0-9][a-z0-9-]{{0,{NAME_MAX - 1}}}$")


def app_name(name: str) -> str:
    return f"{APP_PREFIX}{name}"


def volume_name(name: str) -> str:
    """The box's volume: `devbox-<name>`, shortened with a hash past the limit."""
    full = app_name(name)
    if len(full) <= VOLUME_NAME_MAX:
        return full
    digest = hashlib.sha1(name.encode()).hexdigest()[:6]
    keep = VOLUME_NAME_MAX - len(APP_PREFIX) - len(digest) - 1
    return f"{APP_PREFIX}{name[:keep].rstrip('-')}-{digest}"


def check_env_name(name: str, what: str) -> str:
    """`name` if the box can export it; else a ValueError saying why."""
    if not ENV_VAR_PATTERN.fullmatch(name):
        raise ValueError(f"{what} {name!r} is not a valid environment variable name")
    if name == ADMIN_KEY_SECRET:
        raise ValueError(
            f"{name} is the idle watchdog's key and never reaches your code"
        )
    return name


def user_env_script(env: dict[str, str]) -> str:
    """Shell lines that export `env`, for sh and bash; never the ADMIN key."""
    return "".join(
        f"export {name}={shlex.quote(value)}\n"
        for name, value in env.items()
        if name != ADMIN_KEY_SECRET and ENV_VAR_PATTERN.fullmatch(name)
    )
