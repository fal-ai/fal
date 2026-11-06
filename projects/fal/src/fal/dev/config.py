"""Settings shared by `fal dev` (your laptop) and the box app (the box).

Standard library only, so the CLI starts fast.
"""

from __future__ import annotations

import re
import shlex

# Box apps are named `devbox-<name>` and keep their workspace in
# /data/devboxes/<name>, the same as boxes made with the standalone devbox tool.
APP_PREFIX = "devbox-"
DEFAULT_GPU = "GPU-H100"
DEFAULT_NUM_GPUS = 1
DEFAULT_IDLE_MINUTES = 60
# A GPU above this utilization counts as activity (an unattended job).
GPU_BUSY_PCT = 10

# Team secret holding an ADMIN key, used only by the idle watchdog and /wake:
# the key the runtime gives an app is not allowed to scale apps.
ADMIN_KEY_SECRET = "DEVBOX_ADMIN_KEY"

# Default box name when --name is not given (else your OS username).
ENV_DEFAULT_NAME = "FAL_DEV_NAME"

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

# Claude Code's config, login and transcripts, under the workspace's persisted home.
CLAUDE_CONFIG_SUBDIR = ".home/.claude"
# tmux session running `claude remote-control` on a box deployed with --claude.
CLAUDE_TMUX_SESSION = "claude"
# Where its sessions start. The Claude app names the box after this folder.
CLAUDE_WORKDIR = "/work/fal-devbox"

WORKSPACE_ROOT = "/data/devboxes"
HOLD_FILE = "/tmp/devbox-hold-until"
# Touched after each restore and save; a file under /work newer than it is a change.
SAVE_STAMP = "/tmp/devbox-save-stamp"
# Shown by every new shell when idle shutdown cannot work.
WARNING_FILE = "/tmp/devbox-idle-shutdown-warning"
# The idle watchdog's state each tick, for `fal dev dash`.
IDLE_STATE_FILE = "/tmp/fal-dev-idle.json"
MAX_HOLD_HOURS = 168

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
HOT_SERVER = "/usr/local/lib/devbox-hot/hot_server.py"

NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


def app_name(name: str) -> str:
    return f"{APP_PREFIX}{name}"


# What hold_command prints when a longer hold is already in place.
HOLD_KEPT = "hold-kept"


def hold_command(until: int, hold_file: str = HOLD_FILE) -> str:
    """Shell that holds the box until `until`; prints HOLD_KEPT if a longer one stands.

    Never shortens a longer hold, e.g. one set for an unattended job.
    """
    path = shlex.quote(hold_file)
    return (
        f"old=$(cat {path} 2>/dev/null || echo 0); "
        f'if [ "${{old%.*}}" -gt {until} ] 2>/dev/null; then echo {HOLD_KEPT}; '
        f"else echo {until} > {path}; fi"
    )
