"""What a box is doing, read from the operating system.

Shared by the idle watchdog (box_app.py) and the dashboard's probe (probe.py).
Standard library only: the probe runs on the box's plain python3.
"""

import glob
import os
import subprocess

# First line of the script `fal dev exec` runs; marks a running exec command.
EXEC_TAG = "# fal-dev-exec"
# Cancelling `fal dev exec` leaves its command running, so an exec older than this
# no longer counts as activity: a forgotten job must not keep the box up forever.
EXEC_MAX_AGE_SECONDS = 6 * 3600

GPU_FIELDS = [
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "power.limit",
]


def num(value):
    """A number from text; an int when it is whole, None if it is not a number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def last_tty_input():
    """Time of the newest keyboard input on any pseudo-terminal, 0.0 if none.

    Only input counts: a forgotten terminal that keeps printing (htop, tail -f)
    must not keep the box alive. Reading from a terminal updates its atime, and
    stat() does not.
    """
    latest = 0.0
    for pts in glob.glob("/dev/pts/[0-9]*"):
        try:
            latest = max(latest, os.stat(pts).st_atime)
        except OSError:
            continue
    return latest


def ssh_sessions():
    """How many SSH clients (Cursor, VS Code, a terminal) are connected.

    `fal dev ssh-proxy` runs one `sshd -i` per connection and no sshd daemon
    runs otherwise, so a session is an sshd process whose parent is not an sshd.
    sshd renames itself (`sshd: root@notty`), so this matches the name, not the
    flags. It drops clients that stop answering (a sleeping laptop) within a few
    minutes.
    """
    is_sshd = {}
    parent = {}
    for stat_path in glob.glob("/proc/[0-9]*/stat"):
        pid = stat_path.split("/")[2]
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as f:
                argv0 = f.read().split(b"\0")[0]
            with open(stat_path) as f:
                # The command name may hold spaces; ppid follows the closing paren.
                parent[pid] = f.read().rsplit(")", 1)[1].split()[1]
        except (OSError, IndexError):
            continue
        is_sshd[pid] = argv0.rsplit(b"/", 1)[-1].startswith(b"sshd")
    return sum(
        1 for pid, sshd in is_sshd.items() if sshd and not is_sshd.get(parent[pid])
    )


def _process_age(proc_root, pid):
    """Seconds since process `pid` started, from /proc; None if unreadable."""
    try:
        with open(os.path.join(proc_root, "uptime")) as f:
            uptime = float(f.read().split()[0])
        with open(os.path.join(proc_root, pid, "stat")) as f:
            # starttime is field 22; the command name may hold spaces.
            started = int(f.read().rsplit(")", 1)[1].split()[19])
        return uptime - started / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def exec_commands(proc_root="/proc"):
    """How many `fal dev exec` commands are running: shells run with EXEC_TAG.

    Commands that started more than EXEC_MAX_AGE_SECONDS ago are not counted.
    """
    tag = EXEC_TAG.encode()
    count = 0
    for cmdline in glob.glob(os.path.join(proc_root, "[0-9]*", "cmdline")):
        try:
            with open(cmdline, "rb") as f:
                argv = f.read(256).split(b"\0", 3)
        except OSError:
            continue
        if (
            len(argv) > 2
            and argv[0].rsplit(b"/", 1)[-1] in (b"sh", b"bash", b"dash")
            and argv[1] == b"-c"
            and argv[2].startswith(tag)
        ):
            age = _process_age(proc_root, os.path.basename(os.path.dirname(cmdline)))
            if age is not None and age < EXEC_MAX_AGE_SECONDS:
                count += 1
    return count


def gpus():
    """(one dict per GPU, error): nvidia-smi readings, or why there are none."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(GPU_FIELDS),
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], str(exc)
    if out.returncode != 0:
        return [], (out.stderr or out.stdout).strip()[-200:]
    found = []
    for line in out.stdout.strip().splitlines():
        values = [v.strip() for v in line.split(",")]
        if len(values) != len(GPU_FIELDS):
            continue
        found.append(
            {
                "index": num(values[0]),
                "name": values[1],
                "util_pct": num(values[2]),
                "mem_used_mib": num(values[3]),
                "mem_total_mib": num(values[4]),
                "temp_c": num(values[5]),
                "power_w": num(values[6]),
                "power_limit_w": num(values[7]),
            }
        )
    return found, None


def gpu_busy(readings, threshold):
    """Whether any GPU is at or above `threshold` percent utilization."""
    return any((gpu["util_pct"] or 0) >= threshold for gpu in readings)


def tmux_session(name):
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", name],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def newest_mtime(pattern):
    """Modification time of the newest file matching a glob, 0.0 if none."""
    latest = 0.0
    for path in glob.glob(pattern):
        try:
            latest = max(latest, os.stat(path).st_mtime)
        except OSError:
            continue
    return latest


def hold_until(path):
    """When `fal dev hold` keeps the box up until (epoch seconds), None if no hold."""
    try:
        with open(path) as f:
            return num(f.read().strip())
    except OSError:
        return None


def read_lease(path):
    """(owner, state, heartbeat time) of the /work lease; None if unreadable."""
    try:
        with open(path) as f:
            owner, state, at = f.read().split()
    except (OSError, ValueError):
        return None
    heartbeat = num(at)
    return None if heartbeat is None else (owner, state, heartbeat)


def claude_activity(claude_dir):
    """Time of the newest Claude Code transcript change, 0.0 if none."""
    return newest_mtime(os.path.join(claude_dir, "projects", "*", "*.jsonl"))
