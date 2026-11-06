"""What a box is doing, read from the operating system.

Shared by the idle watchdog (box_app.py) and the dashboard's probe (probe.py).
Standard library only: the probe runs on the box's plain python3.
"""

import glob
import os
import subprocess
import time

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


CGROUP = "/sys/fs/cgroup"


def _first_line(path):
    try:
        with open(path) as f:
            return f.readline().strip()
    except OSError:
        return None


def _cpu_usage_usec(root):
    """CPU time the container has used (microseconds), None if not readable."""
    try:
        with open(os.path.join(root, "cpu.stat")) as f:
            for line in f:
                key, _, value = line.partition(" ")
                if key == "usage_usec":
                    return int(value)
    except (OSError, ValueError):
        pass
    return None


def _cpu_limit(root):
    """How many CPUs the container may use: its cgroup quota, else the visible CPUs."""
    quota, _, period = (_first_line(os.path.join(root, "cpu.max")) or "").partition(" ")
    if quota.isdigit() and period.isdigit() and int(period):
        return int(quota) / int(period)
    if hasattr(os, "sched_getaffinity"):
        return float(len(os.sched_getaffinity(0)))
    return float(os.cpu_count() or 0)


def _working_set(root):
    """Container memory without reclaimable file cache, None if not readable.

    memory.current counts the page cache, which fills up after any big download
    and is dropped before an OOM; inactive_file is the part of it that goes first.
    """
    current = num(_first_line(os.path.join(root, "memory.current")))
    if current is None:
        return None
    try:
        with open(os.path.join(root, "memory.stat")) as f:
            for line in f:
                key, _, value = line.partition(" ")
                if key == "inactive_file":
                    return max(current - int(value), 0)
    except (OSError, ValueError):
        pass
    return current


def _meminfo():
    """(used, total) bytes from /proc/meminfo: the host's view."""
    fields = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                fields[key] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None, None
    total = fields.get("MemTotal")
    available = fields.get("MemAvailable")
    if total is None or available is None:
        return None, None
    return total - available, total


def system(sample_seconds=0.25, root=CGROUP):
    """CPU and memory of the box's container, as its cgroup (v2) limits them.

    CPU percent is of the container's CPU limit, sampled over `sample_seconds`.
    Without a cgroup v2 controller, memory falls back to /proc/meminfo and CPU
    percent is None.
    """
    cpus = _cpu_limit(root)
    before = _cpu_usage_usec(root)
    cpu_pct = None
    if before is not None and cpus:
        start = time.monotonic()
        time.sleep(sample_seconds)
        after = _cpu_usage_usec(root)
        elapsed = time.monotonic() - start
        if after is not None and elapsed > 0:
            cpu_pct = round(
                min((after - before) / 1e6 / elapsed / cpus * 100, 100.0), 1
            )
    used = _working_set(root)
    host_used, host_total = _meminfo()
    if used is None:
        used, total = host_used, host_total
    else:
        limit = num(_first_line(os.path.join(root, "memory.max")))
        # "max" (no cgroup limit) leaves the host's memory as the ceiling.
        total = limit if limit is not None else host_total
    return {
        "cpu_pct": cpu_pct,
        "cpus": num(round(cpus, 2)) if cpus else None,
        "mem_used_bytes": used,
        "mem_total_bytes": total,
    }


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
