"""What a box is doing, read from the operating system.

Shared by the idle watchdog (box_app.py) and the dashboard's probe (probe.py).
Standard library only: the probe runs on the box's plain python3.
"""

import glob
import os
import subprocess
import time

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


def gpu_memory_by_pid():
    """{pid: MiB of GPU memory} over every GPU; empty when nvidia-smi says nothing."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=GPU_MEMORY_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    used = {}
    if out.returncode != 0:
        return used
    for line in out.stdout.strip().splitlines():
        pid, _, mib = (v.strip() for v in line.partition(","))
        pid, mib = num(pid), num(mib)
        if isinstance(pid, int) and mib is not None:
            used[pid] = used.get(pid, 0) + mib
    return used


CGROUP = "/sys/fs/cgroup"
PROC = "/proc"
# Some nodes still run cgroup v1, with one directory per controller.
CPU_V1 = "cpu,cpuacct"
MEMORY_V1 = "memory"
# cgroup v1 reports "no limit" as a number near 2**63.
NO_LIMIT = 2**60
TOP_PROCESSES = 5
# Short: the probe has already waited on nvidia-smi once for the GPU rows.
GPU_MEMORY_TIMEOUT = 3
COMMAND_CHARS = 40


def _first_line(path):
    try:
        with open(path) as f:
            return f.readline().strip()
    except OSError:
        return None


def _stat_value(path, wanted):
    """The value of `wanted` in a "key value" stat file, None if not readable."""
    try:
        with open(path) as f:
            for line in f:
                key, _, value = line.partition(" ")
                if key == wanted:
                    return int(value)
    except (OSError, ValueError):
        pass
    return None


def _cpu_usage_usec(root):
    """CPU time the container has used (microseconds), None if not readable."""
    usage = _stat_value(os.path.join(root, "cpu.stat"), "usage_usec")
    if usage is not None:
        return usage
    nanoseconds = num(_first_line(os.path.join(root, CPU_V1, "cpuacct.usage")))
    return None if nanoseconds is None else nanoseconds // 1000


def _cpu_limit(root):
    """How many CPUs the container may use: its cgroup quota, else the visible CPUs."""
    quota, _, period = (_first_line(os.path.join(root, "cpu.max")) or "").partition(" ")
    if quota.isdigit() and period.isdigit() and int(period):
        return int(quota) / int(period)
    quota = num(_first_line(os.path.join(root, CPU_V1, "cpu.cfs_quota_us")))
    period = num(_first_line(os.path.join(root, CPU_V1, "cpu.cfs_period_us")))
    if quota and quota > 0 and period:
        return quota / period
    if hasattr(os, "sched_getaffinity"):
        return float(len(os.sched_getaffinity(0)))
    return float(os.cpu_count() or 0)


def _working_set(root):
    """Container memory without reclaimable file cache, None if not readable.

    memory.current counts the page cache, which fills up after any big download
    and is dropped before an OOM; inactive_file is the part of it that goes first.
    """
    current = num(_first_line(os.path.join(root, "memory.current")))
    stat = os.path.join(root, "memory.stat")
    if current is None:
        current = num(
            _first_line(os.path.join(root, MEMORY_V1, "memory.usage_in_bytes"))
        )
        stat = os.path.join(root, MEMORY_V1, "memory.stat")
    if current is None:
        return None
    inactive = _stat_value(stat, "inactive_file")
    return current if inactive is None else max(current - inactive, 0)


def _memory_limit(root):
    """The container's memory limit in bytes, None if it has none or is unreadable."""
    limit = num(_first_line(os.path.join(root, "memory.max")))
    if limit is None:
        limit = num(_first_line(os.path.join(root, MEMORY_V1, "memory.limit_in_bytes")))
    return limit if limit is not None and limit < NO_LIMIT else None


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


def disks(paths):
    """One entry per filesystem under `paths`: the paths on it, bytes used, total, free.

    Free is what this box can still write. Used and total can describe a disk the
    node shares with other containers, or a network volume's whole backend.

    Two paths on the same device (st_dev) are one disk, not two. Paths that do
    not exist, or filesystems that report no size, are left out.
    """
    found = {}
    for path in paths:
        try:
            device = os.stat(path).st_dev
            stats = os.statvfs(path)
        except OSError:
            continue
        if device in found:
            found[device]["paths"].append(path)
            continue
        total = stats.f_blocks * stats.f_frsize
        if total:
            found[device] = {
                "paths": [path],
                "used_bytes": total - stats.f_bfree * stats.f_frsize,
                "total_bytes": total,
                "free_bytes": stats.f_bavail * stats.f_frsize,
            }
    return list(found.values())


def _net_bytes(proc=PROC):
    """(received, sent) bytes over every interface but loopback, None if unreadable."""
    received = sent = 0
    try:
        with open(os.path.join(proc, "net", "dev")) as f:
            # Two header lines, then "<interface>: <8 receive fields> <8 send fields>".
            for line in list(f)[2:]:
                name, _, rest = line.partition(":")
                if name.strip() == "lo":
                    continue
                fields = rest.split()
                received += int(fields[0])
                sent += int(fields[8])
    except (OSError, ValueError, IndexError):
        return None
    return received, sent


def _cpu_ticks(proc=PROC):
    """{pid: (command name, CPU ticks used, resident bytes)} for every process."""
    try:
        page = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return {}
    found = {}
    for stat_path in glob.glob(os.path.join(proc, "[0-9]*", "stat")):
        try:
            with open(stat_path) as f:
                head, _, rest = f.read().rpartition(")")
            # The name may hold spaces and parens; fields 14, 15 and 24 follow it
            # (after the state), at indexes 11, 12 and 21.
            fields = rest.split()
            pid = int(os.path.basename(os.path.dirname(stat_path)))
            found[pid] = (
                head.partition("(")[2],
                int(fields[11]) + int(fields[12]),
                int(fields[21]) * page,
            )
        except (OSError, ValueError, IndexError):
            continue
    return found


def _command_name(pid, comm, proc=PROC):
    """What a process is: its script for an interpreter, else its own name."""
    if comm.startswith("python"):
        try:
            with open(os.path.join(proc, str(pid), "cmdline"), "rb") as f:
                argv = f.read().split(b"\0")[1:]
        except OSError:
            argv = []
        script = next((a for a in argv if a and not a.startswith(b"-")), b"")
        if script:
            comm = os.path.basename(script.decode(errors="replace"))
    return comm[:COMMAND_CHARS]


def _busiest(before, after, elapsed, proc=PROC):
    """The most active processes between two _cpu_ticks readings, this one excluded.

    CPU is of one core (a process on two cores shows 200). Ties, which is every
    process on an idle box, go to the larger resident set.
    """
    try:
        per_second = os.sysconf("SC_CLK_TCK") * elapsed
    except (OSError, ValueError):
        return []
    if per_second <= 0:
        return []
    ranked = sorted(
        (
            ((ticks - before[pid][1]) / per_second * 100, rss, pid, comm)
            for pid, (comm, ticks, rss) in after.items()
            if pid in before and pid != os.getpid()
        ),
        reverse=True,
    )
    return [
        {
            "pid": pid,
            "name": _command_name(pid, comm, proc),
            "cpu_pct": round(max(cpu, 0.0), 1),
            "rss_bytes": rss,
        }
        for cpu, rss, pid, comm in ranked[:TOP_PROCESSES]
    ]


def system(sample_seconds=0.25, root=CGROUP, proc=PROC):
    """CPU, memory, network and busiest processes of the box's container.

    CPU percent is of the container's CPU limit (cgroup v2, or v1 on older
    nodes), sampled over `sample_seconds`; network rates and per-process CPU
    share that one window. Without a cgroup, memory falls back to /proc/meminfo
    and CPU percent is None.
    """
    cpus = _cpu_limit(root)
    net_before = _net_bytes(proc)
    ticks_before = _cpu_ticks(proc)
    # Read the cgroup next to the clock so the /proc scan above, and the
    # probe's own CPU spent on it, stay out of the container's CPU percent.
    before = _cpu_usage_usec(root)
    start = time.monotonic()
    time.sleep(sample_seconds)
    after = _cpu_usage_usec(root) if before is not None else None
    elapsed = time.monotonic() - start
    cpu_pct = None
    if after is not None and cpus and elapsed > 0:
        cpu_pct = round(min((after - before) / 1e6 / elapsed / cpus * 100, 100.0), 1)
    net_after = _net_bytes(proc) if net_before else None
    net_in = net_out = None
    if net_after and elapsed > 0:
        net_in = max(net_after[0] - net_before[0], 0) / elapsed
        net_out = max(net_after[1] - net_before[1], 0) / elapsed
    used = _working_set(root)
    host_used, host_total = _meminfo()
    if used is None:
        used, total = host_used, host_total
    else:
        # No cgroup limit leaves the host's memory as the ceiling.
        total = _memory_limit(root) or host_total
    return {
        "cpu_pct": cpu_pct,
        "cpus": num(round(cpus, 2)) if cpus else None,
        "mem_used_bytes": used,
        "mem_total_bytes": total,
        "net_in_bps": None if net_in is None else round(net_in),
        "net_out_bps": None if net_out is None else round(net_out),
        "processes": _busiest(ticks_before, _cpu_ticks(proc), elapsed, proc)
        if ticks_before
        else [],
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
