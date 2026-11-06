"""Status probe that `fal dev dash` runs on the box, once per refresh.

Standard library only: the dashboard sends this file's source over the runner
exec channel and runs it with the box's python3, so it must not import fal.
It prints one JSON document, then END_MARKER, and never writes anything.

Nothing here may count as activity for the idle watchdog: it runs without a
terminal (no /dev/pts input), starts no sshd, and talks to the hot server on
127.0.0.1 directly, not through the app's /hot route.
"""

import glob
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

END_MARKER = "__fal_dev_dash_end__"

# Where the box keeps its state. The dashboard sends overrides as argv[1] (JSON).
DEFAULTS = {
    "workspace": "/data/devboxes/default",
    "local_work": "/work",
    "snapshot": "work.tar.zst",
    "lease": "work.lease",
    "save_stamp": "/tmp/devbox-save-stamp",
    "hold_file": "/tmp/devbox-hold-until",
    "warning_file": "/tmp/devbox-idle-shutdown-warning",
    # Written by a watchdog that publishes its state; preferred over recomputing.
    "idle_state_file": "/tmp/fal-dev-idle.json",
    "admin_key_env": "DEVBOX_ADMIN_KEY",
    "claude_config_dir": ".home/.claude",
    "claude_tmux": "claude",
    "hot_tmux": "hot",
    "hot_port": 7860,
    "hot_log": "/tmp/devbox-hot.log",
    "gpu_busy_pct": 10,
    "check_dirty": False,
}

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


def _num(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _gpus():
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
    gpus = []
    for line in out.stdout.strip().splitlines():
        values = [v.strip() for v in line.split(",")]
        if len(values) != len(GPU_FIELDS):
            continue
        gpus.append(
            {
                "index": _num(values[0]),
                "name": values[1],
                "util_pct": _num(values[2]),
                "mem_used_mib": _num(values[3]),
                "mem_total_mib": _num(values[4]),
                "temp_c": _num(values[5]),
                "power_w": _num(values[6]),
                "power_limit_w": _num(values[7]),
            }
        )
    return gpus, None


def _last_tty_input():
    """Newest keyboard input on any terminal; stat() does not change atime."""
    latest = 0.0
    for pts in glob.glob("/dev/pts/[0-9]*"):
        try:
            latest = max(latest, os.stat(pts).st_atime)
        except OSError:
            continue
    return latest or None


def _ssh_sessions():
    """Connected SSH clients: sshd processes whose parent is not an sshd."""
    names = {}
    parents = {}
    for stat_path in glob.glob("/proc/[0-9]*/stat"):
        pid = stat_path.split("/")[2]
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as f:
                argv0 = f.read().split(b"\0")[0]
            with open(stat_path) as f:
                # The command name may hold spaces; ppid follows the closing paren.
                ppid = f.read().rsplit(")", 1)[1].split()[1]
        except (OSError, IndexError):
            continue
        names[pid] = argv0.rsplit(b"/", 1)[-1].startswith(b"sshd")
        parents[pid] = ppid
    return sum(
        1 for pid, is_sshd in names.items() if is_sshd and not names.get(parents[pid])
    )


def _tmux_session(name):
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


def _newest_mtime(pattern):
    latest = 0.0
    for path in glob.glob(pattern):
        try:
            latest = max(latest, os.stat(path).st_mtime)
        except OSError:
            continue
    return latest or None


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def _stat(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return {"mtime": st.st_mtime, "size": st.st_size}


def _today_at(clock):
    """Epoch of an HH:MM:SS from the hot log (box local time), within the last day."""
    try:
        at = time.mktime(
            time.strptime(time.strftime("%Y-%m-%d ") + clock, "%Y-%m-%d %H:%M:%S")
        )
    except ValueError:
        return None
    return at - 86400 if at > time.time() + 60 else at


def _hot(cfg):
    hot = {"running": _tmux_session(cfg["hot_tmux"])}
    if not hot["running"]:
        return hot
    req = urllib.request.Request(
        "http://127.0.0.1:%d/status" % int(cfg["hot_port"]),
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            status = json.loads(resp.read())
    except (OSError, ValueError) as exc:
        hot["reachable"] = False
        hot["detail"] = str(exc)[:200]
        status = {}
    else:
        hot["reachable"] = True
    for key in (
        "ready",
        "code_version",
        "load_seconds",
        "last_reload",
        "restarting",
        "pid",
    ):
        if key in status:
            hot[key] = status[key]
    if status.get("error"):
        # The last line of a traceback says what went wrong.
        hot["error"] = status["error"].strip().splitlines()[-1][:300]
    lines = status.get("log") or (_read(cfg["hot_log"]) or "").splitlines()[-200:]
    for line in reversed(lines):
        match = re.match(r"(\d\d:\d\d:\d\d) \[hot\] infer (\d+) ms", line)
        if match:
            hot["last_infer"] = {
                "at": _today_at(match.group(1)),
                "ms": int(match.group(2)),
            }
            break
    return hot


def _idle_state(cfg):
    """The watchdog's own view, when it publishes one."""
    raw = _read(cfg["idle_state_file"])
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _work(cfg, workspace):
    snapshot = os.path.join(workspace, cfg["snapshot"])
    work = {
        "snapshot": _stat(snapshot),
        "last_save": None,
        "lease": None,
        "dirty": None,
    }
    stamp = _stat(cfg["save_stamp"])
    if stamp:
        work["last_save"] = stamp["mtime"]
    lease = (_read(os.path.join(workspace, cfg["lease"])) or "").split()
    if len(lease) == 3:
        work["lease"] = {
            "owner": lease[0],
            "state": lease[1],
            "heartbeat": _num(lease[2]),
            "mine": lease[0] == socket.gethostname(),
        }
    if cfg.get("check_dirty") and stamp:
        try:
            out = subprocess.run(
                [
                    "find",
                    cfg["local_work"],
                    "-newer",
                    cfg["save_stamp"],
                    "-print",
                    "-quit",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            work["dirty"] = bool(out.stdout.strip())
        except (OSError, subprocess.TimeoutExpired):
            pass
    return work


def probe(cfg):
    now = time.time()
    workspace = cfg["workspace"]
    claude_dir = cfg["claude_config_dir"]
    if not os.path.isabs(claude_dir):
        claude_dir = os.path.join(workspace, claude_dir)
    gpus, gpu_error = _gpus()
    hold_until = _num((_read(cfg["hold_file"]) or "").strip())
    warnings = [
        line[len("devbox: ") :] if line.startswith("devbox: ") else line
        for line in (_read(cfg["warning_file"]) or "").splitlines()
        if line.strip()
    ]
    return {
        "now": now,
        "hostname": socket.gethostname(),
        "gpus": gpus,
        "gpu_error": gpu_error,
        "activity": {
            "last_tty_input": _last_tty_input(),
            "ssh_sessions": _ssh_sessions(),
            "gpu_busy": any((g["util_pct"] or 0) >= cfg["gpu_busy_pct"] for g in gpus),
            "hold_until": hold_until,
            "held": bool(hold_until and hold_until > now),
            "last_claude_activity": _newest_mtime(
                os.path.join(claude_dir, "projects", "*", "*.jsonl")
            ),
        },
        "idle_state": _idle_state(cfg),
        "admin_key_set": bool(os.environ.get(cfg["admin_key_env"])),
        "warnings": warnings,
        "claude": {"running": _tmux_session(cfg["claude_tmux"])},
        "hot": _hot(cfg),
        "work": _work(cfg, workspace),
    }


def main():
    cfg = dict(DEFAULTS)
    if len(sys.argv) > 1:
        cfg.update(json.loads(sys.argv[1]))
    try:
        result = probe(cfg)
    except Exception as exc:  # report, never hang the dashboard
        result = {"probe_error": repr(exc)[:500]}
    sys.stdout.write(json.dumps(result) + "\n" + END_MARKER + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
