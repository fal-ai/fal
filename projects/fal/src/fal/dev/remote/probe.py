"""Status probe that `fal dev dash` runs on the box, once per refresh.

Standard library only, run by the box's python3 from the directory box_app.py
installs it in, next to config.py and signals.py. It prints one JSON document,
then config.PROBE_END, and never writes anything.

Nothing here may count as activity for the idle watchdog: it runs without a
terminal (no /dev/pts input), starts no sshd, and talks to the hot server on
127.0.0.1 directly, not through the app's /hot route.

Usage: probe.py '{"workspace": "<the box's workspace>", "check_dirty": true}'
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import config
import signals


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


def _hot():
    hot = {"running": signals.tmux_session(config.HOT_TMUX_SESSION)}
    if not hot["running"]:
        return hot
    request = urllib.request.Request(
        "http://127.0.0.1:%d/status" % config.HOT_PORT,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            status = json.loads(response.read())
    except (OSError, ValueError):
        hot["reachable"] = False
        return hot
    hot["reachable"] = True
    for key in (
        "ready",
        "code_version",
        "load_seconds",
        "last_reload",
        "last_infer",
        "restarting",
        "pid",
    ):
        if key in status:
            hot[key] = status[key]
    if status.get("error"):
        # The last line of a traceback says what went wrong.
        lines = status["error"].strip().splitlines()
        hot["error"] = lines[-1][:300] if lines else ""
    return hot


def _gpu_memory():
    try:
        return signals.gpu_memory_by_pid()
    except Exception:
        return {}


def _system(gpus_answered=True):
    """CPU, memory, network, busiest processes; None rather than failing the probe.

    Per-process GPU memory is skipped when nvidia-smi just failed or found no
    GPU: a stuck driver would otherwise cost the probe another timeout.
    """
    gpu_memory = {}
    # nvidia-smi runs while system() sleeps through its sample, so a healthy
    # driver adds no time and a stuck one at most GPU_MEMORY_TIMEOUT.
    reader = threading.Thread(
        target=lambda: gpu_memory.update(_gpu_memory()), daemon=True
    )
    if gpus_answered:
        reader.start()
    try:
        reading = signals.system()
    except Exception:
        return None
    if gpus_answered:
        reader.join(timeout=signals.GPU_MEMORY_TIMEOUT)
    for process in reading.get("processes") or []:
        if process["pid"] in gpu_memory:
            process["gpu_mib"] = gpu_memory[process["pid"]]
    return reading


def _disks():
    """Space on /work, /data and the root filesystem.

    /data is already read for the snapshot and lease, so a stuck volume
    adds no new way for the probe to hang.
    """
    try:
        return signals.disks([config.LOCAL_WORK, config.VOLUME_MOUNT, "/"])
    except Exception:
        return None


def _idle_state():
    """The watchdog's own view, as it last published it."""
    try:
        return json.loads(_read(config.IDLE_STATE_FILE) or "")
    except ValueError:
        return None


def _work(workspace, check_dirty):
    stamp = _stat(config.SAVE_STAMP)
    work = {
        "snapshot": _stat(os.path.join(workspace, config.SNAPSHOT_FILE)),
        "last_save": stamp["mtime"] if stamp else None,
        "lease": None,
        "dirty": None,
    }
    lease = signals.read_lease(os.path.join(workspace, config.LEASE_FILE))
    if lease:
        owner, state, heartbeat = lease
        work["lease"] = {
            "owner": owner,
            "state": state,
            "heartbeat": heartbeat,
            "mine": owner == socket.gethostname(),
        }
    if check_dirty and stamp:
        try:
            out = subprocess.run(
                [
                    "find",
                    config.LOCAL_WORK,
                    # ctime too: rsync -a, cp -p and tar keep an old mtime.
                    "(",
                    "-newer",
                    config.SAVE_STAMP,
                    "-o",
                    "-cnewer",
                    config.SAVE_STAMP,
                    ")",
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


def probe(workspace, check_dirty):
    now = time.time()
    gpus, gpu_error = signals.gpus()
    hold_until = signals.hold_until(config.HOLD_FILE)
    claude_dir = os.path.join(workspace, config.CLAUDE_CONFIG_SUBDIR)
    prefix = "devbox: "
    warnings = [
        line[len(prefix) :] if line.startswith(prefix) else line
        for line in (_read(config.WARNING_FILE) or "").splitlines()
        if line.strip()
    ]
    return {
        "now": now,
        "gpus": gpus,
        "gpu_error": gpu_error,
        "system": _system(gpus_answered=bool(gpus)),
        "disks": _disks(),
        "activity": {
            "ssh_sessions": signals.ssh_sessions(),
            "gpu_busy": signals.gpu_busy(gpus, config.GPU_BUSY_PCT),
            "hold_until": hold_until,
            "held": bool(hold_until and hold_until > now),
            "last_claude_activity": signals.claude_activity(claude_dir) or None,
        },
        "idle_state": _idle_state(),
        "admin_key_set": bool(os.environ.get(config.ADMIN_KEY_SECRET)),
        "warnings": warnings,
        "claude_running": signals.tmux_session(config.CLAUDE_TMUX_SESSION),
        "hot": _hot(),
        "work": _work(workspace, check_dirty),
    }


def main():
    settings = json.loads(sys.argv[1])
    try:
        result = probe(settings["workspace"], settings["check_dirty"])
    except Exception as exc:  # report, never hang the dashboard
        result = {"probe_error": repr(exc)[:500]}
    sys.stdout.write(json.dumps(result) + "\n" + config.PROBE_END + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
