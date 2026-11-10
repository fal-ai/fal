"""Status probe that `fal dev app dash` runs on a `fal dev app` box, per refresh.

Standard library only. The box is the user's own image, where nothing of
fal dev is installed, so the dashboard sends this file's source (with
signals.py) over the runner exec and runs it with the image's python3. It
prints one JSON document, then config.PROBE_END, and never writes anything.

Nothing here is a request to the app on fal.run, so polling never resets the
box's keep_alive: the holder's activity file is read from disk, and the app
process is asked on 127.0.0.1 directly.

Usage: python3 app_probe.py '{"session_dir": "/tmp/fal-dev-app/<box>", ...}'
"""

import json
import os
import sys
import time
import urllib.request

import signals


def _get(url, data=None, timeout=2.0):
    """The JSON a local route answers, or None when it is not there."""
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except (OSError, ValueError):
        return None


def _activity(session_dir):
    try:
        with open(os.path.join(session_dir, "activity.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _app_process(marker, proc="/proc"):
    """The pid and start time of the app process, from /proc, if it runs.

    Matches the argv of ``python -m <marker>``, not a substring: this probe's
    own argv carries the marker in its settings.
    """
    for pid in os.listdir(proc):
        if not pid.isdigit():
            continue
        try:
            with open(os.path.join(proc, pid, "cmdline"), "rb") as f:
                argv = f.read().decode(errors="replace").split("\0")
        except OSError:
            continue
        if any(a == "-m" and b == marker for a, b in zip(argv, argv[1:])):
            try:
                started = os.stat(os.path.join(proc, pid)).st_mtime
            except OSError:
                started = None
            return {"pid": int(pid), "started": started}
    return None


def probe(settings):
    gpus, gpu_error = signals.gpus()
    base = f"http://127.0.0.1:{settings['child_port']}"
    process = _app_process(settings["child_marker"])
    app = None
    if process:
        app = _get(base + settings["child_status"], data=b"{}")
    engine = None
    if app and settings["llm_route"] in (app.get("endpoints") or []):
        engine = _get(base + settings["engine_stats"])
    return {
        "now": time.time(),
        "gpus": gpus,
        "gpu_error": gpu_error,
        "activity": _activity(settings["session_dir"]),
        "process": process,
        "app": app,
        "engine": engine,
    }


def main():
    settings = json.loads(sys.argv[1])
    try:
        result = probe(settings)
    except Exception as exc:
        result = {"probe_error": f"{exc.__class__.__name__}: {exc}"}
    print(json.dumps(result))
    print(settings["end"])


if __name__ == "__main__":
    main()
