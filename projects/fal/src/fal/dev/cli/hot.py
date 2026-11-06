"""`fal dev hot`: keep a model loaded on the box and reload code on save."""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

from rich.markup import escape

from fal.dev import config, hot, state
from fal.dev.box import DevError
from fal.dev.cli.common import get_box


def _project(args) -> tuple[str | None, str]:
    """The local project folder and its folder on the box, remembered per folder."""
    cwd = Path.cwd()
    remembered = state.load(cwd, args.host, args.team)
    local, remote = hot.project_paths(args.local, args.dir, remembered)
    with contextlib.suppress(OSError):
        state.update(cwd, args.host, args.team, hot_local=local, hot_dir=remote)
    return local, remote


def _push(args, hot_box: hot.HotBox, local: str, remote: str) -> None:
    root = Path(local)
    written = hot_box.push(root, remote, sorted(hot.scan(root)))
    args.console.print(f"synced {len(written)} changed files to {escape(remote)}")


def _call(hot_box: hot.HotBox, action: str, payload: dict | None = None) -> dict:
    status, info = hot_box.json(action, payload)
    if status != 200:
        raise DevError(f"could not {action} ({status}): {info.get('error')}")
    return info


def cmd_start(args):
    hot_box = hot.HotBox(get_box(args))
    local, remote = _project(args)
    if local:
        _push(args, hot_box, local, remote)
    start = time.time()
    _call(hot_box, "start", {"dir": remote})
    args.console.print(
        f"started the hot server in {escape(remote)} "
        f"(tmux session {config.HOT_TMUX_SESSION!r}); loading ..."
    )
    info = hot_box.wait_ready()
    args.console.print(
        f"ready in {time.time() - start:.1f}s (model load {info['load_seconds']}s). "
        "Send a request: fal dev hot run"
    )


def cmd_sync(args):
    hot_box = hot.HotBox(get_box(args))
    local, remote = _project(args)
    if not local:
        raise DevError("fal dev hot sync needs a local project folder")
    _push(args, hot_box, local, remote)
    if args.watch:
        params = hot.parse_params(args.param, args.prompt) if args.run else None
        hot_box.watch(Path(local), remote, params, args.output)


def cmd_run(args):
    params = hot.parse_params(args.param, args.prompt)
    return 0 if hot.HotBox(get_box(args)).request(params, args.output) else 1


def cmd_restart(args):
    hot_box = hot.HotBox(get_box(args))
    start = time.time()
    _call(hot_box, "restart")
    time.sleep(2)  # let the old process exit before polling
    info = hot_box.wait_ready()
    args.console.print(
        f"restarted in {time.time() - start:.1f}s (model load {info['load_seconds']}s)"
    )


def cmd_stop(args):
    _call(hot.HotBox(get_box(args)), "stop")
    args.console.print("stopped the hot server")


def cmd_status(args):
    _, info = hot.HotBox(get_box(args)).json("status")
    for line in info.pop("log", None) or []:
        print(line)
    print(json.dumps(info, indent=2))


def cmd_logs(args):
    _, info = hot.HotBox(get_box(args)).json("log")
    for line in info.get("log") or []:
        print(line)
