"""`fal dev hot`: keep a model loaded on the box and reload code on save."""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

from rich.markup import escape

from fal.dev import config, hot, state
from fal.dev.cli.common import get_box
from fal.dev.cli.watch_view import LoopView
from fal.dev.errors import DevError
from fal.dev.loading import STATES
from fal.dev.media import preview


def _project(args) -> tuple[str | None, str]:
    """The local project folder and its folder on the box, remembered per folder."""
    cwd = Path.cwd()
    remembered = state.load(cwd, args.host)
    local, remote = hot.project_paths(args.local, args.dir, remembered)
    with contextlib.suppress(OSError):
        state.update(cwd, args.host, hot_local=local, hot_dir=remote)
    return local, remote


def _push(args, hot_box: hot.HotBox, local: str, remote: str) -> None:
    root = Path(local)
    withheld: list[Path] = []
    written = hot_box.push(root, remote, sorted(hot.scan(root, withheld)))
    if written:
        noun = "file" if len(written) == 1 else "files"
        args.console.print(f"synced {len(written)} changed {noun} to {escape(remote)}")
    else:
        args.console.print(f"{escape(remote)} is already in sync")
    if withheld:
        names = ", ".join(path.as_posix() for path in sorted(withheld)[:3])
        more = f" and {len(withheld) - 3} more" if len(withheld) > 3 else ""
        args.console.print(
            f"skipped {len(withheld)} secret-like files, never synced: "
            f"{escape(names)}{more}"
        )


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
    if not args.watch and args.run:
        args.parser.error("--run needs --watch")
    if not (args.watch and args.run):
        ignored = [
            flag
            for flag, value in (
                ("--prompt", args.prompt),
                ("-p/--param", args.param),
                ("-o/--output", args.output),
            )
            if value
        ]
        if ignored:
            args.parser.error(
                f"{', '.join(ignored)} only apply with --watch --run (sync alone "
                "sends no request; use `fal dev hot run` to send one)"
            )
    hot_box = hot.HotBox(get_box(args))
    hot_box.on_image = preview.HotPreview(args)
    local, remote = _project(args)
    if not local:
        raise DevError("fal dev hot sync needs a local project folder")
    _push(args, hot_box, local, remote)
    if args.watch:
        params = hot.parse_params(args.param, args.prompt) if args.run else None
        code, status = hot_box.json("status")
        view = LoopView(
            args.console, status if code == 200 else None, on_image=hot_box.on_image
        )
        try:
            hot_box.watch(Path(local), remote, params, args.output, view)
        except KeyboardInterrupt:
            args.console.print("stopped watching; the box and the model are still up")


def cmd_run(args):
    params = hot.parse_params(args.param, args.prompt)
    hot_box = hot.HotBox(get_box(args))
    hot_box.on_image = preview.HotPreview(args)
    return 0 if hot_box.request(params, args.output) else 1


def cmd_restart(args):
    hot_box = hot.HotBox(get_box(args))
    start = time.time()
    _call(hot_box, "restart")
    time.sleep(2)  # let the old process exit before polling
    info = hot_box.wait_ready(headline=STATES["hot-restart"][0])
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
