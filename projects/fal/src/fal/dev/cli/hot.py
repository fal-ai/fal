"""`fal dev hot`: keep a model loaded on the box and reload code on save."""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

from rich.markup import escape

from fal.dev import config, hot, state
from fal.dev.cli.common import err_console, get_box, usage_error
from fal.dev.cli.watch_view import LoopView, exception_line
from fal.dev.errors import DevError
from fal.dev.loading import STATES, format_elapsed
from fal.dev.media import preview


def _project(args) -> tuple[str | None, str]:
    """The local project folder and its folder on the box, remembered per folder."""
    cwd = Path.cwd()
    remembered = state.load(cwd, args.host)
    local, remote = hot.project_paths(args.local, args.dir, remembered)
    with contextlib.suppress(OSError):
        state.update(cwd, args.host, hot_local=local, hot_dir=remote)
    return local, remote


def _remembered_remote(args) -> str | None:
    """The project folder on the box last used from here, for relative paths."""
    with contextlib.suppress(OSError, ValueError):
        return state.load(Path.cwd(), args.host).get("hot_dir")
    return None


def _push(args, hot_box: hot.HotBox, local: str, remote: str) -> list[str]:
    root = Path(local)
    withheld: list[Path] = []
    written = hot_box.push(root, remote, sorted(hot.scan(root, withheld)))
    if written:
        noun = "file" if len(written) == 1 else "files"
        err_console(args).print(
            f"synced {len(written)} changed {noun} to {escape(remote)}"
        )
        err_console(args).print(f"  {escape(hot.reload_note(written))}")
    else:
        err_console(args).print(f"{escape(remote)} is already in sync")
    if withheld:
        names = ", ".join(path.as_posix() for path in sorted(withheld)[:3])
        more = f" and {len(withheld) - 3} more" if len(withheld) > 3 else ""
        err_console(args).print(
            f"skipped {len(withheld)} secret-like files, never synced: "
            f"{escape(names)}{more}"
        )
    return written


def _call(hot_box: hot.HotBox, action: str, payload: dict | None = None) -> dict:
    status, info = hot_box.json(action, payload)
    if status != 200:
        raise DevError(f"could not {action} ({status}): {info.get('error')}")
    return info


def _wait_if_restarting(args, hot_box: hot.HotBox) -> tuple[int, dict]:
    """The server's /status; while it restarts, wait for it instead of failing."""
    status, info = hot_box.json("status")
    kind, reason = hot.server_state(status, info)
    if kind == "restarting":
        why = f" ({reason})" if reason else ""
        err_console(args).print(f"restarting{escape(why)}, waiting ...")
        try:
            hot_box.wait_ready(timeout=30, headline=STATES["hot-restart"][0])
        except DevError:
            pass  # A status query should not hang: show the state as it is now.
        status, info = hot_box.json("status")
    return status, info


def cmd_start(args):
    hot_box = hot.HotBox(get_box(args))
    local, remote = _project(args)
    if local:
        _push(args, hot_box, local, remote)
    start = time.time()
    _call(hot_box, "start", {"dir": remote})
    where = f"in {escape(remote)}"
    if getattr(args, "verbose", False):
        where += f" (tmux session {config.HOT_TMUX_SESSION!r})"
    err_console(args).print(f"started the hot server {where}; loading ...")
    err_console(args).print(
        "  the first load downloads weights; later restarts are faster (cached)"
    )
    info = hot_box.wait_ready()
    err_console(args).print(
        f"ready in {format_elapsed(time.time() - start)} "
        f"(model load {info['load_seconds']}s)."
    )
    err_console(args).print(
        "Edit loop: fal dev hot sync --watch --run --prompt '...'   "
        "One request: fal dev hot run --prompt '...'"
    )


def cmd_sync(args):
    if not args.run:
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
            raise usage_error(
                args,
                f"{', '.join(ignored)} only works with --run (sync alone "
                "sends no request; use `fal dev hot run` to send one)",
                [
                    ("fal dev hot sync --run --prompt '...'", "push, then send one"),
                    ("fal dev hot run --prompt '...'", "send one without pushing"),
                ],
            )
    hot_box = hot.HotBox(get_box(args))
    hot_box.on_image = preview.HotPreview(args)
    local, remote = _project(args)
    if not local:
        raise DevError("fal dev hot sync needs a local project folder")
    hot_box.remote = remote
    params = hot.parse_params(args.param, args.prompt) if args.run else None
    if args.watch and args.run and args.output:
        err_console(args).print(
            f"-o: {escape(args.output)} is overwritten on each save "
            "(leave -o off for a timestamped file per save)"
        )
    _push(args, hot_box, local, remote)
    if not args.watch:
        if params is not None:
            return 0 if hot_box.request(params, args.output) else 1
        hot_box.wait_reload("reloading weights")
        return 0
    try:
        hot_box.wait_reload("reloading weights")
    except DevError as exc:
        # A broken first save is what watch mode is for: say so and keep watching.
        err_console(args).print(f"{escape(str(exc))}; fix it and save again")
    code, status = hot_box.json("status")
    view = LoopView(
        args.console, status if code == 200 else None, on_image=hot_box.on_image
    )
    try:
        hot_box.watch(Path(local), remote, params, args.output, view)
    except KeyboardInterrupt:
        err_console(args).print("stopped watching; the box and the model are still up")
    return 0


def cmd_run(args):
    params = hot.parse_params(args.param, args.prompt)
    hot_box = hot.HotBox(get_box(args))
    hot_box.on_image = preview.HotPreview(args)
    hot_box.remote = _remembered_remote(args)
    return 0 if hot_box.request(params, args.output) else 1


def cmd_restart(args):
    hot_box = hot.HotBox(get_box(args))
    start = time.time()
    before = hot_box.server_pid()
    _call(hot_box, "restart")
    info = hot_box.wait_ready(headline=STATES["hot-restart"][0], after_pid=before)
    err_console(args).print(
        f"restarted in {format_elapsed(time.time() - start)} "
        f"(model load {info['load_seconds']}s)"
    )


def cmd_stop(args):
    _call(hot.HotBox(get_box(args)), "stop")
    err_console(args).print(
        "weights unloaded; the box is still up and billing "
        "(fal dev hot start to reload, fal dev down to stop the box)"
    )


def _ago(at: float | None) -> str:
    if not at:
        return ""
    seconds = max(0, time.time() - at)
    return f", {format_elapsed(seconds)} ago" if seconds >= 2 else ", just now"


STATE_TEXT = {
    "ready": "ready",
    "loading": "loading weights",
    "restarting": "restarting",
    "failed": "stopped on a load error (fix load.py and save, or fal dev hot start)",
    "stopped": "stopped (fal dev hot start to load the model)",
    "unreachable": "no answer from the box (fal dev status)",
}


def cmd_status(args):
    hot_box = hot.HotBox(get_box(args))
    status, info = hot_box.json("status")
    if args.json:
        print(json.dumps({"http_status": status, **info}, indent=2))
        return 0
    hot_box.remote = _remembered_remote(args)
    if hot.server_state(status, info)[0] == "restarting":
        status, info = _wait_if_restarting(args, hot_box)
    kind, detail = hot.server_state(status, info)
    text = STATE_TEXT[kind]
    if detail and kind in ("restarting", "unreachable"):
        text += f" ({detail})"
    rows = [("state", text)]
    if status == 200:
        version = info.get("code_version")
        rows.append(("code", hot.code_version(str(version)).removeprefix("code ")))
        if info.get("ready"):
            rows.append(("weights", f"loaded in {info.get('load_seconds')}s"))
        reload = info.get("last_reload") or {}
        if reload:
            what = ", ".join(reload.get("files") or [])
            outcome = "" if reload.get("ok", True) else ", import failed"
            rows.append(
                (
                    "last reload",
                    f"{what}, {reload.get('ms')} ms{outcome}{_ago(reload.get('at'))}",
                )
            )
        infer = info.get("last_infer") or {}
        if infer:
            rows.append(("last infer", f"{infer.get('ms')} ms{_ago(infer.get('at'))}"))
        failure = info.get("last_error") or {}
        if failure:
            rows.append(
                ("last error", f"{failure.get('message')}{_ago(failure.get('at'))}")
            )
        elif info.get("error"):
            rows.append(
                ("last error", exception_line(info["error"]) or "import failed")
            )
    for label, value in rows:
        args.console.print(f"{label:<12}{escape(str(value))}", highlight=False)
    if args.log:
        print()
        for line in hot.clean_log(info.get("log") or []):
            print(line)
    elif kind in ("ready", "failed") and (info.get("last_error") or info.get("error")):
        args.console.print("(fal dev hot status --log for the log tail)")
    return 0


def cmd_logs(args):
    _, info = hot.HotBox(get_box(args)).json("log")
    for line in hot.clean_log(info.get("log") or []):
        print(line)
