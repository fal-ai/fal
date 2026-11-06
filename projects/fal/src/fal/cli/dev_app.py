"""`fal dev app`: a GPU dev box for an existing fal App, with hot reload.

`start` deploys a private box that inherits the app's machine, requirements,
image and secrets, uploads the app's source tree, and runs ``setup()`` once in a
resident process. `sync` uploads changed files: endpoint edits reload in place
with the weights still loaded; edits to ``setup()`` (or what it uses) restart
the process; requirement or image changes need `start` again.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any

from .parser import FalClientParser, RefAction, add_env_argument

# Kept equal to the holder's limits (fal.dev.app_holder) without importing it.
MAX_WARMUPS = 50
MAX_REPEAT = 100

DEFAULT_KEEP_ALIVE = 1800
PROFILES_DIR = Path("~/.fal/dev/profiles").expanduser()
POLL_SECONDS = 1.0


def _bounded(low: int, high: int):
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid integer {value!r}") from None
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return number

    return parse


def escape(text: str) -> str:
    """Show text as is in rich output (package extras look like markup)."""
    from rich.markup import escape as rich_escape

    return rich_escape(text)


def _console():
    from fal.console import console

    return console


def _session(args):
    from fal.dev.app_session import find_session

    return find_session(args.session)


def _print_lines(lines: list[str], width: int = 400) -> None:
    for line in lines:
        print(line if len(line) <= width else f"{line[:width]} ...", flush=True)


def _wait_ready(session, *, after: int | None = None, timeout: float = 3600) -> dict:
    """Stream the app's log until it is ready, or raise if it stops."""
    from fal.dev.app_holder import STOPPED_PHASES
    from fal.dev.app_session import DevAppError, call

    deadline = time.monotonic() + timeout
    if after is None:
        after = call(session, "logs", {"lines": 0})["total"]
    while time.monotonic() < deadline:
        logs = call(session, "logs", {"after": after})
        _print_lines(logs["lines"])
        after = logs["total"]
        status = call(session, "status")
        phase = status["phase"]
        if phase == "ready":
            return status
        if phase in STOPPED_PHASES:
            raise DevAppError(
                f"The app did not start ({phase} {status['detail']}). "
                "Fix the code and run `fal dev app restart`."
            )
        time.sleep(POLL_SECONDS)
    raise DevAppError("Timed out waiting for setup() to finish.")


def _sync(session, *, full: bool = False, reload: bool = True) -> dict[str, Any] | None:
    from fal.dev import app_sync
    from fal.dev.app_session import call

    skipped: list[str] = []
    files = app_sync.collect(Path(session.root), skipped)
    if skipped:
        _console().print(
            f"[yellow]Not synced (over 10 MiB)[/]: {', '.join(skipped[:8])}"
        )
    manifest = app_sync.manifest(files)
    previous = {} if full else session.manifest
    changed, deleted = app_sync.diff(previous, manifest)
    if not changed and not deleted and not full:
        return None
    # Large trees go in several requests. Deletes and pruning ride on the
    # first, before any writes; the reload on the last.
    groups = app_sync.batches(files, changed)
    result: dict[str, Any] = {}
    for index, group in enumerate(groups):
        last = index == len(groups) - 1
        payload: dict[str, Any] = {"reload": reload and last}
        if group:
            payload["archive"] = app_sync.pack(files, group)
        if index == 0:
            payload.update(full=full, deleted=deleted)
            if full:
                payload["all_files"] = sorted(files)
        result = call(session, "files", payload, timeout=600)
    action = (result.get("reload") or {}).get("action")
    if action not in ("rebuild", "error", "not_running"):
        # Otherwise keep the old manifest, so the next sync sends the files
        # again and repeats the warning.
        session.manifest = manifest
        session.save()
    result["changed"] = changed
    result["deleted"] = deleted
    return result


def _report_sync(result: dict[str, Any] | None, started: float) -> None:
    console = _console()
    if result is None:
        console.print("No changes.")
        return
    names = ", ".join((result["changed"] + result["deleted"])[:8])
    count = len(result["changed"]) + len(result["deleted"])
    elapsed = time.perf_counter() - started
    console.print(f"Synced {count} file(s) in {elapsed:.2f}s: {names}")
    reload = result.get("reload")
    if not reload:
        return
    action = reload.get("action")
    if action == "reload":
        console.print(
            f"[green]Reloaded[/] in {reload['ms']} ms "
            f"(code version {reload['code_version']}), setup() kept."
        )
    elif action == "restart":
        console.print(
            f"[yellow]Full restart[/]: {reload['reason']}. setup() runs again."
        )
    elif action == "pending":
        console.print(
            f"Saved; {reload['reason']}, so the edit reloads once it is ready."
        )
    elif action == "rebuild":
        console.print(
            f"[red]Not reloaded[/]: {reload['reason']}. "
            "Run `fal dev app start` again to rebuild the box."
        )
    else:
        console.print(f"[red]Not reloaded[/]: {reload.get('reason')}")
        if reload.get("error"):
            print(reload["error"], file=sys.stderr)


def _boot(session) -> dict:
    """Upload the whole source tree and the SDK, then run setup() on the box."""
    from fal.dev import app_sync
    from fal.dev.app_session import call

    _sync(session, full=True, reload=False)
    _replay_overlay(session)
    sdk = app_sync.sdk_files()
    after = call(session, "logs", {"lines": 0}, timeout=600)["total"]
    call(
        session,
        "start",
        {
            "module": session.module,
            "class_name": session.class_name,
            "sdk_archive": app_sync.pack(sdk, sorted(sdk)),
            "sdk_requirements": app_sync.sdk_requirements(),
        },
        timeout=300,
    )
    _console().print(f"Running {session.class_name}.setup() on the box ...")
    return _wait_ready(session, after=after)


def _replay_overlay(session) -> None:
    """Install the session's dependency overlay on a fresh runner, as pinned.

    A layer that no longer installs is dropped with a warning, so the box
    still starts and `deps` commands keep working.
    """
    from fal.dev.app_deps import redact
    from fal.dev.app_session import DevAppError

    if not session.overlay:
        return
    wanted = [layer["packages"] for layer in session.overlay]
    status = _deps_call(session, {"action": "status"}, remember=False)
    if [layer["packages"] for layer in status["replay"]] == wanted:
        session.overlay = status["replay"]
        session.save()
        return
    console = _console()
    console.print(f"Installing {len(wanted)} dependency overlay layer(s) ...")
    layers = session.overlay
    # Start from a clean overlay: this runner may hold a partial earlier replay.
    _deps_call(session, {"action": "reset"}, remember=False)
    for layer in layers:
        payload = {
            **{key: value for key, value in layer.items() if key != "id"},
            "action": "install",
            "no_deps": True,
            "force": True,
        }
        try:
            _deps_call(session, payload, remember=False)
        except DevAppError as exc:
            pins = redact(", ".join(layer["packages"]))
            console.print(
                f"[red]Dropped an overlay layer[/] ({escape(pins or '')}): "
                f"{escape(redact(str(exc)) or '')}"
            )
    _deps_call(session, {"action": "status"})


def _ensure_booted(session) -> bool:
    """Boot again when the box was scaled to zero or replaced since `start`."""
    from fal.dev.app_session import call

    # A cold box first waits for a runner, which can take minutes.
    status = call(session, "status", timeout=600)
    if status["phase"] != "idle" or status["module"]:
        return False
    _console().print(
        "[yellow]The box is on a new runner[/] (idle shutdown or replacement): "
        "uploading the source and running setup() again."
    )
    _boot(session)
    return True


def _start(args):
    from fal.config import Config
    from fal.dev.app_session import (
        DevAppError,
        Session,
        box_exists,
        deploy_box,
        find_session,
        new_token,
        session_name,
    )

    console = _console()
    file_path, class_name = args.app_ref
    if file_path is None or not file_path.endswith(".py"):
        raise ValueError("Pass the app as path/to/file.py or path/to/file.py::App.")
    file = Path(file_path).resolve()
    root = Path(args.root).resolve() if args.root else file.parent
    if not file.is_relative_to(root):
        raise ValueError(f"{file} is not inside --root {root}")
    module = ".".join(file.relative_to(root).with_suffix("").parts)
    name = session_name(args.session, class_name or file.stem)
    # Pin the team now, so a later `fal team set` cannot retarget the session.
    team = args.team or Config().get_internal("team")
    try:
        existing = find_session(name)
        ours = (
            existing.team,
            existing.host,
            existing.environment,
            existing.root,
        ) == (team, args.host, args.env, str(root))
    except DevAppError:
        ours = False
    exists = box_exists(name, team=team, host=args.host, environment=args.env)
    if exists and not ours and not args.replace:
        raise DevAppError(
            f"An app named {name!r} already exists and was not started from this "
            "machine. Pick another --session, or pass --replace to take it over."
        )

    timings: dict[str, float] = {}
    started = time.perf_counter()
    console.print(f"Deploying dev box [bold]{name}[/] ...")
    token = new_token()
    class_name, run_url = deploy_box(
        str(file),
        class_name,
        root=root,
        app_name=name,
        session=name,
        token=token,
        keep_alive=args.keep_alive,
        machine_type=args.machine_type,
        team=team,
        host=args.host,
        environment=args.env,
    )
    timings["deploy"] = time.perf_counter() - started
    session = Session(
        name=name,
        app_name=name,
        file_path=str(file),
        root=str(root),
        module=module,
        class_name=class_name,
        run_url=run_url,
        token=token,
        team=team,
        host=args.host,
        environment=args.env,
        machine_type=args.machine_type,
    )
    session.save()

    step = time.perf_counter()
    console.print("Waiting for a runner, then uploading the source tree ...")
    status = _boot(session)
    timings["runner+setup"] = time.perf_counter() - step
    app = status.get("app", {})
    summary = ", ".join(f"{k} {v:.1f}s" for k, v in timings.items())
    console.print(f"[green]Ready[/] ({summary}). Endpoints: {app.get('endpoints')}")
    console.print(
        "Next: `fal dev app request / --data '{...}'`, "
        "`fal dev app sync --watch`, `fal dev app stop`."
    )


def _sync_once(session, args, *, full: bool = False) -> None:
    """Sync, wait out a restart, then send the --request if one was given."""
    started = time.perf_counter()
    result = _sync(session, full=full)
    _report_sync(result, started)
    action = ((result or {}).get("reload") or {}).get("action")
    if action in ("restart", "pending"):
        _wait_ready(session)
    if args.request is not None and action in ("reload", "restart", "pending"):
        _do_request(session, args.request, _load_data(args.data), started)


def _sync_cmd(args):
    from fal.dev import app_sync
    from fal.dev.app_session import DevAppError

    session = _session(args)
    _ensure_booted(session)
    _sync_once(session, args, full=args.full)
    if not args.watch:
        return
    root = Path(session.root)
    _console().print(f"Watching {root} (Ctrl+C to stop) ...")
    signature = app_sync.stat_signature(app_sync.collect(root))
    try:
        while True:
            time.sleep(0.3)
            current = app_sync.stat_signature(app_sync.collect(root))
            if current == signature:
                continue
            signature = current
            try:
                _ensure_booted(session)
                _sync_once(session, args)
            except (DevAppError, OSError, ValueError) as exc:
                # Keep watching: the next save may fix it.
                _console().print(f"[red]{exc}[/]")
    except KeyboardInterrupt:
        pass


def _load_data(data: str | None) -> Any:
    if data is None:
        return {}
    if data.startswith("@"):
        return json.loads(Path(data[1:]).read_text())
    return json.loads(data)


def _do_request(
    session,
    path: str,
    body: Any,
    edit_started: float | None,
    *,
    warmups: int = 0,
    repeat: int = 1,
):
    from fal.dev.app_session import DevAppError, call

    console = _console()
    started = time.perf_counter()
    result = call(
        session,
        "request",
        {
            "path": path,
            "body": body,
            "method": "POST",
            "warmups": warmups,
            "repeat": repeat,
        },
        # A little past the box's own 3600 s limit, so its answer arrives.
        timeout=3660,
    )
    total = time.perf_counter() - started
    status = result["status_code"]
    color = "green" if status < 400 else "red"
    timing = f"app {result['elapsed_seconds']:.2f}s, round trip {total:.2f}s"
    if edit_started is not None:
        timing += f", edit to result {time.perf_counter() - edit_started:.2f}s"
    if warmups or repeat > 1:
        if "runs" not in result:
            raise DevAppError(
                "This box predates --warmups/--repeat. "
                "Run `fal dev app start` again to update it."
            )
        _report_runs(path, result, total)
        return result
    console.print(f"[{color}]{status}[/] {path} ({timing})")
    if "body" in result:
        try:
            print(json.dumps(json.loads(result["body"]), indent=2))
        except ValueError:
            print(result["body"])
    else:
        size = len(base64.b64decode(result["body_base64"]))
        print(f"<{size} bytes of {result['content_type']}>")
    return result


def _report_runs(path: str, result: dict, total: float) -> None:
    from fal.dev import app_bench

    console = _console()
    status = result["status_code"]
    color = "green" if status < 400 else "red"
    runs = [run["elapsed_seconds"] for run in result["runs"]]
    console.print(
        f"[{color}]{status}[/] {path}: {len(runs)} run(s) after "
        f"{len(result['warmup_runs'])} warmup(s), round trip {total:.1f}s"
    )
    if result["warmup_runs"]:
        warm = [run["elapsed_seconds"] for run in result["warmup_runs"]]
        console.print(f"  warmups  {app_bench.format_runs(warm)}")
    if runs:
        console.print(f"  runs     {app_bench.format_runs(runs)}")
    if status >= 400:
        console.print(f"[red]Run failed with {status}[/]; no statistics.")
    elif runs:
        console.print(f"  {app_bench.format_stats(app_bench.stats(runs))}")
    if result.get("stopped"):
        console.print(
            f"[red]Stopped early[/] after {len(runs)} run(s): {result['stopped']}"
        )


def _request(args):
    from fal.dev import app_bench

    session = _session(args)
    benchmark = args.warmups > 0 or args.repeat > 1 or args.name is not None
    if args.name is not None:
        app_bench.check_name(args.name)
    body = _load_data(args.data)
    _ensure_booted(session)
    result = _do_request(
        session,
        args.path,
        body,
        None,
        warmups=args.warmups,
        repeat=args.repeat,
    )
    if args.output_file and "body" in result:
        Path(args.output_file).write_text(result["body"])
    if benchmark and result["status_code"] >= 400:
        if args.name:
            _console().print("[red]Not saved[/]: a run failed.")
        if "body" in result:
            print(result["body"][-2000:], file=sys.stderr)
        raise SystemExit(1)
    if result.get("stopped"):
        if args.name:
            _console().print("[red]Not saved[/]: fewer runs than --repeat.")
        raise SystemExit(1)
    if not args.name:
        return
    if "runs" not in result:
        # A single request against a box that predates --repeat.
        result = {**result, "runs": [result], "warmup_runs": []}
    run = app_bench.record(
        name=args.name,
        session=session,
        path=args.path,
        body=body,
        result=result,
        warmups=args.warmups,
    )
    existed = app_bench.path_for(session.name, args.name).exists()
    saved = app_bench.save(run)
    verb = "Overwrote" if existed else "Saved"
    _console().print(f"{verb} benchmark [bold]{args.name}[/] at {saved}")


def _benchmark_compare(args):
    from fal.dev import app_bench
    from fal.dev.app_session import DevAppError, find_session

    try:
        session_name = find_session(args.session).name
    except DevAppError:
        # Files can still be compared by path without a session.
        if args.session:
            raise
        session_name = None
    baseline = app_bench.load(args.baseline, session_name)
    candidate = app_bench.load(args.candidate, session_name)
    console = _console()
    for mismatch in app_bench.mismatches(baseline, candidate):
        console.print(f"[yellow]Not the same benchmark[/] ({mismatch})")
    for line in app_bench.compare(baseline, candidate):
        print(line)


def _benchmark_list(args):
    from fal.dev import app_bench

    session = _session(args)
    runs = app_bench.saved(session.name)
    if not runs:
        _console().print(f"No saved benchmarks for {session.name}.")
        return
    for run in runs:
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(run["created_at"]))
        print(
            f"{run['name']:<20} {created}  {run['path']:<12} "
            f"x{run['repeat']}  {app_bench.format_stats(run['stats'])}"
        )


def _download(session, profile_id: str, name: str, target: Path) -> int:
    """Copy one profile file from the box in chunks; returns its size."""
    from fal.dev.app_session import call

    tmp = target.with_name(f".{target.name}.part")
    offset, size = 0, None
    try:
        with tmp.open("wb") as handle:
            while size is None or offset < size:
                chunk = call(
                    session,
                    "artifact",
                    {"id": profile_id, "name": name, "offset": offset},
                    timeout=600,
                )
                size = chunk["size"]
                data = base64.b64decode(chunk["data"])
                if not data and offset < size:
                    raise OSError(f"{name} stopped at {offset} of {size} bytes")
                handle.write(data)
                offset += len(data)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return size


def _mib(size: int) -> str:
    return f"{size / 1024 / 1024:.1f} MiB"


def _profile_one(session, args, path: str, body: Any, out: Path) -> bool:
    from fal.dev.app_session import call

    console = _console()
    console.print(
        f"Profiling [bold]{path}[/] ({args.mode}, {args.warmups} warmup(s)) ..."
    )
    payload: dict[str, Any] = {
        "path": path,
        "mode": args.mode,
        "warmups": args.warmups,
        "top": args.top,
    }
    if body is not None:
        payload["body"] = body
    result = call(session, "profile", payload, timeout=3660)
    request = result.get("request", {})
    status = request.get("status_code")
    if result.get("status") == "request_failed":
        console.print(f"[red]A warmup request to {path} returned {status}[/].")
        print(str(request.get("body", ""))[-2000:], file=sys.stderr)
        return False
    if (status or 0) >= 400:
        # The profile may exist, but of a failing request; not worth keeping.
        console.print(f"[red]The profiled request to {path} returned {status}[/].")
        print(str(request.get("body", ""))[-2000:], file=sys.stderr)
        return False
    if result.get("status") != "ok":
        detail = result.get("error") or result.get("status")
        console.print(f"[red]No profile for {path}[/]: {detail}")
        return False
    elapsed = result["elapsed_seconds"]
    line = f"[green]{status}[/] {path} in {elapsed:.2f}s under the profiler"
    if result.get("peak_memory_allocated_bytes"):
        peak = result["peak_memory_allocated_bytes"] / 1024**3
        line += f", peak GPU memory {peak:.1f} GiB"
    console.print(line)
    print(result["top_ops"])

    folder = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(path)}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "profile.json").write_text(json.dumps(result, indent=2) + "\n")
    _download(session, result["id"], "ops.txt", folder / "ops.txt")
    size = result["trace_bytes"]
    if size > args.max_trace_mb * 1024 * 1024:
        console.print(
            f"[yellow]Trace not downloaded[/]: {_mib(size)} is over "
            f"--max-trace-mb {args.max_trace_mb}. It stays on the box until "
            "newer profiles replace it. Use --light (GPU kernels only), a "
            "smaller request, or a higher --max-trace-mb."
        )
        console.print(f"Ops table: {folder / 'ops.txt'}")
        return True
    trace = folder / "trace.json.gz"
    _download(session, result["id"], "trace.json.gz", trace)
    console.print(f"Trace: {trace} ({_mib(size)})")
    console.print("Open it in https://ui.perfetto.dev (Open trace file).")
    return True


def _slug(path: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in path).strip("-") or "root"


def _profile(args):
    from fal.dev.app_session import DevAppError, call

    session = _session(args)
    if args.all == (args.path is not None):
        raise ValueError("Pass an endpoint path or --all.")
    if args.all and args.data is not None:
        raise ValueError("--data applies to one endpoint; drop --all.")
    out = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else PROFILES_DIR / session.name
    )
    _ensure_booted(session)
    if not args.all:
        body = _load_data(args.data) if args.data is not None else None
        if not _profile_one(session, args, args.path, body, out):
            raise SystemExit(1)
        return
    endpoints = call(session, "endpoints", timeout=120)["endpoints"]
    failed = False
    for entry in endpoints:
        if "error" in entry:
            _console().print(f"[yellow]Skipping {entry['path']}[/]: {entry['error']}")
            continue
        try:
            failed |= not _profile_one(session, args, entry["path"], None, out)
        except (DevAppError, OSError) as exc:
            _console().print(f"[red]{entry['path']}[/]: {exc}")
            failed = True
    if failed:
        raise SystemExit(1)


def _remember_overlay(session, result: dict[str, Any]) -> None:
    # The box's overlay is the truth; keep the session in step for replays.
    session.overlay = result["replay"]
    session.save()


def _print_layers(layers: list[dict[str, Any]]) -> None:
    console = _console()
    if not layers:
        console.print("No dependency overlay: the app runs on its own environment.")
        return
    console.print(f"{len(layers)} overlay layer(s), newest first:")
    for layer in layers:
        requested = escape(", ".join(layer["requested"]))
        console.print(f"  [bold]{layer['id']}[/] ({requested})")
        for package in layer["installed"]:
            base = package.get("base_version")
            shadows = f" (app has {base})" if base else " (new)"
            console.print(f"    {package['name']}=={package['version']}{shadows}")


def _deps_call(
    session, payload: dict[str, Any], *, remember: bool = True
) -> dict[str, Any]:
    from fal.dev.app_session import call

    result = call(session, "deps", payload, timeout=3600)
    if remember:
        _remember_overlay(session, result)
    return result


def _after_change(session, result: dict[str, Any], restart: bool) -> None:
    console = _console()
    if result.get("restarted"):
        console.print("Restarting the app so the overlay takes effect ...")
        _wait_ready(session)
        status = _deps_call(session, {"action": "status"})
        _print_app_versions(status)
    elif result.get("restart_needed") and not restart:
        console.print("Run `fal dev app restart` to use it.")


def _print_app_versions(status: dict[str, Any]) -> None:
    versions = status.get("app_versions") or {}
    if "error" in versions:
        error = escape(versions["error"])
        _console().print(f"[yellow]Could not ask the app[/]: {error}")
        return
    for name, found in sorted(versions.items()):
        if found:
            print(f"  app imports {name}=={found['version']} from {found['path']}")


def _deps_install(args):
    session = _session(args)
    _ensure_booted(session)
    payload = {
        "action": "install",
        "packages": args.packages,
        "no_deps": args.no_deps,
        "pre": args.pre,
        "index_url": args.index_url,
        "extra_index_urls": args.extra_index_url or [],
        "force": args.force,
        "restart": args.restart,
    }
    packages = escape(" ".join(args.packages))
    _console().print(f"Installing {packages} into an overlay ...")
    result = _deps_call(session, payload)
    layer = result.get("layer")
    if layer is None:
        _console().print(
            f"Nothing to install: {result.get('reason', 'already satisfied')}."
        )
        return
    pins = ", ".join(
        f"{p['name']}=={p['version']}"
        + (f" (app has {p['base_version']})" if p.get("base_version") else "")
        for top in result["layers"][:1]
        for p in top["installed"]
    )
    _console().print(f"[green]Installed[/] layer {layer['id']}: {pins}")
    _after_change(session, result, args.restart)


def _deps_uninstall(args):
    session = _session(args)
    _ensure_booted(session)
    result = _deps_call(
        session,
        {"action": "uninstall", "packages": args.packages, "restart": args.restart},
    )
    for layer in result["removed"]:
        names = ", ".join(p["name"] for p in layer["installed"])
        _console().print(f"Removed layer {layer['id']} ({names})")
    _after_change(session, result, args.restart)


def _deps_reset(args):
    session = _session(args)
    _ensure_booted(session)
    result = _deps_call(session, {"action": "reset", "restart": args.restart})
    _console().print(f"Removed {result['removed']} overlay layer(s).")
    _after_change(session, result, args.restart)


def _deps_status(args):
    session = _session(args)
    _ensure_booted(session)
    status = _deps_call(session, {"action": "status", "names": args.packages})
    if args.json:
        # replay has index URLs as given, credentials included.
        status.pop("replay", None)
        print(json.dumps(status, indent=2))
        return
    _print_layers(status["layers"])
    if status.get("restart_needed"):
        _console().print(
            "[yellow]The running app predates the last change[/]: "
            "run `fal dev app restart`."
        )
    _print_app_versions(status)


def _restart(args):
    from fal.dev.app_session import call

    session = _session(args)
    started = time.perf_counter()
    if _ensure_booted(session):
        _console().print(f"[green]Ready[/] after {time.perf_counter() - started:.1f}s.")
        return
    if args.sync:
        _sync(session, reload=False)
    after = call(session, "logs", {"lines": 0})["total"]
    call(session, "start", {}, timeout=300)
    _wait_ready(session, after=after)
    _console().print(f"[green]Ready[/] after {time.perf_counter() - started:.1f}s.")


def _logs(args):
    from fal.dev.app_session import call

    session = _session(args)
    logs = call(session, "logs", {"lines": args.lines})
    _print_lines(logs["lines"])
    after = logs["total"]
    while args.follow:
        time.sleep(POLL_SECONDS)
        logs = call(session, "logs", {"after": after})
        _print_lines(logs["lines"])
        after = logs["total"]


def _status(args):
    from fal.dev.app_session import DevAppError, call

    session = _session(args)
    info: dict[str, Any] = {
        "session": session.name,
        "app": f"{session.file_path}::{session.class_name}",
        "box": session.run_url,
        "team": session.team,
    }
    try:
        info["remote"] = call(session, "status")
    except DevAppError as exc:
        info["remote"] = {"error": str(exc)}
    print(json.dumps(info, indent=2, default=str))


def _stop(args):
    from fal.dev.app_session import stop_box

    session = _session(args)
    killed, existed = stop_box(session)
    session.delete()
    deleted = "deleted the app" if existed else "the app was already gone"
    _console().print(
        f"Stopped {session.name}: {deleted}, killed {len(killed)} runner(s)."
    )


def add_parser(main_subparsers, parents=None):
    """Register `app` under a `fal dev` subparser group."""
    parents = parents or []
    app_help = "A dev box for an existing fal App, with hot reload."
    parser = main_subparsers.add_parser(
        "app",
        description=__doc__,
        help=app_help,
        parents=parents,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        dest="cmd",
        required=True,
        parser_class=FalClientParser,
    )
    session_parser = argparse.ArgumentParser(add_help=False)
    session_parser.add_argument(
        "--session",
        "-s",
        help="Box name, always prefixed dev- (defaults to the only session).",
    )

    start = subparsers.add_parser(
        "start",
        help="Deploy a dev box from an app and run its setup() once.",
        parents=[*parents, session_parser],
    )
    start.add_argument(
        "app_ref", action=RefAction, help="path/to/app.py or path/to/app.py::App"
    )
    start.add_argument("--machine-type", help="Override the app's machine type.")
    start.add_argument(
        "--root", help="Source tree to sync (defaults to the app file's folder)."
    )
    add_env_argument(start)
    start.add_argument(
        "--replace",
        action="store_true",
        help="Take over an existing app with the same dev- name.",
    )
    start.add_argument(
        "--keep-alive",
        type=int,
        default=DEFAULT_KEEP_ALIVE,
        help="Seconds without requests before the box shuts down "
        f"(default {DEFAULT_KEEP_ALIVE}).",
    )
    start.set_defaults(func=_start)

    sync = subparsers.add_parser(
        "sync",
        help="Upload changed files; endpoints reload in place.",
        parents=[*parents, session_parser],
    )
    sync.add_argument("--watch", action="store_true", help="Keep syncing on save.")
    sync.add_argument("--full", action="store_true", help="Upload every file.")
    sync.add_argument(
        "--request",
        metavar="PATH",
        help="Send a request to PATH after each sync (with --data).",
    )
    sync.add_argument("--data", help="JSON body, or @file.json.")
    sync.set_defaults(func=_sync_cmd)

    request = subparsers.add_parser(
        "request",
        help="Send a request to the app on the box.",
        parents=[*parents, session_parser],
    )
    request.add_argument("path", help="Endpoint path, e.g. /")
    request.add_argument("--data", help="JSON body, or @file.json.")
    request.add_argument("--output-file", help="Write the response body here.")
    request.add_argument(
        "--warmups",
        type=_bounded(0, MAX_WARMUPS),
        default=0,
        help="Requests to send first and leave out of the timings.",
    )
    request.add_argument(
        "--repeat",
        type=_bounded(1, MAX_REPEAT),
        default=1,
        help="Time this many requests (run back to back on the box) and report "
        "median, p90, min and max.",
    )
    request.add_argument(
        "--name", help="Save the timings as a named benchmark, for `benchmark compare`."
    )
    request.set_defaults(func=_request)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Compare saved benchmark runs (`request --repeat N --name NAME`).",
        parents=parents,
    )
    benchmark_commands = benchmark.add_subparsers(
        title="Commands",
        metavar="command",
        dest="benchmark_cmd",
        required=True,
        parser_class=FalClientParser,
    )
    compare = benchmark_commands.add_parser(
        "compare",
        help="Show two saved runs side by side.",
        parents=[*parents, session_parser],
    )
    compare.add_argument("baseline", help="Benchmark name or JSON file.")
    compare.add_argument("candidate", help="Benchmark name or JSON file.")
    compare.set_defaults(func=_benchmark_compare)
    listing = benchmark_commands.add_parser(
        "list", help="List saved runs.", parents=[*parents, session_parser]
    )
    listing.set_defaults(func=_benchmark_list)

    profile = subparsers.add_parser(
        "profile",
        help="Run one request under the torch profiler and download the trace.",
        description="Warm the endpoint up, run one request under torch.profiler, "
        "print the top ops and save a Perfetto/Chrome trace locally. Without "
        "--data the request is built from the endpoint's input model "
        "(examples and defaults). CPU ops are recorded in the thread that runs "
        "the endpoint: work an async endpoint hands to other threads shows up "
        "as GPU kernels only.",
        parents=[*parents, session_parser],
    )
    profile.add_argument("path", nargs="?", help="Endpoint path, e.g. /")
    profile.add_argument(
        "--all", action="store_true", help="Profile every endpoint in turn."
    )
    profile.add_argument("--data", help="JSON body, or @file.json.")
    depth = profile.add_mutually_exclusive_group()
    depth.add_argument(
        "--light",
        dest="mode",
        action="store_const",
        const="light",
        help="GPU kernels only: the smallest trace.",
    )
    depth.add_argument(
        "--deep",
        dest="mode",
        action="store_const",
        const="deep",
        help="Add shapes, memory, stacks and FLOPs (large traces).",
    )
    profile.set_defaults(mode="standard")
    profile.add_argument(
        "--warmups",
        type=_bounded(0, 10),
        default=1,
        help="Unprofiled requests to send first (default 1).",
    )
    profile.add_argument(
        "--top", type=_bounded(1, 100), default=15, help="Ops to print (default 15)."
    )
    profile.add_argument(
        "--output-dir",
        help="Where to save profiles (default ~/.fal/dev/profiles/<session>).",
    )
    profile.add_argument(
        "--max-trace-mb",
        type=_bounded(1, 100_000),
        default=200,
        help="Skip downloading compressed traces over this size (default 200).",
    )
    profile.set_defaults(func=_profile)

    deps = subparsers.add_parser(
        "deps",
        help="Try other package versions without rebuilding the box.",
        description="Install packages into an overlay that the app imports ahead "
        "of its own environment (which is never modified), then restart the app. "
        "Only what the app's environment does not already satisfy is installed. "
        "The overlay is kept with the session and installed again on a new runner.",
        parents=parents,
    )
    deps_commands = deps.add_subparsers(
        title="Commands",
        metavar="command",
        dest="deps_cmd",
        required=True,
        parser_class=FalClientParser,
    )
    no_restart = argparse.ArgumentParser(add_help=False)
    no_restart.add_argument(
        "--no-restart",
        dest="restart",
        action="store_false",
        help="Change the overlay without restarting the app.",
    )
    install = deps_commands.add_parser(
        "install",
        help="Install packages into a new overlay layer.",
        parents=[*parents, session_parser, no_restart],
    )
    install.add_argument("packages", nargs="+", help="Requirement specs.")
    install.add_argument(
        "--no-deps", action="store_true", help="Install exactly these packages."
    )
    install.add_argument("--pre", action="store_true", help="Allow pre-releases.")
    install.add_argument("--index-url", help="Package index to use.")
    install.add_argument(
        "--extra-index-url",
        action="append",
        help="Another index (repeatable). The highest version across indexes "
        "wins, so a public package can shadow a private one of the same name.",
    )
    install.add_argument(
        "--force",
        action="store_true",
        help="Allow changing packages the box's app server runs on "
        "(fastapi, pydantic, uvicorn, ...).",
    )
    install.set_defaults(func=_deps_install)
    uninstall = deps_commands.add_parser(
        "uninstall",
        help="Remove the overlay layers that installed these packages.",
        parents=[*parents, session_parser, no_restart],
    )
    uninstall.add_argument("packages", nargs="+")
    uninstall.set_defaults(func=_deps_uninstall)
    reset = deps_commands.add_parser(
        "reset",
        help="Remove every overlay layer.",
        parents=[*parents, session_parser, no_restart],
    )
    reset.set_defaults(func=_deps_reset)
    deps_status = deps_commands.add_parser(
        "status",
        help="Show overlay layers and the versions the app imports.",
        parents=[*parents, session_parser],
    )
    deps_status.add_argument(
        "packages", nargs="*", help="Also show which version the app imports."
    )
    deps_status.add_argument("--json", action="store_true")
    deps_status.set_defaults(func=_deps_status)

    restart = subparsers.add_parser(
        "restart",
        help="Restart the app process; setup() runs again.",
        parents=[*parents, session_parser],
    )
    restart.add_argument(
        "--no-sync", dest="sync", action="store_false", help="Skip syncing first."
    )
    restart.set_defaults(func=_restart)

    logs = subparsers.add_parser(
        "logs",
        help="Show the app process's output.",
        parents=[*parents, session_parser],
    )
    logs.add_argument("--lines", type=int, default=200)
    logs.add_argument("--follow", "-f", action="store_true")
    logs.set_defaults(func=_logs)

    status = subparsers.add_parser(
        "status", help="Show the box and app state.", parents=[*parents, session_parser]
    )
    status.set_defaults(func=_status)

    stop = subparsers.add_parser(
        "stop",
        help="Shut the box down and delete it.",
        parents=[*parents, session_parser],
    )
    stop.set_defaults(func=_stop)
    return parser
