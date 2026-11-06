"""`fal dev app`: handlers for a GPU dev box of an existing fal App."""

from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from rich.markup import escape

from fal.config import Config
from fal.console import console
from fal.dev.app import bench, sync
from fal.dev.app.session import (
    DevAppError,
    Session,
    box_exists,
    call,
    deploy_box,
    find_session,
    new_token,
    package_root,
    resolve_app,
    session_name,
    stop_box,
)
from fal.dev.remote.deps import redact
from fal.dev.remote.holder import STOPPED_PHASES, check_session_env

PROFILES_DIR = Path("~/.fal/dev/profiles").expanduser()
POLL_SECONDS = 1.0
PLACE_BATCH = 2000
PLACE_BATCH_BYTES = 512 * 1024 * 1024


def _print_lines(lines: list[str], width: int = 400) -> None:
    for line in lines:
        print(line if len(line) <= width else f"{line[:width]} ...", flush=True)


def _wait_ready(session, *, after: int | None = None, timeout: float = 3600) -> dict:
    """Stream the app's log until it is ready, or raise if it stops."""
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


def _collect(
    session, skipped: list[str] | None = None, withheld: list[str] | None = None
):
    return sync.collect(
        Path(session.root),
        skipped,
        ignore=session.ignore,
        ignore_base=session.ignore_base,
        withheld=withheld,
    )


def _place_from_store(
    session, files, manifest: dict[str, str]
) -> dict[str, str] | None:
    """Have the box copy the files its team's object store already holds.

    Returns the manifest of the files now in place (the rest must be sent),
    or None when the box has no store.
    """
    try:
        found = call(
            session,
            "objects",
            {"action": "missing", "digests": sorted(set(manifest.values()))},
            timeout=600,
        )
    except DevAppError as exc:
        # 404: a box from before the object store. Anything else: say so.
        if "(404)" not in str(exc):
            console.print(f"[yellow]Object store skipped[/]: {escape(str(exc))}")
        return None
    if not found.get("enabled"):
        return None
    missing = set(found["missing"])
    executable: dict[str, bool] = {}
    for rel, digest in manifest.items():
        if digest in missing:
            continue
        try:
            executable[rel] = bool(files[rel].stat().st_mode & 0o111)
        except OSError:
            continue  # gone since collect; the sync sorts it out
    placed: dict[str, str] = {}
    for batch in sync.batches(
        files, sorted(executable), PLACE_BATCH_BYTES, PLACE_BATCH
    ):
        entries = {rel: [manifest[rel], executable[rel]] for rel in batch}
        try:
            result = call(
                session, "objects", {"action": "place", "files": entries}, timeout=600
            )
        except DevAppError as exc:
            # Whatever was not placed is uploaded instead.
            console.print(f"[yellow]Object store place failed[/]: {escape(str(exc))}")
            continue
        placed.update({rel: manifest[rel] for rel in result["placed"]})
    return placed


def _sync(
    session, *, full: bool = False, reload: bool = True, from_store: bool = False
) -> dict[str, Any] | None:
    skipped: list[str] = []
    withheld: list[str] = []
    files = _collect(session, skipped, withheld)
    if skipped:
        console.print(
            f"[yellow]Not synced (over 10 MiB)[/]: {escape(', '.join(skipped[:8]))}"
        )
    if withheld and full:
        console.print(
            "[yellow]Not synced (key or credential files)[/]: "
            f"{escape(', '.join(withheld[:8]))}"
        )
    manifest = sync.manifest(files)
    previous = {} if full else session.manifest
    placed = None
    if full and from_store:
        stored = _place_from_store(session, files, manifest)
        if stored is not None:
            previous, placed = stored, len(stored)
    changed, deleted = sync.diff(previous, manifest)
    if not changed and not deleted and not full:
        return None
    # Large trees go in several requests. Deletes and pruning ride on the
    # first, before any writes; the reload on the last.
    groups = sync.batches(files, changed)
    result: dict[str, Any] = {}
    for index, group in enumerate(groups):
        last = index == len(groups) - 1
        payload: dict[str, Any] = {"reload": reload and last}
        if group:
            payload["archive"] = sync.pack(files, group)
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
    result["placed"] = placed
    return result


def _report_sync(result: dict[str, Any] | None, started: float) -> None:
    if result is None:
        console.print("No changes.")
        return
    names = escape(", ".join((result["changed"] + result["deleted"])[:8]))
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
            f"[yellow]Full restart[/]: {escape(reload['reason'])}. setup() runs again."
        )
    elif action == "pending":
        console.print(
            f"Saved; {escape(reload['reason'])}, so the edit reloads once it is ready."
        )
    elif action == "rebuild":
        console.print(
            f"[red]Not reloaded[/]: {escape(reload['reason'])}. "
            "Run `fal dev app start` again to rebuild the box."
        )
    else:
        console.print(f"[red]Not reloaded[/]: {escape(str(reload.get('reason')))}")
        if reload.get("error"):
            print(reload["error"], file=sys.stderr)


def _boot(session) -> dict:
    """Upload the whole source tree and the SDK, then run setup() on the box."""
    started = time.perf_counter()
    result = _sync(session, full=True, reload=False, from_store=True) or {}
    uploaded = len(result.get("changed", []))
    elapsed = time.perf_counter() - started
    if result.get("placed") is None:
        console.print(f"Source: {uploaded} file(s) uploaded ({elapsed:.1f}s).")
    else:
        console.print(
            f"Source: {result['placed']} file(s) copied from the team's object "
            f"store on the box, {uploaded} uploaded ({elapsed:.1f}s)."
        )
    _replay_overlay(session)
    sdk = sync.sdk_files()
    after = call(session, "logs", {"lines": 0}, timeout=600)["total"]
    call(
        session,
        "start",
        {
            "module": session.module,
            "class_name": session.class_name,
            "sdk_archive": sync.pack(sdk, sorted(sdk)),
            "sdk_requirements": sync.sdk_requirements(),
            "layout": {"paths": session.paths, "workdir": session.workdir},
            "session_env": session.session_env,
        },
        timeout=300,
    )
    console.print(f"Running {session.class_name}.setup() on the box ...")
    return _wait_ready(session, after=after)


def _replay_overlay(session) -> None:
    """Install the session's dependency overlay on a fresh runner, as pinned.

    A layer that no longer installs is dropped with a warning, so the box
    still starts and `deps` commands keep working.
    """
    if not session.overlay:
        return
    wanted = [layer["packages"] for layer in session.overlay]
    status = _deps_call(session, {"action": "status"}, remember=False)
    if [layer["packages"] for layer in status["replay"]] == wanted:
        session.overlay = status["replay"]
        session.save()
        return
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
    # A cold box first waits for a runner, which can take minutes.
    status = call(session, "status", timeout=600)
    if status["phase"] != "idle" or status["module"]:
        return False
    console.print(
        "[yellow]The box is on a new runner[/] (idle shutdown or replacement): "
        "syncing the source and running setup() again."
    )
    _boot(session)
    return True


def _booted(args) -> Session:
    """The named session, booted again if its box was replaced."""
    session = find_session(args.name)
    _ensure_booted(session)
    return session


def cmd_start(args):
    session_env = check_session_env(list(args.session_env or []))
    file_path, class_name, app_data = resolve_app(args.app_ref)
    file = Path(file_path)
    import_root = Path(args.root).resolve() if args.root else package_root(file)
    if not file.is_relative_to(import_root):
        raise ValueError(f"{file} is not inside --root {import_root}")
    module = ".".join(file.relative_to(import_root).with_suffix("").parts)
    # A pyproject.toml app is named after its entry there, as on a deploy.
    ref_is_file = (args.app_ref[0] or "").endswith(".py")
    default_name = class_name or file.stem if ref_is_file else args.app_ref[0]
    name = session_name(args.name, default_name)
    # Pin the team now, so a later `fal team set` cannot retarget the session.
    # A team in pyproject.toml is where the app is deployed for real; the box
    # always goes to the caller's own team.
    team = args.team or Config().get_internal("team")
    if app_data.team and app_data.team != team:
        console.print(
            escape(
                f"Ignoring team {app_data.team!r} from pyproject.toml: the box is "
                f"deployed to {team or 'your personal account'}."
            )
        )
    try:
        existing = find_session(name)
        ours = (
            existing.team,
            existing.host,
            existing.environment,
            existing.file_path,
        ) == (team, args.host, args.env, str(file))
    except DevAppError:
        ours = False
    exists = box_exists(name, team=team, host=args.host, environment=args.env)
    if exists and not ours and not args.replace:
        raise DevAppError(
            f"An app named {name!r} already exists and was not started from this "
            "machine. Pick another --name, or pass --replace to take it over."
        )

    timings: dict[str, float] = {}
    started = time.perf_counter()
    console.print(f"Deploying dev box [bold]{escape(name)}[/] ...")
    token = new_token()
    deployed = deploy_box(
        str(file),
        class_name,
        app_data=app_data,
        root=import_root,
        explicit_root=bool(args.root),
        app_name=name,
        session=name,
        token=token,
        keep_alive=args.keep_alive,
        machine_type=args.machine_type,
        team=team,
        host=args.host,
        environment=args.env,
    )
    layout = deployed.layout
    timings["deploy"] = time.perf_counter() - started
    session = Session(
        name=name,
        app_name=name,
        file_path=str(file),
        root=str(layout.root),
        module=module,
        class_name=deployed.class_name,
        run_url=deployed.run_url,
        token=token,
        team=team,
        host=args.host,
        environment=args.env,
        machine_type=args.machine_type,
        paths=layout.paths,
        workdir=layout.workdir,
        ignore=layout.ignore,
        ignore_base=layout.ignore_base,
        session_env=session_env,
    )
    session.save()

    step = time.perf_counter()
    console.print(
        f"Waiting for a runner, then uploading {escape(str(layout.root))} ..."
    )
    status = _boot(session)
    timings["runner+setup"] = time.perf_counter() - step
    app = status.get("app", {})
    summary = ", ".join(f"{k} {v:.1f}s" for k, v in timings.items())
    console.print(
        f"[green]Ready[/] ({summary}). Endpoints: {escape(str(app.get('endpoints')))}"
    )
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


def cmd_sync(args):
    session = _booted(args)
    _sync_once(session, args, full=args.full)
    if not args.watch:
        return
    root = Path(session.root)
    console.print(f"Watching {escape(str(root))} (Ctrl+C to stop) ...")
    signature = sync.stat_signature(_collect(session))
    try:
        while True:
            time.sleep(0.3)
            current = sync.stat_signature(_collect(session))
            if current == signature:
                continue
            signature = current
            try:
                _ensure_booted(session)
                _sync_once(session, args)
            except (DevAppError, OSError, ValueError) as exc:
                # Keep watching: the next save may fix it.
                console.print(f"[red]{escape(str(exc))}[/]")
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
    console.print(f"[{color}]{status}[/] {escape(path)} ({timing})")
    if "body" in result:
        try:
            print(json.dumps(json.loads(result["body"]), indent=2))
        except ValueError:
            print(result["body"])
    else:
        size = len(base64.b64decode(result["body_base64"]))
        console.print(escape(f"<{size} bytes of {result['content_type']}>"))
    return result


def _report_runs(path: str, result: dict, total: float) -> None:
    status = result["status_code"]
    color = "green" if status < 400 else "red"
    runs = [run["elapsed_seconds"] for run in result["runs"]]
    console.print(
        f"[{color}]{status}[/] {escape(path)}: {len(runs)} run(s) after "
        f"{len(result['warmup_runs'])} warmup(s), round trip {total:.1f}s"
    )
    if result["warmup_runs"]:
        warm = [run["elapsed_seconds"] for run in result["warmup_runs"]]
        console.print(f"  warmups  {escape(bench.format_runs(warm))}")
    if runs:
        console.print(f"  runs     {escape(bench.format_runs(runs))}")
    if status >= 400:
        console.print(f"[red]Run failed with {status}[/]; no statistics.")
    elif runs:
        console.print(f"  {escape(bench.format_stats(bench.stats(runs)))}")
    if result.get("stopped"):
        console.print(
            f"[red]Stopped early[/] after {len(runs)} run(s): "
            f"{escape(str(result['stopped']))}"
        )


def cmd_request(args):
    benchmark = args.warmups > 0 or args.repeat > 1 or args.save is not None
    if args.save is not None:
        bench.check_name(args.save)
    body = _load_data(args.data)
    session = _booted(args)
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
        if args.save:
            console.print("[red]Not saved[/]: a run failed.")
        if "body" in result:
            print(result["body"][-2000:], file=sys.stderr)
        raise SystemExit(1)
    if result.get("stopped"):
        if args.save:
            console.print("[red]Not saved[/]: fewer runs than --repeat.")
        raise SystemExit(1)
    if not args.save:
        return
    if "runs" not in result:
        # A single request against a box that predates --repeat.
        result = {**result, "runs": [result], "warmup_runs": []}
    run = bench.record(
        name=args.save,
        session=session,
        path=args.path,
        body=body,
        result=result,
        warmups=args.warmups,
    )
    existed = bench.path_for(session.name, args.save).exists()
    saved = bench.save(run)
    verb = "Overwrote" if existed else "Saved"
    console.print(
        f"{verb} benchmark [bold]{escape(args.save)}[/] at {escape(str(saved))}"
    )


def cmd_benchmark_compare(args):
    try:
        name = find_session(args.name).name
    except DevAppError:
        # Files can still be compared by path without a session.
        if args.name:
            raise
        name = None
    baseline = bench.load(args.baseline, name)
    candidate = bench.load(args.candidate, name)
    for mismatch in bench.mismatches(baseline, candidate):
        console.print(f"[yellow]Not the same benchmark[/] ({escape(mismatch)})")
    for line in bench.compare(baseline, candidate):
        console.print(escape(line))


def cmd_benchmark_list(args):
    session = find_session(args.name)
    runs = bench.saved(session.name)
    if not runs:
        console.print(f"No saved benchmarks for {escape(session.name)}.")
        return
    for run in runs:
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(run["created_at"]))
        console.print(
            escape(
                f"{run['name']:<20} {created}  {run['path']:<12} "
                f"x{run['repeat']}  {bench.format_stats(run['stats'])}"
            )
        )


def _download(session, profile_id: str, name: str, target: Path) -> int:
    """Copy one profile file from the box in chunks; returns its size."""
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
    console.print(
        f"Profiling [bold]{escape(path)}[/] ({args.mode}, {args.warmups} warmup(s)) ..."
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
        console.print(f"[red]A warmup request to {escape(path)} returned {status}[/].")
        print(str(request.get("body", ""))[-2000:], file=sys.stderr)
        return False
    if (status or 0) >= 400:
        # The profile may exist, but of a failing request; not worth keeping.
        console.print(
            f"[red]The profiled request to {escape(path)} returned {status}[/]."
        )
        print(str(request.get("body", ""))[-2000:], file=sys.stderr)
        return False
    if result.get("status") != "ok":
        detail = result.get("error") or result.get("status")
        console.print(f"[red]No profile for {escape(path)}[/]: {escape(str(detail))}")
        return False
    elapsed = result["elapsed_seconds"]
    line = f"[green]{status}[/] {escape(path)} in {elapsed:.2f}s under the profiler"
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
        console.print(f"Ops table: {escape(str(folder / 'ops.txt'))}")
        return True
    trace = folder / "trace.json.gz"
    _download(session, result["id"], "trace.json.gz", trace)
    console.print(f"Trace: {escape(str(trace))} ({_mib(size)})")
    console.print("Open it in https://ui.perfetto.dev (Open trace file).")
    return True


def _slug(path: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in path).strip("-") or "root"


def cmd_profile(args):
    if args.all == (args.path is not None):
        raise ValueError("Pass an endpoint path or --all.")
    if args.all and args.data is not None:
        raise ValueError("--data applies to one endpoint; drop --all.")
    session = _booted(args)
    out = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else PROFILES_DIR / session.name
    )
    if not args.all:
        body = _load_data(args.data) if args.data is not None else None
        if not _profile_one(session, args, args.path, body, out):
            raise SystemExit(1)
        return
    endpoints = call(session, "endpoints", timeout=120)["endpoints"]
    failed = False
    for entry in endpoints:
        if "error" in entry:
            console.print(
                f"[yellow]Skipping {escape(entry['path'])}[/]: {escape(entry['error'])}"
            )
            continue
        try:
            failed |= not _profile_one(session, args, entry["path"], None, out)
        except (DevAppError, OSError) as exc:
            console.print(f"[red]{escape(entry['path'])}[/]: {escape(str(exc))}")
            failed = True
    if failed:
        raise SystemExit(1)


def _remember_overlay(session, result: dict[str, Any]) -> None:
    # The box's overlay is the truth; keep the session in step for replays.
    session.overlay = result["replay"]
    session.save()


def _print_layers(layers: list[dict[str, Any]]) -> None:
    if not layers:
        console.print("No dependency overlay: the app runs on its own environment.")
        return
    console.print(f"{len(layers)} overlay layer(s), newest first:")
    for layer in layers:
        requested = escape(", ".join(layer["requested"]))
        console.print(f"  [bold]{escape(layer['id'])}[/] ({requested})")
        for package in layer["installed"]:
            base = package.get("base_version")
            shadows = f" (app has {base})" if base else " (new)"
            console.print(
                escape(f"    {package['name']}=={package['version']}{shadows}")
            )


def _deps_call(
    session, payload: dict[str, Any], *, remember: bool = True
) -> dict[str, Any]:
    result = call(session, "deps", payload, timeout=3600)
    if remember:
        _remember_overlay(session, result)
    return result


def _after_change(session, result: dict[str, Any], restart: bool) -> None:
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
        console.print(f"[yellow]Could not ask the app[/]: {error}")
        return
    for name, found in sorted(versions.items()):
        if found:
            console.print(
                escape(f"  app imports {name}=={found['version']} from {found['path']}")
            )


# A package index that refused the box's credentials (or lack of them).
AUTH_REFUSED = re.compile(
    r"unauthori[sz]ed|forbidden|\b40[13]\b (client error|error)", re.IGNORECASE
)
INDEX_AUTH_HINT = (
    "A package index refused the box. The platform authenticates some indexes "
    "only while it builds an app's environment, so a running box cannot install "
    "from them: add the package to the app's requirements and run "
    "`fal dev app start` again."
)


def cmd_deps_install(args):
    session = _booted(args)
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
    console.print(f"Installing {packages} into an overlay ...")
    try:
        result = _deps_call(session, payload)
    except DevAppError as exc:
        # Only the box's own install errors (409), not a refused CLI key.
        message = str(exc)
        if message.startswith("deps failed (409)") and AUTH_REFUSED.search(message):
            raise DevAppError(f"{exc}\n{INDEX_AUTH_HINT}") from exc
        raise
    layer = result.get("layer")
    if layer is None:
        reason = escape(str(result.get("reason", "already satisfied")))
        console.print(f"Nothing to install: {reason}.")
        return
    pins = ", ".join(
        f"{p['name']}=={p['version']}"
        + (f" (app has {p['base_version']})" if p.get("base_version") else "")
        for top in result["layers"][:1]
        for p in top["installed"]
    )
    console.print(f"[green]Installed[/] layer {escape(layer['id'])}: {escape(pins)}")
    _after_change(session, result, args.restart)


def cmd_deps_uninstall(args):
    session = _booted(args)
    result = _deps_call(
        session,
        {"action": "uninstall", "packages": args.packages, "restart": args.restart},
    )
    for layer in result["removed"]:
        names = ", ".join(p["name"] for p in layer["installed"])
        console.print(f"Removed layer {escape(layer['id'])} ({escape(names)})")
    _after_change(session, result, args.restart)


def cmd_deps_reset(args):
    session = _booted(args)
    result = _deps_call(session, {"action": "reset", "restart": args.restart})
    console.print(f"Removed {result['removed']} overlay layer(s).")
    _after_change(session, result, args.restart)


def cmd_deps_status(args):
    session = _booted(args)
    status = _deps_call(session, {"action": "status", "names": args.packages})
    if args.json:
        # replay has index URLs as given, credentials included.
        status.pop("replay", None)
        print(json.dumps(status, indent=2))
        return
    _print_layers(status["layers"])
    if status.get("restart_needed"):
        console.print(
            "[yellow]The running app predates the last change[/]: "
            "run `fal dev app restart`."
        )
    _print_app_versions(status)


def cmd_restart(args):
    session = find_session(args.name)
    started = time.perf_counter()
    if _ensure_booted(session):
        console.print(f"[green]Ready[/] after {time.perf_counter() - started:.1f}s.")
        return
    if args.sync:
        _sync(session, reload=False)
    after = call(session, "logs", {"lines": 0})["total"]
    call(session, "start", {}, timeout=300)
    _wait_ready(session, after=after)
    console.print(f"[green]Ready[/] after {time.perf_counter() - started:.1f}s.")


def cmd_logs(args):
    session = find_session(args.name)
    logs = call(session, "logs", {"lines": args.lines})
    _print_lines(logs["lines"])
    after = logs["total"]
    while args.follow:
        time.sleep(POLL_SECONDS)
        logs = call(session, "logs", {"after": after})
        _print_lines(logs["lines"])
        after = logs["total"]


def cmd_status(args):
    session = find_session(args.name)
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


def cmd_stop(args):
    session = find_session(args.name)
    killed, existed = stop_box(session)
    session.delete()
    deleted = "deleted the app" if existed else "the app was already gone"
    console.print(f"Stopped {session.name}: {deleted}, killed {len(killed)} runner(s).")
