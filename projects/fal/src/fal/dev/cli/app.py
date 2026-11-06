"""`fal dev app`: handlers for a GPU dev box of an existing fal App."""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path
from typing import Any

from rich.markup import escape

from fal.api.client import SyncServerlessClient
from fal.console import console
from fal.dev.app import bench, client, deploy
from fal.dev.app.session import Session, find_session, session_name, slug
from fal.dev.box import log
from fal.dev.errors import DevError
from fal.dev.remote.protocol import BODY_TAIL_CHARS, check_session_env, redact
from fal.dev.state import STATE_DIR
from fal.dev.team import announce, confirm_deploy, key_account, resolve_account

PROFILES_DIR = STATE_DIR / "profiles"


def _check_account(session: Session) -> None:
    """Refuse to act on a box started under an API key without that key's account.

    A login would reach fal as another account. When the key's account name
    cannot be read the check is presence-only: any API key passes.
    """
    if not session.account.via_key:
        return
    current = key_account()
    if current is None:
        raise DevError(
            f"{session.name} was started with an API key: set FAL_KEY to use it"
        )
    started, now = session.account.team, current.team
    if started and now and started.lower() != now.lower():
        raise DevError(
            f"{session.name} was started with the API key of {started!r}, "
            f"but FAL_KEY belongs to {now!r}"
        )


def _session(args, *, acts_as_account: bool = False) -> Session:
    """The named session, with the account it runs as announced on stderr.

    Commands that act on the box as its account pass `acts_as_account`; the
    ones that only look at it or stop it (status, logs, stop) never refuse.
    """
    session = find_session(args.name)
    if acts_as_account:
        _check_account(session)
    announce(session.account)
    if args.team and args.team.lower() != (session.account.team or "").lower():
        log(f"--team {args.team} is ignored: the box keeps its team")
    return session


def _booted(args) -> Session:
    """The named session, booted again if its box was replaced."""
    session = _session(args, acts_as_account=True)
    with client.explain_index_refusal(session.account):
        client.ensure_booted(session)
    return session


def _load_data(data: str | None) -> Any:
    if data is None:
        return {}
    if data.startswith("@"):
        return json.loads(Path(data[1:]).read_text())
    return json.loads(data)


def cmd_start(args):
    session_env = check_session_env(list(args.session_env or []))
    file_path, class_name, app_data = deploy.resolve_app(args.app_ref)
    file = Path(file_path)
    import_root = Path(args.root).resolve() if args.root else deploy.package_root(file)
    if not file.is_relative_to(import_root):
        raise DevError(f"{file} is not inside --root {import_root}")
    # A pyproject.toml app is named after its entry there, as on a deploy.
    if (args.app_ref[0] or "").endswith(".py"):
        default_name = class_name or file.stem
    else:
        default_name = args.app_ref[0]
    name = session_name(args.name, default_name)
    # The account is pinned in the session, so a later `fal account set` cannot
    # retarget it. A team in pyproject.toml is where the app is deployed for
    # real; the box goes to the account chosen with --team, else your own.
    account = resolve_account(args.team, name, args.host)
    announce(account)
    if app_data.team and app_data.team != account.team:
        console.print(
            escape(
                f"Ignoring team {app_data.team!r} from pyproject.toml: the box is "
                f"deployed to {account.runs_as}."
            )
        )
    confirm_deploy(
        SyncServerlessClient(host=args.host, team=account.fal_team),
        account,
        name,
        args.host,
        yes=args.yes,
    )
    console.print(f"Deploying dev box [bold]{escape(name)}[/] ...")
    started = time.perf_counter()
    with client.explain_index_refusal(account):
        session = deploy.start_box(
            file,
            class_name,
            app_data,
            name=name,
            root=import_root,
            explicit_root=bool(args.root),
            replace=args.replace,
            keep_alive=args.keep_alive,
            machine_type=args.machine_type,
            account=account,
            host=args.host,
            environment=args.env,
            session_env=session_env,
        )
        deployed = time.perf_counter() - started
        console.print(
            f"Waiting for a runner, then uploading {escape(session.root)} ..."
        )
        status = client.boot(session)
    booted = time.perf_counter() - started - deployed
    endpoints = escape(str(status.get("app", {}).get("endpoints")))
    console.print(
        f"[green]Ready[/] (deploy {deployed:.1f}s, runner+setup {booted:.1f}s). "
        f"Endpoints: {endpoints}"
    )
    console.print(
        "Next: `fal dev app request / --data '{...}'`, "
        "`fal dev app sync --watch`, `fal dev app stop`."
    )


def _report_sync(result: client.SyncResult, started: float) -> None:
    if not result.changed and not result.deleted:
        console.print("No changes.")
        return
    names = escape(", ".join((result.changed + result.deleted)[:8]))
    count = len(result.changed) + len(result.deleted)
    elapsed = time.perf_counter() - started
    console.print(f"Synced {count} file(s) in {elapsed:.2f}s: {names}")
    reload = result.reload
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
    elif action:
        console.print(f"[red]Not reloaded[/]: {escape(str(reload.get('reason')))}")
        if reload.get("error"):
            print(reload["error"], file=sys.stderr)


def _sync_once(session: Session, args, *, full: bool = False) -> None:
    """Sync, wait out a restart, then send the --request if one was given."""
    started = time.perf_counter()
    result = client.sync_files(session, full=full)
    _report_sync(result, started)
    action = result.reload.get("action")
    if action in ("restart", "pending"):
        client.wait_ready(session)
    if args.request is not None and action in ("reload", "restart", "pending"):
        _request(
            session,
            args.request,
            _load_data(args.data),
            edit_started=started,
        )


def cmd_sync(args):
    session = _booted(args)
    _sync_once(session, args, full=args.full)
    if not args.watch:
        return
    console.print(f"Watching {escape(session.root)} (Ctrl+C to stop) ...")
    signature = client.tree_signature(session)
    try:
        while True:
            time.sleep(0.3)
            current = client.tree_signature(session)
            if current == signature:
                continue
            signature = current
            try:
                client.ensure_booted(session)
                _sync_once(session, args)
            except (DevError, OSError, ValueError) as exc:
                # Keep watching: the next save may fix it.
                console.print(f"[red]{escape(str(exc))}[/]")
    except KeyboardInterrupt:
        pass


def _request(
    session: Session,
    path: str,
    body: Any,
    *,
    warmups: int = 0,
    repeat: int = 1,
    edit_started: float | None = None,
) -> dict[str, Any]:
    """Send the request and report what came back."""
    started = time.perf_counter()
    result = client.send_request(session, path, body, warmups=warmups, repeat=repeat)
    total = time.perf_counter() - started
    if warmups or repeat > 1:
        _report_runs(path, result, total)
        return result
    status = result["status_code"]
    color = "green" if status < 400 else "red"
    timing = f"app {result['elapsed_seconds']:.2f}s, round trip {total:.2f}s"
    if edit_started is not None:
        timing += f", edit to result {time.perf_counter() - edit_started:.2f}s"
    console.print(f"[{color}]{status}[/] {escape(path)} ({timing})")
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
    status = result["status_code"]
    color = "green" if status < 400 else "red"
    runs = [run["elapsed_seconds"] for run in result["runs"]]
    console.print(
        f"[{color}]{status}[/] {escape(path)}: {len(runs)} run(s) after "
        f"{len(result['warmup_runs'])} warmup(s), round trip {total:.1f}s"
    )
    if result["warmup_runs"]:
        warm = [run["elapsed_seconds"] for run in result["warmup_runs"]]
        print(f"  warmups  {bench.format_runs(warm)}")
    if runs:
        print(f"  runs     {bench.format_runs(runs)}")
    if status >= 400:
        console.print(f"[red]Run failed with {status}[/]; no statistics.")
    elif runs:
        print(f"  {bench.format_stats(bench.stats(runs))}")
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
    result = _request(
        session, args.path, body, warmups=args.warmups, repeat=args.repeat
    )
    if args.output_file and "body" in result:
        Path(args.output_file).write_text(result["body"])
    if benchmark and result["status_code"] >= 400:
        if args.save:
            console.print("[red]Not saved[/]: a run failed.")
        if "body" in result:
            print(result["body"][-BODY_TAIL_CHARS:], file=sys.stderr)
        raise SystemExit(1)
    if result.get("stopped"):
        if args.save:
            console.print("[red]Not saved[/]: fewer runs than --repeat.")
        raise SystemExit(1)
    if not args.save:
        return
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
    except DevError:
        # Files can still be compared by path without a session.
        if args.name:
            raise
        name = None
    baseline = bench.load(args.baseline, name)
    candidate = bench.load(args.candidate, name)
    for mismatch in bench.mismatches(baseline, candidate):
        console.print(f"[yellow]Not the same benchmark[/] ({escape(mismatch)})")
    for line in bench.compare(baseline, candidate):
        print(line)


def cmd_benchmark_list(args):
    session = find_session(args.name)
    runs = bench.saved(session.name)
    if not runs:
        console.print(f"No saved benchmarks for {escape(session.name)}.")
        return
    for run in runs:
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(run["created_at"]))
        print(
            f"{run['name']:<20} {created}  {run['path']:<12} "
            f"x{run['repeat']}  {bench.format_stats(run['stats'])}"
        )


def _mib(size: int) -> str:
    return f"{size / 1024 / 1024:.1f} MiB"


def _profile(session: Session, args, body: Any, out: Path) -> bool:
    path = args.path
    console.print(
        f"Profiling [bold]{escape(path)}[/] ({args.mode}, {args.warmups} warmup(s)) ..."
    )
    result = client.run_profile(
        session, path, body, mode=args.mode, warmups=args.warmups, top=args.top
    )
    request = result.get("request", {})
    status = request.get("status_code")
    if result.get("status") == "request_failed":
        console.print(f"[red]A warmup request to {escape(path)} returned {status}[/].")
        print(str(request.get("body", ""))[-BODY_TAIL_CHARS:], file=sys.stderr)
        return False
    if (status or 0) >= 400:
        # The profile may exist, but of a failing request; not worth keeping.
        console.print(
            f"[red]The profiled request to {escape(path)} returned {status}[/]."
        )
        print(str(request.get("body", ""))[-BODY_TAIL_CHARS:], file=sys.stderr)
        return False
    if result.get("status") != "ok":
        detail = result.get("error") or result.get("status")
        console.print(f"[red]No profile for {escape(path)}[/]: {escape(str(detail))}")
        return False
    line = (
        f"[green]{status}[/] {escape(path)} in {result['elapsed_seconds']:.2f}s "
        "under the profiler"
    )
    if result.get("peak_memory_allocated_bytes"):
        peak = result["peak_memory_allocated_bytes"] / 1024**3
        line += f", peak GPU memory {peak:.1f} GiB"
    console.print(line)
    print(result["top_ops"])

    folder = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug(path.strip('/') or 'root')}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "profile.json").write_text(json.dumps(result, indent=2) + "\n")
    client.download(session, result["id"], "ops.txt", folder / "ops.txt")
    size = result["trace_bytes"]
    if size > args.max_trace_mb * 1024 * 1024:
        console.print(
            f"[yellow]Trace not downloaded[/]: {_mib(size)} is over "
            f"--max-trace-mb {args.max_trace_mb}. It stays on the box until its "
            "runner stops. Use --light (GPU kernels only), a smaller request, "
            "or a higher --max-trace-mb."
        )
        console.print(f"Ops table: {escape(str(folder / 'ops.txt'))}")
        return True
    trace = folder / "trace.json.gz"
    client.download(session, result["id"], "trace.json.gz", trace)
    console.print(f"Trace: {escape(str(trace))} ({_mib(size)})")
    console.print("Open it in https://ui.perfetto.dev (Open trace file).")
    return True


def cmd_profile(args):
    session = _booted(args)
    out = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else PROFILES_DIR / session.name
    )
    if not _profile(session, args, _load_data(args.data), out):
        raise SystemExit(1)


def _print_packages(packages: list[dict[str, Any]]) -> None:
    if not packages:
        console.print("No dependency overlay: the app runs on its own environment.")
        return
    console.print(f"{len(packages)} overlay package(s):")
    for package in packages:
        base = package["base_version"]
        shadows = f" (app has {base})" if base else " (new)"
        print(f"  {package['name']}=={package['version']}{shadows}")


def _print_app_versions(status: dict[str, Any]) -> None:
    versions = status.get("app_versions") or {}
    if "error" in versions:
        console.print(f"[yellow]Could not ask the app[/]: {escape(versions['error'])}")
        return
    for name, found in sorted(versions.items()):
        if found:
            print(f"  app imports {name}=={found['version']} from {found['path']}")


def _after_change(session: Session, result: dict[str, Any], restart: bool) -> None:
    if result.get("restarted"):
        console.print("Restarting the app so the overlay takes effect ...")
        client.wait_ready(session)
        _print_app_versions(client.deps_call(session, {"action": "status"}))
    elif result.get("restart_needed") and not restart:
        console.print("Run `fal dev app restart` to use it.")


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
    console.print(f"Installing {escape(' '.join(args.packages))} into the overlay ...")
    with client.explain_index_refusal(session.account):
        result = client.deps_call(session, payload)
    if not result["pins"]:
        reason = escape(str(result.get("reason", "already satisfied")))
        console.print(f"Nothing to install: {reason}.")
        return
    pins = escape(", ".join(redact(pin) for pin in result["pins"]))
    console.print(f"[green]Installed[/] {pins}")
    _after_change(session, result, args.restart)


def cmd_deps_uninstall(args):
    session = _booted(args)
    result = client.deps_call(
        session,
        {"action": "uninstall", "packages": args.packages, "restart": args.restart},
    )
    pins = escape(", ".join(redact(pin) for pin in result["removed"]))
    console.print(f"Removed {pins}")
    _after_change(session, result, args.restart)


def cmd_deps_reset(args):
    session = _booted(args)
    result = client.deps_call(session, {"action": "reset", "restart": args.restart})
    console.print(f"Removed {result['removed']} overlay package(s).")
    _after_change(session, result, args.restart)


def cmd_deps_status(args):
    session = _booted(args)
    status = client.deps_call(session, {"action": "status", "names": args.packages})
    if args.json:
        # replay has index URLs as given, credentials included.
        status.pop("replay", None)
        print(json.dumps(status, indent=2))
        return
    _print_packages(status["packages"])
    if status.get("restart_needed"):
        console.print(
            "[yellow]The running app predates the last change[/]: "
            "run `fal dev app restart`."
        )
    _print_app_versions(status)


def cmd_restart(args):
    session = _session(args, acts_as_account=True)
    started = time.perf_counter()
    with client.explain_index_refusal(session.account):
        client.restart(session, sync_first=args.sync)
    console.print(f"[green]Ready[/] after {time.perf_counter() - started:.1f}s.")


def cmd_logs(args):
    session = _session(args)
    logs = client.logs(session, lines=args.lines)
    client.print_log_lines(logs["lines"])
    after = logs["total"]
    while args.follow:
        time.sleep(client.POLL_SECONDS)
        logs = client.logs(session, after=after)
        client.print_log_lines(logs["lines"])
        after = logs["total"]


def cmd_status(args):
    session = _session(args)
    info: dict[str, Any] = {
        "session": session.name,
        "app": f"{session.file_path}::{session.class_name}",
        "box": session.run_url,
        "team": session.account.team,
    }
    try:
        info["remote"] = client.status(session)
    except DevError as exc:
        info["remote"] = {"error": str(exc)}
    print(json.dumps(info, indent=2, default=str))


def cmd_stop(args):
    session = _session(args)
    killed, existed = deploy.stop_box(session)
    session.delete()
    deleted = "deleted the app" if existed else "the app was already gone"
    console.print(f"Stopped {session.name}: {deleted}, killed {len(killed)} runner(s).")
