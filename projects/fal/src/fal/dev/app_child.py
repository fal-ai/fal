"""The resident process behind `fal dev app`: runs ``setup()`` once, reloads endpoints.

Started on the runner by the dev box holder (``app_holder``) as
``python -m fal.dev.app_child``. It imports the user's app from the synced
source tree, runs ``setup()`` through the app's own lifespan, and serves the
app's FastAPI routes on 127.0.0.1. The port only opens once ``setup()`` is done.

``POST /__fal_dev_child/reload`` re-imports the source tree and either swaps
the app class in place (keeping the instance state ``setup()`` built), or says
a restart or rebuild is needed; the holder acts on that answer. See
``app_reload`` for the rules.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import linecache
import os
import site
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import uvicorn

import fal.ref
from fal.app import App
from fal.dev import app_profile, app_reload
from fal.dev.app_holder import CHILD_PREFIX


def _log(message: str) -> None:
    print(f"[fal dev] {message}", flush=True)


async def _read_body(receive) -> bytes:
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            return body


async def _send_json(send, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, default=str).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _versions(names: list[str]) -> dict[str, Any]:
    """The version of each package this process imports, and where from."""
    found: dict[str, Any] = {}
    for name in names:
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
            continue
        found[name] = {"version": dist.version, "path": str(dist.locate_file(""))}
    return found


def _activate_overlay() -> None:
    """Process .pth files in the dependency overlay (already on PYTHONPATH)."""
    for path in reversed(os.environ.get("FAL_DEV_OVERLAY", "").split(os.pathsep)):
        if path:
            site.addsitedir(path)


class DevAppServer:
    """An ASGI app that owns the user's app instance and swaps its code on reload."""

    def __init__(self, root: Path, module_name: str, class_name: str) -> None:
        self.root = root.resolve()
        self.module_name = module_name
        self.class_name = class_name
        self.generation = 0
        self.error: str | None = None
        self.last_reload: dict[str, Any] = {}
        self.setup_seconds: float | None = None
        self.started_at = time.time()
        self.profiler = app_profile.Profiler()
        self.instance, self.fastapi_app = self._load()
        self.setup_fp = app_reload.setup_fingerprint(type(self.instance), self.root)
        self.setup_writes = app_reload.setup_global_writes(
            type(self.instance), self.root
        )
        # Values of those globals, kept here too: a failed reload drops the
        # modules that held them.
        self.kept_globals: dict[str, dict[str, Any]] = {}
        self.env_fp = app_reload.environment_fingerprint(type(self.instance))
        self._lifespan: Any = None

    def _import_class(self) -> type[App]:
        module = importlib.import_module(self.module_name)
        cls = getattr(module, self.class_name, None)
        if not isinstance(cls, type) or not issubclass(cls, App):
            raise TypeError(f"{self.module_name}.{self.class_name} is not a fal.App")
        return cls

    def _load(self) -> tuple[App, Any]:
        cls = self._import_class()
        instance = cls(_allow_init=True)
        fal.ref.set_current_app(instance)
        return instance, self._build(instance)

    def _build(self, instance: App) -> Any:
        """The app's FastAPI app, with endpoints that can run under a profile."""
        routes = app_profile.wrap_routes(instance.collect_routes(), self.profiler)
        # _build_app() reads the routes from collect_routes(); shadow it on
        # the instance only while building, so reloads never copy it over.
        instance.collect_routes = lambda: routes  # type: ignore[method-assign]
        try:
            return instance._build_app()
        finally:
            del instance.collect_routes

    def _drop_local_modules(self) -> None:
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if path and Path(path).resolve().is_relative_to(self.root):
                del sys.modules[name]
        importlib.invalidate_caches()
        linecache.checkcache()

    def _setup_globals(self) -> dict[str, dict[str, Any]]:
        """Current values of the module globals that setup() assigned."""
        values: dict[str, dict[str, Any]] = {}
        for module_name, names in self.setup_writes.items():
            module = sys.modules.get(module_name)
            if module is not None:
                values[module_name] = {
                    name: vars(module)[name] for name in names if name in vars(module)
                }
        return values

    def reload(self, changed: list[str]) -> dict[str, Any]:
        start = time.perf_counter()
        for module_name, values in self._setup_globals().items():
            self.kept_globals.setdefault(module_name, {}).update(values)
        kept_globals = self.kept_globals
        self._drop_local_modules()
        try:
            try:
                cls = self._import_class()
            finally:
                # Re-importing reset what setup() stored in module globals,
                # including in helpers imported before a failure.
                for module_name, values in kept_globals.items():
                    module = sys.modules.get(module_name)
                    if module is not None:
                        vars(module).update(values)
            setup_fp = app_reload.setup_fingerprint(cls, self.root)
            env_fp = app_reload.environment_fingerprint(cls)
            action, reason = app_reload.decide(
                self.setup_fp, setup_fp, self.env_fp, env_fp
            )
            if action == "reload":
                instance = object.__new__(cls)
                # The state setup() built (models, pipelines) lives on the instance.
                instance.__dict__.update(self.instance.__dict__)
                fastapi_app = self._build(instance)
                self.instance, self.fastapi_app = instance, fastapi_app
                # set_current_app() refuses to replace the app, so set it here.
                fal.ref.current_app = instance
                self.setup_writes = app_reload.setup_global_writes(cls, self.root)
                self.generation += 1
            self.error = None
        except Exception:
            self.error = traceback.format_exc()
            action, reason = "error", "the code failed to import"
            _log(f"reload failed:\n{self.error}")
        result = {
            "action": action,
            "reason": reason,
            "files": changed,
            "ms": round((time.perf_counter() - start) * 1000, 1),
            "code_version": self.generation,
            "error": self.error,
        }
        self.last_reload = result
        _log(f"{action}: {reason} ({result['ms']} ms, code version {self.generation})")
        return result

    def status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "pid": os.getpid(),
            "code_version": self.generation,
            "setup_seconds": self.setup_seconds,
            "last_reload": self.last_reload,
            "error": self.error,
            "endpoints": sorted(
                signature.path for signature in self.instance.collect_routes()
            ),
        }

    async def _run_lifespan(self, receive, send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                start = time.perf_counter()
                try:
                    self._lifespan = self.instance.lifespan(self.fastapi_app)
                    await self._lifespan.__aenter__()
                except Exception:
                    tb = traceback.format_exc()
                    _log(f"setup() failed:\n{tb}")
                    await send({"type": "lifespan.startup.failed", "message": tb})
                    return
                self.setup_seconds = round(time.perf_counter() - start, 2)
                _log(f"setup() finished in {self.setup_seconds}s")
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                if self._lifespan is not None:
                    try:
                        await self._lifespan.__aexit__(None, None, None)
                    except Exception:
                        _log(f"teardown() failed:\n{traceback.format_exc()}")
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            await self._run_lifespan(receive, send)
            return
        path = scope.get("path", "")
        if scope["type"] == "http" and path.startswith(CHILD_PREFIX):
            action = path[len(CHILD_PREFIX) :]
            body = await _read_body(receive)
            try:
                payload = json.loads(body or b"{}")
            except ValueError:
                payload = None
            if not isinstance(payload, dict):
                await _send_json(send, 422, {"detail": "expected a JSON object"})
                return
            if action == "reload":
                await _send_json(send, 200, self.reload(payload.get("files", [])))
            elif action == "status":
                await _send_json(send, 200, self.status())
            elif action == "versions":
                found = _versions(payload.get("names") or [])
                await _send_json(send, 200, found)
            elif action == "endpoints":
                routes = self.instance.collect_routes()
                found = {"endpoints": app_profile.endpoints(routes)}
                await _send_json(send, 200, found)
            elif action == "profile/arm":
                try:
                    await _send_json(send, 200, self.profiler.arm(payload))
                except (KeyError, ValueError) as exc:
                    await _send_json(send, 422, {"detail": str(exc)})
            elif action == "profile/result":
                result = self.profiler.result(str(payload.get("id")))
                await _send_json(send, 200, result)
            else:
                await _send_json(send, 404, {"detail": f"unknown action {action}"})
            return
        if self.error is not None and scope["type"] == "http":
            # 422, not 5xx: the holder passes statuses through and fal replaces
            # runners whose apps answer 503.
            await _read_body(receive)
            await _send_json(
                send,
                422,
                {"detail": "the app code failed to import", "traceback": self.error},
            )
            return
        await self.fastapi_app(scope, receive, send)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m fal.dev.app_child")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--module", required=True)
    parser.add_argument("--cls", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)

    # A same-size edit within the same second would otherwise reuse a stale .pyc.
    sys.dont_write_bytecode = True
    _activate_overlay()
    sys.path.insert(0, str(args.root.resolve()))
    os.chdir(args.root)
    server = DevAppServer(args.root, args.module, args.cls)
    _log(f"running setup() for {args.module}.{args.cls}")
    uvicorn.run(
        server,
        host="127.0.0.1",
        port=args.port,
        lifespan="on",
        timeout_keep_alive=300,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
