"""`fal dev hot` server: keeps a model loaded and reloads the inference code around it.

Runs on the box inside the project's own Python (standard library only, so it
works in any virtualenv). A project is a folder with:

    load.py   load() -> state       slow: loads weights, called once per process
    infer.py  infer(state, **params) fast: called per request, reloaded on save

Saving any other .py file in the folder reloads it and infer.py, in the same
process, so the next request runs the new code with the weights still on the
GPU. Saving load.py (or requirements.txt) is a full
restart: the process exits with RESTART_EXIT_CODE and `devbox-hot` reinstalls
requirements and starts it again, which reloads the weights.

`infer()` returns a PIL image (sent as PNG), bytes, or anything JSON-serializable.

Errors use 409 (restarting) and 422 (your code failed), never 5xx: the box's
/hot route passes them through, and fal replaces a runner that answers 503.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import io
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Changes to these mean new weights or new dependencies: restart the process.
RESTART_FILES = {"load.py", "requirements.txt"}
RESTART_EXIT_CODE = 3
SKIP_DIRS = {".venv", "__pycache__", ".git", "node_modules"}
POLL_SECONDS = 0.25

LOG: collections.deque[str] = collections.deque(maxlen=200)


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} [hot] {message}"
    LOG.append(line)
    print(line, flush=True)


def scan(root: Path) -> dict[str, float]:
    """Modification time of every watched file under root, by relative path."""
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
        ]
        for filename in filenames:
            if filename.endswith(".py") or filename in RESTART_FILES:
                path = Path(dirpath) / filename
                try:
                    found[str(path.relative_to(root))] = path.stat().st_mtime
                except OSError:
                    continue
    return found


class Project:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock = (
            threading.Lock()
        )  # one request or reload at a time: they share the GPU
        self.files = scan(root)
        self.state = None
        self.infer = None
        self.error: str | None = None
        self.generation = 0
        self.load_seconds = 0.0
        self.last_change = max(self.files.values(), default=time.time())
        self.last_reload: dict = {}
        self.restart_reason: str | None = None

    def load(self) -> None:
        sys.path.insert(0, str(self.root))
        start = time.time()
        log("loading the model (load.load()) ...")
        import load

        self.state = load.load()
        self.load_seconds = time.time() - start
        log(f"model loaded in {self.load_seconds:.1f}s")
        self._import_infer()

    def _import_infer(self) -> None:
        try:
            self.infer = importlib.import_module("infer").infer
            self.error = None
        except Exception:
            self.infer = None
            self.error = traceback.format_exc()
            log(f"infer.py failed to import:\n{self.error}")

    def _reload_code(self, changed: list[str]) -> None:
        """Drop the project's modules (not load) from sys.modules; reimport infer."""
        start = time.time()
        root = str(self.root)
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None) or ""
            if (
                name != "load"
                and path.startswith(root + os.sep)
                and SKIP_DIRS.isdisjoint(Path(path).parts)
            ):
                del sys.modules[name]
        importlib.invalidate_caches()
        self._import_infer()
        self.generation += 1
        ms = (time.time() - start) * 1000
        self.last_reload = {
            "files": changed,
            "ms": round(ms, 1),
            "ok": self.error is None,
        }
        log(
            f"reloaded ({', '.join(changed)}) in {ms:.0f} ms, "
            f"code version {self.generation}"
        )

    def check(self) -> str | None:
        """Apply file changes; a restart reason if the process must restart.

        Call with the lock held.
        """
        if self.restart_reason:
            return self.restart_reason
        current = scan(self.root)
        changed = sorted(
            p
            for p in current.keys() | self.files.keys()
            if current.get(p) != self.files.get(p)
        )
        if not changed:
            return None
        self.files = current
        self.last_change = max(
            [current[p] for p in changed if p in current], default=time.time()
        )
        restart = [
            p
            for p in changed
            if Path(p).name in RESTART_FILES and Path(p).parent == Path(".")
        ]
        if restart:
            self.restart_reason = f"{', '.join(restart)} changed"
            return self.restart_reason
        self._reload_code(changed)
        return None

    def run(self, params: dict) -> tuple[int, str, bytes, dict]:
        """(status, content type, body, timing headers) for one request."""
        with self.lock:
            reason = self.check()
            if reason:
                return (
                    409,
                    "application/json",
                    json.dumps({"restarting": reason}).encode(),
                    {},
                )
            if self.infer is None:
                body = {
                    "error": (
                        "infer.py, or a module it imports, does not import; "
                        "fix it and save"
                    ),
                    "traceback": self.error,
                }
                return 422, "application/json", json.dumps(body).encode(), {}
            start = time.time()
            try:
                result = self.infer(self.state, **params)
                cuda_sync()
            except Exception:
                tb = traceback.format_exc()
                log(f"infer() raised:\n{tb}")
                return (
                    422,
                    "application/json",
                    json.dumps({"error": "infer() raised", "traceback": tb}).encode(),
                    {},
                )
            infer_ms = (time.time() - start) * 1000
        content_type, body = encode(result)
        headers = {
            "X-Hot-Infer-Ms": f"{infer_ms:.0f}",
            "X-Hot-Code-Version": str(self.generation),
            # Time from the newest saved change to this result, as seen on the box.
            "X-Hot-Since-Change-Ms": f"{(time.time() - self.last_change) * 1000:.0f}",
        }
        log(
            f"infer {infer_ms:.0f} ms (code version {self.generation}) "
            f"params={json.dumps(params)[:200]}"
        )
        return 200, content_type, body, headers

    def status(self) -> dict:
        return {
            "ready": self.state is not None,
            "code_version": self.generation,
            "load_seconds": round(self.load_seconds, 1),
            "last_reload": self.last_reload,
            "error": self.error,
            "restarting": self.restart_reason,
            "pid": os.getpid(),
            "log": list(LOG)[-40:],
        }


def cuda_sync() -> None:
    """Wait for queued GPU work, so timings cover the whole request."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def encode(result) -> tuple[str, bytes]:
    if hasattr(result, "save"):  # a PIL image
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return "image/png", buf.getvalue()
    if isinstance(result, (bytes, bytearray)):
        return "application/octet-stream", bytes(result)
    return "application/json", json.dumps(result, default=str).encode()


def watch(project: Project, server: ThreadingHTTPServer) -> None:
    """Reload as soon as a file is saved; restart when load code or deps change."""
    while True:
        time.sleep(POLL_SECONDS)
        with project.lock:
            reason = project.check()
        if reason:
            log(f"FULL RESTART: {reason}; the weights reload")
            server.shutdown()
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    project = Project(args.project.resolve())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:
            pass

        def _send(
            self,
            status: int,
            content_type: str,
            body: bytes,
            headers: dict | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path == "/status":
                self._send(
                    200, "application/json", json.dumps(project.status()).encode()
                )
            elif self.path == "/restart":
                with project.lock:
                    project.restart_reason = "restart requested"
                self._send(200, "application/json", b'{"restarting": true}')
            elif self.path == "/infer":
                try:
                    params = json.loads(raw or b"{}")
                except ValueError:
                    self._send(
                        400, "application/json", b'{"error": "body must be JSON"}'
                    )
                    return
                self._send(*project.run(params))
            else:
                self._send(404, "application/json", b'{"error": "not found"}')

    try:
        project.load()
    except Exception:
        log(f"load() raised:\n{traceback.format_exc()}")
        # Last, so it stays in the log tail the CLI reads however long the traceback.
        log("load.load() failed; fix load.py and save to retry")
        # Wait for a fix instead of exiting into a tight restart loop.
        while scan(project.root) == project.files:
            time.sleep(POLL_SECONDS)
        sys.exit(RESTART_EXIT_CODE)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=watch, args=(project, server), daemon=True).start()
    log(f"serving {project.root} on 127.0.0.1:{args.port}")
    server.serve_forever()
    # Only the watcher stops the server: a restart was requested or load code changed.
    sys.exit(RESTART_EXIT_CODE)


if __name__ == "__main__":
    main()
