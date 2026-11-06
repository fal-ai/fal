"""The dependency overlay of `fal dev app deps`: packages ahead of the app's own.

The overlay is one folder, built with ``uv pip install --target``, that goes
first on the app process's path, so a restart picks it up. The app's own
environment is never modified, and ``reset`` deletes the folder.

An install is planned against the app's environment: only the packages it does
not already satisfy are installed, pinned to the versions uv chose, with
``--no-deps``. The pins (and the index options they came from) are recorded,
and every change rebuilds the folder from them, so the same record installs the
same overlay on a new runner. Uninstalling removes the named packages only; the
dependencies installed with them stay until ``reset``.

The installer runs with the runner's environment minus ``FAL_*`` variables, so
a package's build scripts see the app's other secrets, as the app itself does.

Runs in the box holder, which is shipped by value: standard library only.
"""

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fal.dev.errors import DevError
from fal.dev.remote.protocol import redact

# Packages the box's own processes run on; changing them can break the app
# process's server rather than the app.
CONTROL_PLANE = frozenset(
    {
        "cloudpickle",
        "fastapi",
        "pydantic",
        "pydantic-core",
        "starlette",
        "uvicorn",
    }
)
# What `uv pip install --dry-run` prints for each package it would install.
UV_PLANNED = re.compile(
    r"^\s*\+\s*([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?(==\S+| @ \S+)"
)
# One install (its plan and its build) ends within this, inside the box's request
# timeout.
INSTALL_TIMEOUT_SECONDS = 3000


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(spec: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
    return canonical(match.group(1)) if match else spec


def base_version(name: str) -> str | None:
    """The version in the app's own environment (this process has no overlay)."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class Overlay:
    """The overlay under ``root``: its folder and the record that rebuilds it."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.site = root / "site"
        self.state_file = root / "state.json"
        # Raised on every change, so the holder can tell when the running app
        # process predates the overlay.
        self.version = 0
        self._lock = threading.Lock()
        self._deadline = 0.0

    def state(self) -> dict[str, Any]:
        """The recorded pins and index options, as given (credentials included)."""
        try:
            return json.loads(self.state_file.read_text())
        except (OSError, ValueError):
            return {"packages": []}

    def _installer(self) -> list[str]:
        uv = shutil.which("uv")
        if not uv:
            raise DevError("uv is not available on the runner")
        return [uv, "pip", "install", "--python", sys.executable]

    def _run(self, command: list[str]) -> str:
        env = {k: v for k, v in os.environ.items() if not k.startswith("FAL_")}
        left = self._deadline - time.monotonic()
        if left <= 0:
            raise DevError("the install ran out of time")
        try:
            result = subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=left,
            )
        except subprocess.TimeoutExpired as exc:
            raise DevError(f"the installer took over {exc.timeout:.0f}s") from exc
        output = result.stdout + result.stderr
        if result.returncode != 0:
            raise DevError(redact(f"uv pip install failed:\n{output[-3000:]}"))
        return output

    def _plan(self, options: list[str], packages: list[str]) -> list[str]:
        """Pins for what the app's environment does not already have."""
        output = self._run([*self._installer(), "--dry-run", *options, *packages])
        return [
            f"{match.group(1)}{match.group(2)}"
            for line in output.splitlines()
            if (match := UV_PLANNED.match(line))
        ]

    def _rebuild(self, state: dict[str, Any]) -> None:
        """Install the recorded pins into a fresh folder, then replace the old."""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / "site.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            if state["packages"]:
                command = [*self._installer(), "--target", str(tmp), "--no-deps"]
                self._run([*command, *_index_options(state), *state["packages"]])
                shutil.rmtree(self.site, ignore_errors=True)
                os.replace(tmp, self.site)
            else:
                shutil.rmtree(self.site, ignore_errors=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        scratch = self.state_file.with_suffix(".tmp")
        # The record can hold index credentials.
        fd = os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(state, handle, indent=2)
        os.replace(scratch, self.state_file)
        self.version += 1

    def install(
        self,
        packages: list[str],
        *,
        no_deps: bool = False,
        pre: bool = False,
        index_url: str | None = None,
        extra_index_urls: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Add packages to the overlay; returns the pins this added.

        Index options accumulate: the latest ``index_url`` wins, the extra
        indexes are all kept, and ``pre`` stays on once given.
        """
        if not packages:
            raise DevError("no packages given")
        for spec in packages:
            if spec.startswith("-"):
                raise DevError(f"refusing installer flag {spec!r}")
        with self._lock:
            self._deadline = time.monotonic() + INSTALL_TIMEOUT_SECONDS
            state = self.state()
            state = {
                "packages": state["packages"],
                "pre": pre or state.get("pre", False),
                "index_url": index_url or state.get("index_url"),
                "extra_index_urls": list(
                    dict.fromkeys(
                        [*state.get("extra_index_urls", []), *(extra_index_urls or [])]
                    )
                ),
            }
            pins = packages
            if not no_deps:
                pins = self._plan(_index_options(state), packages)
            known = set(state["packages"])
            pins = [pin for pin in pins if pin not in known]
            if not pins:
                return {"pins": [], "reason": "already satisfied"}
            touched = sorted(CONTROL_PLANE & {requirement_name(pin) for pin in pins})
            if touched and not force:
                raise DevError(
                    f"this would change {', '.join(touched)}, which the box's app "
                    "server runs on. Pass --force to install anyway, or --no-deps."
                )
            # A newer pin of a package replaces the older one.
            replaced = {requirement_name(pin) for pin in pins}
            kept = [p for p in state["packages"] if requirement_name(p) not in replaced]
            state["packages"] = [*kept, *pins]
            self._rebuild(state)
            return {"pins": pins}

    def uninstall(self, names: list[str]) -> list[str]:
        """Remove the pins of the named packages; returns the pins removed."""
        wanted = {canonical(name) for name in names}
        with self._lock:
            self._deadline = time.monotonic() + INSTALL_TIMEOUT_SECONDS
            state = self.state()
            removed = [p for p in state["packages"] if requirement_name(p) in wanted]
            if not removed:
                raise DevError(f"the overlay has no {', '.join(sorted(wanted))}")
            state["packages"] = [p for p in state["packages"] if p not in removed]
            self._rebuild(state)
            return removed

    def reset(self) -> int:
        """Delete the overlay; returns how many pins it had."""
        with self._lock:
            self._deadline = time.monotonic() + INSTALL_TIMEOUT_SECONDS
            count = len(self.state()["packages"])
            if count:
                self._rebuild({"packages": []})
            return count

    def packages(self) -> list[dict[str, Any]]:
        """Installed distributions, each with the version it shadows."""
        found = {}
        for dist in importlib.metadata.distributions(path=[str(self.site)]):
            name = dist.metadata["Name"]
            if name:
                found[canonical(name)] = {
                    "name": name,
                    "version": dist.version,
                    "base_version": base_version(name),
                }
        return [found[key] for key in sorted(found)]


def _index_options(state: dict[str, Any]) -> list[str]:
    options = ["--pre"] if state.get("pre") else []
    if state.get("index_url"):
        options += ["--index-url", state["index_url"]]
    for url in state.get("extra_index_urls", []):
        options += ["--extra-index-url", url]
    return options
