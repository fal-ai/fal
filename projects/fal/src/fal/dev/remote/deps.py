"""Dependency overlays for `fal dev app deps`: packages layered ahead of the app's venv.

Each ``deps install`` adds a layer: a folder of packages installed with
``--target``. The app process gets the layers on its path ahead of the app's
own environment (newest layer first), so ``restart`` picks them up. The app's
environment itself is never modified, and ``reset`` deletes every layer.

To keep layers small, the install is planned against the app's environment
first: only the packages it does not already satisfy are installed into the new
layer, pinned to the versions the resolver chose, with ``--no-deps``. pip also
counts the existing layers; uv does not, so with uv a later layer can repeat a
package an earlier layer has in another version (exact repeats are skipped).
``--no-deps`` on the command line skips the plan and installs exactly the
packages given.

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
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

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
UV_PLANNED = re.compile(
    r"^\s*\+\s*([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?(==\S+| @ \S+)"
)
URL_CREDENTIALS = re.compile(r"(?<=://)[^/@\s]+@")
OUTPUT_LINES = 40
INSTALL_TIMEOUT_SECONDS = 3000


class DepsError(Exception):
    pass


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def redact(text: str | None) -> str | None:
    """Hide credentials in URLs (``https://user:token@host``)."""
    return URL_CREDENTIALS.sub("***@", text) if text else text


def requirement_name(spec: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
    return canonical(match.group(1)) if match else spec


def installed(path: Path) -> list[dict[str, str]]:
    """Distributions in one layer folder."""
    found = {}
    for dist in importlib.metadata.distributions(path=[str(path)]):
        name = dist.metadata["Name"]
        if name:
            found[canonical(name)] = {"name": name, "version": dist.version}
    return [found[key] for key in sorted(found)]


def base_version(name: str) -> str | None:
    """The version in the app's own environment (this process has no layers)."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class Overlay:
    """The layers under ``root``, newest last in ``state.json``."""

    def __init__(self, root: Path, extra_path: list[Path] | None = None) -> None:
        self.root = root
        self.layers_dir = root / "layers"
        self.state_file = root / "state.json"
        # Folders the app process also sees (the shipped SDK's dependencies),
        # so the plan counts them as installed.
        self.extra_path = extra_path or []
        self.version = 0

    def layers(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.state_file.read_text())["layers"]
        except (OSError, ValueError, KeyError):
            return []

    def _save(self, layers: list[dict[str, Any]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"layers": layers}, indent=2))
        os.replace(tmp, self.state_file)
        self.version += 1

    def paths(self) -> list[str]:
        """Layer folders, newest first."""
        return [str(self.layers_dir / layer["id"]) for layer in reversed(self.layers())]

    def _env(self) -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            # The installer never needs fal credentials.
            if not key.startswith("FAL_")
        }
        env["PYTHONPATH"] = os.pathsep.join([*self.paths(), *map(str, self.extra_path)])
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        return env

    def _installer(self) -> list[str]:
        probe = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            return [sys.executable, "-m", "pip", "install"]
        uv = shutil.which("uv")
        if uv:
            return [uv, "pip", "install", "--python", sys.executable]
        raise DepsError("neither pip nor uv is available on the runner")

    def _run(self, command: list[str], log: list[str]) -> str:
        try:
            result = subprocess.run(
                command,
                env=self._env(),
                capture_output=True,
                text=True,
                check=False,
                timeout=INSTALL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise DepsError(f"the installer took over {exc.timeout:.0f}s") from exc
        output = result.stdout + result.stderr
        log.extend(output.splitlines()[-OUTPUT_LINES:])
        if result.returncode != 0:
            raise DepsError(f"{' '.join(command[:4])} ... failed:\n{output[-3000:]}")
        return output

    def _plan(
        self, installer: list[str], options: list[str], packages: list[str], log
    ) -> list[str]:
        """Pins for what the app's environment and layers do not already have."""
        if installer[0] == sys.executable:
            with tempfile.TemporaryDirectory() as tmp:
                report = Path(tmp) / "report.json"
                self._run(
                    [
                        *installer,
                        "--dry-run",
                        "--quiet",
                        "--report",
                        str(report),
                        *options,
                        *packages,
                    ],
                    log,
                )
                items = json.loads(report.read_text()).get("install", [])
            pins = []
            for item in items:
                info = item.get("download_info") or {}
                metadata = item.get("metadata") or {}
                if "name" not in metadata or "version" not in metadata:
                    raise DepsError(f"unexpected pip report entry: {item}")
                if item.get("is_direct") and info.get("url"):
                    pins.append(f"{metadata['name']} @ {info['url']}")
                else:
                    pins.append(f"{metadata['name']}=={metadata['version']}")
            return pins
        # uv prints the plan, one "+ name==version" per package to install.
        output = self._run([*installer, "--dry-run", *options, *packages], log)
        planned = [
            f"{match.group(1)}{match.group(2)}"
            for line in output.splitlines()
            if (match := UV_PLANNED.match(line))
        ]
        # uv plans against the app's environment only: skip exact repeats of
        # what a layer already has.
        have = {
            f"{canonical(dist['name'])}=={dist['version']}"
            for layer in self.layers()
            for dist in layer["installed"]
        }
        return [
            pin
            for pin in planned
            if f"{requirement_name(pin)}=={pin.partition('==')[2]}" not in have
        ]

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
        if not packages:
            raise DepsError("no packages given")
        for spec in packages:
            if spec.startswith("-"):
                raise DepsError(f"refusing installer flag {spec!r}")
        options = ["--pre"] if pre else []
        if index_url:
            options += ["--index-url", index_url]
        for url in extra_index_urls or []:
            options += ["--extra-index-url", url]
        installer = self._installer()
        log: list[str] = []
        pins = packages if no_deps else self._plan(installer, options, packages, log)
        if not pins:
            return {"layer": None, "log": log, "reason": "already satisfied"}
        touched = sorted(CONTROL_PLANE & {requirement_name(pin) for pin in pins})
        if touched and not force:
            raise DepsError(
                f"this would change {', '.join(touched)}, which the box's app "
                "server runs on. Pass --force to install anyway, or --no-deps."
            )
        layer_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self.layers_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.layers_dir / f".{layer_id}.tmp"
        try:
            self._run(
                [*installer, "--target", str(tmp), "--no-deps", *options, *pins], log
            )
            os.replace(tmp, self.layers_dir / layer_id)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        layer = {
            "id": layer_id,
            "created_at": time.time(),
            "requested": packages,
            "packages": pins,
            "no_deps": no_deps,
            "pre": pre,
            "index_url": index_url,
            "extra_index_urls": list(extra_index_urls or []),
            "installed": installed(self.layers_dir / layer_id),
        }
        self._save([*self.layers(), layer])
        return {"layer": layer, "log": log}

    def uninstall(self, names: list[str]) -> list[dict[str, Any]]:
        """Drop the oldest layer that has one of ``names``, and every newer one.

        Newer layers were planned assuming the older ones were there, so they
        go too.
        """
        wanted = {canonical(name) for name in names}
        layers = self.layers()
        first = None
        for index, layer in enumerate(layers):
            provided = {canonical(dist["name"]) for dist in layer["installed"]}
            provided |= {requirement_name(spec) for spec in layer["requested"]}
            if provided & wanted:
                first = index
                break
        if first is None:
            raise DepsError(f"no layer has {', '.join(sorted(wanted))}")
        keep, removed = layers[:first], layers[first:]
        self._save(keep)
        for layer in removed:
            shutil.rmtree(self.layers_dir / layer["id"], ignore_errors=True)
        return removed

    def reset(self) -> int:
        count = len(self.layers())
        if not count:
            return 0
        self._save([])
        shutil.rmtree(self.layers_dir, ignore_errors=True)
        return count

    def status(self) -> list[dict[str, Any]]:
        """Layers, newest first, each package with the version it shadows.

        URLs are redacted; ``replay`` has them as given.
        """
        result = []
        for layer in reversed(self.layers()):
            packages = [
                {**dist, "base_version": base_version(dist["name"])}
                for dist in layer["installed"]
            ]
            result.append(
                {
                    **layer,
                    "installed": packages,
                    "requested": [redact(spec) for spec in layer["requested"]],
                    "packages": [redact(spec) for spec in layer["packages"]],
                    "index_url": redact(layer.get("index_url")),
                    "extra_index_urls": [
                        redact(url) for url in layer.get("extra_index_urls", [])
                    ],
                }
            )
        return result

    def replay(self) -> list[dict[str, Any]]:
        """What it takes to install the same layers again, oldest first."""
        keys = ("id", "packages", "pre", "index_url", "extra_index_urls")
        return [{key: layer.get(key) for key in keys} for layer in self.layers()]
