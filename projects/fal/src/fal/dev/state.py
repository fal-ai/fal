"""Local state for `fal dev`.

A folder remembers the box it was last used with (and its hot reload project),
so commands run there later reattach to the same box without flags. A box
remembers the team it runs as, and the team whose production apps you have
already accepted a deploy next to.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

STATE_DIR = Path.home() / ".fal" / "dev"


def _path(kind: str, subject: str, host: str | None) -> Path:
    digest = hashlib.sha256(f"{subject}\0{host or ''}".encode()).hexdigest()[:16]
    return STATE_DIR / kind / f"{digest}.json"


def _load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _update(path: Path, values: dict) -> None:
    current = _load(path)
    data = {
        key: value for key, value in {**current, **values}.items() if value is not None
    }
    if data == current:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique per process: editors may run several `fal dev ssh-proxy` at once.
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def _folder(project_dir: Path, host: str | None) -> Path:
    return _path("projects", str(project_dir.resolve()), host)


def load(project_dir: Path, host: str | None) -> dict:
    return _load(_folder(project_dir, host))


def update(project_dir: Path, host: str | None, **values) -> None:
    """Merge values into the folder's state; a value of None removes its key."""
    _update(
        _folder(project_dir, host),
        {**values, "project_dir": str(project_dir.resolve())},
    )


def load_box(app: str, host: str | None) -> dict:
    return _load(_path("boxes", app, host))


def update_box(app: str, host: str | None, **values) -> None:
    """Merge values into the state of box app `app`; None removes a key."""
    _update(_path("boxes", app, host), {**values, "app": app})
