"""Per-project local state for `fal dev`.

A folder remembers the box it was last used with (and its hot reload project),
so commands run there later reattach to the same box without flags.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

STATE_DIR = Path.home() / ".fal" / "dev"


def project_key(project_dir: Path, host: str | None, team: str | None) -> str:
    identity = f"{project_dir.resolve()}\0{host or ''}\0{team or ''}"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def _path(project_dir: Path, host: str | None, team: str | None) -> Path:
    return STATE_DIR / "projects" / f"{project_key(project_dir, host, team)}.json"


def load(project_dir: Path, host: str | None, team: str | None) -> dict:
    try:
        data = json.loads(_path(project_dir, host, team).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def update(project_dir: Path, host: str | None, team: str | None, **values) -> None:
    """Merge values into the folder's state; a value of None removes its key."""
    path = _path(project_dir, host, team)
    current = load(project_dir, host, team)
    data = {**current, **values, "project_dir": str(project_dir.resolve())}
    data = {key: value for key, value in data.items() if value is not None}
    if data == current:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique per process: editors may run several `fal dev ssh-proxy` at once.
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)
