"""`fal dev cp`: copy files between this laptop and the box with scp.

scp runs over the same ssh-proxy as `ssh`, so a stopped box is started on
connect. The box is authenticated by the fal API (the proxy is an exec stream
on your own runner), so its host key is not checked and nothing is written to
~/.ssh/known_hosts.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from fal.dev.errors import DevError

# Overrides on the command line win over the ssh-config entry.
SCP_OPTIONS = (
    "UserKnownHostsFile=/dev/null",
    "StrictHostKeyChecking=no",
    "LogLevel=ERROR",
    "ControlMaster=no",
    "ControlPath=none",
    # The box authorizes the ~/.ssh/id_*.pub keys; offering every agent key risks
    # "Too many authentication failures".
    "IdentitiesOnly=yes",
)


@dataclass(frozen=True)
class CopyPlan:
    """What to copy: `upload` is laptop to box."""

    upload: bool
    local: str
    remote: str
    # A download into a folder that existed before scp ran lands inside it.
    into_dir: bool = False


def plan(src: str, dst: str) -> CopyPlan:
    """Split `src` and `dst` into a local and a remote side; ':' marks the remote."""
    src_remote, dst_remote = src.startswith(":"), dst.startswith(":")
    if src_remote == dst_remote:
        raise DevError(
            "exactly one side must be on the box, marked with a ':' prefix",
            fixes=[
                ("fal dev cp :/work/out.mp4 .", "download"),
                ("fal dev cp ./data :/data/x", "upload"),
            ],
        )
    remote, local = (src, dst) if src_remote else (dst, src)
    remote = remote[1:]
    if not remote:
        raise DevError("the path after ':' is empty; use ':/path' on the box")
    if not local:
        raise DevError("the local path is empty")
    # scp reads `name:path` as a remote host, so keep a colon out of the first part.
    if ":" in local.split("/", 1)[0]:
        local = f"./{local}"
    return CopyPlan(
        upload=dst_remote,
        local=local,
        remote=remote,
        into_dir=not dst_remote and Path(local).is_dir(),
    )


def scp_command(
    copy: CopyPlan, alias: str, config_path: str, recursive: bool, scp: str
) -> list[str]:
    remote = f"{alias}:{copy.remote}"
    argv = [scp, "-F", config_path]
    for option in SCP_OPTIONS:
        argv += ["-o", option]
    if recursive:
        argv.append("-r")
    argv.append("--")
    argv += [copy.local, remote] if copy.upload else [remote, copy.local]
    return argv


def tree_size(path: str) -> int:
    """Bytes in a file or a directory tree (symlinks are not followed)."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        try:
            return root.stat().st_size
        except OSError:
            return 0
    total = 0
    for base, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += os.lstat(os.path.join(base, name)).st_size
            except OSError:
                continue
    return total


def human_size(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def check_local(copy: CopyPlan, recursive: bool) -> None:
    """Fail before waking the box when the local side cannot work."""
    path = Path(copy.local)
    if copy.upload:
        if not path.exists():
            raise DevError(f"{copy.local} does not exist")
        if path.is_dir() and not recursive:
            raise DevError(f"{copy.local} is a directory", fixes=[("-r", "copy it")])
    elif not path.parent.exists():
        raise DevError(f"{path.parent} does not exist")


def downloaded_path(copy: CopyPlan) -> str | None:
    """Where a download landed: inside the local path when it was already a folder.

    None when that cannot be told (a glob, or a remote like `/` or `.`).
    """
    name = Path(copy.remote.rstrip("/")).name
    if copy.into_dir:
        if name in ("", ".", "..") or any(c in name for c in "*?[~"):
            return None
        return str(Path(copy.local) / name)
    return copy.local


def find_scp() -> str:
    scp = shutil.which("scp")
    if scp is None:
        raise DevError("scp is not installed; install an OpenSSH client")
    return scp
