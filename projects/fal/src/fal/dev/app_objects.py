"""A content-addressed store of synced files, shared by a team's dev boxes.

A fresh runner would otherwise get the whole source tree from the laptop on
every `fal dev app start` (and after every idle shutdown). Instead each file
the box receives is also kept on the team's persistent ``/data`` volume, named
by its sha256. On boot the CLI sends the tree's manifest; the box copies every
file it already holds into place and the CLI uploads only the rest. Files are
shared across sessions, branches and teammates, so after the first upload a
new box costs only the files that changed since any box last saw them, with
no git hosting token anywhere.

Objects are checked against their name when written and again when copied
out, so a damaged object is treated as missing and sent again; symlinks in the
store are never followed. Nothing is ever deleted: the store grows with every
version synced, and stops growing while the volume has under 20 GiB free.
Delete /data/.fal-dev to reset it. Files the dev sync never sends (``.env``,
key and credential files, files over 10 MiB) never reach it.

Runs on the runner inside the holder, so: standard library only.
"""

import hashlib
import os
import re
import shutil
import stat
import tempfile
import time
from pathlib import Path

DATA_ROOT = Path("/data")
STORE_DIR = ".fal-dev/objects"
DIGEST = re.compile(r"[0-9a-f]{64}")
CHUNK = 1024 * 1024
# Stop adding objects when the team's volume gets this full: /data also holds
# the team's weights and caches.
MIN_FREE_BYTES = 20 * 1024**3
# The dev sync never sends files over 10 MiB.
MAX_OBJECT_BYTES = 10 * 1024 * 1024
TMP_MAX_AGE_SECONDS = 3600


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked(digest: str) -> str:
    if not DIGEST.fullmatch(digest):
        raise ValueError(f"not a sha256 digest: {digest!r}")
    return digest


def _regular(path: Path) -> bool:
    """A regular file, not reached through a symlink.

    Other apps on the team can write to /data too; never follow their links.
    """
    try:
        if path.parent.is_symlink():
            return False
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def open(cls, data_root: Path = DATA_ROOT) -> "Store | None":
        """The store on the persistent volume, or None when there is none."""
        if not data_root.is_dir():
            return None
        root = data_root / STORE_DIR
        try:
            # Checked before creating anything: never write through a link.
            if any(p.is_symlink() for p in (root.parent, root)):
                return None
            root.mkdir(parents=True, exist_ok=True)
            if any(p.is_symlink() for p in (root.parent, root)):
                return None
            probe = tempfile.NamedTemporaryFile(dir=root, prefix=".probe-")
            probe.close()
        except OSError:
            return None
        store = cls(root)
        store.sweep()
        return store

    def sweep(self, older_than: float = TMP_MAX_AGE_SECONDS) -> None:
        """Delete copies left half-written by boxes that stopped mid-put."""
        cutoff = time.time() - older_than
        for tmp in self.root.glob("*/.tmp-*"):
            try:
                if os.lstat(tmp).st_mtime < cutoff:
                    tmp.unlink()
            except OSError:
                continue

    def path(self, digest: str) -> Path:
        digest = _checked(digest)
        return self.root / digest[:2] / digest

    def missing(self, digests: list[str]) -> list[str]:
        return sorted({d for d in digests if not _regular(self.path(d))})

    def put(self, source: Path) -> str | None:
        """Keep a copy of ``source``; returns its digest, or None if not kept.

        Best effort: a full or failing volume never fails the sync.
        """
        try:
            digest = _digest(source)
            target = self.path(digest)
            if _regular(target):
                return digest
            if shutil.disk_usage(self.root).free < MIN_FREE_BYTES:
                return None
            target.parent.mkdir(exist_ok=True)
            if target.parent.is_symlink():
                return None
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
            try:
                with os.fdopen(fd, "wb") as writer, source.open("rb") as reader:
                    shutil.copyfileobj(reader, writer, CHUNK)
                # The source may change while it is copied; keep only a true copy.
                if _digest(Path(tmp)) != digest:
                    return None
                # Readable by the team's other boxes (mkstemp makes it 0600).
                os.chmod(tmp, 0o644)
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except OSError:
            return None
        return digest

    def get(self, digest: str, target: Path, executable: bool = False) -> bool:
        """Copy the object into ``target``; False if it is missing or damaged,
        or cannot be written there."""
        source = self.path(digest)
        if source.parent.is_symlink():
            return False
        tmp = target.with_name(f".{target.name}.fal-dev-tmp")
        try:
            # No link, no device, nothing past the sync's size limit: checked on
            # the open file, since a team peer can swap the path meanwhile.
            # O_NONBLOCK: a FIFO planted here must not hang the open.
            fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as reader:
                if not stat.S_ISREG(os.fstat(reader.fileno()).st_mode):
                    return False
                target.parent.mkdir(parents=True, exist_ok=True)
                hashed = hashlib.sha256()
                size = 0
                with tmp.open("wb") as writer:
                    for block in iter(lambda: reader.read(CHUNK), b""):
                        size += len(block)
                        if size > MAX_OBJECT_BYTES:
                            break
                        hashed.update(block)
                        writer.write(block)
            if size > MAX_OBJECT_BYTES or hashed.hexdigest() != digest:
                tmp.unlink(missing_ok=True)
                # Drop the damaged object, so the next put replaces it.
                source.unlink(missing_ok=True)
                return False
            tmp.chmod(0o755 if executable else 0o644)
            os.replace(tmp, target)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        return True
