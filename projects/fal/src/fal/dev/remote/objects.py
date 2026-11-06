"""A content-addressed store of synced files, shared by a team's dev boxes.

Every file the box receives is also kept on the team's persistent ``/data``
volume, named by its sha256. When the CLI syncs, the box copies the files it
already holds into place and the CLI uploads only the rest, so a new box (or a
switch back to an earlier branch) costs only the files no box has seen. The
store is shared by every box and teammate of the team, so synced source is
readable by teammates. Files the sync never sends (``.env``, key and credential
files, files over 10 MiB) never reach it.

Objects are checked against their name when written and again when copied out,
so a damaged object is treated as missing and sent again. Symlink checks guard
against links planted in the store, not against a concurrent swap, which the
hash check catches. An object that no box has used for 30 days is deleted
(``expire``), and no objects are added while the volume has under 20 GiB free.
Delete /data/.fal-dev to reset the store.

Runs on the runner inside the holder, so: standard library only.
"""

import contextlib
import hashlib
import os
import queue
import re
import shutil
import stat
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from fal.dev.remote.protocol import MAX_FILE_BYTES

DATA_ROOT = Path("/data")
STORE_DIR = ".fal-dev/objects"
DIGEST = re.compile(r"[0-9a-f]{64}")
CHUNK = 1024 * 1024
# Stop adding objects when the team's volume gets this full: /data also holds
# the team's weights and caches.
MIN_FREE_BYTES = 20 * 1024**3
# Files waiting to be stored; more are dropped, as storing is best effort.
KEEP_QUEUE_SIZE = 256
TMP_MAX_AGE_SECONDS = 3600
# An object nobody has used for this long is deleted.
OBJECT_MAX_AGE_SECONDS = 30 * 24 * 3600


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


def _touch(path: Path) -> None:
    """Mark an object as used now, which is what keeps it from expiring."""
    with contextlib.suppress(OSError):
        os.utime(path, follow_symlinks=False)


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
        return cls(root)

    def expire(self) -> None:
        """Delete objects nobody has used for 30 days, and half-written copies
        left by boxes that stopped mid-put."""
        now = time.time()
        for entry in self.root.glob("*/*"):
            max_age = (
                TMP_MAX_AGE_SECONDS
                if entry.name.startswith(".tmp-")
                else OBJECT_MAX_AGE_SECONDS
            )
            try:
                if not entry.parent.is_symlink() and (
                    now - os.lstat(entry).st_mtime > max_age
                ):
                    entry.unlink()
            except OSError:
                continue

    def path(self, digest: str) -> Path:
        digest = _checked(digest)
        return self.root / digest[:2] / digest

    def put(self, data: bytes) -> str | None:
        """Keep a copy of ``data``; returns its digest, or None if not kept.

        Best effort: a full or failing volume never fails the sync.
        """
        if len(data) > MAX_FILE_BYTES:
            return None
        digest = hashlib.sha256(data).hexdigest()
        try:
            target = self.path(digest)
            if _regular(target):
                _touch(target)
                return digest
            if shutil.disk_usage(self.root).free < MIN_FREE_BYTES:
                return None
            target.parent.mkdir(exist_ok=True)
            if target.parent.is_symlink():
                return None
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
            try:
                with os.fdopen(fd, "wb") as writer:
                    writer.write(data)
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
                        if size > MAX_FILE_BYTES:
                            break
                        hashed.update(block)
                        writer.write(block)
            if size > MAX_FILE_BYTES or hashed.hexdigest() != digest:
                tmp.unlink(missing_ok=True)
                # Drop the damaged object, so the next put replaces it.
                source.unlink(missing_ok=True)
                return False
            tmp.chmod(0o755 if executable else 0o644)
            os.replace(tmp, target)
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            return False
        _touch(source)
        return True


class Keeper:
    """Copies file contents into the store on one background thread, so a sync
    never waits on /data. The same thread expires old objects first."""

    def __init__(self, store: Store, log: Callable[[str], None]) -> None:
        self.store = store
        self.log = log
        self.queue: queue.Queue[bytes] = queue.Queue(KEEP_QUEUE_SIZE)
        threading.Thread(target=self._run, daemon=True).start()

    def keep(self, data: bytes) -> None:
        with contextlib.suppress(queue.Full):
            self.queue.put_nowait(data)

    def _run(self) -> None:
        # Whatever fails, the worker stays alive for the next file.
        try:
            self.store.expire()
        except Exception as exc:
            self.log(f"[fal dev] could not expire stored files: {exc}")
        while True:
            data = self.queue.get()
            try:
                self.store.put(data)
            except Exception as exc:
                self.log(f"[fal dev] could not store a synced file: {exc}")
