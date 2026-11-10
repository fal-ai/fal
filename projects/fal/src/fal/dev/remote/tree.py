"""The synced source tree on the box: what `fal dev app sync` writes and removes."""

import base64
import io
import os
import tarfile
import threading
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from fal.dev.remote.objects import Keeper, Store
from fal.dev.remote.protocol import MAX_PLACE_FILES


def relative(raw: str) -> PurePosixPath:
    """A path inside the tree; refuses absolute paths and ``..``."""
    rel = PurePosixPath(raw)
    if not raw or rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"refusing path {raw!r}")
    return rel


def extract(
    archive_b64: str,
    dest: Path,
    on_write: Callable[[str, bytes], None] | None = None,
) -> list[str]:
    """Unpack a tar.gz into dest, refusing paths that escape it.

    ``on_write`` is called with each file's path and bytes as it is written."""
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(archive_b64))) as archive:
        for member in archive.getmembers():
            rel = relative(member.name)
            if not member.isfile():
                raise ValueError(f"refusing archive entry {member.name!r}")
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            assert source is not None
            tmp = target.with_name(f".{target.name}.fal-dev-tmp")
            data = source.read()
            tmp.write_bytes(data)
            tmp.chmod(0o755 if member.mode & 0o111 else 0o644)
            # The running process never sees a half-written file.
            os.replace(tmp, target)
            written.append(str(rel))
            if on_write is not None:
                on_write(str(rel), data)
    return written


class SourceTree:
    """The files the CLI synced, and the team's object store they are kept in."""

    def __init__(
        self, root: Path, store: Store | None, log: Callable[[str], None]
    ) -> None:
        self.root = root
        self.store = store
        self.keeper = Keeper(store, log) if store else None
        # Files written by syncs, so a full sync only prunes what it put there.
        self.synced: set[str] = set()
        # Syncs and places run in separate request threads.
        self.lock = threading.Lock()

    def _remove(self, rel: str) -> bool:
        """Delete one file, then any folders it leaves empty."""
        target = self.root / relative(rel)
        if not target.is_file():
            return False
        target.unlink()
        parent = target.parent
        while parent != self.root and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
        return True

    def place(self, entries: dict[str, Any]) -> list[str]:
        """Copy files from the object store into the tree; returns those placed.

        ``entries`` maps a path to ``[digest, executable]``. A file the store
        lacks, or holds damaged, is left for the CLI to upload.
        """
        if self.store is None:
            return []
        if not isinstance(entries, dict) or len(entries) > MAX_PLACE_FILES:
            raise ValueError(f"files must map at most {MAX_PLACE_FILES} paths")
        checked = []
        # Check every entry before placing any.
        for raw, entry in sorted(entries.items()):
            if not isinstance(entry, list) or len(entry) != 2:
                raise ValueError(f"expected [digest, executable] for {raw!r}")
            self.store.path(str(entry[0]))
            checked.append((str(relative(raw)), str(entry[0]), bool(entry[1])))
        with self.lock:
            placed = [
                rel
                for rel, digest, executable in checked
                if self.store.get(digest, self.root / rel, executable)
            ]
            self.synced.update(placed)
        return placed

    def sync(
        self,
        *,
        archive: str | None = None,
        deleted: list[str],
        all_files: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Apply one batch of a sync; returns the paths written and deleted.

        Deletes go first, since a path can turn from a file into a folder or
        back. ``all_files``, on a full sync, lists every file the tree should
        have: the synced files it leaves out are deleted too.
        """
        with self.lock:
            removed = [rel for rel in deleted if self._remove(rel)]
            if all_files is not None:
                wanted = {str(relative(raw)) for raw in all_files}
                removed += [
                    rel for rel in sorted(self.synced - wanted) if self._remove(rel)
                ]
                self.synced = wanted
            self.synced.difference_update(removed)

            def wrote(rel: str, data: bytes) -> None:
                # Recorded as it lands, so a failure mid-archive can be pruned.
                self.synced.add(rel)
                if self.keeper is not None:
                    self.keeper.keep(data)

            written = extract(archive, self.root, wrote) if archive else []
            return written, removed
