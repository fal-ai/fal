import base64
import io
import tarfile
from types import SimpleNamespace

import pytest

from dev.remote import fakes
from fal.dev.app import sync as app_sync
from fal.dev.remote import tree as app_tree
from fal.dev.remote.objects import Store
from fal.dev.remote.tree import SourceTree, extract

roomy = fakes.roomy

pytestmark = pytest.mark.usefixtures("dev_account")


def test_extract_refuses_escaping_paths(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        data = b"x"
        info = tarfile.TarInfo("../evil.py")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError):
        extract(base64.b64encode(buffer.getvalue()).decode(), tmp_path / "dest")


def test_sync_handles_file_becoming_folder_and_back(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    (src / "data").write_text("file\n")
    tree = SourceTree(dest, None, print)
    files = app_sync.collect(src)
    tree.sync(archive=app_sync.pack(files, ["data"]), deleted=[])

    (src / "data").unlink()
    (src / "data").mkdir()
    (src / "data" / "x.txt").write_text("x\n")
    files = app_sync.collect(src)
    written, deleted = tree.sync(
        archive=app_sync.pack(files, ["data/x.txt"]), deleted=["data"]
    )
    assert (written, deleted) == (["data/x.txt"], ["data"])
    assert (dest / "data" / "x.txt").read_text() == "x\n"

    _, deleted = tree.sync(deleted=["data/x.txt"])
    assert deleted == ["data/x.txt"]
    assert not (dest / "data").exists()


def test_full_sync_deletes_only_synced_files_that_are_gone(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    for name in ("a.py", "b.py"):
        (src / name).write_text(name)
    tree = SourceTree(dest, None, print)
    files = app_sync.collect(src)
    tree.sync(archive=app_sync.pack(files, sorted(files)), deleted=[])
    # A file the app wrote itself is not the sync's to delete.
    (dest / "cache.bin").write_text("x")

    _, deleted = tree.sync(deleted=[], all_files=["a.py"])
    assert deleted == ["b.py"]
    assert (dest / "a.py").exists() and (dest / "cache.bin").exists()
    assert not (dest / "b.py").exists()


def _synced(tmp_path, roomy_store=True):
    (tmp_path / "data").mkdir()
    store = Store.open(tmp_path / "data")
    assert store is not None
    return store


def test_tree_places_stored_files_and_hands_synced_ones_to_the_store(
    tmp_path, roomy, monkeypatch
):
    store = _synced(tmp_path)
    kept = []
    monkeypatch.setattr(
        app_tree,
        "Keeper",
        lambda store, log: SimpleNamespace(keep=kept.append),
    )
    tree = SourceTree(tmp_path / "session" / "src", store, print)
    local = tmp_path / "local"
    (local / "pkg").mkdir(parents=True)
    (local / "pkg" / "a.py").write_text("A = 1\n")
    files = app_sync.collect(local)
    digest = app_sync.manifest(files)["pkg/a.py"]

    # Nothing stored yet: nothing is placed.
    assert tree.place({"pkg/a.py": [digest, False]}) == []

    # A synced file is handed to the store's keeper...
    written, _ = tree.sync(archive=app_sync.pack(files, ["pkg/a.py"]), deleted=[])
    assert kept == [b"A = 1\n"]
    store.put(kept[0])
    # ...so a fresh runner can copy it into place.
    fresh = SourceTree(tmp_path / "fresh" / "src", store, print)
    assert fresh.place({"pkg/a.py": [digest, False]}) == ["pkg/a.py"]
    assert (fresh.root / "pkg" / "a.py").read_text() == "A = 1\n"
    assert "pkg/a.py" in fresh.synced
    with pytest.raises(ValueError, match="refusing"):
        fresh.place({"../x": [digest, 0]})
    with pytest.raises(ValueError, match="digest"):
        fresh.place({"x.py": ["../../etc", 0]})

    # No store (no /data volume): nothing placed, everything is uploaded.
    assert SourceTree(tmp_path / "bare", None, print).place({"a.py": [digest, 0]}) == []


def test_files_written_before_an_archive_fails_are_recorded(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("a")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ("a.py", "../evil.py"):
            info = tarfile.TarInfo(name)
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
    tree = SourceTree(tmp_path / "dest", None, print)
    with pytest.raises(ValueError):
        tree.sync(archive=base64.b64encode(buffer.getvalue()).decode(), deleted=[])
    assert tree.synced == {"a.py"}
