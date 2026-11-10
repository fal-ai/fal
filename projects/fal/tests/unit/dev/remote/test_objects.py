import hashlib
import os
from pathlib import Path

import pytest

from dev.remote import fakes
from fal.dev.remote.objects import Store

roomy = fakes.roomy

pytestmark = pytest.mark.usefixtures("dev_account")


def test_object_store_keeps_checked_copies(tmp_path, roomy, monkeypatch):
    from fal.dev.remote import objects as app_objects

    assert Store.open(tmp_path / "missing") is None
    store = Store.open(tmp_path)
    assert store is not None
    source = tmp_path / "a.py"
    source.write_text("print(1)\n")
    digest = store.put(source.read_bytes())
    assert digest == hashlib.sha256(b"print(1)\n").hexdigest()

    target = tmp_path / "src" / "pkg" / "a.py"
    assert store.get(digest, target, executable=True)
    assert target.read_text() == "print(1)\n"
    assert target.stat().st_mode & 0o111

    # A damaged object is never copied out, and is dropped so the next put
    # repairs it.
    store.path(digest).write_text("tampered")
    assert not store.get(digest, tmp_path / "other.py")
    assert not (tmp_path / "other.py").exists()
    assert not store.path(digest).exists()
    assert store.put(source.read_bytes()) == digest
    assert store.get(digest, tmp_path / "other.py")

    # Links planted in the store are never followed.
    planted = tmp_path / "planted.py"
    planted.write_text("print(2)\n")
    planted_digest = hashlib.sha256(b"print(2)\n").hexdigest()
    store.path(planted_digest).parent.mkdir(exist_ok=True)
    store.path(planted_digest).symlink_to(planted)
    assert not store.get(planted_digest, tmp_path / "x.py")

    # A FIFO planted in the store neither hangs nor gets copied.
    fifo_digest = "f" * 64
    store.path(fifo_digest).parent.mkdir(exist_ok=True)
    os.mkfifo(store.path(fifo_digest))
    assert not store.get(fifo_digest, tmp_path / "fifo.py")

    # Nothing bigger than the sync's file limit is copied out.
    big = b"x" * (app_objects.MAX_FILE_BYTES + 1)
    big_digest = hashlib.sha256(big).hexdigest()
    store.path(big_digest).parent.mkdir(exist_ok=True)
    store.path(big_digest).write_bytes(big)
    assert not store.get(big_digest, tmp_path / "big.bin")
    assert not (tmp_path / "big.bin").exists()

    # A nearly full volume is left alone.
    monkeypatch.setattr(app_objects, "MIN_FREE_BYTES", 1 << 62)
    assert store.put(planted.read_bytes()) is None
    with pytest.raises(ValueError):
        store.path("../../etc/passwd")


def _age(path: Path, days: float) -> None:
    import time

    then = time.time() - days * 86400
    os.utime(path, (then, then), follow_symlinks=False)


def test_object_store_expires_objects_nobody_uses(tmp_path, roomy):
    store = Store.open(tmp_path)
    assert store is not None
    digests = {}
    for name in ("old", "used", "kept", "fresh"):
        source = tmp_path / f"{name}.py"
        source.write_text(name)
        digests[name] = store.put(source.read_bytes())
    for name in ("old", "used", "kept"):
        _age(store.path(digests[name]), 31)
    # Copying an object out, or finding it already stored, counts as use.
    assert store.get(digests["used"], tmp_path / "out" / "used.py")
    assert store.put((tmp_path / "kept.py").read_bytes()) == digests["kept"]
    # Half-written copies go much sooner.
    tmp_copy = store.path(digests["fresh"]).parent / ".tmp-abc"
    tmp_copy.write_text("x")
    _age(tmp_copy, 0.1)
    young = store.path(digests["fresh"]).parent / ".tmp-new"
    young.write_text("x")

    store.expire()

    assert not store.path(digests["old"]).exists()
    for name in ("used", "kept", "fresh"):
        assert store.path(digests[name]).exists()
    assert not tmp_copy.exists() and young.exists()


def test_object_store_expiry_does_not_follow_planted_folders(tmp_path, roomy):
    store = Store.open(tmp_path / "data") or Store.open(tmp_path)
    assert store is not None
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "precious").write_text("keep")
    _age(victim / "precious", 400)
    (store.root / "ab").symlink_to(victim)
    store.expire()
    assert (victim / "precious").exists()


def test_object_store_refuses_oversized_content(tmp_path, roomy):
    (tmp_path / "data").mkdir()
    store = Store.open(tmp_path / "data")
    assert store is not None
    assert store.put(b"x" * (10 * 1024 * 1024 + 1)) is None
    assert not list(store.root.glob("*/*"))
