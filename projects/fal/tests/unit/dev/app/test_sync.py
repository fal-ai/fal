import os

import pytest

from dev.app import fakes
from fal.dev.app import sync as app_sync
from fal.dev.remote.tree import extract

repo = fakes.repo

pytestmark = pytest.mark.usefixtures("dev_account")


def test_sync_roundtrip(tmp_path):
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "a.py").write_text("a = 1\n")
    (src / "pkg" / "b.py").write_text("b = 1\n")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"x")

    files = app_sync.collect(src)
    assert sorted(files) == ["a.py", "pkg/b.py"]
    first = app_sync.manifest(files)

    (src / "a.py").write_text("a = 2\n")
    (src / "pkg" / "b.py").unlink()
    (src / "c.py").write_text("c = 1\n")
    files = app_sync.collect(src)
    changed, deleted = app_sync.diff(first, app_sync.manifest(files))
    assert changed == ["a.py", "c.py"]
    assert deleted == ["pkg/b.py"]

    written = extract(app_sync.pack(files, changed), tmp_path / "dest")
    assert sorted(written) == ["a.py", "c.py"]
    assert (tmp_path / "dest" / "a.py").read_text() == "a = 2\n"


def test_sync_skips_log_files_so_a_logged_watch_cannot_loop(tmp_path):
    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / "watch.log").write_text("synced 1 file\n")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "app.log").write_text("x")

    assert sorted(app_sync.collect(tmp_path)) == ["a.py"]


def test_sync_sends_symlinks_and_skips_env(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("a = 1\n")
    (src / ".env").write_text("SECRET=1\n")
    os.symlink(src / "a.py", src / "b.py")
    (tmp_path / "outside.py").write_text("x = 1\n")
    os.symlink(tmp_path / "outside.py", src / "c.py")
    os.symlink(src / ".env", src / "config.txt")
    files = app_sync.collect(src)
    assert sorted(files) == ["a.py", "b.py"]
    written = extract(app_sync.pack(files, sorted(files)), tmp_path / "dest")
    assert sorted(written) == ["a.py", "b.py"]
    assert (tmp_path / "dest" / "b.py").read_text() == "a = 1\n"


def test_sync_batches_large_trees(tmp_path, monkeypatch):
    monkeypatch.setattr(app_sync, "BATCH_BYTES", 10)
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("x = 1\n")  # 6 bytes each
    files = app_sync.collect(tmp_path)
    assert app_sync.batches(files, sorted(files)) == [["a.py"], ["b.py"], ["c.py"]]
    assert app_sync.batches(files, []) == [[]]


def test_app_sync_withholds_what_hot_sync_withholds(tmp_path):
    for name in ["app.py", "secrets/token.txt", "prod.tfvars", ".env.local", "id_rsa"]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x")
    withheld: list[str] = []
    files = app_sync.collect(tmp_path, withheld=withheld)
    assert list(files) == ["app.py"]
    assert sorted(withheld) == [
        ".env.local",
        "id_rsa",
        "prod.tfvars",
        "secrets/token.txt",
    ]


def test_app_sync_keeps_code_named_like_secrets(tmp_path):
    for name in ["credentials.py", "app/secrets/store.py", "secrets/key.json"]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x")
    assert sorted(app_sync.collect(tmp_path)) == [
        "app/secrets/store.py",
        "credentials.py",
    ]


def test_sync_applies_app_files_ignore_from_the_context_folder(repo):
    files = app_sync.collect(repo, ignore=[r"\.bin$", r"^assets/"])
    assert "assets/prompt.txt" not in files
    files = app_sync.collect(repo, ignore=[r"^prompt"], ignore_base="assets")
    assert "assets/prompt.txt" not in files
    assert "assets/weights.bin" in files
    # Patterns only apply under their base, as on a deploy.
    assert "pkg/helpers.py" in files


def test_sync_never_sends_keys_or_credentials(tmp_path):
    for name in ("app.py", "server.pem", "tls.key", "credentials.json", "id_ed25519"):
        (tmp_path / name).write_text("x")
    for name in (
        "conf/.npmrc",
        ".aws/credentials",
        "certs/ca.p8",
        ".SSH/config",
        "deploy/kubeconfig",
        "conf/secrets.yaml",
    ):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text("x")
    # Not even through a link with an innocent name.
    (tmp_path / "ok.txt").symlink_to(tmp_path / "server.pem")
    withheld: list[str] = []
    (tmp_path / "settings").mkdir()
    (tmp_path / "settings" / "secrets.py").write_text("x")
    assert sorted(app_sync.collect(tmp_path, withheld=withheld)) == [
        "app.py",
        "settings/secrets.py",
    ]
    assert "server.pem" in withheld
    assert ".aws/credentials" in withheld
