import os
import subprocess
from pathlib import Path

import pytest

import fal.toolkit.utils.download_utils as download_utils
from fal.toolkit import clone_repository, download_file, download_model_weights
from fal.toolkit.utils.download_utils import _filename_from_response, _hash_url
from fal.toolkit.utils.ssrf import SafeResponse

# data: URLs exercise the whole cache and force logic without any network.
SAMPLE_URL = "data:text/plain,hello"


@pytest.fixture
def persistent_dirs(tmp_path, monkeypatch):
    """Point the module's /data constants at tmp_path.

    All three are patched: the derived ones were computed from
    FAL_PERSISTENT_DIR at import time.
    """
    monkeypatch.setattr(download_utils, "FAL_PERSISTENT_DIR", tmp_path)
    monkeypatch.setattr(download_utils, "FAL_REPOSITORY_DIR", tmp_path / "repos")
    monkeypatch.setattr(
        download_utils, "FAL_MODEL_WEIGHTS_DIR", tmp_path / "model_weights"
    )
    return tmp_path


def test_filename_from_response_strips_directory_components():
    file_name = _filename_from_response(
        "https://example.com/file/with/slash.txt",
        SafeResponse(
            200,
            headers={
                "content-disposition": 'attachment; filename="file/with/slash.txt"'
            },
        ),
    )

    assert "/" not in file_name


def test_download_file_resolves_relative_target_under_persistent_dir(persistent_dirs):
    path = download_file(SAMPLE_URL, target_dir="downloads")

    assert path == persistent_dirs / "downloads" / _hash_url(SAMPLE_URL)
    assert path.read_bytes() == b"hello"


def test_download_file_keeps_cached_file_unless_forced(tmp_path):
    path = download_file(SAMPLE_URL, tmp_path)
    assert path == tmp_path / _hash_url(SAMPLE_URL)
    # Same size as the real content, so the cache check cannot tell it apart.
    path.write_bytes(b"stale")

    assert download_file(SAMPLE_URL, tmp_path) == path
    assert path.read_bytes() == b"stale"

    assert download_file(SAMPLE_URL, tmp_path, force=True) == path
    assert path.read_bytes() == b"hello"


def test_download_model_weights_caches_per_url_unless_forced(persistent_dirs):
    expected = persistent_dirs / "model_weights" / _hash_url(SAMPLE_URL)

    path = download_model_weights(SAMPLE_URL)
    assert path == expected / _hash_url(SAMPLE_URL)
    path.write_bytes(b"stale")

    assert download_model_weights(SAMPLE_URL).read_bytes() == b"stale"
    assert download_model_weights(SAMPLE_URL, force=True).read_bytes() == b"hello"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        text=True,
    ).strip()


@pytest.fixture
def source_repo(tmp_path):
    """A local repository with two commits; git clone accepts a path as URL."""
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    commits = []
    for n in (1, 2):
        (repo / "file.txt").write_text(f"revision {n}\n")
        _git(repo, "add", "file.txt")
        _git(repo, "commit", "-q", "-m", f"revision {n}")
        commits.append(_git(repo, "rev-parse", "HEAD"))
    return repo, commits


def test_clone_repository_names_checkout_after_url_and_commit(
    persistent_dirs, source_repo
):
    repo, (first, second) = source_repo
    repos = persistent_dirs / "repos"

    assert clone_repository(str(repo)) == repos / "source"
    assert (repos / "source" / "file.txt").read_text() == "revision 2\n"

    first_path = clone_repository(str(repo), commit_hash=first)
    assert first_path == repos / f"source-{first[:8]}"
    assert _git(first_path, "rev-parse", "HEAD") == first
    assert (first_path / "file.txt").read_text() == "revision 1\n"

    # Each pinned commit gets its own checkout beside the others.
    second_path = clone_repository(str(repo), commit_hash=second)
    assert second_path == repos / f"source-{second[:8]}"
    assert _git(second_path, "rev-parse", "HEAD") == second
    assert first_path.exists()


# Re-cloning renames onto a directory TemporaryDirectory already created, which
# Windows refuses; clone_repository only runs on Linux runners.
@pytest.mark.skipif(os.name == "nt", reason="clone_repository targets Linux runners")
def test_clone_repository_reuses_matching_checkout_unless_forced(
    persistent_dirs, source_repo
):
    repo, (first, _) = source_repo

    path = clone_repository(str(repo), commit_hash=first)
    marker = path / "local-edit"
    marker.touch()

    assert clone_repository(str(repo), commit_hash=first) == path
    assert marker.exists()

    assert clone_repository(str(repo), commit_hash=first, force=True) == path
    assert not marker.exists()
