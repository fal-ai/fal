"""Tests for ``fal.api._sdist`` — local-path requirement materialization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from fal.api import _sdist


@pytest.mark.parametrize(
    "req,expected",
    [
        (".", True),
        (".[func]", True),
        (".[func,app]", True),
        ("  .  ", True),
        ("  .[app]  ", True),
        ("simple", False),
        ("./subdir", False),
        ("..", False),
        (".[]", False),
        ("-e .", False),
        ("file:///abs/path", False),
        ("git+https://example.com/foo.git", False),
        ("simple[extra]", False),
    ],
)
def test_local_path_regex(req, expected):
    matched = _sdist._LOCAL_PATH_RE.match(req.strip()) is not None
    assert matched is expected


@pytest.mark.parametrize(
    "requirements,expected",
    [
        ([], False),
        (["fal"], False),
        (["fal", "."], True),
        (["fal", ".[func]"], True),
        ([["fal"], [".[app]"]], True),
        ([["fal"], ["numpy"]], False),
    ],
)
def test_has_local_path(requirements, expected):
    assert _sdist.has_local_path(requirements) is expected


def test_rewrite_one_dot():
    assert _sdist._rewrite_one(".", "simple", "https://cdn/simple.tgz") == (
        "simple @ https://cdn/simple.tgz"
    )


def test_rewrite_one_with_extras():
    assert _sdist._rewrite_one(".[func]", "simple", "https://cdn/simple.tgz") == (
        "simple[func] @ https://cdn/simple.tgz"
    )


def test_rewrite_one_passthrough():
    assert _sdist._rewrite_one("numpy", "simple", "https://cdn/simple.tgz") == "numpy"


def test_materialize_noop_when_no_local_path(tmp_path):
    """No build/upload triggered when no `.` in requirements."""
    with patch.object(_sdist, "_build_sdist") as build, patch.object(
        _sdist, "_upload_sdist"
    ) as upload:
        out = _sdist.materialize_local_paths(["fal", "numpy"], tmp_path)
    assert out == ["fal", "numpy"]
    build.assert_not_called()
    upload.assert_not_called()


def test_materialize_rewrites_flat_list(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "simple"\nversion = "0.1.0"\n')

    fake_sdist = tmp_path / "build_out" / "simple-0.1.0.tar.gz"
    fake_sdist.parent.mkdir()
    fake_sdist.write_bytes(b"fake-sdist-bytes")

    with patch.object(_sdist, "_build_sdist", return_value=fake_sdist) as build, patch(
        "fal.api._sdist._upload_sdist", return_value="https://cdn/simple-0.1.0.tar.gz"
    ) as upload:
        out = _sdist.materialize_local_paths(["fal", ".[func]", "numpy"], tmp_path)

    assert out == [
        "fal",
        "simple[func] @ https://cdn/simple-0.1.0.tar.gz",
        "numpy",
    ]
    build.assert_called_once_with(tmp_path)
    upload.assert_called_once()


def test_materialize_preserves_layered_shape(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "simple"\nversion = "0.1.0"\n')

    fake_sdist = tmp_path / "build_out" / "simple-0.1.0.tar.gz"
    fake_sdist.parent.mkdir()
    fake_sdist.write_bytes(b"fake-sdist-bytes")

    with patch.object(_sdist, "_build_sdist", return_value=fake_sdist), patch(
        "fal.api._sdist._upload_sdist", return_value="https://cdn/simple.tgz"
    ):
        out = _sdist.materialize_local_paths([["fal"], [".", "numpy"]], tmp_path)

    assert out == [["fal"], ["simple @ https://cdn/simple.tgz", "numpy"]]


def test_materialize_reuploads_each_time(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "simple"\nversion = "0.1.0"\n')

    build_count = 0

    def build_sdist(_project_root):
        nonlocal build_count
        build_count += 1
        fake_sdist = tmp_path / f"build_out_{build_count}" / "simple-0.1.0.tar.gz"
        fake_sdist.parent.mkdir()
        fake_sdist.write_bytes(b"sdist-bytes")
        return fake_sdist

    with patch.object(_sdist, "_build_sdist", side_effect=build_sdist) as build, patch(
        "fal.api._sdist._upload_sdist",
        side_effect=["https://cdn/simple-1.tgz", "https://cdn/simple-2.tgz"],
    ) as upload:
        first = _sdist.materialize_local_paths([".[func]"], tmp_path)
        second = _sdist.materialize_local_paths([".[app]"], tmp_path)

    assert first == ["simple[func] @ https://cdn/simple-1.tgz"]
    assert second == ["simple[app] @ https://cdn/simple-2.tgz"]
    assert build.call_count == 2
    assert upload.call_count == 2


def test_materialize_progress_events(tmp_path):
    """``on_progress`` receives the documented phase events on first build."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "simple"\nversion = "0.1.0"\n')

    fake_sdist = tmp_path / "build_out" / "simple-0.1.0.tar.gz"
    fake_sdist.parent.mkdir()
    fake_sdist.write_bytes(b"x" * 1024)

    events: list[tuple[str, dict]] = []

    with patch.object(_sdist, "_build_sdist", return_value=fake_sdist), patch(
        "fal.api._sdist._upload_sdist", return_value="https://cdn/simple.tgz"
    ):
        _sdist.materialize_local_paths(
            [".[func]"], tmp_path, on_progress=lambda e, p: events.append((e, p))
        )

    names = [name for name, _ in events]
    assert names == [
        "build_started",
        "build_finished",
        "upload_started",
        "upload_finished",
    ]
    assert events[0][1]["package_name"] == "simple"
    assert events[3][1]["url"] == "https://cdn/simple.tgz"
    assert "cached" not in events[3][1]


def test_materialize_progress_repeats_after_previous_call(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "simple"\nversion = "0.1.0"\n')

    build_count = 0

    def build_sdist(_project_root):
        nonlocal build_count
        build_count += 1
        fake_sdist = tmp_path / f"build_out_{build_count}" / "simple-0.1.0.tar.gz"
        fake_sdist.parent.mkdir()
        fake_sdist.write_bytes(b"x")
        return fake_sdist

    with patch.object(_sdist, "_build_sdist", side_effect=build_sdist), patch(
        "fal.api._sdist._upload_sdist",
        side_effect=["https://cdn/simple-1.tgz", "https://cdn/simple-2.tgz"],
    ):
        _sdist.materialize_local_paths([".[func]"], tmp_path)
        events: list[tuple[str, dict]] = []
        _sdist.materialize_local_paths(
            [".[app]"], tmp_path, on_progress=lambda e, p: events.append((e, p))
        )

    assert [name for name, _ in events] == [
        "build_started",
        "build_finished",
        "upload_started",
        "upload_finished",
    ]
    assert events[3][1]["url"] == "https://cdn/simple-2.tgz"


def test_upload_sdist_uses_one_hour_lifecycle_for_primary_and_fallback(tmp_path):
    sdist = tmp_path / "simple-0.1.0.tar.gz"
    sdist.write_bytes(b"sdist")
    uploaded_file = type("UploadedFile", (), {"url": "https://cdn/simple.tgz"})()

    with patch("fal.toolkit.File.from_path", return_value=uploaded_file) as from_path:
        url = _sdist._upload_sdist(sdist)

    assert url == "https://cdn/simple.tgz"
    from_path.assert_called_once()
    _, kwargs = from_path.call_args
    expected = {"expiration_duration_seconds": 3600}
    assert kwargs["save_kwargs"]["object_lifecycle_preference"] == expected
    assert kwargs["fallback_save_kwargs"]["object_lifecycle_preference"] == expected
    assert kwargs["save_kwargs"] is kwargs["fallback_save_kwargs"]


def test_read_package_name_missing_pyproject(tmp_path):
    with pytest.raises(RuntimeError, match="no pyproject.toml"):
        _sdist._read_package_name(tmp_path)


def test_read_package_name_missing_name(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "0.1.0"\n')
    with pytest.raises(RuntimeError, match=r"\[project\].name"):
        _sdist._read_package_name(tmp_path)


def test_build_sdist_nonzero_exit_raises_and_cleans_up(tmp_path):
    """A failed ``python -m build`` raises and removes the temp outdir."""
    created_dirs: list = []

    real_mkdtemp = _sdist.tempfile.mkdtemp

    def _tracking_mkdtemp(*args, **kwargs):
        d = real_mkdtemp(*args, **kwargs)
        created_dirs.append(d)
        return d

    fake_result = type("R", (), {"returncode": 1})()
    with patch.object(
        _sdist.tempfile, "mkdtemp", side_effect=_tracking_mkdtemp
    ), patch.object(_sdist.subprocess, "run", return_value=fake_result):
        with pytest.raises(RuntimeError, match=r"sdist build .* failed"):
            _sdist._build_sdist(tmp_path)

    assert created_dirs, "mkdtemp was not invoked"
    for d in created_dirs:
        assert not Path(d).exists(), f"temp dir {d} leaked after failure"


def test_build_sdist_missing_artefact_raises_and_cleans_up(tmp_path):
    """A successful exit with no .tar.gz still raises and cleans up."""
    created_dirs: list = []

    real_mkdtemp = _sdist.tempfile.mkdtemp

    def _tracking_mkdtemp(*args, **kwargs):
        d = real_mkdtemp(*args, **kwargs)
        created_dirs.append(d)
        return d

    fake_result = type("R", (), {"returncode": 0})()
    with patch.object(
        _sdist.tempfile, "mkdtemp", side_effect=_tracking_mkdtemp
    ), patch.object(_sdist.subprocess, "run", return_value=fake_result):
        with pytest.raises(RuntimeError, match="expected exactly one"):
            _sdist._build_sdist(tmp_path)

    assert created_dirs
    for d in created_dirs:
        assert not Path(d).exists(), f"temp dir {d} leaked after failure"
