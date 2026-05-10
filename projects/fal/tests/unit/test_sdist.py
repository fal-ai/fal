"""Tests for ``fal.api._sdist`` — local-path requirement materialization."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fal.api import _sdist


@pytest.fixture(autouse=True)
def _clear_cache():
    _sdist._SDIST_URL_CACHE.clear()
    yield
    _sdist._SDIST_URL_CACHE.clear()


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


def test_materialize_caches_by_project_root(tmp_path):
    """Subsequent calls for the same project root skip both build and upload.

    Within a single process, ``_resolve_sdist_url`` short-circuits on the
    cached ``(package_name, url)`` rather than re-running ``python -m build``.
    """
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "simple"\nversion = "0.1.0"\n')

    fake_sdist = tmp_path / "build_out" / "simple-0.1.0.tar.gz"
    fake_sdist.parent.mkdir()
    fake_sdist.write_bytes(b"sdist-bytes")

    with patch.object(_sdist, "_build_sdist", return_value=fake_sdist) as build, patch(
        "fal.api._sdist._upload_sdist", return_value="https://cdn/simple.tgz"
    ) as upload:
        first = _sdist.materialize_local_paths([".[func]"], tmp_path)
        second = _sdist.materialize_local_paths([".[app]"], tmp_path)

    assert first == ["simple[func] @ https://cdn/simple.tgz"]
    assert second == ["simple[app] @ https://cdn/simple.tgz"]
    assert build.call_count == 1  # second call short-circuits on cache
    assert upload.call_count == 1


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
    assert events[3][1]["cached"] is False


def test_materialize_progress_emits_cached_on_second_call(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "simple"\nversion = "0.1.0"\n')

    fake_sdist = tmp_path / "build_out" / "simple-0.1.0.tar.gz"
    fake_sdist.parent.mkdir()
    fake_sdist.write_bytes(b"x")

    with patch.object(_sdist, "_build_sdist", return_value=fake_sdist), patch(
        "fal.api._sdist._upload_sdist", return_value="https://cdn/simple.tgz"
    ):
        _sdist.materialize_local_paths([".[func]"], tmp_path)
        events: list[tuple[str, dict]] = []
        _sdist.materialize_local_paths(
            [".[app]"], tmp_path, on_progress=lambda e, p: events.append((e, p))
        )

    assert events == [("upload_finished", events[0][1])]
    assert events[0][1]["cached"] is True


def test_read_package_name_missing_pyproject(tmp_path):
    with pytest.raises(RuntimeError, match="no pyproject.toml"):
        _sdist._read_package_name(tmp_path)


def test_read_package_name_missing_name(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "0.1.0"\n')
    with pytest.raises(RuntimeError, match=r"\[project\].name"):
        _sdist._read_package_name(tmp_path)
