"""Tests for ``fal.api._sdist`` — local-path requirement materialization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from fal.api import _sdist


@pytest.mark.parametrize(
    "req,expected_parts,expected_extras",
    [
        (".", ("project",), ""),
        (".[func]", ("project",), "[func]"),
        (".[func,app]", ("project",), "[func,app]"),
        ("  .  ", ("project",), ""),
        ("  .[app]  ", ("project",), "[app]"),
        ("./subdir", ("project", "subdir"), ""),
        ("./subdir[func]", ("project", "subdir"), "[func]"),
        ("./subdir [func]", ("project", "subdir"), "[func]"),
        ("../shared", ("shared",), ""),
        ("..", (), ""),
    ],
)
def test_parse_local_path_requirement_for_existing_dirs(
    tmp_path, req, expected_parts, expected_extras
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "subdir").mkdir()
    (tmp_path / "shared").mkdir()

    parsed = _sdist._parse_local_path_requirement(req, project_root)
    assert parsed is not None
    assert parsed.path == tmp_path.joinpath(*expected_parts).resolve()
    assert parsed.extras == expected_extras


@pytest.mark.parametrize(
    "req",
    [
        "simple",
        "./subdir",
        ".[]",
        "-e .",
        "file:///abs/path",
        "git+https://example.com/foo.git",
        "simple[extra]",
    ],
)
def test_parse_local_path_requirement_for_non_dirs(tmp_path, req):
    parsed = _sdist._parse_local_path_requirement(req, tmp_path)
    assert parsed is None


@pytest.mark.parametrize(
    "requirements,expected",
    [
        ([], False),
        (["fal"], False),
        (["fal", "."], True),
        (["fal", ".[func]"], True),
        (["fal", "./subdir"], False),
        ([["fal"], [".[app]"]], True),
        ([["fal"], ["numpy"]], False),
    ],
)
def test_has_local_path(tmp_path, requirements, expected):
    assert _sdist.has_local_path(requirements, tmp_path) is expected


def test_has_local_path_detects_existing_relative_dir(tmp_path):
    (tmp_path / "subdir").mkdir()
    assert _sdist.has_local_path(["fal", "./subdir"], tmp_path) is True


def test_rewrite_one_dot(tmp_path):
    resolved_sdists = {tmp_path.resolve(): ("simple", "https://cdn/simple.tgz")}
    assert _sdist._rewrite_one(".", tmp_path, resolved_sdists, None) == (
        "simple @ https://cdn/simple.tgz"
    )


def test_rewrite_one_with_extras(tmp_path):
    resolved_sdists = {tmp_path.resolve(): ("simple", "https://cdn/simple.tgz")}
    assert _sdist._rewrite_one(".[func]", tmp_path, resolved_sdists, None) == (
        "simple[func] @ https://cdn/simple.tgz"
    )


def test_rewrite_one_passthrough(tmp_path):
    assert _sdist._rewrite_one("numpy", tmp_path, {}, None) == "numpy"


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
    build.assert_called_once_with(tmp_path.resolve())
    upload.assert_called_once()


def test_materialize_rewrites_relative_project_path(tmp_path):
    package_root = tmp_path / "packages" / "common"
    package_root.mkdir(parents=True)
    pyproject = package_root / "pyproject.toml"
    pyproject.write_text('[project]\nname = "common"\nversion = "0.1.0"\n')

    fake_sdist = tmp_path / "build_out" / "common-0.1.0.tar.gz"
    fake_sdist.parent.mkdir()
    fake_sdist.write_bytes(b"fake-sdist-bytes")

    with patch.object(_sdist, "_build_sdist", return_value=fake_sdist) as build, patch(
        "fal.api._sdist._upload_sdist", return_value="https://cdn/common-0.1.0.tar.gz"
    ):
        out = _sdist.materialize_local_paths(
            ["fal", "./packages/common[func]"], tmp_path
        )

    assert out == [
        "fal",
        "common[func] @ https://cdn/common-0.1.0.tar.gz",
    ]
    build.assert_called_once_with(package_root.resolve())


def test_materialize_rewrites_bare_local_project_dir(tmp_path):
    package_root = tmp_path / "common"
    package_root.mkdir()
    pyproject = package_root / "pyproject.toml"
    pyproject.write_text('[project]\nname = "common"\nversion = "0.1.0"\n')

    fake_sdist = tmp_path / "build_out" / "common-0.1.0.tar.gz"
    fake_sdist.parent.mkdir()
    fake_sdist.write_bytes(b"fake-sdist-bytes")

    with patch.object(_sdist, "_build_sdist", return_value=fake_sdist), patch(
        "fal.api._sdist._upload_sdist", return_value="https://cdn/common-0.1.0.tar.gz"
    ):
        out = _sdist.materialize_local_paths(["common[func]"], tmp_path)

    assert out == ["common[func] @ https://cdn/common-0.1.0.tar.gz"]


def test_materialize_bare_directory_without_pyproject_raises(tmp_path):
    (tmp_path / "common").mkdir()

    with pytest.raises(RuntimeError, match="no pyproject.toml"):
        _sdist.materialize_local_paths(["common"], tmp_path)


def test_materialize_rewrites_multiple_local_project_paths(tmp_path):
    package_a = tmp_path / "packages" / "package_a"
    package_b = tmp_path / "packages" / "package_b"
    package_a.mkdir(parents=True)
    package_b.mkdir(parents=True)
    (package_a / "pyproject.toml").write_text(
        '[project]\nname = "package-a"\nversion = "0.1.0"\n'
    )
    (package_b / "pyproject.toml").write_text(
        '[project]\nname = "package-b"\nversion = "0.1.0"\n'
    )
    fake_sdist_a = tmp_path / "build_out_a" / "package-a-0.1.0.tar.gz"
    fake_sdist_b = tmp_path / "build_out_b" / "package-b-0.1.0.tar.gz"
    fake_sdist_a.parent.mkdir()
    fake_sdist_b.parent.mkdir()
    fake_sdist_a.write_bytes(b"fake-sdist-a")
    fake_sdist_b.write_bytes(b"fake-sdist-b")

    with patch.object(
        _sdist, "_build_sdist", side_effect=[fake_sdist_a, fake_sdist_b]
    ) as build, patch(
        "fal.api._sdist._upload_sdist",
        side_effect=["https://cdn/package-a.tgz", "https://cdn/package-b.tgz"],
    ):
        out = _sdist.materialize_local_paths(
            ["./packages/package_a[api]", "./packages/package_b"], tmp_path
        )

    assert out == [
        "package-a[api] @ https://cdn/package-a.tgz",
        "package-b @ https://cdn/package-b.tgz",
    ]
    assert build.call_args_list[0].args == (package_a.resolve(),)
    assert build.call_args_list[1].args == (package_b.resolve(),)


def test_materialize_reuses_sdist_for_same_local_project_path(tmp_path):
    package_root = tmp_path / "packages" / "common"
    package_root.mkdir(parents=True)
    (package_root / "pyproject.toml").write_text(
        '[project]\nname = "common"\nversion = "0.1.0"\n'
    )
    fake_sdist = tmp_path / "build_out" / "common-0.1.0.tar.gz"
    fake_sdist.parent.mkdir()
    fake_sdist.write_bytes(b"fake-sdist-bytes")

    with patch.object(_sdist, "_build_sdist", return_value=fake_sdist) as build, patch(
        "fal.api._sdist._upload_sdist", return_value="https://cdn/common.tgz"
    ):
        out = _sdist.materialize_local_paths(
            ["./packages/common[api]", "./packages/common[worker]"], tmp_path
        )

    assert out == [
        "common[api] @ https://cdn/common.tgz",
        "common[worker] @ https://cdn/common.tgz",
    ]
    build.assert_called_once_with(package_root.resolve())


def test_materialize_reuses_sdist_without_duplicate_progress_events(tmp_path):
    package_root = tmp_path / "packages" / "common"
    package_root.mkdir(parents=True)
    (package_root / "pyproject.toml").write_text(
        '[project]\nname = "common"\nversion = "0.1.0"\n'
    )
    fake_sdist = tmp_path / "build_out" / "common-0.1.0.tar.gz"
    fake_sdist.parent.mkdir()
    fake_sdist.write_bytes(b"fake-sdist-bytes")
    events: list[tuple[str, dict]] = []

    with patch.object(_sdist, "_build_sdist", return_value=fake_sdist) as build, patch(
        "fal.api._sdist._upload_sdist", return_value="https://cdn/common.tgz"
    ):
        out = _sdist.materialize_local_paths(
            ["./packages/common[api]", "./packages/common[worker]"],
            tmp_path,
            on_progress=lambda e, p: events.append((e, p)),
        )

    assert out == [
        "common[api] @ https://cdn/common.tgz",
        "common[worker] @ https://cdn/common.tgz",
    ]
    build.assert_called_once_with(package_root.resolve())
    assert [name for name, _ in events] == [
        "build_started",
        "build_finished",
        "upload_started",
        "upload_finished",
    ]


def test_materialize_missing_local_project_path_passthrough(tmp_path):
    with patch.object(_sdist, "_build_sdist") as build, patch.object(
        _sdist, "_upload_sdist"
    ) as upload:
        out = _sdist.materialize_local_paths(["./missing"], tmp_path)

    assert out == ["./missing"]
    build.assert_not_called()
    upload.assert_not_called()


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
