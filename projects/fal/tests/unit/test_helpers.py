from __future__ import annotations

from pathlib import Path

import pytest

import fal.helpers as helpers


def test_warm_file_reads_file_and_validates_inputs(tmp_path: Path):
    file_path = tmp_path / "model.bin"
    file_path.write_bytes(b"abcdef")

    helpers.warm_file(file_path, chunk_size=2)

    with pytest.raises(ValueError, match="chunk_size"):
        helpers.warm_file(file_path, chunk_size=0)

    with pytest.raises(FileNotFoundError, match="File not found"):
        helpers.warm_file(tmp_path / "missing.bin")

    with pytest.raises(IsADirectoryError, match="Expected a file"):
        helpers.warm_file(tmp_path)


def test_warm_dir_warms_nested_files_with_parallelism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    top_level_file = tmp_path / "model.bin"
    top_level_file.write_bytes(b"top-level")

    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_file = nested_dir / "tokenizer.json"
    nested_file.write_text("{}", encoding="utf-8")

    symlink_path = tmp_path / "model-link.bin"
    try:
        symlink_path.symlink_to(top_level_file)
    except OSError:
        symlink_path = None

    warmed_paths: list[Path] = []

    def fake_warm_file(
        file_path: str | Path,
        chunk_size: int = helpers.DEFAULT_WARM_FILE_CHUNK_SIZE,
    ) -> None:
        del chunk_size
        warmed_paths.append(Path(file_path))

    monkeypatch.setattr(helpers, "warm_file", fake_warm_file)

    helpers.warm_dir(tmp_path, parallelism=2)

    assert set(warmed_paths) == {top_level_file, nested_file}
    if symlink_path is not None:
        assert symlink_path not in warmed_paths


def test_warm_dir_skips_empty_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "empty").mkdir()
    monkeypatch.setattr(
        helpers,
        "warm_file",
        lambda *args, **kwargs: pytest.fail("warm_file should not be called"),
    )

    helpers.warm_dir(tmp_path)


def test_warm_dir_validates_inputs(tmp_path: Path):
    file_path = tmp_path / "model.bin"
    file_path.write_bytes(b"abcdef")

    with pytest.raises(ValueError, match="parallelism"):
        helpers.warm_dir(tmp_path, parallelism=0)

    with pytest.raises(FileNotFoundError, match="Directory not found"):
        helpers.warm_dir(tmp_path / "missing")

    with pytest.raises(NotADirectoryError, match="Expected a directory"):
        helpers.warm_dir(file_path)
