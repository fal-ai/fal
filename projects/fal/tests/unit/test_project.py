import os

from fal.project import find_pyproject_toml, parse_pyproject_toml


def _write_project(tmp_path, name: str, body: str):
    project_dir = tmp_path / name
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(body)
    return project_dir


def test_find_pyproject_toml_follows_chdir_within_one_process(tmp_path, monkeypatch):
    project_a = _write_project(tmp_path, "project_a", '[tool.fal]\nmarker = "a"\n')
    project_b = _write_project(tmp_path, "project_b", '[tool.fal]\nmarker = "b"\n')

    monkeypatch.chdir(project_a)
    found_a = find_pyproject_toml()

    monkeypatch.chdir(project_b)
    found_b = find_pyproject_toml()

    assert found_a == str(project_a / "pyproject.toml")
    assert found_b == str(project_b / "pyproject.toml")


def test_parse_pyproject_toml_observes_in_place_edit(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text('[tool.fal]\nmarker = "v1"\n')

    first = parse_pyproject_toml(str(path))

    # Force the mtime forward: some filesystems have coarse mtime
    # resolution, and a same-second edit must still be observed.
    new_mtime = path.stat().st_mtime + 1
    path.write_text('[tool.fal]\nmarker = "v2"\n')
    os.utime(path, (new_mtime, new_mtime))

    second = parse_pyproject_toml(str(path))

    assert first == {"marker": "v1"}
    assert second == {"marker": "v2"}
