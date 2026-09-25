import pytest

from fal.project import parse_pyproject_toml


def _write_pyproject(tmp_path, body: str) -> str:
    path = tmp_path / "pyproject.toml"
    path.write_text(body)
    return str(path)


def test_dash_and_underscore_keys_collide(tmp_path):
    path = _write_pyproject(
        tmp_path,
        """
        [tool.fal]
        my-key = "a"
        my_key = "b"
        """,
    )

    with pytest.raises(ValueError, match="my-key.*my_key|my_key.*my-key"):
        parse_pyproject_toml(path)


def test_double_dash_key_collides_with_its_stripped_form(tmp_path):
    # "--" is stripped rather than folded to "_", so "keep--alive" normalizes
    # to "keepalive" and collides with a literal "keepalive" key, not with
    # "keep_alive".
    path = _write_pyproject(
        tmp_path,
        """
        [tool.fal]
        keep--alive = "a"
        keepalive = "b"
        """,
    )

    with pytest.raises(ValueError):
        parse_pyproject_toml(path)


def test_no_collision_when_keys_are_distinct(tmp_path):
    path = _write_pyproject(
        tmp_path,
        """
        [tool.fal]
        my-key = "a"
        other-key = "b"
        """,
    )

    config = parse_pyproject_toml(path)

    assert config == {"my_key": "a", "other_key": "b"}


def test_collision_check_is_not_recursive_into_apps(tmp_path):
    path = _write_pyproject(
        tmp_path,
        """
        [tool.fal.apps."my-app"]
        my-key = "a"
        my_key = "b"
        """,
    )

    # Only [tool.fal] top-level keys are checked for collisions; keys nested
    # under an individual app are untouched here (app-level validation lives
    # in fal.cli._utils.get_app_data_from_toml instead).
    config = parse_pyproject_toml(path)

    assert config == {"apps": {"my-app": {"my-key": "a", "my_key": "b"}}}


def test_realistic_manifest_parses_unchanged(tmp_path):
    path = _write_pyproject(
        tmp_path,
        """
        [tool.fal.apps."my-app"]
        ref = "src/my_app/inference.py::MyApp"
        auth = "shared"

        [tool.fal.apps.override-app]
        ref = "src/override_app/inference.py::OverrideApp"
        name = "override-name"
        auth = "private"
        requirements = ["numpy==1.26.4"]
        machine_type = "GPU-H100"
        num_gpus = 2
        min_concurrency = 2
        regions = ["us-east"]
        keep_alive = 300
        """,
    )

    config = parse_pyproject_toml(path)

    assert config == {
        "apps": {
            "my-app": {
                "ref": "src/my_app/inference.py::MyApp",
                "auth": "shared",
            },
            "override-app": {
                "ref": "src/override_app/inference.py::OverrideApp",
                "name": "override-name",
                "auth": "private",
                "requirements": ["numpy==1.26.4"],
                "machine_type": "GPU-H100",
                "num_gpus": 2,
                "min_concurrency": 2,
                "regions": ["us-east"],
                "keep_alive": 300,
            },
        }
    }
