from unittest.mock import patch

import pytest

from fal.cli._utils import get_app_data_from_toml

BASE_APP = {
    "ref": "src/my_app/inference.py::MyApp",
}


def _manifest(app_data: dict, **top_level) -> dict:
    data = dict(top_level)
    data["apps"] = {"my-app": app_data}
    return data


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_unknown_top_level_key_warns_but_does_not_raise(mock_parse, mock_find, capsys):
    mock_parse.return_value = _manifest(dict(BASE_APP), some_stray_key="oops")

    app_data = get_app_data_from_toml("my-app")

    assert app_data.ref is not None
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "some_stray_key" in captured.out


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_no_warning_when_only_apps_key_present(mock_parse, mock_find, capsys):
    mock_parse.return_value = _manifest(dict(BASE_APP))

    get_app_data_from_toml("my-app")

    captured = capsys.readouterr()
    assert "WARNING" not in captured.out


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_invalid_auth_raises(mock_parse, mock_find):
    mock_parse.return_value = _manifest({**BASE_APP, "auth": "privat"})

    with pytest.raises(ValueError, match="auth"):
        get_app_data_from_toml("my-app")


@pytest.mark.parametrize("mode", ["public", "private", "shared"])
@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_valid_auth_modes_pass(mock_parse, mock_find, mode):
    mock_parse.return_value = _manifest({**BASE_APP, "auth": mode})

    app_data = get_app_data_from_toml("my-app")

    assert app_data.auth == mode


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_negative_keep_alive_raises(mock_parse, mock_find):
    mock_parse.return_value = _manifest({**BASE_APP, "keep_alive": -1})

    with pytest.raises(ValueError, match="keep_alive"):
        get_app_data_from_toml("my-app")


@pytest.mark.parametrize("value", [0, 300])
@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_non_negative_keep_alive_passes(mock_parse, mock_find, value):
    mock_parse.return_value = _manifest({**BASE_APP, "keep_alive": value})

    app_data = get_app_data_from_toml("my-app")

    assert app_data.options.host["keep_alive"] == value


@patch("fal.cli._utils.find_pyproject_toml", return_value="pyproject.toml")
@patch("fal.cli._utils.parse_pyproject_toml")
def test_realistic_manifest_parses_unchanged(mock_parse, mock_find):
    mock_parse.return_value = {
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

    my_app = get_app_data_from_toml("my-app")
    override_app = get_app_data_from_toml("override-app")

    assert my_app.auth == "shared"
    assert override_app.name == "override-name"
    assert override_app.options.host["machine_type"] == "GPU-H100"
    assert override_app.options.host["keep_alive"] == 300
    assert override_app.options.host["regions"] == ["us-east"]
    assert override_app.options.environment["requirements"] == ["numpy==1.26.4"]
