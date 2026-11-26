"""--strategy and --reset-scale/--no-scale for pyproject.toml app-name deploys.

Before this fix, `fal deploy <app-name>` replaced the whole deploy config with
the pyproject.toml entry, so both flags were silently dropped.
"""

from unittest.mock import patch

import pytest

from fal.api.deploy import _resolve_deployment_reference
from fal.cli._utils import AppData
from fal.cli.main import parse_args

TOML_APP = AppData(
    ref="/project/app.py::MyApp",
    deployment_strategy="recreate",
    reset_scale=True,
)


@pytest.fixture
def toml_app():
    with patch("fal.cli._utils.get_app_data_from_toml", return_value=TOML_APP):
        yield


def resolve(app_ref, **kwargs):
    return _resolve_deployment_reference(app_ref, **kwargs)[1]


def test_pyproject_values_apply_when_no_flag_is_given(toml_app):
    app_data = resolve(("my-app", None))
    assert app_data.deployment_strategy == "recreate"
    assert app_data.reset_scale is True


def test_explicit_strategy_wins_over_pyproject(toml_app):
    assert resolve(("my-app", None), strategy="rolling").deployment_strategy == (
        "rolling"
    )


def test_explicit_no_scale_wins_over_pyproject(toml_app):
    assert resolve(("my-app", None), reset_scale=False).reset_scale is False


def test_explicit_reset_scale_wins_over_pyproject(toml_app):
    with patch(
        "fal.cli._utils.get_app_data_from_toml",
        return_value=AppData(ref="/project/app.py::MyApp", reset_scale=False),
    ):
        assert resolve(("my-app", None), reset_scale=True).reset_scale is True


def test_file_ref_defaults_are_unchanged():
    # None registers as rolling; scale settings come from the live app.
    app_data = resolve(("/project/app.py", "MyApp"))
    assert app_data.deployment_strategy is None
    assert app_data.reset_scale is False


def test_app_name_and_auth_flags_are_still_rejected(toml_app):
    with pytest.raises(ValueError, match="Cannot use --app-name or --auth"):
        resolve(("my-app", None), app_name="other")


@pytest.mark.parametrize(
    ("flags", "strategy", "scale"),
    [
        ([], None, None),
        (["--strategy", "recreate"], "recreate", None),
        (["--reset-scale"], None, True),
        (["--no-scale"], None, False),
    ],
)
def test_cli_leaves_unset_flags_as_none(flags, strategy, scale):
    args = parse_args(["deploy", "my-app", *flags])
    assert args.strategy == strategy
    assert args.app_scale_settings == scale


def test_inherited_scale_note_follows_the_pyproject_entry(toml_app):
    from argparse import Namespace

    from fal.cli.deploy import _effective_reset_scale

    unset = Namespace(app_scale_settings=None)
    assert _effective_reset_scale(unset, ("my-app", None)) is True
    assert _effective_reset_scale(unset, ("/project/app.py", "MyApp")) is False
    explicit = Namespace(app_scale_settings=False)
    assert _effective_reset_scale(explicit, ("my-app", None)) is False
