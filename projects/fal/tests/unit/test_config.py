import os
import subprocess
import sys
from pathlib import Path

from fal.config import Config


def test_global_config_is_separate_from_profiles(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("check_updates = false\n\n[default]\n")
    monkeypatch.setenv("FAL_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("FAL_PROFILE", raising=False)

    config = Config()

    assert config.get_global("check_updates") is False
    assert config.profiles() == ["default"]

    config.save()
    reloaded_config = Config()
    assert reloaded_config.get_global("check_updates") is False
    assert reloaded_config.profiles() == ["default"]


def test_unit_collection_ignores_local_config(tmp_path):
    config_path = tmp_path / "invalid-config.toml"
    config_path.write_text("not valid TOML [")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_sdk.py::"
            "test_get_credentials_returns_key_credentials_when_available",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={
            **os.environ,
            "FAL_CONFIG_PATH": str(config_path),
            "FAL_PROFILE": "local-profile",
            "FAL_KEY": "dummy:dummy",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
