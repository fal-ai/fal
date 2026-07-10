from types import SimpleNamespace
from unittest.mock import MagicMock

import fal.cli.profile as profile


def test_profile_commands_do_not_treat_global_options_as_profiles(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("check_updates = false\n")
    monkeypatch.setenv("FAL_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("FAL_PROFILE", raising=False)

    args = SimpleNamespace(PROFILE="check_updates", console=MagicMock())

    profile._set(args)
    profile._create(args)
    profile._delete(args)

    assert config_path.read_text() == "check_updates = false\n"
    assert args.console.print.call_count == 3
