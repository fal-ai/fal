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
