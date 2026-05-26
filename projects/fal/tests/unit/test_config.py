from fal.config import Config


def test_profiles_ignore_top_level_settings(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("check_updates = false\n\n[default]\n")
    monkeypatch.setenv("FAL_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("FAL_PROFILE", raising=False)

    config = Config()

    assert config.get_global("check_updates") is False
    assert config.profiles() == ["default"]
