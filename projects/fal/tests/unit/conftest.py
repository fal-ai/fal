import pytest


@pytest.fixture(autouse=True)
def isolate_fal_config(monkeypatch, tmp_path):
    # Subprocesses must inherit the same isolation as in-process credential lookup.
    monkeypatch.setenv("FAL_CONFIG_PATH", str(tmp_path / "fal-config.toml"))
    monkeypatch.delenv("FAL_PROFILE", raising=False)
