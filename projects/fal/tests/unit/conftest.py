import pytest


@pytest.fixture(autouse=True)
def isolate_fal_config(monkeypatch, tmp_path):
    # Subprocesses must inherit the same isolation as in-process credential lookup.
    monkeypatch.setenv("FAL_CONFIG_PATH", str(tmp_path / "fal-config.toml"))
    monkeypatch.delenv("FAL_PROFILE", raising=False)


ACCOUNTS = [
    {"nickname": "me", "is_personal": True},
    *({"nickname": name, "is_personal": False} for name in ("mine", "t", "red")),
]


@pytest.fixture
def dev_account(monkeypatch, tmp_path):
    """`fal dev` commands run as the personal account `me`, with no API lookups."""
    from fal.dev import state

    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "dev-state")
    monkeypatch.setattr("fal.dev.team.key_account", lambda: None)
    monkeypatch.setattr("fal.dev.team.accounts", lambda: ACCOUNTS)
    monkeypatch.setattr("fal.dev.team.production_apps", lambda client: [])
