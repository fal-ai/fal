from unittest.mock import patch

from fal.auth.local import load_preference, save_preference


def test_save_and_load_preference(tmp_path):
    with patch("fal.auth.local._FAL_HOME_DIR", str(tmp_path)):
        save_preference("last_auth_connection", "github")
        assert load_preference("last_auth_connection") == "github"


def test_load_preference_returns_none_when_missing(tmp_path):
    with patch("fal.auth.local._FAL_HOME_DIR", str(tmp_path)):
        assert load_preference("last_auth_connection") is None


def test_save_preference_overwrites(tmp_path):
    with patch("fal.auth.local._FAL_HOME_DIR", str(tmp_path)):
        save_preference("last_auth_connection", "github")
        save_preference("last_auth_connection", "google")
        assert load_preference("last_auth_connection") == "google"


def test_load_preference_strips_whitespace(tmp_path):
    with patch("fal.auth.local._FAL_HOME_DIR", str(tmp_path)):
        (tmp_path / "test_key").write_text("  value  \n")
        assert load_preference("test_key") == "value"


def test_load_preference_returns_none_for_empty_file(tmp_path):
    with patch("fal.auth.local._FAL_HOME_DIR", str(tmp_path)):
        (tmp_path / "test_key").write_text("   \n")
        assert load_preference("test_key") is None
