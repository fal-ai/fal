import importlib
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

cli_main = importlib.import_module("fal.cli.main")
fal_version = importlib.import_module("fal._version")


def _setup_version_check(monkeypatch, tmp_path, config_text=None):
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("FAL_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("FAL_PROFILE", raising=False)

    if config_text is not None:
        config_path.write_text(config_text)

    monkeypatch.setattr(
        fal_version,
        "_PYPI_CACHE_PATH",
        str(tmp_path / "cache" / "pypi.json"),
    )

    console = SimpleNamespace(is_terminal=True, print=MagicMock())
    monkeypatch.setattr(cli_main, "console", console)
    return console


def _check_latest_version():
    with patch(
        "fal._version._fetch_pypi_data",
        return_value={"info": {"version": "99.0.0"}},
    ) as latest:
        with patch("fal._version.version_tuple", (1, 0, 0)):
            cli_main._check_latest_version()

    return latest


def test_update_check_prints_once_per_day_by_default(monkeypatch, tmp_path):
    console = _setup_version_check(monkeypatch, tmp_path)

    latest = _check_latest_version()
    assert latest.call_count == 1
    assert console.print.call_count == 1

    console.print.reset_mock()
    latest = _check_latest_version()
    assert latest.call_count == 0
    console.print.assert_not_called()

    expired_mtime = time.time() - 24 * 60 * 60 - 1
    os.utime(fal_version._PYPI_CACHE_PATH, (expired_mtime, expired_mtime))
    latest = _check_latest_version()
    assert latest.call_count == 1
    assert console.print.call_count == 1


def test_update_check_can_be_disabled(monkeypatch, tmp_path):
    console = _setup_version_check(
        monkeypatch,
        tmp_path,
        "check_updates = false\n",
    )

    latest = _check_latest_version()

    assert latest.call_count == 0
    console.print.assert_not_called()
