import importlib

installer = importlib.import_module("fal._installer")


def _fake_prefix(monkeypatch, tmp_path, marker=None):
    prefix = tmp_path / "env"
    prefix.mkdir()
    if marker:
        (prefix / marker).write_text("{}")
    monkeypatch.setattr(installer.sys, "prefix", str(prefix))
    monkeypatch.setattr(installer.sys, "base_prefix", str(tmp_path / "base"))
    return prefix


def _fake_installer_metadata(monkeypatch, value):
    monkeypatch.setattr(installer, "_read_installer", lambda _package: value)


def test_pipx_install_suggests_pipx_upgrade(monkeypatch, tmp_path) -> None:
    # pipx shells out to uv these days, so INSTALLER says "uv" even here.
    _fake_prefix(monkeypatch, tmp_path, marker="pipx_metadata.json")
    _fake_installer_metadata(monkeypatch, "uv")

    assert installer.get_upgrade_command() == "pipx upgrade fal"


def test_uv_tool_install_suggests_uv_tool_upgrade(monkeypatch, tmp_path) -> None:
    _fake_prefix(monkeypatch, tmp_path, marker="uv-receipt.toml")
    _fake_installer_metadata(monkeypatch, "uv")

    assert installer.get_upgrade_command() == "uv tool upgrade fal"


def test_uv_venv_suggests_uv_pip(monkeypatch, tmp_path) -> None:
    _fake_prefix(monkeypatch, tmp_path)
    _fake_installer_metadata(monkeypatch, "uv")

    assert installer.get_upgrade_command() == "uv pip install --upgrade fal"


def test_uv_project_suggests_lock_and_sync(monkeypatch, tmp_path) -> None:
    prefix = _fake_prefix(monkeypatch, tmp_path)
    (prefix.parent / "uv.lock").write_text("")
    _fake_installer_metadata(monkeypatch, "uv")

    assert installer.get_upgrade_command() == "uv lock --upgrade-package fal && uv sync"


def test_poetry_install_suggests_poetry_update(monkeypatch, tmp_path) -> None:
    _fake_prefix(monkeypatch, tmp_path)
    _fake_installer_metadata(monkeypatch, "poetry")

    assert installer.get_upgrade_command() == "poetry update fal"


def test_activated_venv_suggests_bare_pip(monkeypatch, tmp_path) -> None:
    prefix = _fake_prefix(monkeypatch, tmp_path)
    monkeypatch.setenv("VIRTUAL_ENV", str(prefix))
    _fake_installer_metadata(monkeypatch, "pip")

    assert installer.get_upgrade_command() == "pip install --upgrade fal"


def test_global_install_spells_out_the_interpreter(monkeypatch, tmp_path) -> None:
    _fake_prefix(monkeypatch, tmp_path)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(installer.sys, "executable", "/usr/bin/python3")
    _fake_installer_metadata(monkeypatch, "pip")

    assert (
        installer.get_upgrade_command()
        == "/usr/bin/python3 -m pip install --upgrade fal"
    )


def test_interpreter_path_with_spaces_is_quoted(monkeypatch, tmp_path) -> None:
    _fake_prefix(monkeypatch, tmp_path)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(installer.sys, "executable", "/opt/my python/bin/python")
    _fake_installer_metadata(monkeypatch, "pip")

    assert installer.get_upgrade_command() == (
        '"/opt/my python/bin/python" -m pip install --upgrade fal'
    )


def test_missing_metadata_falls_back_to_pip(monkeypatch, tmp_path) -> None:
    prefix = _fake_prefix(monkeypatch, tmp_path)
    monkeypatch.setenv("VIRTUAL_ENV", str(prefix))
    _fake_installer_metadata(monkeypatch, None)

    assert installer.get_upgrade_command() == "pip install --upgrade fal"


def test_detection_errors_fall_back_to_pip(monkeypatch, tmp_path) -> None:
    prefix = _fake_prefix(monkeypatch, tmp_path)
    monkeypatch.setenv("VIRTUAL_ENV", str(prefix))

    def boom(_package):
        raise RuntimeError("no metadata for you")

    monkeypatch.setattr(installer, "_read_installer", boom)

    assert installer.get_upgrade_command() == "pip install --upgrade fal"
