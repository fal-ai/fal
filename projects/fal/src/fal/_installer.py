"""Detect how fal was installed, so upgrade hints suggest a command that works.

`pip install --upgrade fal` is wrong (or a no-op) for the common non-pip
installs: pipx and `uv tool` put fal in a managed virtualenv of their own, and
uv projects revert anything installed behind the lockfile's back.
"""

import os
import sys
from typing import Optional

_PACKAGE = "fal"

# `pipx` and `uv tool` both drop a receipt in the root of the virtualenv they
# manage, which is the only reliable way to tell them apart from a plain venv:
# pipx installs with uv under the hood these days, so the dist-info INSTALLER
# says "uv" for both.
_PIPX_MARKER = "pipx_metadata.json"
_UV_TOOL_MARKER = "uv-receipt.toml"


def _installer_from_disk(package: str) -> Optional[str]:
    """Read INSTALLER from the dist-info sitting next to this package.

    `importlib.metadata` resolves by name against `sys.path`, so a stale
    `fal.egg-info` in the working directory can shadow the real install. The
    dist-info next to the module we are actually running cannot.
    """
    import glob

    site_packages = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern = os.path.join(site_packages, f"{package}-*.dist-info", "INSTALLER")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as fobj:
                installer = fobj.read().strip().lower()
        except OSError:
            continue

        if installer:
            return installer

    return None


def _installer_from_metadata(package: str) -> Optional[str]:
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        installer = distribution(package).read_text("INSTALLER")
    except (PackageNotFoundError, OSError, ValueError):
        return None

    if not installer:
        return None

    return installer.strip().lower()


def _read_installer(package: str) -> Optional[str]:
    return _installer_from_disk(package) or _installer_from_metadata(package)


def _is_uv_project(prefix: str) -> bool:
    """Is this venv the `.venv` of a uv project with a lockfile?"""
    return os.path.exists(os.path.join(os.path.dirname(prefix), "uv.lock"))


def _same_path(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return False

    try:
        return os.path.samefile(left, right)
    except OSError:
        # One of them does not exist; compare them textually instead.
        return os.path.normcase(os.path.normpath(left)) == os.path.normcase(
            os.path.normpath(right)
        )


def _pip_command() -> str:
    # `pip` on PATH is the right one only when the venv fal runs from is also
    # the activated one. If a *different* venv is activated, that `pip` would
    # upgrade the wrong environment, so spell out the interpreter instead.
    if sys.prefix != sys.base_prefix and _same_path(
        os.environ.get("VIRTUAL_ENV"), sys.prefix
    ):
        return "pip"

    executable = sys.executable or "python"
    if " " in executable:
        executable = f'"{executable}"'

    return f"{executable} -m pip"


def detect_install_manager(package: str = _PACKAGE) -> str:
    prefix = sys.prefix

    if os.path.exists(os.path.join(prefix, _PIPX_MARKER)):
        return "pipx"

    if os.path.exists(os.path.join(prefix, _UV_TOOL_MARKER)):
        return "uv-tool"

    installer = _read_installer(package)

    if installer == "uv":
        return "uv-project" if _is_uv_project(prefix) else "uv"

    if installer in ("poetry", "pdm"):
        return installer

    return "pip"


def get_upgrade_command(package: str = _PACKAGE) -> str:
    try:
        manager = detect_install_manager(package)
    except Exception:
        manager = "pip"

    if manager == "pipx":
        return f"pipx upgrade {package}"
    if manager == "uv-tool":
        return f"uv tool upgrade {package}"
    if manager == "uv-project":
        return f"uv lock --upgrade-package {package} && uv sync"
    if manager == "uv":
        return f"uv pip install --upgrade {package}"
    if manager == "poetry":
        return f"poetry update {package}"
    if manager == "pdm":
        return f"pdm update {package}"

    return f"{_pip_command()} install --upgrade {package}"
