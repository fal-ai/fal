from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import portalocker

from fal.console import console
from fal.console.ux import maybe_open_browser_tab

_DEFAULT_HOME_DIR = str(Path.home() / ".fal")
_FAL_HOME_DIR = os.getenv("FAL_HOME_DIR", _DEFAULT_HOME_DIR)
_AUTH0_TOKEN_FILE = "auth0_token"
_WORKOS_TOKEN_FILE = "workos_token"
_LOCK_FILE = ".portalock"

AuthProvider = Literal["auth0", "workos"]


@dataclass
class SavedAuthConfig:
    provider: AuthProvider
    refresh_token: str
    access_token: str | None


@dataclass
class ActiveAuthConfig(SavedAuthConfig):
    access_token: str


@dataclass
class UserInfo:
    name: str
    id: str


def _check_dir_exist():
    """
    Checks if a specific directory exists, creates if not.
    In case the user didn't set a custom dir, will turn to the default home
    """
    dir = Path(_FAL_HOME_DIR).expanduser()

    if not dir.exists():
        dir.mkdir(parents=True)

    return dir


def _get_existing_token_file() -> tuple[Path, AuthProvider]:
    dir = _check_dir_exist()
    workos_token_file = dir / _WORKOS_TOKEN_FILE

    if workos_token_file.exists():
        return workos_token_file, "workos"

    return dir / _AUTH0_TOKEN_FILE, "auth0"


def _get_provider_token_file(provider: AuthProvider) -> Path:
    if provider == "workos":
        return _check_dir_exist() / _WORKOS_TOKEN_FILE
    return _check_dir_exist() / _AUTH0_TOKEN_FILE


def _read_token_file(path: Path) -> list[str] | None:
    if path.exists():
        return path.read_text().splitlines()
    return None


def _write_file(path: Path, contents: list[str]):
    path.write_text("\n".join(contents))


def load_auth_config() -> SavedAuthConfig | None:
    token_file, provider = _get_existing_token_file()
    lines = _read_token_file(token_file)
    if not lines:
        return None

    refresh_token = lines[0]
    access_token = None
    if len(lines) > 1:
        access_token = lines[1]

    return SavedAuthConfig(
        provider=provider, refresh_token=refresh_token, access_token=access_token
    )


def save_auth_config(
    auth_config: SavedAuthConfig,
) -> None:
    tokens = [auth_config.refresh_token]
    if auth_config.access_token:
        tokens.append(auth_config.access_token)

    token_file = _get_provider_token_file(auth_config.provider)
    _write_file(token_file, tokens)


def delete_token() -> None:
    dir = _check_dir_exist()

    try:
        (dir / _AUTH0_TOKEN_FILE).unlink()
    except Exception:
        pass

    try:
        (dir / _WORKOS_TOKEN_FILE).unlink()
    except Exception:
        pass


@contextmanager
def lock_token():
    """
    Lock the access to the token file to avoid race conditions when running multiple
    scripts at the same time.
    """
    lock_file = _check_dir_exist() / _LOCK_FILE
    with portalocker.utils.TemporaryFileLock(
        str(lock_file),
        fail_when_locked=False,
        timeout=20,
    ):
        yield


def open_browser(url: str, code: str | None) -> None:
    maybe_open_browser_tab(url)

    console.print(
        "If browser didn't open automatically, "
        "on your computer or mobile device navigate to"
    )
    console.print(url)

    if code:
        console.print(
            f"\nConfirm it shows the following code: [markdown.code]{code}[/]\n"
        )


def verify_access_token_expiration(token: str) -> None:
    from jwt import decode

    leeway = 30 * 60  # 30 minutes
    decode(
        token,
        leeway=-leeway,  # negative to consider expired before actual expiration
        options={"verify_exp": True, "verify_signature": False},
    )
