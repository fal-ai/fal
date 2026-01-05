from __future__ import annotations

import base64
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

import httpx


class GoogleColabState:
    def __init__(self):
        self.is_checked = False
        self.lock = Lock()
        self.secret: Optional[str] = None


_colab_state = GoogleColabState()


def is_google_colab() -> bool:
    try:
        from IPython import get_ipython

        return "google.colab" in str(get_ipython())
    except ModuleNotFoundError:
        return False
    except NameError:
        return False


def get_colab_token() -> Optional[str]:
    if not is_google_colab():
        return None
    with _colab_state.lock:
        if _colab_state.is_checked:  # request access only once
            return _colab_state.secret

        try:
            from google.colab import userdata  # noqa: I001
        except ImportError:
            return None

        try:
            token = userdata.get("FAL_KEY")
            _colab_state.secret = token.strip()
        except Exception:
            _colab_state.secret = None

        _colab_state.is_checked = True
        return _colab_state.secret


class MissingCredentialsError(Exception):
    pass


@dataclass(frozen=True)
class AuthCredentials:
    """Represents an authorization header value."""

    scheme: str
    token: str

    @property
    def header_value(self) -> str:
        return f"{self.scheme} {self.token}"

    def as_headers(self) -> dict[str, str]:
        return {"Authorization": self.header_value}


FAL_RUN_HOST = os.environ.get("FAL_RUN_HOST", "fal.run")
FAL_QUEUE_RUN_HOST = os.environ.get("FAL_QUEUE_RUN_HOST", f"queue.{FAL_RUN_HOST}")

AUTH0_DOMAIN = "auth.fal.ai"
AUTH0_TOKEN_URL = f"https://{AUTH0_DOMAIN}/oauth/token"
AUTH0_CLIENT_ID = "TwXR51Vz8JbY8GUUMy6EyuVR0fTO7N4N"
AUTH_TOKEN_FILENAME = "auth0_token"
AUTH_LOCK_FILENAME = ".portalock"
DEFAULT_FAL_HOME = Path.home() / ".fal"


def _get_fal_home_dir() -> Path:
    return Path(os.getenv("FAL_HOME_DIR", str(DEFAULT_FAL_HOME))).expanduser()


@contextmanager
def _token_lock():
    """Best effort lock shared with fal-cli."""

    try:
        import portalocker
    except Exception:
        yield
        return

    lock_file = _get_fal_home_dir() / AUTH_LOCK_FILENAME
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.utils.TemporaryFileLock(
        str(lock_file), fail_when_locked=False, timeout=20
    ):
        yield


def _read_auth_tokens() -> tuple[Optional[str], Optional[str]]:
    path = _get_fal_home_dir() / AUTH_TOKEN_FILENAME
    if not path.exists():
        return None, None

    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return None, None

    refresh_token = lines[0]
    access_token: Optional[str] = None
    if len(lines) > 1:
        access_token = lines[1]

    return refresh_token or None, access_token or None


def _write_auth_tokens(refresh_token: str, access_token: Optional[str]) -> None:
    path = _get_fal_home_dir() / AUTH_TOKEN_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)

    contents = [refresh_token]
    if access_token:
        contents.append(access_token)

    path.write_text("\n".join(contents))


def _decode_jwt_exp(token: str) -> Optional[int]:
    """Returns exp claim in seconds since epoch or None if unreadable."""

    try:
        payload_segment = token.split(".")[1]
        padding = "=" * (-len(payload_segment) % 4)
        payload = base64.urlsafe_b64decode(payload_segment + padding)
        claims = json.loads(payload.decode("utf-8"))
        return int(claims.get("exp"))
    except Exception:
        return None


def _is_access_token_expired(token: str, *, leeway_seconds: int = 300) -> bool:
    exp = _decode_jwt_exp(token)
    if exp is None:
        # If we cannot parse the token, assume it is expired to force refresh.
        return True

    return time.time() + leeway_seconds >= exp


def _refresh_access_token(refresh_token: str) -> dict:
    response = httpx.post(
        AUTH0_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "client_id": AUTH0_CLIENT_ID,
            "refresh_token": refresh_token,
        },
        timeout=30,
    )

    try:
        token_data = response.json()
    except Exception:
        token_data = {}

    if response.status_code != 200 or "access_token" not in token_data:
        raise MissingCredentialsError(
            "Failed to refresh fal auth token. Please run `fal auth login` again."
        )

    return token_data


def _load_bearer_token_from_login() -> Optional[str]:
    """
    Try to reuse tokens created by `fal auth login`.

    We share the same file layout as fal-cli:
    - refresh token on first line
    - optional cached access token on second line
    """

    with _token_lock():
        refresh_token, access_token = _read_auth_tokens()

    if not refresh_token:
        return None

    if access_token and not _is_access_token_expired(access_token):
        return access_token

    token_data = _refresh_access_token(refresh_token)
    new_refresh = token_data.get("refresh_token", refresh_token)
    new_access = token_data["access_token"]

    with _token_lock():
        _write_auth_tokens(new_refresh, new_access)

    return new_access


def fetch_auth_credentials() -> AuthCredentials:
    """
    Return credentials using this priority:
    1) FAL_KEY / FAL_KEY_ID+FAL_KEY_SECRET / Colab secret (unless FAL_FORCE_AUTH_BY_USER=1)
    2) Tokens saved by `fal auth login`
    """

    force_user_auth = os.environ.get("FAL_FORCE_AUTH_BY_USER") == "1"

    if not force_user_auth:
        if key := os.getenv("FAL_KEY"):
            return AuthCredentials("Key", key)
        elif (key_id := os.getenv("FAL_KEY_ID")) and (
            fal_key_secret := os.getenv("FAL_KEY_SECRET")
        ):
            return AuthCredentials("Key", f"{key_id}:{fal_key_secret}")
        elif colab_token := get_colab_token():
            return AuthCredentials("Key", colab_token)

    if bearer := _load_bearer_token_from_login():
        return AuthCredentials("Bearer", bearer)

    raise MissingCredentialsError(
        "No credentials found. Set FAL_KEY (or FAL_KEY_ID/FAL_KEY_SECRET) or login via `fal auth login`."
    )


def fetch_credentials() -> str:
    """
    Legacy helper kept for backwards compatibility.

    It only returns Key-based credentials; user-based auth (Bearer) will raise
    MissingCredentialsError so callers don't accidentally send it as a key.
    """

    auth = fetch_auth_credentials()
    if auth.scheme.lower() != "key":
        raise MissingCredentialsError(
            "Key credentials not found. Set FAL_KEY (or FAL_KEY_ID/FAL_KEY_SECRET)."
        )

    return auth.token
