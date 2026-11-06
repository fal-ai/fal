"""What the CLI and the box share: routes, limits, phases and input checks.

Standard library only, so the CLI can import it without FastAPI or ``fal.app``.
"""

import re
from typing import Any

from fal.dev.errors import DevError

ROUTE_PREFIX = "/__fal_dev/"
# The app process's own control routes (127.0.0.1 on the box only).
CHILD_PREFIX = "/__fal_dev_child/"
# Requests carry the box token in this body field.
TOKEN_FIELD = "token"
# Phases in which the app process is being brought up; edits made then are
# applied by a reload once it is ready.
STARTING_PHASES = ("starting", "installing", "setup")
# Phases in which no app process is running or coming up.
STOPPED_PHASES = ("idle", "exited", "failed", "stopped")
# The box's own limit on one request; the CLI waits a little longer.
REQUEST_TIMEOUT_SECONDS = 3600
# `request` and `profile` limits: the CLI checks them as flags, and the box
# enforces them too.
MAX_WARMUPS = 50
MAX_REPEAT = 100
MAX_PROFILE_WARMUPS = 10
# The sync never sends a file over this size.
MAX_FILE_BYTES = 10 * 1024 * 1024
# Source files the CLI asks the box to copy from its object store at once.
MAX_PLACE_FILES = 5000
# How much of a failed request's response body is reported back.
BODY_TAIL_CHARS = 2000

# Set to the box name in the app process. Apps that write shared state (a
# compile cache on /data, say) namespace it by this so dev runs never read or
# overwrite what deployed runners use. `start --session-env NAME` sets more.
SESSION_ENV = "FAL_DEV_SESSION"
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Names the loader, Python, the shell or the fal SDK read: never overwritten.
# A unit test keeps this in step with the FAL_ names the SDK reads.
RESERVED_ENV = {
    "PATH",
    "HOME",
    "TMPDIR",
    "USER",
    "SHELL",
    "PWD",
    "LANG",
    "FAL_API_SOCKET",
    "FAL_APP_NAME",
    "FAL_APP_ROOT_PATH",
    "FAL_CDN_V3_HOST",
    "FAL_CONFIG_PATH",
    "FAL_DEPLOY_CHECK",
    "FAL_DONT_OPEN_LINKS",
    "FAL_ENV",
    "FAL_FORCE_AUTH_BY_USER",
    "FAL_GRPC_HOST",
    "FAL_HOME_DIR",
    "FAL_PLATFORM_URL",
    "FAL_PROFILE",
    "FAL_QUEUE_RUN_HOST",
    "FAL_REST_HOST",
    "FAL_RUNNER_STATE",
    "FAL_RUN_HOST",
    "FAL_URL_OUTPUT",
    "FAL_USE_LOCAL_UPLOADER",
    "FAL_WS_RUN_HOST",
}
RESERVED_ENV_PREFIXES = (
    "LD_",
    "DYLD_",
    "PYTHON",
    "LC_",
    "FAL_KEY",
    "FAL_HOST",
    "FAL_DEV_",
)
# A URL's scheme, and what follows "://" up to whitespace.
URL = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*)://(\S+)")
# Credentials passed as query parameters.
QUERY_CREDENTIALS = re.compile(
    r"([?&;](?:token|access[_-]?token|api[_-]?key|key|password|passwd|secret|"
    r"auth|signature|sig)=)[^&\s'\"]+",
    re.IGNORECASE,
)


def number(payload: dict, key: str, default: float, low: float, high: float) -> Any:
    """``payload[key]`` as a number in [low, high]: an int when ``default`` is one."""
    value = payload.get(key, default)
    kind = "an integer" if isinstance(default, int) else "a number"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be {kind}")
    if isinstance(default, int) and not isinstance(value, int):
        raise ValueError(f"{key} must be {kind}")
    if not low <= value <= high:
        raise ValueError(f"{key} must be between {low} and {high}")
    return value


def check_session_env(names: list[str]) -> list[str]:
    """Names `start --session-env` may set; raises DevError for any other."""
    for name in names:
        upper = name.upper()
        if (
            not ENV_NAME.fullmatch(name)
            or upper in RESERVED_ENV
            or upper.startswith(RESERVED_ENV_PREFIXES)
        ):
            raise DevError(f"cannot set environment variable {name!r}")
    return names


def _userinfo_end(rest: str) -> int:
    """Where the userinfo of a URL's text after ``://`` ends (0 if it has none)."""
    authority = re.split(r"[/?#]", rest, maxsplit=1)[0]
    if "@" in authority:
        # The last @ before the path: a password may hold "@".
        return authority.rindex("@") + 1
    # A password may also hold "/": `user:pa/ss@host`. Only a colon followed by
    # more than a port number makes the part before the "/" a user and password,
    # so `host/o/r.git@v1` (a ref after a path) is left alone.
    _, _, password = authority.partition(":")
    if password and not password.isdigit() and "@" in rest:
        return rest.index("@") + 1
    return 0


def redact(text: str) -> str:
    """Hide credentials in URLs (``https://user:token@host``, ``?token=...``)."""

    def userinfo(match: re.Match[str]) -> str:
        scheme, rest = match.groups()
        end = _userinfo_end(rest)
        if not end:
            return match.group()
        # The conventional `git@host` ssh user is not a secret; any other
        # user alone may be a token.
        if scheme.lower() in ("ssh", "git+ssh") and rest[: end - 1] == "git":
            return match.group()
        return f"{scheme}://***@{rest[end:]}"

    return QUERY_CREDENTIALS.sub(r"\1***", URL.sub(userinfo, text))
