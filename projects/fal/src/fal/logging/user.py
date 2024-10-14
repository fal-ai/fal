from __future__ import annotations

from structlog.typing import EventDict, WrappedLogger

from fal.auth import USER


def get_key_from_user_info(key: str) -> str | None:
    try:
        return USER.info.get(key)
    except Exception:
        # logs are fail-safe, so any exception is safe to ignore
        # this is expected to happen only when user is logged out
        # or there's no internet connection
        return None


def add_user_info(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """The structlog processor that sends the logged user id on every log"""

    event_dict["usr.id"] = get_key_from_user_info("sub")
    event_dict["usr.name"] = get_key_from_user_info("nickname")

    return event_dict
