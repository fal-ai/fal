from __future__ import annotations

from structlog.typing import EventDict, WrappedLogger

from fal.auth import USER


def add_user_info(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """The structlog processor that sends the logged user id on every log"""
    user_id: str | None = None
    try:
        user_id = USER.info.get("sub")
        user_name = USER.info.get("nickname")
    except Exception:
        # logs are fail-safe, so any exception is safe to ignore
        # this is expected to happen only when user is logged out
        # or there's no internet connection
        pass

    event_dict["usr.id"] = user_id
    event_dict["usr.name"] = user_name

    return event_dict
