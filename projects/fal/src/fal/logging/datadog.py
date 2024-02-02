from __future__ import annotations

import sys
import traceback
import warnings

from datadog_api_client import Configuration, ThreadedApiClient
from datadog_api_client.v2.api.logs_api import LogsApi
from datadog_api_client.v2.model.http_log import HTTPLog
from datadog_api_client.v2.model.http_log_item import HTTPLogItem
from structlog.typing import EventDict, WrappedLogger

from fal.env import CLI_ENV, DATADOG_API_KEY, DATADOG_APP_KEY
from fal.logging.trace import get_current_span_context

if sys.version_info >= (3, 10):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata


configuration = Configuration()
configuration.api_key["apiKeyAuth"] = DATADOG_API_KEY
configuration.api_key["appKeyAuth"] = DATADOG_APP_KEY


def _is_error_level(level: str) -> bool:
    return level in ["error", "exception", "critical"]


def submit_to_datadog(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    if configuration.api_key["apiKeyAuth"] is None:
        return event_dict

    log_data = dict(event_dict)
    event = log_data.pop("event")
    level = log_data.pop("level")

    current_span = get_current_span_context()
    attributes = log_data.copy()
    tags: dict[str, str] = {}
    if current_span is not None:
        tags["invocation_id"] = current_span.invocation_id
        attributes["dd.trace_id"] = current_span.trace_id
        attributes["dd.span_id"] = current_span.span_id

    stack = None
    if _is_error_level(method_name):
        attributes["error.message"] = str(event)
        attributes["error.kind"] = type(event).__name__
        stack = traceback.format_exc()

    ddtags = ",".join([f"{key}:{value}" for (key, value) in tags.items()])
    log_item = HTTPLogItem(
        message=str(event),
        level=level,
        hostname="client",
        service="fal-serverless-cli",
        env=CLI_ENV,
        version=importlib_metadata.version("fal"),
        ddsource="python",
        ddtags=ddtags,
        traceback=stack,
        **attributes,
    )
    with ThreadedApiClient(configuration) as api_client:
        # Deprecation warning of underlying dependencies should not be shown to users
        # TODO enable it only in the prod distribution (better: remove when fixed)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO improve this - add batching
        api_instance = LogsApi(api_client)
        _ = api_instance.submit_log(HTTPLog([log_item]))
        api_client.close()

    return event_dict
