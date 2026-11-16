from __future__ import annotations

import os

from fal.config import Config


def bool_envvar(name: str):
    if name in os.environ:
        val = os.environ[name].lower().strip()
        return val != "0" and val != "false" and val != ""
    return False


def swap_api_label(host: str, replacement: str) -> str:
    """Swap the `api` label of a host for `replacement`.

    One DNS label names the service, so the REST host is the gRPC host with
    that label swapped. The label is not always the first one: an environment
    may put its own labels in front of it. A host with no `api` label is
    returned unchanged.
    """
    labels = host.split(".")
    for index, label in enumerate(labels):
        if label == "api":
            labels[index] = replacement
            return ".".join(labels)
    return host


def api_label_index(host: str) -> int:
    """Position of the `api` label in a host, or 0 when it has none."""
    labels = host.split(".")
    return labels.index("api") if "api" in labels else 0


def insert_label(host: str, label: str, index: int) -> str:
    """Insert `label` into `host` at `index`."""
    labels = host.split(".")
    labels.insert(min(index, len(labels)), label)
    return ".".join(labels)


DEBUG = bool_envvar("DEBUG")
TEST_MODE = bool_envvar("ISOLATE_TEST_MODE")
AUTH_DISABLED = bool_envvar("ISOLATE_AUTH_DISABLED")

config = Config()
config_host = config.get("host")

# Explicit host overrides (highest precedence)
_GRPC_HOST_ENV = os.getenv("FAL_GRPC_HOST")
_REST_HOST_ENV = os.getenv("FAL_REST_HOST")
_RUN_HOST_ENV = os.getenv("FAL_RUN_HOST")
_QUEUE_RUN_HOST_ENV = os.getenv("FAL_QUEUE_RUN_HOST")
_WS_RUN_HOST_ENV = os.getenv("FAL_WS_RUN_HOST")

# Legacy: FAL_HOST takes precedence over config.host (unless FAL_GRPC_HOST is set)
GRPC_HOST = _GRPC_HOST_ENV or os.getenv("FAL_HOST") or config_host or "api.alpha.fal.ai"

REST_HOST = _REST_HOST_ENV or swap_api_label(GRPC_HOST, "rest")
REST_SCHEME = "http" if TEST_MODE or AUTH_DISABLED else "https"
REST_URL = f"{REST_SCHEME}://{REST_HOST}"

# fal.run / env.fal.run
FAL_RUN_HOST = _RUN_HOST_ENV or (
    GRPC_HOST.replace("api.", "", 1).replace("alpha.", "", 1).replace(".ai", ".run", 1)
)

# The queue and ws hosts name their service in the same position the gRPC host
# names `api`, so any environment labels in front of it stay where they are.
_SERVICE_INDEX = api_label_index(GRPC_HOST)

FAL_QUEUE_RUN_HOST = _QUEUE_RUN_HOST_ENV or insert_label(
    FAL_RUN_HOST, "queue", _SERVICE_INDEX
)

FAL_WS_RUN_HOST = _WS_RUN_HOST_ENV or insert_label(FAL_RUN_HOST, "ws", _SERVICE_INDEX)

DONT_OPEN_LINKS = bool_envvar("FAL_DONT_OPEN_LINKS")

# Controls which URLs are printed when running/deploying apps
# Valid values: "all" (default), "playground", "none"
URL_OUTPUT = os.getenv("FAL_URL_OUTPUT", "all").lower().strip()
