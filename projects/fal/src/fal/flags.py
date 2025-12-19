from __future__ import annotations

import os

from fal.config import Config


def bool_envvar(name: str):
    if name in os.environ:
        val = os.environ[name].lower().strip()
        return val != "0" and val != "false" and val != ""
    return False


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

# Legacy: FAL_HOST takes precedence over config.host (unless FAL_GRPC_HOST is set)
GRPC_HOST = _GRPC_HOST_ENV or os.getenv("FAL_HOST") or config_host or "api.alpha.fal.ai"
if not TEST_MODE:
    # Keep legacy safety check, but allow explicit overrides via FAL_GRPC_HOST.
    if not _GRPC_HOST_ENV:
        assert GRPC_HOST.startswith(
            "api"
        ), "FAL_HOST must start with 'api' (use FAL_GRPC_HOST to override)"

REST_HOST = _REST_HOST_ENV or (
    GRPC_HOST.replace("api", "rest", 1) if GRPC_HOST.startswith("api") else GRPC_HOST
)
REST_SCHEME = "http" if TEST_MODE or AUTH_DISABLED else "https"
REST_URL = f"{REST_SCHEME}://{REST_HOST}"

# fal.run / env.fal.run
FAL_RUN_HOST = _RUN_HOST_ENV or (
    GRPC_HOST.replace("fal.dev", "run.fal.dev")
    .replace("api.", "", 1)
    .replace("alpha.", "", 1)
    if GRPC_HOST.endswith("fal.dev")
    else GRPC_HOST.replace("api.", "", 1)
    .replace("alpha.", "", 1)
    .replace(".ai", ".run", 1)
)

FAL_QUEUE_RUN_HOST = _QUEUE_RUN_HOST_ENV or (
    FAL_RUN_HOST.replace("run.fal.dev", "queue.run.fal.dev")
    if FAL_RUN_HOST.endswith("run.fal.dev")
    else f"queue.{FAL_RUN_HOST}"
    if FAL_RUN_HOST.endswith(".run")
    else FAL_RUN_HOST
)

DONT_OPEN_LINKS = bool_envvar("FAL_DONT_OPEN_LINKS")
