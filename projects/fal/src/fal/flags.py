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

# FAL_HOST takes precedence over config.host
GRPC_HOST = os.getenv("FAL_HOST") or config_host or "api.alpha.fal.ai"
if not TEST_MODE:
    assert GRPC_HOST.startswith("api"), "FAL_HOST must start with 'api'"

REST_HOST = GRPC_HOST.replace("api", "rest", 1)
REST_SCHEME = "http" if TEST_MODE or AUTH_DISABLED else "https"
REST_URL = f"{REST_SCHEME}://{REST_HOST}"

# fal.run / env.fal.run
FAL_RUN_HOST = (
    GRPC_HOST.replace("api.", "", 1).replace("alpha.", "", 1).replace(".ai", ".run", 1)
)

DONT_OPEN_LINKS = bool_envvar("FAL_DONT_OPEN_LINKS")
