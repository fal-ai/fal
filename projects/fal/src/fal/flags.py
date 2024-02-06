from __future__ import annotations

import os


def bool_envvar(name: str):
    if name in os.environ:
        val = os.environ[name].lower().strip()
        return val != "0" and val != "false" and val != ""
    return False


DEBUG = bool_envvar("DEBUG")
TEST_MODE = bool_envvar("ISOLATE_TEST_MODE")
AUTH_DISABLED = bool_envvar("ISOLATE_AUTH_DISABLED")

GRPC_HOST = os.getenv("FAL_HOST", "api.alpha.fal.ai")
if not TEST_MODE:
    assert GRPC_HOST.startswith("api"), "FAL_HOST must start with 'api'"

GATEWAY_HOST = GRPC_HOST.replace("api", "gateway", 1)

REST_HOST = GRPC_HOST.replace("api", "rest", 1)
REST_SCHEME = "http" if TEST_MODE or AUTH_DISABLED else "https"
REST_URL = f"{REST_SCHEME}://{REST_HOST}"

# fal.run / env.fal.run
FAL_RUN_HOST = (
    GRPC_HOST.replace("api.", "", 1).replace("alpha.", "", 1).replace(".ai", ".run", 1)
)

FORCE_SETUP = bool_envvar("FAL_FORCE_SETUP")
