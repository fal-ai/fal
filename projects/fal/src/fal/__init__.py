from __future__ import annotations

from fal import apps  # noqa: F401
from fal.api import FalServerlessHost, LocalHost, cached, function
from fal.api import function as isolated  # noqa: F401
from fal.app import App, endpoint, realtime, wrap_app  # noqa: F401
from fal.sdk import FalServerlessKeyCredentials
from fal.sync import sync_dir

local = LocalHost()
serverless = FalServerlessHost()

# DEPRECATED - use serverless instead
cloud = FalServerlessHost()

__all__ = [
    "function",
    "cached",
    "App",
    "endpoint",
    "realtime",
    # "wrap_app",
    "FalServerlessKeyCredentials",
    "sync_dir",
]
