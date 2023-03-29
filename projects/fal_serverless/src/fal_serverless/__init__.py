from __future__ import annotations

from fal_serverless.api import FalServerlessHost, LocalHost, cached, isolated
from fal_serverless.sdk import FalServerlessKeyCredentials

local = LocalHost()
serverless = FalServerlessHost()

# DEPRECATED - use serverless instead
cloud = FalServerlessHost()
