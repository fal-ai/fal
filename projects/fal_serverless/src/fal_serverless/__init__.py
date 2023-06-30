from __future__ import annotations

from fal_serverless.api import FalServerlessHost, LocalHost, cached, isolated
from fal_serverless.decorators import download_file, download_weights
from fal_serverless.sdk import FalServerlessKeyCredentials
from fal_serverless.sync import sync_dir

local = LocalHost()
serverless = FalServerlessHost()

# DEPRECATED - use serverless instead
cloud = FalServerlessHost()
