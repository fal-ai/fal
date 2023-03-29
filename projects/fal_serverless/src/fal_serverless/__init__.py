from __future__ import annotations

from fal_serverless.api import KoldstartHost, LocalHost, cached, isolated
from fal_serverless.sdk import CloudKeyCredentials

local = LocalHost()
cloud = KoldstartHost()
