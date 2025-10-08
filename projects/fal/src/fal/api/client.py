from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from fal.api import FAL_SERVERLESS_DEFAULT_URL
from fal.sdk import (
    AliasInfo,
    Credentials,
    RunnerInfo,
)

from . import apps as apps_api
from . import deploy as deploy_api
from . import runners as runners_api


class _AppsNamespace:
    def __init__(self, client: SyncServerlessClient):
        self.client = client

    def list(self, *, filter: str | None = None) -> List[AliasInfo]:
        return apps_api.list_apps(self.client, filter=filter)

    def runners(
        self, app_name: str, *, since=None, state: List[str] | None = None
    ) -> List[RunnerInfo]:
        return apps_api.apps_runners(self.client, app_name, since=since, state=state)

    def scale(
        self,
        app_name: str,
        *,
        keep_alive: int | None = None,
        max_multiplexing: int | None = None,
        max_concurrency: int | None = None,
        min_concurrency: int | None = None,
        concurrency_buffer: int | None = None,
        concurrency_buffer_perc: int | None = None,
        request_timeout: int | None = None,
        startup_timeout: int | None = None,
        machine_types: List[str] | None = None,
        regions: List[str] | None = None,
    ) -> apps_api.AliasInfo:
        return apps_api.scale_app(
            self.client,
            app_name,
            keep_alive=keep_alive,
            max_multiplexing=max_multiplexing,
            max_concurrency=max_concurrency,
            min_concurrency=min_concurrency,
            concurrency_buffer=concurrency_buffer,
            concurrency_buffer_perc=concurrency_buffer_perc,
            request_timeout=request_timeout,
            startup_timeout=startup_timeout,
            machine_types=machine_types,
            regions=regions,
        )


class _RunnersNamespace:
    def __init__(self, client: SyncServerlessClient):
        self.client = client

    def list(self, *, since=None) -> List[RunnerInfo]:
        return runners_api.list_runners(self.client, since=since)


@dataclass
class SyncServerlessClient:
    host: Optional[str] = None
    api_key: Optional[str] = None
    profile: Optional[str] = None
    team: Optional[str] = None

    def __post_init__(self) -> None:
        self.apps = _AppsNamespace(self)
        self.runners = _RunnersNamespace(self)

    # Top-level verbs
    def deploy(self, *args, **kwargs):
        return deploy_api.deploy(self, *args, **kwargs)

    # Internal helpers
    @property
    def _grpc_host(self) -> str:
        return self.host or FAL_SERVERLESS_DEFAULT_URL

    @property
    def _credentials(self) -> Credentials:
        from fal.sdk import FalServerlessKeyCredentials, get_default_credentials

        if self.api_key:
            if self.team:
                raise ValueError(
                    "Using explicit team with key credentials is not allowed"
                )
            try:
                key_id, key_secret = self.api_key.split(":", 1)
            except ValueError:
                raise ValueError("api_key must be in 'KEY_ID:KEY_SECRET' format")
            return FalServerlessKeyCredentials(key_id, key_secret)

        if self.profile:
            prev = os.environ.get("FAL_PROFILE")
            os.environ["FAL_PROFILE"] = self.profile
            try:
                return get_default_credentials(team=self.team)
            finally:
                if prev is None:
                    os.environ.pop("FAL_PROFILE", None)
                else:
                    os.environ["FAL_PROFILE"] = prev

        return get_default_credentials(team=self.team)
