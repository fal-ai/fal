from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from fal.api import FAL_SERVERLESS_DEFAULT_URL, FalServerlessHost
from fal.sdk import (
    AliasInfo,
    Credentials,
    KeyScope,
    RunnerInfo,
    ServerlessSecret,
    UserKeyInfo,
)

if TYPE_CHECKING:
    from openapi_fal_rest.client import Client

from . import apps as apps_api
from . import deploy as deploy_api
from . import keys as keys_api
from . import runners as runners_api
from . import secrets as secrets_api


class _AppsNamespace:
    def __init__(self, client: SyncServerlessClient):
        self.client = client

    def list(
        self, *, filter: str | None = None, environment_name: str | None = None
    ) -> List[AliasInfo]:
        return apps_api.list_apps(
            self.client, filter=filter, environment_name=environment_name
        )

    def runners(
        self,
        app_name: str,
        *,
        since=None,
        state: List[str] | None = None,
        environment_name: str | None = None,
    ) -> List[RunnerInfo]:
        return apps_api.apps_runners(
            self.client,
            app_name,
            since=since,
            state=state,
            environment_name=environment_name,
        )

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
        scaling_delay: int | None = None,
        request_timeout: int | None = None,
        startup_timeout: int | None = None,
        machine_types: List[str] | None = None,
        regions: List[str] | None = None,
        environment_name: str | None = None,
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
            scaling_delay=scaling_delay,
            request_timeout=request_timeout,
            startup_timeout=startup_timeout,
            machine_types=machine_types,
            regions=regions,
            environment_name=environment_name,
        )

    def rollout(
        self, app_name: str, *, force: bool = False, environment_name: str | None = None
    ) -> None:
        return apps_api.rollout_app(
            self.client, app_name, force=force, environment_name=environment_name
        )


class _RunnersNamespace:
    def __init__(self, client: SyncServerlessClient):
        self.client = client

    def list(self, *, since=None) -> List[RunnerInfo]:
        return runners_api.list_runners(self.client, since=since)

    def stop(self, runner_id: str) -> None:
        return runners_api.stop_runner(self.client, runner_id)

    def kill(self, runner_id: str) -> None:
        return runners_api.kill_runner(self.client, runner_id)


class _KeysNamespace:
    def __init__(self, client: SyncServerlessClient):
        self.client = client

    def create(
        self, *, scope: KeyScope, description: str | None = None
    ) -> tuple[str, str]:
        return keys_api.create_key(self.client, scope=scope, description=description)

    def list(self) -> List[UserKeyInfo]:
        return keys_api.list_keys(self.client)

    def revoke(self, key_id: str) -> None:
        return keys_api.revoke_key(self.client, key_id)


class _SecretsNamespace:
    def __init__(self, client: SyncServerlessClient):
        self.client = client

    def set(self, name: str, value: str) -> None:
        return secrets_api.set_secret(self.client, name, value)

    def list(self) -> List[ServerlessSecret]:
        return secrets_api.list_secrets(self.client)

    def unset(self, name: str) -> None:
        return secrets_api.unset_secret(self.client, name)


@dataclass
class SyncServerlessClient:
    host: Optional[str] = None
    api_key: Optional[str] = None
    profile: Optional[str] = None
    team: Optional[str] = None

    def __post_init__(self) -> None:
        self.apps = _AppsNamespace(self)
        self.runners = _RunnersNamespace(self)
        self.keys = _KeysNamespace(self)
        self.secrets = _SecretsNamespace(self)

    # Top-level verbs
    def deploy(self, *args, **kwargs):
        return deploy_api.deploy(self, *args, **kwargs)

    # Internal helpers
    @property
    def _grpc_host(self) -> str:
        return self.host or FAL_SERVERLESS_DEFAULT_URL

    @property
    def _rest_url(self) -> str:
        from fal.flags import REST_SCHEME

        return f"{REST_SCHEME}://{self._grpc_host.replace('api', 'rest', 1)}"

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

    def _create_host(
        self, *, local_file_path: str = "", environment_name: str | None = None
    ) -> FalServerlessHost:
        return FalServerlessHost(
            self._grpc_host,
            local_file_path=local_file_path,
            credentials=self._credentials,
            environment_name=environment_name,
        )

    def _create_rest_client(self) -> Client:
        from openapi_fal_rest.client import Client

        import fal.flags as flags

        return Client(
            self._rest_url,
            headers=self._credentials.to_headers(),
            timeout=30,
            verify_ssl=not flags.TEST_MODE,
            raise_on_unexpected_status=False,
            follow_redirects=True,
        )
