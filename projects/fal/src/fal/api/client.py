from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from fal.api import FAL_SERVERLESS_DEFAULT_URL, FalServerlessHost
from fal.sdk import (
    AliasInfo,
    Credentials,
    EnvironmentInfo,
    KeyScope,
    RunnerInfo,
    ServerlessSecret,
    UserKeyInfo,
)

if TYPE_CHECKING:
    from openapi_fal_rest.client import Client

from . import apps as apps_api
from . import deploy as deploy_api
from . import environments as environments_api
from . import keys as keys_api
from . import runners as runners_api
from . import secrets as secrets_api


class AppsNamespace:
    """Namespace for app management operations.

    Corresponds to `fal apps ...` CLI commands.

    Accessed via `client.apps`.

    Example:
        from fal.api import SyncServerlessClient

        client = SyncServerlessClient()
        apps = client.apps.list()
        client.apps.scale("my-app", max_concurrency=10)
    """

    def __init__(self, client: SyncServerlessClient):
        self.client = client

    def list(
        self, *, filter: str | None = None, environment_name: str | None = None
    ) -> List[AliasInfo]:
        """List all applications. Corresponds to `fal apps list`.

        Args:
            filter: Optional app name filter string.
            environment_name: Optional environment name.

        Example:
            apps = client.apps.list()
            filtered = client.apps.list(filter="stable")
        """
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
        """List runners for a specific app. Corresponds to `fal apps runners <app>`.

        Args:
            app_name: Name of the application.
            since: Only return runners started after this datetime.
            state: Filter by runner state (e.g., ["running"]).
            environment_name: Optional environment name.

        Example:
            from datetime import datetime, timedelta

            runners = client.apps.runners("my-app")
            recent = client.apps.runners(
                "my-app", since=datetime.now() - timedelta(hours=1)
            )
            running = client.apps.runners("my-app", state=["running"])
        """
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
        """Adjust scaling settings for an application. Corresponds to `fal apps scale`.

        Any omitted option keeps the current value.

        Args:
            app_name: Name of the application.
            keep_alive: Keep-alive time in seconds.
            max_multiplexing: Maximum request multiplexing.
            max_concurrency: Maximum concurrent runners.
            min_concurrency: Minimum concurrent runners.
            request_timeout: Request timeout in seconds.
            startup_timeout: Startup timeout in seconds.
            machine_types: List of allowed machine types (e.g., ["GPU-H100"]).
            regions: List of allowed regions (e.g., ["us-east-1"]).

        Example:
            client.apps.scale(
                "my-app",
                keep_alive=300,
                max_concurrency=10,
                min_concurrency=1,
                machine_types=["GPU-H100", "GPU-H200"],
            )
        """
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


class RunnersNamespace:
    """Namespace for runner management operations.

    Corresponds to `fal runners ...` CLI commands.

    Accessed via `client.runners`.

    Example:
        from fal.api import SyncServerlessClient

        client = SyncServerlessClient()
        runners = client.runners.list()
        client.runners.stop("runner-id")
    """

    def __init__(self, client: SyncServerlessClient):
        self.client = client

    def list(self, *, since=None) -> List[RunnerInfo]:
        """List all runners. Corresponds to `fal runners list`.

        Args:
            since: Only return runners started after this datetime.

        Example:
            from datetime import datetime, timedelta

            all_runners = client.runners.list()
            recent = client.runners.list(since=datetime.now() - timedelta(minutes=10))
        """
        return runners_api.list_runners(self.client, since=since)

    def stop(self, runner_id: str, replace_first: bool = False) -> None:
        """Gracefully stop a runner.

        Args:
            runner_id: The ID of the runner to stop.
            replace_first: Whether to replace the runner before stopping it.
        """
        return runners_api.stop_runner(
            self.client, runner_id, replace_first=replace_first
        )

    def kill(self, runner_id: str) -> None:
        """Forcefully kill a runner.

        Args:
            runner_id: The ID of the runner to kill.
        """
        return runners_api.kill_runner(self.client, runner_id)


class KeysNamespace:
    """Namespace for API key management. Corresponds to `fal keys ...` CLI commands.

    Accessed via `client.keys`.

    Example:
        from fal.api import SyncServerlessClient

        client = SyncServerlessClient()
        keys = client.keys.list()
        key_id, key_secret = client.keys.create(scope="admin")
    """

    def __init__(self, client: SyncServerlessClient):
        self.client = client

    def create(
        self, *, scope: KeyScope, description: str | None = None
    ) -> tuple[str, str]:
        """Create a new API key.

        Args:
            scope: Key scope (e.g., "admin").
            description: Optional description for the key.

        Returns:
            Tuple of (key_id, key_secret).
        """
        return keys_api.create_key(self.client, scope=scope, description=description)

    def list(self) -> List[UserKeyInfo]:
        """List all API keys."""
        return keys_api.list_keys(self.client)

    def revoke(self, key_id: str) -> None:
        """Revoke an API key.

        Args:
            key_id: The ID of the key to revoke.
        """
        return keys_api.revoke_key(self.client, key_id)


class SecretsNamespace:
    """Namespace for secrets management. Corresponds to `fal secrets ...` CLI commands.

    Accessed via `client.secrets`.

    Example:
        from fal.api import SyncServerlessClient

        client = SyncServerlessClient()
        client.secrets.set("API_KEY", "my-secret-value")
        secrets = client.secrets.list()
    """

    def __init__(self, client: SyncServerlessClient):
        self.client = client

    def set(self, name: str, value: str, environment_name: str | None = None) -> None:
        """Set a secret value.

        Args:
            name: Name of the secret.
            value: Value to store.
        """
        return secrets_api.set_secret(
            self.client, name, value, environment_name=environment_name
        )

    def list(self, environment_name: str | None = None) -> List[ServerlessSecret]:
        """List all secrets (names only, not values)."""
        return secrets_api.list_secrets(self.client, environment_name=environment_name)

    def unset(self, name: str, environment_name: str | None = None) -> None:
        """Delete a secret.

        Args:
            name: Name of the secret to delete.
            environment_name: Optional environment name.
        """
        return secrets_api.unset_secret(
            self.client, name, environment_name=environment_name
        )


class _EnvironmentsNamespace:
    def __init__(self, client: SyncServerlessClient):
        self.client = client

    def create(self, name: str, description: str | None = None) -> EnvironmentInfo:
        return environments_api.create_environment(
            self.client, name, description=description
        )

    def list(self) -> List[EnvironmentInfo]:
        return environments_api.list_environments(self.client)

    def delete(self, name: str) -> None:
        return environments_api.delete_environment(self.client, name)


@dataclass
class SyncServerlessClient:
    """Synchronous Python client for fal Serverless.

    Manage apps, runners, and deployments programmatically. The namespaces
    and methods mirror the CLI so you can automate the same workflows from Python.

    Args:
        host: Optional. Override API host.
        api_key: Optional. If omitted, read from env/profile.
        profile: Optional. Named profile to use.
        team: Optional. Team context for runner operations.

    Example:
        from fal.api import SyncServerlessClient

        client = SyncServerlessClient()

        # List apps
        apps = client.apps.list()

        # Scale an app
        client.apps.scale("my-app", max_concurrency=10)

        # Deploy
        client.deploy("path/to/myfile.py::MyApp")

    Namespaces:
        - client.apps.*    - corresponds to `fal apps ...`
        - client.runners.* - corresponds to `fal runners ...`
        - client.keys.*    - corresponds to `fal keys ...`
        - client.secrets.* - corresponds to `fal secrets ...`
        - client.deploy()  - corresponds to `fal deploy ...`
    """

    host: Optional[str] = None
    api_key: Optional[str] = None
    profile: Optional[str] = None
    team: Optional[str] = None

    def __post_init__(self) -> None:
        self.apps = AppsNamespace(self)
        self.runners = RunnersNamespace(self)
        self.keys = KeysNamespace(self)
        self.secrets = SecretsNamespace(self)
        self.environments = _EnvironmentsNamespace(self)

    def deploy(self, *args, **kwargs):
        """Deploy an application. Corresponds to `fal deploy`.

        If app_ref is omitted, discovery behavior matches the CLI
        (e.g., uses pyproject.toml).

        Args:
            app_ref: Path to file, file::Class, or existing app name.
            app_name: Override the application name.
            auth: Authentication mode ("private" | "public").
            strategy: Deployment strategy ("recreate" | "rolling").
            reset_scale: If False, use previous scaling settings.

        Example:
            # Auto-discover from pyproject.toml
            client.deploy()

            # Deploy from a file path
            client.deploy("path/to/myfile.py")

            # Deploy a specific class
            client.deploy("path/to/myfile.py::MyApp")

            # With options
            client.deploy(
                app_ref="path/to/myfile.py::MyApp",
                app_name="myapp",
                auth="public",
                strategy="rolling",
            )
        """
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
