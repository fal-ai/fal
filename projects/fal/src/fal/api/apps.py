from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from fal.sdk import AliasInfo, FalServerlessClient, RunnerInfo

if TYPE_CHECKING:
    from .client import SyncServerlessClient


def list_apps(
    client: SyncServerlessClient,
    *,
    filter: Optional[str] = None,
) -> List[AliasInfo]:
    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        apps = conn.list_aliases()

    if filter:
        apps = [a for a in apps if filter in a.alias]
    return apps


def apps_runners(
    client: SyncServerlessClient,
    app_name: str,
    *,
    since: Optional[datetime] = None,
    state: Optional[list[str]] = None,
) -> List[RunnerInfo]:
    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        alias_runners = conn.list_alias_runners(alias=app_name, start_time=since)

    if state and "all" not in set(state):
        states = set(state)
        alias_runners = [
            r
            for r in alias_runners
            if r.state.value.lower() in states
            or (
                "terminated" in states and r.state.value.lower() == "dead"
            )  # TODO for backwards compatibility. remove later
        ]
    return alias_runners


def scale_app(
    client: SyncServerlessClient,
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
    machine_types: list[str] | None = None,
    regions: list[str] | None = None,
) -> AliasInfo:
    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        return conn.update_application(
            application_name=app_name,
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
            valid_regions=regions,
        )


def rollout_app(
    client: SyncServerlessClient,
    app_name: str,
    *,
    force: bool = False,
) -> None:
    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        conn.rollout_application(
            application_name=app_name,
            force=force,
        )
