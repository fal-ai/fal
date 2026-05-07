from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from fal.sdk import AliasInfo, FalServerlessClient, RunnerInfo

from ._metrics import _get_metrics, _normalize_gpu_counts

if TYPE_CHECKING:
    from .client import SyncServerlessClient


def list_apps(
    client: SyncServerlessClient,
    *,
    filter: Optional[str] = None,
    environment_name: Optional[str] = None,
) -> List[AliasInfo]:
    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        apps = conn.list_aliases(environment_name=environment_name)

    if filter:
        apps = [a for a in apps if filter in a.alias]
    return apps


def apps_runners(
    client: SyncServerlessClient,
    app_name: str,
    *,
    since: Optional[datetime] = None,
    state: Optional[list[str]] = None,
    environment_name: Optional[str] = None,
) -> List[RunnerInfo]:
    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        alias_runners = conn.list_alias_runners(
            alias=app_name, start_time=since, environment_name=environment_name
        )

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
    environment_name: str | None = None,
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
            environment_name=environment_name,
        )


def rollout_app(
    client: SyncServerlessClient,
    app_name: str,
    *,
    force: bool = False,
    environment_name: str | None = None,
) -> None:
    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        conn.rollout_application(
            application_name=app_name,
            force=force,
            environment_name=environment_name,
        )


def app_gpus(client: SyncServerlessClient, app_name: str) -> dict:
    """Get GPU usage for a single application.

    Accepts either a bare name ("flux") or a namespaced one ("fal-ai/flux").
    The metrics endpoint keys apps as "<namespace>/<name>", but the rest of
    `fal apps` accepts bare names, so we match by basename for parity.

    Returns {"gpus": {<type>: <count>, ...}, "total": <int>} where `total`
    equals the sum of values in `gpus`. GPU type names are normalized to
    short SKUs (e.g. "H100"); the metrics endpoint includes every machine
    type with non-GPU plans at count 0, which we filter out.
    """
    data = _get_metrics(client)
    apps = data.get("apps") or {}
    target = app_name.rsplit("/", 1)[-1]
    matches = [info for key, info in apps.items() if key.rsplit("/", 1)[-1] == target]
    if not matches:
        raise RuntimeError(f"Application {app_name!r} not found in metrics.")
    if len(matches) > 1:
        candidates = sorted(key for key in apps if key.rsplit("/", 1)[-1] == target)
        raise RuntimeError(
            f"Application name {app_name!r} is ambiguous; matches: "
            f"{', '.join(candidates)}"
        )
    gpu_by_type = _normalize_gpu_counts(matches[0].get("gpu_count_by_type"))
    return {"gpus": gpu_by_type, "total": sum(gpu_by_type.values())}
