from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from fal.sdk import FalServerlessClient, RunnerInfo

from ._metrics import _get_metrics, _normalize_gpu_counts

if TYPE_CHECKING:
    from .client import SyncServerlessClient


def list_runners(
    client: SyncServerlessClient, *, since: Optional[datetime] = None
) -> List[RunnerInfo]:
    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        return conn.list_runners(start_time=since)


def stop_runner(
    client: SyncServerlessClient, runner_id: str, replace_first: bool = False
) -> None:
    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        conn.stop_runner(runner_id, replace_first=replace_first)


def kill_runner(client: SyncServerlessClient, runner_id: str) -> None:
    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        conn.kill_runner(runner_id)


def runners_gpus(client: SyncServerlessClient) -> dict:
    """Get team-wide GPU usage summary.

    Returns {"gpus": {<type>: <count>, ...}, "total": <int>} where `total`
    equals the sum of values in `gpus`. GPU type names are normalized to
    short SKUs (e.g. "H100"); the metrics endpoint includes every machine
    type with non-GPU plans at count 0, which we filter out.
    """
    data = _get_metrics(client)
    summary = data.get("summary") or {}
    gpu_by_type = _normalize_gpu_counts(summary.get("total_gpu_by_type"))
    return {"gpus": gpu_by_type, "total": sum(gpu_by_type.values())}
