from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from fal.sdk import FalServerlessClient, RunnerInfo

if TYPE_CHECKING:
    from .client import SyncServerlessClient


def list_runners(
    client: SyncServerlessClient, *, since: Optional[datetime] = None
) -> List[RunnerInfo]:
    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        return conn.list_runners(start_time=since)


def stop_runner(client: SyncServerlessClient, runner_id: str) -> None:
    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        conn.stop_runner(runner_id)


def kill_runner(client: SyncServerlessClient, runner_id: str) -> None:
    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        conn.kill_runner(runner_id)
