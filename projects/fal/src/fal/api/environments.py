from __future__ import annotations

from typing import TYPE_CHECKING, List

from .api import _handle_grpc_error

if TYPE_CHECKING:
    from fal.sdk import EnvironmentInfo

    from .client import SyncServerlessClient


@_handle_grpc_error()
def create_environment(
    client: SyncServerlessClient,
    name: str,
    description: str | None = None,
) -> EnvironmentInfo:
    from fal.sdk import FalServerlessClient

    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        return conn.create_environment(name, description=description)


@_handle_grpc_error()
def list_environments(client: SyncServerlessClient) -> List[EnvironmentInfo]:
    from fal.sdk import FalServerlessClient

    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        return list(conn.list_environments())


@_handle_grpc_error()
def delete_environment(client: SyncServerlessClient, name: str) -> None:
    from fal.sdk import FalServerlessClient

    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        conn.delete_environment(name)

