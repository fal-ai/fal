from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fal.sdk import ServerlessSecret

    from .client import SyncServerlessClient


def set_secret(client: SyncServerlessClient, name: str, value: str) -> None:
    from fal.sdk import FalServerlessClient

    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        conn.set_secret(name, value)


def list_secrets(client: SyncServerlessClient) -> List[ServerlessSecret]:
    from fal.sdk import FalServerlessClient

    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        return list(conn.list_secrets())


def unset_secret(client: SyncServerlessClient, name: str) -> None:
    from fal.sdk import FalServerlessClient

    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        conn.delete_secret(name)
