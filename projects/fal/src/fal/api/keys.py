from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fal.sdk import KeyPreset, UserKeyInfo

    from .client import SyncServerlessClient


def create_key(
    client: SyncServerlessClient,
    *,
    preset: KeyPreset | None = None,
    permissions: List[str] | None = None,
    description: str | None = None,
) -> tuple[str, str]:
    from fal.sdk import FalServerlessClient

    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        return conn.create_user_key(preset, description, permissions)


def list_keys(client: SyncServerlessClient) -> List[UserKeyInfo]:
    from fal.sdk import FalServerlessClient

    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        return conn.list_user_keys()


def revoke_key(client: SyncServerlessClient, key_id: str) -> None:
    from fal.sdk import FalServerlessClient

    with FalServerlessClient(client._grpc_host, client._credentials).connect() as conn:
        conn.revoke_user_key(key_id)
