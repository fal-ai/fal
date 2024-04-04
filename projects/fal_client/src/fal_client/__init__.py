from fal_client.client import (
    AsyncClient,
    SyncClient,
    Status,
    Queued,
    InProgress,
    Completed,
    SyncRequestHandle,
    AsyncRequestHandle,
)

__all__ = [
    "SyncClient",
    "AsyncClient",
    "Status",
    "Queued",
    "InProgress",
    "Completed",
    "SyncRequestHandle",
    "AsyncRequestHandle",
    "run",
    "submit",
    "stream",
    "run_async",
    "submit_async",
    "stream_async",
]

sync_client = SyncClient()
run = sync_client.run
submit = sync_client.submit
stream = sync_client.stream

async_client = AsyncClient()
run_async = async_client.run
submit_async = async_client.submit
stream_async = async_client.stream
