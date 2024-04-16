from fal_client.client import (
    AsyncClient,
    SyncClient,
    Status,
    Queued,
    InProgress,
    Completed,
    SyncRequestHandle,
    AsyncRequestHandle,
    encode,
    encode_file,
    encode_image,
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
    "encode",
    "encode_file",
    "encode_image",
]

sync_client = SyncClient()
run = sync_client.run
submit = sync_client.submit
stream = sync_client.stream
upload = sync_client.upload
upload_file = sync_client.upload_file
upload_image = sync_client.upload_image

async_client = AsyncClient()
run_async = async_client.run
submit_async = async_client.submit
stream_async = async_client.stream
upload_async = async_client.upload
upload_file_async = async_client.upload_file
upload_image_async = async_client.upload_image
