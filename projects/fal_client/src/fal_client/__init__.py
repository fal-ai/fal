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
subscribe = sync_client.subscribe
submit = sync_client.submit
status = sync_client.status
result = sync_client.result
stream = sync_client.stream
upload = sync_client.upload
upload_file = sync_client.upload_file
upload_image = sync_client.upload_image

async_client = AsyncClient()
run_async = async_client.run
subscribe_async = async_client.subscribe
submit_async = async_client.submit
status_async = async_client.status
result_async = async_client.result
stream_async = async_client.stream
upload_async = async_client.upload
upload_file_async = async_client.upload_file
upload_image_async = async_client.upload_image
