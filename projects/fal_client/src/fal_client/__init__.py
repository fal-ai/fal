from fal_client.client import (
    AsyncClient,
    AsyncRealtimeConnection,
    SyncClient,
    RealtimeConnection,
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
    "RealtimeConnection",
    "AsyncRealtimeConnection",
    "Status",
    "Queued",
    "InProgress",
    "Completed",
    "SyncRequestHandle",
    "AsyncRequestHandle",
    "run",
    "subscribe_async",
    "subscribe",
    "submit",
    "stream",
    "run_async",
    "submit_async",
    "stream_async",
    "realtime",
    "realtime_async",
    "cancel",
    "cancel_async",
    "status",
    "status_async",
    "result",
    "result_async",
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
cancel = sync_client.cancel
stream = sync_client.stream
realtime = sync_client.realtime
ws_connect = sync_client.ws_connect
upload = sync_client.upload
upload_file = sync_client.upload_file
upload_image = sync_client.upload_image

async_client = AsyncClient()
run_async = async_client.run
subscribe_async = async_client.subscribe
submit_async = async_client.submit
status_async = async_client.status
result_async = async_client.result
cancel_async = async_client.cancel
stream_async = async_client.stream
realtime_async = async_client.realtime
ws_connect_async = async_client.ws_connect
upload_async = async_client.upload
upload_file_async = async_client.upload_file
upload_image_async = async_client.upload_image
