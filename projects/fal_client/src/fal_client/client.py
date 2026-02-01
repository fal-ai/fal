from __future__ import annotations

import io
import json
import math
import os
import mimetypes
import asyncio
from pathlib import Path
import random
import time
import base64
import threading
import logging
import concurrent.futures
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from dataclasses import dataclass, field
from functools import cached_property
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    TYPE_CHECKING,
    Optional,
    Literal,
    Callable,
    Union,
)
from urllib.parse import urlencode
import warnings

import httpx
from httpx_sse import aconnect_sse, connect_sse

from fal_client.auth import (
    AuthCredentials,
    FAL_QUEUE_RUN_HOST,
    FAL_RUN_HOST,
    MissingCredentialsError,
    fetch_auth_credentials,
)
from fal_client._headers import (
    Priority,
    add_priority_header,
    add_timeout_header,
    add_hint_header,
    REQUEST_TIMEOUT_TYPE_HEADER,
    REQUEST_TIMEOUT_HEADER,
)

if TYPE_CHECKING:
    from websockets.client import WebSocketClientProtocol
    from websockets.sync.connection import Connection

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from PIL import Image

AnyJSON = Dict[str, Any]

RUN_URL_FORMAT = f"https://{FAL_RUN_HOST}/"
QUEUE_URL_FORMAT = f"https://{FAL_QUEUE_RUN_HOST}/"
REALTIME_URL_FORMAT = f"wss://{FAL_RUN_HOST}/"
REST_URL = "https://rest.alpha.fal.ai"
CDN_URL = "https://v3.fal.media"
USER_AGENT = "fal-client/0.2.2 (python)"

MIN_REQUEST_TIMEOUT_SECONDS = 1

# Global executor for sync client timeout operations
EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    thread_name_prefix="FAL_CLIENT_EXECUTOR"
)


@dataclass
class CDNToken:
    token: str
    token_type: str
    base_upload_url: str
    expires_at: datetime

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at


class CDNTokenManager:
    def __init__(self, auth: AuthCredentials) -> None:
        self._auth = auth
        self._token: CDNToken = CDNToken(
            token="",
            token_type="",
            base_upload_url="",
            expires_at=datetime.min.replace(tzinfo=timezone.utc),
        )
        self._lock: threading.Lock = threading.Lock()
        self._url = f"{REST_URL}/storage/auth/token?storage_type=fal-cdn-v3"
        self._headers = {
            "Authorization": self._auth.header_value,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _refresh_token(self) -> CDNToken:
        with httpx.Client() as client:
            response = client.post(self._url, headers=self._headers, json={})
            response.raise_for_status()
            data = response.json()

        return CDNToken(
            token=data["token"],
            token_type=data["token_type"],
            base_upload_url=data["base_url"],
            expires_at=datetime.fromisoformat(data["expires_at"]),
        )

    def get_token(self) -> CDNToken:
        with self._lock:
            if self._token.is_expired():
                self._token = self._refresh_token()
            return self._token


class AsyncCDNTokenManager:
    def __init__(self, auth: AuthCredentials) -> None:
        self._auth = auth
        self._token: CDNToken = CDNToken(
            token="",
            token_type="",
            base_upload_url="",
            expires_at=datetime.min.replace(tzinfo=timezone.utc),
        )
        self._lock: asyncio.Lock = asyncio.Lock()
        self._url = f"{REST_URL}/storage/auth/token?storage_type=fal-cdn-v3"
        self._headers = {
            "Authorization": self._auth.header_value,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def _refresh_token(self) -> CDNToken:
        async with httpx.AsyncClient() as client:
            response = await client.post(self._url, headers=self._headers, json={})
            response.raise_for_status()
            data = response.json()

        return CDNToken(
            token=data["token"],
            token_type=data["token_type"],
            base_upload_url=data["base_url"],
            expires_at=datetime.fromisoformat(data["expires_at"]),
        )

    async def get_token(self) -> CDNToken:
        async with self._lock:
            if self._token.is_expired():
                self._token = await self._refresh_token()
            return self._token


MULTIPART_THRESHOLD = 100 * 1024 * 1024
MULTIPART_CHUNK_SIZE = 10 * 1024 * 1024
MULTIPART_MAX_CONCURRENCY = 10


class MultipartUpload:
    def __init__(
        self,
        *,
        file_name: str,
        client: httpx.Client,
        token_manager: CDNTokenManager,
        chunk_size: int | None = None,
        content_type: str | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self.file_name = file_name
        self._client = client
        self._token_manager = token_manager
        self.chunk_size = chunk_size or MULTIPART_CHUNK_SIZE
        self.content_type = content_type or "application/octet-stream"
        self.max_concurrency = max_concurrency or MULTIPART_MAX_CONCURRENCY
        self._access_url: str | None = None
        self._upload_id: str | None = None
        self._parts: list[dict] = []

    @property
    def access_url(self) -> str:
        if not self._access_url:
            raise ValueError("Upload not initiated")
        return self._access_url

    @property
    def upload_id(self) -> str:
        if not self._upload_id:
            raise ValueError("Upload not initiated")
        return self._upload_id

    @property
    def auth_headers(self) -> dict[str, str]:
        token = self._token_manager.get_token()
        return {
            "Authorization": f"{token.token_type} {token.token}",
            "User-Agent": "fal/0.1.0",
        }

    def create(self):
        token = self._token_manager.get_token()
        url = f"{token.base_upload_url}/files/upload/multipart"
        response = _maybe_retry_request(
            self._client,
            "POST",
            url,
            headers={
                **self.auth_headers,
                "Accept": "application/json",
                "Content-Type": self.content_type,
                "X-Fal-File-Name": self.file_name,
            },
        )
        result = response.json()
        self._access_url = result["access_url"]
        self._upload_id = result["uploadId"]

    def upload_part(self, part_number: int, data: bytes) -> None:
        url = f"{self.access_url}/multipart/{self.upload_id}/{part_number}"

        response = _request(
            self._client,
            "PUT",
            url,
            headers={
                **self.auth_headers,
                "Content-Type": self.content_type,
                "Accept-Encoding": "identity",  # Keep this to ensure we get ETag headers
            },
            content=data,
            timeout=None,
        )

        etag = response.headers["etag"]
        self._parts.append(
            {
                "partNumber": part_number,
                "etag": etag,
            }
        )

    def complete(self) -> str:
        url = f"{self.access_url}/multipart/{self.upload_id}/complete"
        _maybe_retry_request(
            self._client,
            "POST",
            url,
            headers=self.auth_headers,
            json={"parts": self._parts},
        )
        return self.access_url

    @classmethod
    def save(
        cls,
        *,
        client: httpx.Client,
        token_manager: CDNTokenManager,
        file_name: str,
        data: bytes,
        content_type: str | None = None,
        chunk_size: int | None = None,
        max_concurrency: int | None = None,
    ):
        import concurrent.futures

        multipart = cls(
            file_name=file_name,
            client=client,
            token_manager=token_manager,
            chunk_size=chunk_size,
            content_type=content_type,
            max_concurrency=max_concurrency,
        )
        multipart.create()
        parts = math.ceil(len(data) / multipart.chunk_size)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=multipart.max_concurrency
        ) as executor:
            futures = []
            for part_number in range(1, parts + 1):
                start = (part_number - 1) * multipart.chunk_size
                data = data[start : start + multipart.chunk_size]
                futures.append(
                    executor.submit(multipart.upload_part, part_number, data)
                )
            for future in concurrent.futures.as_completed(futures):
                future.result()
        return multipart.complete()

    @classmethod
    def save_file(
        cls,
        *,
        client: httpx.Client,
        token_manager: CDNTokenManager,
        file_path: str | Path,
        chunk_size: int | None = None,
        content_type: str | None = None,
        max_concurrency: int | None = None,
    ) -> str:
        import concurrent.futures

        file_name = os.path.basename(file_path)
        size = os.path.getsize(file_path)
        multipart = cls(
            file_name=file_name,
            client=client,
            token_manager=token_manager,
            chunk_size=chunk_size,
            content_type=content_type,
            max_concurrency=max_concurrency,
        )
        multipart.create()
        parts = math.ceil(size / multipart.chunk_size)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=multipart.max_concurrency
        ) as executor:
            futures = []
            for part_number in range(1, parts + 1):

                def _upload_part(pn: int) -> None:
                    with open(file_path, "rb") as f:
                        start = (pn - 1) * multipart.chunk_size
                        f.seek(start)
                        data = f.read(multipart.chunk_size)
                        multipart.upload_part(pn, data)

                futures.append(executor.submit(_upload_part, part_number))
            for future in concurrent.futures.as_completed(futures):
                future.result()
        return multipart.complete()


class AsyncMultipartUpload:
    def __init__(
        self,
        *,
        file_name: str,
        client: httpx.AsyncClient,
        token_manager: AsyncCDNTokenManager,
        chunk_size: int | None = None,
        content_type: str | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self.file_name = file_name
        self._client = client
        self._token_manager = token_manager
        self.chunk_size = chunk_size or MULTIPART_CHUNK_SIZE
        self.content_type = content_type or "application/octet-stream"
        self.max_concurrency = max_concurrency or MULTIPART_MAX_CONCURRENCY
        self._access_url: str | None = None
        self._upload_id: str | None = None
        self._parts: list[dict] = []

    @property
    def access_url(self) -> str:
        if not self._access_url:
            raise ValueError("Upload not initiated")
        return self._access_url

    @property
    def upload_id(self) -> str:
        if not self._upload_id:
            raise ValueError("Upload not initiated")
        return self._upload_id

    @property
    async def auth_headers(self) -> dict[str, str]:
        token = await self._token_manager.get_token()
        return {
            "Authorization": f"{token.token_type} {token.token}",
            "User-Agent": "fal/0.1.0",
        }

    async def create(self):
        token = await self._token_manager.get_token()
        url = f"{token.base_upload_url}/files/upload/multipart"
        headers = await self.auth_headers
        response = await _async_maybe_retry_request(
            self._client,
            "POST",
            url,
            headers={
                **headers,
                "Accept": "application/json",
                "Content-Type": self.content_type,
                "X-Fal-File-Name": self.file_name,
            },
        )
        result = response.json()
        self._access_url = result["access_url"]
        self._upload_id = result["uploadId"]

    async def upload_part(self, part_number: int, data: bytes) -> None:
        url = f"{self.access_url}/multipart/{self.upload_id}/{part_number}"
        headers = await self.auth_headers

        response = await _async_request(
            self._client,
            "PUT",
            url,
            headers={
                **headers,
                "Content-Type": self.content_type,
                "Accept-Encoding": "identity",  # Keep this to ensure we get ETag headers
            },
            content=data,
            timeout=None,
        )

        etag = response.headers["etag"]
        self._parts.append(
            {
                "partNumber": part_number,
                "etag": etag,
            }
        )

    async def complete(self) -> str:
        url = f"{self.access_url}/multipart/{self.upload_id}/complete"
        headers = await self.auth_headers
        await _async_maybe_retry_request(
            self._client,
            "POST",
            url,
            headers=headers,
            json={"parts": self._parts},
        )
        return self.access_url

    @classmethod
    async def save(
        cls,
        *,
        client: httpx.AsyncClient,
        token_manager: AsyncCDNTokenManager,
        file_name: str,
        data: bytes,
        content_type: str | None = None,
        chunk_size: int | None = None,
        max_concurrency: int | None = None,
    ) -> str:
        multipart = cls(
            file_name=file_name,
            client=client,
            token_manager=token_manager,
            chunk_size=chunk_size,
            content_type=content_type,
            max_concurrency=max_concurrency,
        )
        await multipart.create()
        parts = math.ceil(len(data) / multipart.chunk_size)

        async def upload_part(part_number: int) -> None:
            start = (part_number - 1) * multipart.chunk_size
            chunk = data[start : start + multipart.chunk_size]
            await multipart.upload_part(part_number, chunk)

        tasks = [
            asyncio.create_task(upload_part(part_number))
            for part_number in range(1, parts + 1)
        ]

        # Process concurrent uploads with semaphore to limit concurrency
        sem = asyncio.Semaphore(multipart.max_concurrency)

        async def bounded_upload(task):
            async with sem:
                await task

        await asyncio.gather(*[bounded_upload(task) for task in tasks])
        return await multipart.complete()

    @classmethod
    async def save_file(
        cls,
        *,
        client: httpx.AsyncClient,
        token_manager: AsyncCDNTokenManager,
        file_path: str | Path,
        chunk_size: int | None = None,
        content_type: str | None = None,
        max_concurrency: int | None = None,
    ) -> str:
        file_name = os.path.basename(file_path)
        size = os.path.getsize(file_path)
        multipart = cls(
            file_name=file_name,
            client=client,
            token_manager=token_manager,
            chunk_size=chunk_size,
            content_type=content_type,
            max_concurrency=max_concurrency,
        )
        await multipart.create()
        parts = math.ceil(size / multipart.chunk_size)

        async def upload_part(part_number: int) -> None:
            with open(file_path, "rb") as f:
                start = (part_number - 1) * multipart.chunk_size
                f.seek(start)
                data = f.read(multipart.chunk_size)
                await multipart.upload_part(part_number, data)

        tasks = [
            asyncio.create_task(upload_part(part_number))
            for part_number in range(1, parts + 1)
        ]

        # Process concurrent uploads with semaphore to limit concurrency
        sem = asyncio.Semaphore(multipart.max_concurrency)

        async def bounded_upload(task):
            async with sem:
                await task

        await asyncio.gather(*[bounded_upload(task) for task in tasks])
        return await multipart.complete()


class FalClientError(Exception):
    pass


@dataclass
class FalClientHTTPError(FalClientError):
    message: str
    status_code: int
    response_headers: dict[str, str]
    response: httpx.Response

    def __str__(self) -> str:
        return f"{self.message}"


@dataclass
class FalClientTimeoutError(FalClientError, TimeoutError):
    timeout: float
    request_id: Optional[str] = None

    def __str__(self) -> str:
        if self.request_id is None:
            return f"Request timed out after {self.timeout} seconds"
        else:
            return f"Request {self.request_id} timed out after {self.timeout} seconds"


def _raise_for_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            msg = response.json()["detail"]
        except (ValueError, KeyError):
            msg = response.text

        raise FalClientHTTPError(
            msg,
            response.status_code,
            # converting to dict to avoid httpx.Headers,
            # which means we don't support multiple values per header
            dict(response.headers),
            response=response,
        ) from exc


@dataclass
class Status: ...


@dataclass
class Queued(Status):
    """Indicates the request is enqueued and waiting to be processed. The position
    field indicates the relative position in the queue (0-indexed)."""

    position: int


@dataclass
class InProgress(Status):
    """Indicates the request is currently being processed. If the status operation called
    with the `with_logs` parameter set to True, the logs field will be a list of
    log objects."""

    # TODO: Type the log object structure so we can offer editor completion
    logs: list[dict[str, Any]] | None = field()


@dataclass
class Completed(Status):
    """Indicates the request has been completed and the result can be gathered. The logs field will
    contain the logs if the status operation was called with the `with_logs` parameter set to True. Metrics
    might contain the inference time, and other internal metadata (number of tokens
    processed, etc.)."""

    logs: list[dict[str, Any]] | None = field()
    metrics: dict[str, Any] = field()


@dataclass(frozen=True)
class _BaseRequestHandle:
    request_id: str
    response_url: str = field(repr=False)
    status_url: str = field(repr=False)
    cancel_url: str = field(repr=False)

    def _parse_status(self, data: AnyJSON) -> Status:
        if data["status"] == "IN_QUEUE":
            return Queued(position=data["queue_position"])
        elif data["status"] == "IN_PROGRESS":
            return InProgress(logs=data["logs"])
        elif data["status"] == "COMPLETED":
            # NOTE: legacy apps might not return metrics
            metrics = data.get("metrics", {})
            return Completed(logs=data["logs"], metrics=metrics)
        else:
            raise ValueError(f"Unknown status: {data['status']}")


APP_NAMESPACES = ["workflows", "comfy"]


def _ensure_app_id_format(id: str) -> str:
    import re

    parts = id.split("/")
    if len(parts) > 1:
        return id

    match = re.match(r"^([0-9]+)-([a-zA-Z0-9-]+)$", id)
    if match:
        app_owner, app_id = match.groups()
        return f"{app_owner}/{app_id}"

    raise ValueError(f"Invalid app id: {id}. Must be in the format <appOwner>/<appId>")


@dataclass(frozen=True)
class AppId:
    owner: str
    alias: str
    path: Optional[str]
    namespace: Optional[str]

    @classmethod
    def from_endpoint_id(cls, endpoint_id: str) -> AppId:
        normalized_id = _ensure_app_id_format(endpoint_id)
        parts = normalized_id.split("/")

        if parts[0] in APP_NAMESPACES:
            return cls(
                owner=parts[1],
                alias=parts[2],
                path="/".join(parts[3:]) or None,
                namespace=parts[0],
            )

        return cls(
            owner=parts[0],
            alias=parts[1],
            path="/".join(parts[2:]) or None,
            namespace=None,
        )


REALTIME_TOKEN_EXPIRATION_SECONDS = 120
REALTIME_OPEN_TIMEOUT = 90.0
REALTIME_MAX_BUFFERING = (1, 60)


def _format_app_path(app_id: AppId) -> str:
    prefix = f"{app_id.namespace}/" if app_id.namespace else ""
    suffix = f"/{app_id.path}" if app_id.path else ""
    return f"{prefix}{app_id.owner}/{app_id.alias}{suffix}"


def _serialize_max_buffering(value: int | None) -> str | None:
    if value is None:
        return None

    min_value, max_value = REALTIME_MAX_BUFFERING
    if not (min_value <= value <= max_value):
        raise ValueError(
            f"max_buffering must be between {min_value} and {max_value} (inclusive)"
        )
    return str(value)


def _build_runner_ws_url(
    application: str,
    token: str | None,
    *,
    path: str = "",
    max_buffering: int | None = None,
) -> str:
    app_id = AppId.from_endpoint_id(application)
    app_path = _format_app_path(app_id)
    url = f"{REALTIME_URL_FORMAT}{app_path}"
    if path:
        url += "/" + path.lstrip("/")
    query: dict[str, str] = {}
    if token:
        query["fal_jwt_token"] = token
    serialized_buffering = _serialize_max_buffering(max_buffering)
    if serialized_buffering is not None:
        query["max_buffering"] = serialized_buffering
    if query:
        return f"{url}?{urlencode(query)}"
    return url


def _build_realtime_url(
    application: str,
    token: str | None,
    max_buffering: int | None,
    *,
    path: str = "realtime",
) -> str:
    return _build_runner_ws_url(
        application,
        token,
        path=path,
        max_buffering=max_buffering,
    )


def _parse_token_response(data: Any) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("token"), str):
            return data["token"]
        if isinstance(data.get("detail"), str):
            return data["detail"]
    raise RuntimeError("Unexpected realtime token response format")


class RealtimeError(RuntimeError):
    """Raised when the realtime endpoint sends an error payload."""

    def __init__(
        self,
        error: str,
        reason: str | None = None,
        payload: Optional[dict[str, Any]] = None,
    ):
        self.error = error
        self.reason = reason or ""
        self.payload = payload or {}
        message = error if not self.reason else f"{error}: {self.reason}"
        super().__init__(message)


def msgpack_decode_message(message: bytes) -> Any:
    import msgpack

    return msgpack.unpackb(message, raw=False)


def msgpack_encode_message(message: Any) -> bytes:
    import msgpack

    return msgpack.packb(message, use_bin_type=True)


def _decode_realtime_message(
    message: Any, decode_message: Callable[[bytes], Any] | None
) -> dict[str, Any] | None:
    if isinstance(message, memoryview):
        message = message.tobytes()

    if isinstance(message, (bytes, bytearray)):
        decode = decode_message or msgpack_decode_message
        return decode(message)

    if isinstance(message, str):
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return {"type": "text", "payload": message}

        msg_type = payload.get("type")
        if msg_type == "x-fal-error":
            raise RealtimeError(
                payload.get("error", "UNKNOWN_ERROR"),
                payload.get("reason"),
                payload,
            )
        if msg_type == "x-fal-message":
            # meta message, skip it silently
            return None
        return payload

    return {"payload": message}


def _encode_realtime_message(
    message: dict[str, Any],
    encode_message: Callable[[Any], bytes] | None,
) -> bytes:
    encode = encode_message or msgpack_encode_message
    return encode(message)


@dataclass
class RealtimeConnection:
    """Synchronous realtime connection wrapper."""

    _ws: "Connection"
    _encode_message: Callable[[Any], bytes] | None = None
    _decode_message: Callable[[bytes], Any] | None = None

    def send(self, arguments: dict[str, Any]) -> None:
        payload = _encode_realtime_message(arguments, self._encode_message)
        self._ws.send(payload)

    def recv(self) -> dict[str, Any]:
        while True:
            response = self._ws.recv()
            decoded = _decode_realtime_message(response, self._decode_message)
            if decoded is None:
                continue
            return decoded

    def close(self) -> None:
        close = getattr(self._ws, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> RealtimeConnection:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass
class AsyncRealtimeConnection:
    """Asynchronous realtime connection wrapper."""

    _ws: "WebSocketClientProtocol"
    _encode_message: Callable[[Any], bytes] | None = None
    _decode_message: Callable[[bytes], Any] | None = None

    async def send(self, arguments: dict[str, Any]) -> None:
        payload = _encode_realtime_message(arguments, self._encode_message)
        await self._ws.send(payload)

    async def recv(self) -> dict[str, Any]:
        while True:
            response = await self._ws.recv()
            decoded = _decode_realtime_message(response, self._decode_message)
            if decoded is None:
                continue
            return decoded

    async def close(self) -> None:
        close = getattr(self._ws, "close", None)
        if callable(close):
            await close()

    async def __aenter__(self) -> AsyncRealtimeConnection:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


@contextmanager
def _connect_sync_ws(
    url: str, headers: dict[str, str] | None = None
) -> Iterator["Connection"]:
    from websockets.sync import client

    with client.connect(
        url,
        additional_headers=headers,
        open_timeout=REALTIME_OPEN_TIMEOUT,
        max_size=None,
    ) as ws:
        yield ws


@asynccontextmanager
async def _connect_async_ws(
    url: str, headers: dict[str, str] | None = None
) -> AsyncIterator["WebSocketClientProtocol"]:
    import websockets

    async with websockets.connect(
        url,
        extra_headers=headers,
        open_timeout=REALTIME_OPEN_TIMEOUT,
        max_size=None,
    ) as ws:
        yield ws


def _request(
    client: httpx.Client, method: str, url: str, **kwargs: Any
) -> httpx.Response:
    response = client.request(method, url, **kwargs)
    _raise_for_status(response)
    return response


async def _async_request(
    client: httpx.AsyncClient, method: str, url: str, **kwargs: Any
) -> httpx.Response:
    response = await client.request(method, url, **kwargs)
    _raise_for_status(response)
    return response


MAX_ATTEMPTS = 10
BASE_DELAY = 0.1
MAX_DELAY = 30
RETRY_CODES = [408, 409, 429]
INGRESS_ERROR_CODES = [502, 503, 504]


def _is_ingress_error(response: httpx.Response) -> bool:
    """Tell apart ingress errors from client errors."""

    if response.status_code not in INGRESS_ERROR_CODES:
        return False

    if "x-fal-request-id" in response.headers:
        # this is clearly returned from our server
        return False

    # heuristic to detect an ingress error
    if "nginx" in response.text:
        return True

    return False


def _should_retry_response(
    response: httpx.Response,
    extra_retry_codes: list[int] = [],
) -> bool:
    # Honor user defined timeouts, do not retry
    if response.status_code == 504 and response.headers.get(
        REQUEST_TIMEOUT_TYPE_HEADER
    ):
        return False

    if _is_ingress_error(response):
        return True

    if response.status_code in RETRY_CODES or response.status_code in extra_retry_codes:
        return True

    return False


def _should_retry(exc: Exception, extra_retry_codes: list[int] = []) -> bool:
    # Check TimeoutException FIRST, before TransportError
    if isinstance(exc, httpx.TimeoutException):
        try:
            request = exc.request
            # User set a timeout - honor it, don't retry
            if isinstance(request, httpx.Request):
                if REQUEST_TIMEOUT_HEADER in request.headers:
                    return False
        except RuntimeError:
            pass  # .request property not set
        return True  # No user timeout specified, safe to retry

    if isinstance(exc, httpx.TransportError):
        return True

    if isinstance(exc, (httpx.HTTPStatusError, FalClientHTTPError)):
        return _should_retry_response(exc.response, extra_retry_codes)

    return False


def _get_retry_delay(
    num_retry: int,
    base_delay: float,
    max_delay: float,
    backoff_type: Literal["exponential", "fixed"] = "exponential",
    jitter: bool = False,
) -> float:
    if backoff_type == "exponential":
        delay = min(base_delay * (2 ** (num_retry - 1)), max_delay)
    else:
        delay = min(base_delay, max_delay)

    if jitter:
        delay *= random.uniform(0.5, 1.5)

    return min(delay, max_delay)


def _maybe_retry_request(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    extra_retry_codes: list[int] = [],
    **kwargs: Any,
) -> httpx.Response:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _request(client, method, url, **kwargs)
        except (httpx.HTTPError, FalClientHTTPError) as exc:
            if _should_retry(exc, extra_retry_codes) and attempt < MAX_ATTEMPTS:
                delay = _get_retry_delay(
                    attempt, BASE_DELAY, MAX_DELAY, "exponential", True
                )
                logger.debug(
                    f"Retrying request to {url} due to {exc} ({MAX_ATTEMPTS - attempt} attempts left)"
                )
                time.sleep(delay)
                continue
            raise
    # Should be unreachable, added for type checkers
    raise RuntimeError("Failed to perform request")


async def _async_maybe_retry_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    extra_retry_codes: list[int] = [],
    **kwargs: Any,
) -> httpx.Response:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return await _async_request(client, method, url, **kwargs)
        except (httpx.HTTPError, FalClientHTTPError) as exc:
            if _should_retry(exc, extra_retry_codes) and attempt < MAX_ATTEMPTS:
                delay = _get_retry_delay(attempt, 0.1, 10, "exponential", True)
                logger.debug(
                    f"Retrying request to {url} due to {exc} ({MAX_ATTEMPTS - attempt} attempts left)"
                )
                await asyncio.sleep(delay)
                continue
            raise
    # Should be unreachable, added for type checkers
    raise RuntimeError("Failed to perform request")


def _maybe_cancel_request(handle: SyncRequestHandle) -> None:
    try:
        handle.cancel()
    except Exception:
        pass


async def _async_maybe_cancel_request(handle: AsyncRequestHandle) -> None:
    try:
        await handle.cancel()
    except Exception:
        pass


@dataclass(frozen=True)
class SyncRequestHandle(_BaseRequestHandle):
    client: httpx.Client = field(repr=False)

    @classmethod
    def from_request_id(
        cls,
        client: httpx.Client,
        application: str,
        request_id: str,
    ) -> SyncRequestHandle:
        app_id = AppId.from_endpoint_id(application)
        prefix = f"{app_id.namespace}/" if app_id.namespace else ""
        base_url = f"{QUEUE_URL_FORMAT}{prefix}{app_id.owner}/{app_id.alias}/requests/{request_id}"
        return cls(
            request_id=request_id,
            response_url=base_url,
            status_url=base_url + "/status",
            cancel_url=base_url + "/cancel",
            client=client,
        )

    def status(self, *, with_logs: bool = False) -> Status:
        """Returns the status of the request (which can be one of the following:
        Queued, InProgress, Completed). If `with_logs` is True, logs will be included
        for InProgress and Completed statuses."""

        response = _maybe_retry_request(
            self.client,
            "GET",
            self.status_url,
            params={
                "logs": with_logs,
            },
        )
        _raise_for_status(response)

        return self._parse_status(response.json())

    def iter_events(
        self, *, with_logs: bool = False, interval: float = 0.1
    ) -> Iterator[Status]:
        """Continuously poll for the status of the request and yield it at each interval till
        the request is completed. If `with_logs` is True, logs will be included in the response.
        """

        while True:
            status = self.status(with_logs=with_logs)
            yield status
            if isinstance(status, Completed):
                break

            time.sleep(interval)

    def get(self) -> AnyJSON:
        """Wait till the request is completed and return the result of the inference call."""
        for _ in self.iter_events(with_logs=False):
            continue

        response = _maybe_retry_request(self.client, "GET", self.response_url)
        _raise_for_status(response)
        return response.json()

    def cancel(self) -> None:
        """Cancel the request."""
        response = _maybe_retry_request(
            self.client,
            "PUT",
            self.cancel_url,
        )
        _raise_for_status(response)


@dataclass(frozen=True)
class AsyncRequestHandle(_BaseRequestHandle):
    client: httpx.AsyncClient = field(repr=False)

    @classmethod
    def from_request_id(
        cls,
        client: httpx.AsyncClient,
        application: str,
        request_id: str,
    ) -> AsyncRequestHandle:
        app_id = AppId.from_endpoint_id(application)
        prefix = f"{app_id.namespace}/" if app_id.namespace else ""
        base_url = f"{QUEUE_URL_FORMAT}{prefix}{app_id.owner}/{app_id.alias}/requests/{request_id}"
        return cls(
            request_id=request_id,
            response_url=base_url,
            status_url=base_url + "/status",
            cancel_url=base_url + "/cancel",
            client=client,
        )

    async def status(self, *, with_logs: bool = False) -> Status:
        """Returns the status of the request (which can be one of the following:
        Queued, InProgress, Completed). If `with_logs` is True, logs will be included
        for InProgress and Completed statuses."""

        response = await _async_maybe_retry_request(
            self.client,
            "GET",
            self.status_url,
            params={
                "logs": with_logs,
            },
        )
        _raise_for_status(response)

        return self._parse_status(response.json())

    async def iter_events(
        self, *, with_logs: bool = False, interval: float = 0.1
    ) -> AsyncIterator[Status]:
        """Continuously poll for the status of the request and yield it at each interval till
        the request is completed. If `with_logs` is True, logs will be included in the response.
        """

        while True:
            status = await self.status(with_logs=with_logs)
            yield status
            if isinstance(status, Completed):
                break

            await asyncio.sleep(interval)

    async def get(self) -> AnyJSON:
        """Wait till the request is completed and return the result."""
        async for _ in self.iter_events(with_logs=False):
            continue

        response = await _async_maybe_retry_request(
            self.client,
            "GET",
            self.response_url,
        )
        _raise_for_status(response)
        return response.json()

    async def cancel(self) -> None:
        """Cancel the request."""
        response = await _async_maybe_retry_request(
            self.client,
            "PUT",
            self.cancel_url,
        )
        _raise_for_status(response)


@dataclass(frozen=True)
class AsyncClient:
    key: str | None = field(default=None, repr=False)
    default_timeout: float = 120.0

    @cached_property
    def _auth(self) -> AuthCredentials:
        if self.key is None:
            return fetch_auth_credentials()
        return AuthCredentials("Key", self.key)

    def _get_auth(self) -> AuthCredentials:
        return self._auth

    def _get_key(self) -> str:
        auth = self._get_auth()
        if auth.scheme.lower() != "key":
            raise MissingCredentialsError(
                "Key credentials are required for this operation. Set FAL_KEY or FAL_KEY_ID/FAL_KEY_SECRET."
            )
        return auth.token

    @cached_property
    def _token_manager(self) -> AsyncCDNTokenManager:
        return AsyncCDNTokenManager(self._get_auth())

    @cached_property
    def _client(self) -> httpx.AsyncClient:
        auth = self._get_auth()
        return httpx.AsyncClient(
            headers={
                "Authorization": auth.header_value,
                "User-Agent": USER_AGENT,
            },
            timeout=self.default_timeout,
        )

    async def _get_cdn_client(self) -> httpx.AsyncClient:
        token = await self._token_manager.get_token()
        return httpx.AsyncClient(
            headers={
                "Authorization": f"{token.token_type} {token.token}",
                "User-Agent": USER_AGENT,
            },
            timeout=self.default_timeout,
        )

    async def _get_realtime_token(
        self,
        application: str,
        *,
        token_expiration: int = REALTIME_TOKEN_EXPIRATION_SECONDS,
    ) -> str:
        payload = {
            "allowed_apps": [AppId.from_endpoint_id(application).alias],
            "token_expiration": token_expiration,
        }
        response = await _async_maybe_retry_request(
            self._client,
            "POST",
            f"{REST_URL}/tokens/",
            json=payload,
        )
        return _parse_token_response(response.json())

    async def run(
        self,
        application: str,
        arguments: AnyJSON,
        *,
        path: str = "",
        timeout: Optional[Union[int, float]] = None,
        start_timeout: Optional[Union[int, float]] = None,
        hint: str | None = None,
        headers: dict[str, str] = {},
    ) -> AnyJSON:
        """Run an application with the given arguments (which will be JSON serialized). The path parameter can be used to
        specify a subpath when applicable. This method will return the result of the inference call directly.

        Args:
            timeout: Client-side HTTP timeout in seconds. Controls how long the HTTP
                client waits for a response. Defaults to the client's default_timeout.
            start_timeout: Server-side request timeout in seconds. Limits total time spent
                waiting before processing starts. Does not apply once the application begins processing.
        """

        url = RUN_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

        _headers: dict[str, str] = {**headers}

        # Set the headers
        if hint is not None:
            add_hint_header(hint, _headers)

        if start_timeout is not None:
            add_timeout_header(start_timeout, _headers)

        response = await _async_maybe_retry_request(
            self._client,
            "POST",
            url,
            json=arguments,
            timeout=timeout,
            headers=_headers,
        )

        _raise_for_status(response)
        return response.json()

    async def submit(
        self,
        application: str,
        arguments: AnyJSON,
        *,
        path: str = "",
        hint: str | None = None,
        webhook_url: str | None = None,
        priority: Optional[Priority] = None,
        headers: dict[str, str] = {},
        start_timeout: Optional[Union[int, float]] = None,
    ) -> AsyncRequestHandle:
        """Submit an application with the given arguments (which will be JSON serialized). The path parameter can be used to
        specify a subpath when applicable. This method will return a handle to the request that can be used to check the status
        and retrieve the result of the inference call when it is done.

        Args:
            start_timeout: Server-side request timeout in seconds. Limits total time spent
                waiting before processing starts (includes queue wait, retries, and
                routing). Does not apply once the application begins processing.
        """

        url = QUEUE_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

        if webhook_url is not None:
            url += "?" + urlencode({"fal_webhook": webhook_url})

        _headers: dict[str, str] = {**headers}

        # Set the headers
        if hint is not None:
            add_hint_header(hint, _headers)

        if priority is not None:
            add_priority_header(priority, _headers)

        if start_timeout is not None:
            add_timeout_header(start_timeout, _headers)

        response = await _async_maybe_retry_request(
            self._client,
            "POST",
            url,
            json=arguments,
            timeout=self.default_timeout,
            headers=_headers,
        )
        _raise_for_status(response)

        data = response.json()
        return AsyncRequestHandle(
            request_id=data["request_id"],
            response_url=data["response_url"],
            status_url=data["status_url"],
            cancel_url=data["cancel_url"],
            client=self._client,
        )

    async def subscribe(
        self,
        application: str,
        arguments: AnyJSON,
        *,
        path: str = "",
        hint: str | None = None,
        with_logs: bool = False,
        on_enqueue: Optional[Callable[[str], None]] = None,
        on_queue_update: Optional[Callable[[Status], None]] = None,
        priority: Optional[Priority] = None,
        headers: dict[str, str] = {},
        start_timeout: Optional[Union[int, float]] = None,
        client_timeout: Optional[Union[int, float]] = None,
    ) -> AnyJSON:
        """Subscribe to an application and wait for the result.

        Args:
            start_timeout: Server-side request timeout in seconds. Limits total time spent
                waiting before processing starts (includes queue wait, retries, and
                routing). Does not apply once the application begins processing.
            client_timeout: Client-side total timeout in seconds. Limits the total time
                spent waiting for the entire request to complete (including queue wait
                and processing). If not set, waits indefinitely.
        """
        if client_timeout is not None:
            if start_timeout is None:
                start_timeout = client_timeout
            elif start_timeout > client_timeout:
                warnings.warn(
                    f"start_timeout ({start_timeout}s) is larger than client_timeout ({client_timeout}s). "
                    "The request may timeout on the client before the server-side timeout is reached.",
                    UserWarning,
                    stacklevel=2,
                )

        handle_ref: list[Optional[AsyncRequestHandle]] = [None]

        async def _do_subscribe() -> AnyJSON:
            handle = await self.submit(
                application,
                arguments,
                path=path,
                hint=hint,
                priority=priority,
                headers=headers,
                start_timeout=start_timeout,
            )
            handle_ref[0] = handle

            if on_enqueue is not None:
                on_enqueue(handle.request_id)

            if on_queue_update is not None:
                async for event in handle.iter_events(with_logs=with_logs):
                    on_queue_update(event)

            return await handle.get()

        if client_timeout is None:
            return await _do_subscribe()

        try:
            return await asyncio.wait_for(_do_subscribe(), timeout=client_timeout)
        except asyncio.TimeoutError as e:
            request_id = None
            handle = handle_ref[0]
            if handle is not None:
                request_id = handle.request_id
                await _async_maybe_cancel_request(handle)

            raise FalClientTimeoutError(
                timeout=client_timeout,
                request_id=request_id,
            ) from e

    def get_handle(self, application: str, request_id: str) -> AsyncRequestHandle:
        return AsyncRequestHandle.from_request_id(self._client, application, request_id)

    async def status(
        self, application: str, request_id: str, *, with_logs: bool = False
    ) -> Status:
        handle = self.get_handle(application, request_id)
        return await handle.status(with_logs=with_logs)

    async def result(self, application: str, request_id: str) -> AnyJSON:
        handle = self.get_handle(application, request_id)
        return await handle.get()

    async def cancel(self, application: str, request_id: str) -> None:
        handle = self.get_handle(application, request_id)
        await handle.cancel()

    async def stream(
        self,
        application: str,
        arguments: AnyJSON,
        *,
        path: str = "/stream",
        timeout: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream the output of an application with the given arguments (which will be JSON serialized). This is only supported
        at a few select applications at the moment, so be sure to first consult with the documentation of individual applications
        to see if this is supported.

        The function will iterate over each event that is streamed from the server.
        """

        url = RUN_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

        async with aconnect_sse(
            self._client,
            "POST",
            url,
            json=arguments,
            timeout=timeout,
        ) as events:
            async for event in events.aiter_sse():
                yield event.json()

    async def upload(
        self, data: str | bytes, content_type: str, file_name: str | None = None
    ) -> str:
        """Upload the given data blob to the CDN and return the access URL. The content type should be specified
        as the second argument. Use upload_file or upload_image for convenience."""

        client = await self._get_cdn_client()

        if isinstance(data, str):
            data = data.encode("utf-8")

        if len(data) > MULTIPART_THRESHOLD:
            if file_name is None:
                file_name = "upload.bin"
            return await AsyncMultipartUpload.save(
                client=client,
                token_manager=self._token_manager,
                file_name=file_name,
                data=data,
                content_type=content_type,
            )

        headers = {"Content-Type": content_type}
        if file_name is not None:
            headers["X-Fal-File-Name"] = file_name

        response = await client.post(
            CDN_URL + "/files/upload",
            content=data,
            headers=headers,
        )
        _raise_for_status(response)

        return response.json()["access_url"]

    async def upload_file(self, path: os.PathLike) -> str:
        """Upload a file from the local filesystem to the CDN and return the access URL."""

        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        if os.path.getsize(path) > MULTIPART_THRESHOLD:
            client = await self._get_cdn_client()
            return await AsyncMultipartUpload.save_file(
                file_path=str(path),
                client=client,
                token_manager=self._token_manager,
                content_type=mime_type,
            )

        with open(path, "rb") as file:
            return await self.upload(
                file.read(), mime_type, file_name=os.path.basename(path)
            )

    async def upload_image(self, image: Image.Image, format: str = "jpeg") -> str:
        """Upload a pillow image object to the CDN and return the access URL."""

        with io.BytesIO() as buffer:
            image.save(buffer, format=format)
            return await self.upload(buffer.getvalue(), f"image/{format}")

    @asynccontextmanager
    async def realtime(
        self,
        application: str,
        *,
        use_jwt: bool = True,
        path: str = "/realtime",
        max_buffering: int | None = None,
        token_expiration: int = REALTIME_TOKEN_EXPIRATION_SECONDS,
        encode_message: Callable[[Any], bytes] | None = None,
        decode_message: Callable[[bytes], Any] | None = None,
    ) -> AsyncIterator[AsyncRealtimeConnection]:
        headers: dict[str, str] | None = None
        token: str | None = None
        if use_jwt:
            token = await self._get_realtime_token(
                application, token_expiration=token_expiration
            )
        else:
            auth = self._get_auth()
            headers = {"Authorization": auth.header_value, "User-Agent": USER_AGENT}

        url = _build_realtime_url(application, token, max_buffering, path=path)
        async with _connect_async_ws(url, headers=headers) as ws:
            yield AsyncRealtimeConnection(
                ws, _encode_message=encode_message, _decode_message=decode_message
            )

    @asynccontextmanager
    async def ws_connect(
        self,
        application: str,
        *,
        use_jwt: bool = True,
        path: str = "",
        max_buffering: int | None = None,
        token_expiration: int = REALTIME_TOKEN_EXPIRATION_SECONDS,
    ) -> AsyncIterator["WebSocketClientProtocol"]:
        headers: dict[str, str] | None = None
        token: str | None = None
        if use_jwt:
            token = await self._get_realtime_token(
                application, token_expiration=token_expiration
            )
        else:
            auth = self._get_auth()
            headers = {"Authorization": auth.header_value, "User-Agent": USER_AGENT}

        url = _build_runner_ws_url(
            application,
            token,
            path=path,
            max_buffering=max_buffering,
        )
        async with _connect_async_ws(url, headers=headers) as ws:
            yield ws


@dataclass(frozen=True)
class SyncClient:
    key: str | None = field(default=None, repr=False)
    default_timeout: float = 120.0

    @cached_property
    def _auth(self) -> AuthCredentials:
        if self.key is None:
            return fetch_auth_credentials()
        return AuthCredentials("Key", self.key)

    def _get_auth(self) -> AuthCredentials:
        return self._auth

    def _get_key(self) -> str:
        auth = self._get_auth()
        if auth.scheme.lower() != "key":
            raise MissingCredentialsError(
                "Key credentials are required for this operation. Set FAL_KEY or FAL_KEY_ID/FAL_KEY_SECRET."
            )
        return auth.token

    @cached_property
    def _client(self) -> httpx.Client:
        auth = self._get_auth()
        return httpx.Client(
            headers={
                "Authorization": auth.header_value,
                "User-Agent": USER_AGENT,
            },
            timeout=self.default_timeout,
            follow_redirects=True,
        )

    @cached_property
    def _token_manager(self) -> CDNTokenManager:
        return CDNTokenManager(self._get_auth())

    @property
    def _executor(self) -> concurrent.futures.ThreadPoolExecutor:
        return EXECUTOR

    def _get_cdn_client(self) -> httpx.Client:
        token = self._token_manager.get_token()
        return httpx.Client(
            headers={
                "Authorization": f"{token.token_type} {token.token}",
                "User-Agent": USER_AGENT,
            },
            timeout=self.default_timeout,
        )

    def _get_realtime_token(
        self,
        application: str,
        *,
        token_expiration: int = REALTIME_TOKEN_EXPIRATION_SECONDS,
    ) -> str:
        payload = {
            "allowed_apps": [AppId.from_endpoint_id(application).alias],
            "token_expiration": token_expiration,
        }
        response = _maybe_retry_request(
            self._client,
            "POST",
            f"{REST_URL}/tokens/",
            json=payload,
        )
        return _parse_token_response(response.json())

    def run(
        self,
        application: str,
        arguments: AnyJSON,
        *,
        path: str = "",
        timeout: Optional[Union[int, float]] = None,
        start_timeout: Optional[Union[int, float]] = None,
        hint: str | None = None,
        headers: dict[str, str] = {},
    ) -> AnyJSON:
        """Run an application with the given arguments (which will be JSON serialized).

        Args:
            timeout: Client-side HTTP timeout in seconds. Controls how long the HTTP
                client waits for a response. Defaults to the client's default_timeout.
            start_timeout: Server-side request timeout in seconds. Limits total time spent
                waiting before processing starts. Does not apply once the application begins processing.
        """

        url = RUN_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

        _headers: dict[str, str] = {**headers}
        if hint is not None:
            add_hint_header(hint, _headers)

        if start_timeout is not None:
            add_timeout_header(start_timeout, _headers)

        response = _maybe_retry_request(
            self._client,
            "POST",
            url,
            json=arguments,
            timeout=timeout,
            headers=_headers,
        )
        _raise_for_status(response)
        return response.json()

    def submit(
        self,
        application: str,
        arguments: AnyJSON,
        *,
        path: str = "",
        hint: str | None = None,
        webhook_url: str | None = None,
        priority: Optional[Priority] = None,
        headers: dict[str, str] = {},
        start_timeout: Optional[Union[int, float]] = None,
    ) -> SyncRequestHandle:
        """Submit an application with the given arguments (which will be JSON serialized).

        Args:
            start_timeout: Server-side request timeout in seconds. Limits total time spent
                waiting before processing starts (includes queue wait, retries, and
                routing). Does not apply once the application begins processing.
        """

        url = QUEUE_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

        if webhook_url is not None:
            url += "?" + urlencode({"fal_webhook": webhook_url})

        _headers: dict[str, str] = {**headers}

        if hint is not None:
            add_hint_header(hint, _headers)

        if priority is not None:
            add_priority_header(priority, _headers)

        if start_timeout is not None:
            add_timeout_header(start_timeout, _headers)

        response = _maybe_retry_request(
            self._client,
            "POST",
            url,
            json=arguments,
            timeout=self.default_timeout,
            headers=_headers,
        )
        _raise_for_status(response)

        data = response.json()
        return SyncRequestHandle(
            request_id=data["request_id"],
            response_url=data["response_url"],
            status_url=data["status_url"],
            cancel_url=data["cancel_url"],
            client=self._client,
        )

    def subscribe(
        self,
        application: str,
        arguments: AnyJSON,
        *,
        path: str = "",
        hint: str | None = None,
        with_logs: bool = False,
        on_enqueue: Optional[Callable[[str], None]] = None,
        on_queue_update: Optional[Callable[[Status], None]] = None,
        priority: Optional[Priority] = None,
        headers: dict[str, str] = {},
        start_timeout: Optional[Union[int, float]] = None,
        client_timeout: Optional[Union[int, float]] = None,
    ) -> AnyJSON:
        """Subscribe to an application and wait for the result.

        Args:
            start_timeout: Server-side request timeout in seconds. Limits total time spent
                waiting before processing starts (includes queue wait, retries, and
                routing). Does not apply once the application begins processing.
            client_timeout: Client-side total timeout in seconds. Limits the total time
                spent waiting for the entire request to complete (including queue wait
                and processing). If not set, waits indefinitely.
        """
        if client_timeout is not None:
            if start_timeout is None:
                start_timeout = client_timeout
            elif start_timeout > client_timeout:
                warnings.warn(
                    f"start_timeout ({start_timeout}s) is larger than client_timeout ({client_timeout}s). "
                    "The request may timeout on the client before the server-side timeout is reached.",
                    UserWarning,
                    stacklevel=2,
                )

        handle_ref: list[Optional[SyncRequestHandle]] = [None]

        def _do_subscribe() -> AnyJSON:
            handle = self.submit(
                application,
                arguments,
                path=path,
                hint=hint,
                priority=priority,
                headers=headers,
                start_timeout=start_timeout,
            )
            handle_ref[0] = handle

            if on_enqueue is not None:
                on_enqueue(handle.request_id)

            if on_queue_update is not None:
                for event in handle.iter_events(with_logs=with_logs):
                    on_queue_update(event)

            return handle.get()

        if client_timeout is None:
            return _do_subscribe()

        future = self._executor.submit(_do_subscribe)
        try:
            return future.result(timeout=client_timeout)
        except concurrent.futures.TimeoutError as e:
            request_id = None
            handle = handle_ref[0]
            if handle is not None:
                request_id = handle.request_id
                _maybe_cancel_request(handle)

            raise FalClientTimeoutError(
                timeout=client_timeout,
                request_id=request_id,
            ) from e

    def get_handle(self, application: str, request_id: str) -> SyncRequestHandle:
        return SyncRequestHandle.from_request_id(self._client, application, request_id)

    def status(
        self, application: str, request_id: str, *, with_logs: bool = False
    ) -> Status:
        handle = self.get_handle(application, request_id)
        return handle.status(with_logs=with_logs)

    def result(self, application: str, request_id: str) -> AnyJSON:
        handle = self.get_handle(application, request_id)
        return handle.get()

    def cancel(self, application: str, request_id: str) -> None:
        handle = self.get_handle(application, request_id)
        handle.cancel()

    def stream(
        self,
        application: str,
        arguments: AnyJSON,
        *,
        path: str = "/stream",
        timeout: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream the output of an application with the given arguments (which will be JSON serialized). This is only supported
        at a few select applications at the moment, so be sure to first consult with the documentation of individual applications
        to see if this is supported.

        The function will iterate over each event that is streamed from the server.
        """

        url = RUN_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

        with connect_sse(
            self._client, "POST", url, json=arguments, timeout=timeout
        ) as events:
            for event in events.iter_sse():
                yield event.json()

    def upload(
        self, data: str | bytes, content_type: str, file_name: str | None = None
    ) -> str:
        """Upload the given data blob to the CDN and return the access URL. The content type should be specified
        as the second argument. Use upload_file or upload_image for convenience."""

        client = self._get_cdn_client()

        if isinstance(data, str):
            data = data.encode("utf-8")

        if len(data) > MULTIPART_THRESHOLD:
            if file_name is None:
                file_name = "upload.bin"
            return MultipartUpload.save(
                client=client,
                token_manager=self._token_manager,
                file_name=file_name,
                data=data,
                content_type=content_type,
            )

        headers = {"Content-Type": content_type}
        if file_name is not None:
            headers["X-Fal-File-Name"] = file_name

        response = client.post(
            CDN_URL + "/files/upload",
            content=data,
            headers=headers,
        )
        _raise_for_status(response)

        return response.json()["access_url"]

    def upload_file(self, path: os.PathLike) -> str:
        """Upload a file from the local filesystem to the CDN and return the access URL."""

        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        if os.path.getsize(path) > MULTIPART_THRESHOLD:
            client = self._get_cdn_client()
            return MultipartUpload.save_file(
                file_path=str(path),
                client=client,
                token_manager=self._token_manager,
                content_type=mime_type,
            )

        with open(path, "rb") as file:
            return self.upload(file.read(), mime_type, file_name=os.path.basename(path))

    def upload_image(self, image: Image.Image, format: str = "jpeg") -> str:
        """Upload a pillow image object to the CDN and return the access URL."""

        with io.BytesIO() as buffer:
            image.save(buffer, format=format)
            return self.upload(buffer.getvalue(), f"image/{format}")

    @contextmanager
    def realtime(
        self,
        application: str,
        *,
        use_jwt: bool = True,
        path: str = "/realtime",
        max_buffering: int | None = None,
        token_expiration: int = REALTIME_TOKEN_EXPIRATION_SECONDS,
        encode_message: Callable[[Any], bytes] | None = None,
        decode_message: Callable[[bytes], Any] | None = None,
    ) -> Iterator[RealtimeConnection]:
        headers: dict[str, str] | None = None
        token: str | None = None
        if use_jwt:
            token = self._get_realtime_token(
                application, token_expiration=token_expiration
            )
        else:
            auth = self._get_auth()
            headers = {"Authorization": auth.header_value, "User-Agent": USER_AGENT}

        url = _build_realtime_url(application, token, max_buffering, path=path)
        with _connect_sync_ws(url, headers=headers) as ws:
            yield RealtimeConnection(
                ws, _encode_message=encode_message, _decode_message=decode_message
            )

    @contextmanager
    def ws_connect(
        self,
        application: str,
        *,
        use_jwt: bool = True,
        path: str = "",
        max_buffering: int | None = None,
        token_expiration: int = REALTIME_TOKEN_EXPIRATION_SECONDS,
    ) -> Iterator["Connection"]:
        headers: dict[str, str] | None = None
        token: str | None = None
        if use_jwt:
            token = self._get_realtime_token(
                application, token_expiration=token_expiration
            )
        else:
            auth = self._get_auth()
            headers = {"Authorization": auth.header_value, "User-Agent": USER_AGENT}

        url = _build_runner_ws_url(
            application,
            token,
            path=path,
            max_buffering=max_buffering,
        )
        with _connect_sync_ws(url, headers=headers) as ws:
            yield ws


def encode(data: str | bytes, content_type: str) -> str:
    """Encode the given data blob to a data URL with the specified content type."""
    if isinstance(data, str):
        data = data.encode("utf-8")

    return f"data:{content_type};base64,{base64.b64encode(data).decode()}"


def encode_file(path: os.PathLike) -> str:
    """Encode a file from the local filesystem to a data URL with the inferred content type."""
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(path, "rb") as file:
        return encode(file.read(), mime_type)


def encode_image(image: Image.Image, format: str = "jpeg") -> str:
    """Encode a pillow image object to a data URL with the specified format."""
    with io.BytesIO() as buffer:
        image.save(buffer, format=format)
        return encode(buffer.getvalue(), f"image/{format}")
