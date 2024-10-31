from __future__ import annotations

import io
import os
import mimetypes
import asyncio
import time
import base64
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, AsyncIterator, Dict, Iterator, TYPE_CHECKING, Optional, Literal
from urllib.parse import urlencode

import httpx
from httpx_sse import aconnect_sse, connect_sse
from fal_client.auth import FAL_RUN_HOST, fetch_credentials

if TYPE_CHECKING:
    from PIL import Image

AnyJSON = Dict[str, Any]
Priority = Literal["normal", "low"]

RUN_URL_FORMAT = f"https://{FAL_RUN_HOST}/"
QUEUE_URL_FORMAT = f"https://queue.{FAL_RUN_HOST}/"
REALTIME_URL_FORMAT = f"wss://{FAL_RUN_HOST}/"
REST_URL = "https://rest.alpha.fal.ai"
CDN_URL = "https://v3.fal.media"
USER_AGENT = "fal-client/0.2.2 (python)"


@dataclass
class CDNToken:
    token: str
    token_type: str
    base_upload_url: str
    expires_at: datetime

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at


class CDNTokenManager:
    def __init__(self, key: str) -> None:
        self._key = key
        self._token: CDNToken = CDNToken(
            token="",
            token_type="",
            base_upload_url="",
            expires_at=datetime.min.replace(tzinfo=timezone.utc),
        )
        self._lock: threading.Lock = threading.Lock()
        self._url = f"{REST_URL}/storage/auth/token?storage_type=fal-cdn-v3"
        self._headers = {
            "Authorization": f"Key {self._key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _refresh_token(self) -> CDNToken:
        with httpx.Client() as client:
            response = client.post(self._url, headers=self._headers, data=b"{}")
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
    def __init__(self, key: str) -> None:
        self._key = key
        self._token: CDNToken = CDNToken(
            token="",
            token_type="",
            base_upload_url="",
            expires_at=datetime.min.replace(tzinfo=timezone.utc),
        )
        self._lock: asyncio.Lock = asyncio.Lock()
        self._url = f"{REST_URL}/storage/auth/token?storage_type=fal-cdn-v3"
        self._headers = {
            "Authorization": f"Key {self._key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def _refresh_token(self) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.post(self._url, headers=self._headers, data=b"{}")
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


class FalClientError(Exception):
    pass


def _raise_for_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            msg = response.json()["detail"]
        except (ValueError, KeyError):
            msg = response.text

        raise FalClientError(msg) from exc


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


@dataclass(frozen=True)
class SyncRequestHandle(_BaseRequestHandle):
    client: httpx.Client = field(repr=False)

    @classmethod
    def from_request_id(
        cls, client: httpx.Client, application: str, request_id: str
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

        response = self.client.get(
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

        response = self.client.get(self.response_url)
        _raise_for_status(response)
        return response.json()

    def cancel(self) -> None:
        """Cancel the request."""
        response = self.client.put(self.cancel_url)
        _raise_for_status(response)


@dataclass(frozen=True)
class AsyncRequestHandle(_BaseRequestHandle):
    client: httpx.AsyncClient = field(repr=False)

    @classmethod
    def from_request_id(
        cls, client: httpx.AsyncClient, application: str, request_id: str
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

        response = await self.client.get(
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

        response = await self.client.get(self.response_url)
        _raise_for_status(response)
        return response.json()

    async def cancel(self) -> None:
        """Cancel the request."""
        response = await self.client.put(self.cancel_url)
        _raise_for_status(response)


@dataclass(frozen=True)
class AsyncClient:
    key: str | None = field(default=None, repr=False)
    default_timeout: float = 120.0

    def _get_key(self) -> str:
        if self.key is None:
            return fetch_credentials()
        return self.key

    @cached_property
    def _token_manager(self) -> AsyncCDNTokenManager:
        return AsyncCDNTokenManager(self._get_key())

    @cached_property
    def _client(self) -> httpx.AsyncClient:
        key = self._get_key()
        return httpx.AsyncClient(
            headers={
                "Authorization": f"Key {key}",
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

    async def run(
        self,
        application: str,
        arguments: AnyJSON,
        *,
        path: str = "",
        timeout: float | None = None,
        hint: str | None = None,
    ) -> AnyJSON:
        """Run an application with the given arguments (which will be JSON serialized). The path parameter can be used to
        specify a subpath when applicable. This method will return the result of the inference call directly.
        """

        url = RUN_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

        headers = {}
        if hint is not None:
            headers["X-Fal-Runner-Hint"] = hint

        response = await self._client.post(
            url,
            json=arguments,
            timeout=timeout,
            headers=headers,
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
    ) -> AsyncRequestHandle:
        """Submit an application with the given arguments (which will be JSON serialized). The path parameter can be used to
        specify a subpath when applicable. This method will return a handle to the request that can be used to check the status
        and retrieve the result of the inference call when it is done."""

        url = QUEUE_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

        if webhook_url is not None:
            url += "?" + urlencode({"fal_webhook": webhook_url})

        headers = {}
        if hint is not None:
            headers["X-Fal-Runner-Hint"] = hint

        if priority is not None:
            headers["X-Fal-Queue-Priority"] = priority

        response = await self._client.post(
            url,
            json=arguments,
            timeout=self.default_timeout,
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
        on_enqueue: Optional[callable[[Queued], None]] = None,
        on_queue_update: Optional[callable[[Status], None]] = None,
        priority: Optional[Priority] = None,
    ) -> AnyJSON:
        handle = await self.submit(
            application,
            arguments,
            path=path,
            hint=hint,
            priority=priority,
        )

        if on_enqueue is not None:
            on_enqueue(handle.request_id)

        if on_queue_update is not None:
            async for event in handle.iter_events(with_logs=with_logs):
                on_queue_update(event)

        return await handle.get()

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

        headers = {"Content-Type": content_type}
        if file_name is not None:
            headers["X-Fal-File-Name"] = file_name

        response = await client.post(
            CDN_URL + "/files/upload",
            data=data,
            headers=headers,
        )
        _raise_for_status(response)

        return response.json()["access_url"]

    async def upload_file(self, path: os.PathLike) -> str:
        """Upload a file from the local filesystem to the CDN and return the access URL."""

        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(path, "rb") as file:
            return await self.upload(
                file.read(), mime_type, file_name=os.path.basename(path)
            )

    async def upload_image(self, image: Image.Image, format: str = "jpeg") -> str:
        """Upload a pillow image object to the CDN and return the access URL."""

        with io.BytesIO() as buffer:
            image.save(buffer, format=format)
            return await self.upload(buffer.getvalue(), f"image/{format}")


@dataclass(frozen=True)
class SyncClient:
    key: str | None = field(default=None, repr=False)
    default_timeout: float = 120.0

    def _get_key(self) -> str:
        if self.key is None:
            return fetch_credentials()
        return self.key

    @cached_property
    def _client(self) -> httpx.Client:
        key = self._get_key()
        return httpx.Client(
            headers={
                "Authorization": f"Key {key}",
                "User-Agent": USER_AGENT,
            },
            timeout=self.default_timeout,
        )

    @cached_property
    def _token_manager(self) -> CDNTokenManager:
        return CDNTokenManager(self._get_key())

    def _get_cdn_client(self) -> httpx.Client:
        token = self._token_manager.get_token()
        return httpx.Client(
            headers={
                "Authorization": f"{token.token_type} {token.token}",
                "User-Agent": USER_AGENT,
            },
            timeout=self.default_timeout,
        )

    def run(
        self,
        application: str,
        arguments: AnyJSON,
        *,
        path: str = "",
        timeout: float | None = None,
        hint: str | None = None,
    ) -> AnyJSON:
        """Run an application with the given arguments (which will be JSON serialized). The path parameter can be used to
        specify a subpath when applicable. This method will return the result of the inference call directly.
        """

        url = RUN_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

        headers = {}
        if hint is not None:
            headers["X-Fal-Runner-Hint"] = hint

        response = self._client.post(
            url,
            json=arguments,
            timeout=timeout,
            headers=headers,
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
    ) -> SyncRequestHandle:
        """Submit an application with the given arguments (which will be JSON serialized). The path parameter can be used to
        specify a subpath when applicable. This method will return a handle to the request that can be used to check the status
        and retrieve the result of the inference call when it is done."""

        url = QUEUE_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

        if webhook_url is not None:
            url += "?" + urlencode({"fal_webhook": webhook_url})

        headers = {}
        if hint is not None:
            headers["X-Fal-Runner-Hint"] = hint

        if priority is not None:
            headers["X-Fal-Queue-Priority"] = priority

        response = self._client.post(
            url,
            json=arguments,
            timeout=self.default_timeout,
            headers=headers,
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
        on_enqueue: Optional[callable[[Queued], None]] = None,
        on_queue_update: Optional[callable[[Status], None]] = None,
        priority: Optional[Priority] = None,
    ) -> AnyJSON:
        handle = self.submit(
            application,
            arguments,
            path=path,
            hint=hint,
            priority=priority,
        )

        if on_enqueue is not None:
            on_enqueue(handle.request_id)

        if on_queue_update is not None:
            for event in handle.iter_events(with_logs=with_logs):
                on_queue_update(event)

        return handle.get()

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

        headers = {"Content-Type": content_type}
        if file_name is not None:
            headers["X-Fal-File-Name"] = file_name

        response = client.post(
            CDN_URL + "/files/upload",
            data=data,
            headers=headers,
        )
        _raise_for_status(response)

        return response.json()["access_url"]

    def upload_file(self, path: os.PathLike) -> str:
        """Upload a file from the local filesystem to the CDN and return the access URL."""

        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(path, "rb") as file:
            return self.upload(file.read(), mime_type, file_name=os.path.basename(path))

    def upload_image(self, image: Image.Image, format: str = "jpeg") -> str:
        """Upload a pillow image object to the CDN and return the access URL."""

        with io.BytesIO() as buffer:
            image.save(buffer, format=format)
            return self.upload(buffer.getvalue(), f"image/{format}")


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
