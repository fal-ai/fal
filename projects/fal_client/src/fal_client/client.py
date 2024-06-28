from __future__ import annotations

import io
import os
import mimetypes
import asyncio
import time
import base64
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, AsyncIterator, Iterator, TYPE_CHECKING

import httpx
from httpx_sse import aconnect_sse, connect_sse
from fal_client.auth import FAL_RUN_HOST, fetch_credentials

if TYPE_CHECKING:
    from PIL import Image

AnyJSON = dict[str, Any]

RUN_URL_FORMAT = f"https://{FAL_RUN_HOST}/"
QUEUE_URL_FORMAT = f"https://queue.{FAL_RUN_HOST}/"
REALTIME_URL_FORMAT = f"wss://{FAL_RUN_HOST}/"
CDN_URL = "https://fal.media"
USER_AGENT = "fal-client/0.2.2 (python)"


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


@dataclass(frozen=True)
class SyncRequestHandle(_BaseRequestHandle):
    client: httpx.Client = field(repr=False)

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
        response.raise_for_status()

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
        response.raise_for_status()
        return response.json()


@dataclass(frozen=True)
class AsyncRequestHandle(_BaseRequestHandle):
    client: httpx.AsyncClient = field(repr=False)

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
        response.raise_for_status()

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
        response.raise_for_status()
        return response.json()


@dataclass(frozen=True)
class AsyncClient:
    key: str | None = field(default=None, repr=False)
    default_timeout: float = 120.0

    @cached_property
    def _client(self) -> httpx.AsyncClient:
        if self.key is None:
            key = fetch_credentials()
        else:
            key = self.key

        return httpx.AsyncClient(
            headers={
                "Authorization": f"Key {key}",
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
        response.raise_for_status()
        return response.json()

    async def submit(
        self,
        application: str,
        arguments: AnyJSON,
        *,
        path: str = "",
        hint: str | None = None,
    ) -> AsyncRequestHandle:
        """Submit an application with the given arguments (which will be JSON serialized). The path parameter can be used to
        specify a subpath when applicable. This method will return a handle to the request that can be used to check the status
        and retrieve the result of the inference call when it is done."""

        url = QUEUE_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

        headers = {}
        if hint is not None:
            headers["X-Fal-Runner-Hint"] = hint

        response = await self._client.post(
            url,
            json=arguments,
            timeout=self.default_timeout,
        )
        response.raise_for_status()

        data = response.json()
        return AsyncRequestHandle(
            request_id=data["request_id"],
            response_url=data["response_url"],
            status_url=data["status_url"],
            cancel_url=data["cancel_url"],
            client=self._client,
        )

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

    async def upload(self, data: str | bytes, content_type: str) -> str:
        """Upload the given data blob to the CDN and return the access URL. The content type should be specified
        as the second argument. Use upload_file or upload_image for convenience."""

        response = await self._client.post(
            CDN_URL + "/files/upload",
            data=data,
            headers={"Content-Type": content_type},
        )
        response.raise_for_status()

        return response.json()["access_url"]

    async def upload_file(self, path: os.PathLike) -> str:
        """Upload a file from the local filesystem to the CDN and return the access URL."""

        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(path, "rb") as file:
            return await self.upload(file.read(), mime_type)

    async def upload_image(self, image: Image.Image, format: str = "jpeg") -> str:
        """Upload a pillow image object to the CDN and return the access URL."""

        with io.BytesIO() as buffer:
            image.save(buffer, format=format)
            return await self.upload(buffer.getvalue(), f"image/{format}")


@dataclass(frozen=True)
class SyncClient:
    key: str | None = field(default=None, repr=False)
    default_timeout: float = 120.0

    @cached_property
    def _client(self) -> httpx.Client:
        if self.key is None:
            key = fetch_credentials()
        else:
            key = self.key
        return httpx.Client(
            headers={
                "Authorization": f"Key {key}",
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
        response.raise_for_status()
        return response.json()

    def submit(
        self,
        application: str,
        arguments: AnyJSON,
        *,
        path: str = "",
        hint: str | None = None,
    ) -> SyncRequestHandle:
        """Submit an application with the given arguments (which will be JSON serialized). The path parameter can be used to
        specify a subpath when applicable. This method will return a handle to the request that can be used to check the status
        and retrieve the result of the inference call when it is done."""

        url = QUEUE_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

        headers = {}
        if hint is not None:
            headers["X-Fal-Runner-Hint"] = hint

        response = self._client.post(
            url,
            json=arguments,
            timeout=self.default_timeout,
            headers=headers,
        )
        response.raise_for_status()

        data = response.json()
        return SyncRequestHandle(
            request_id=data["request_id"],
            response_url=data["response_url"],
            status_url=data["status_url"],
            cancel_url=data["cancel_url"],
            client=self._client,
        )

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

    def upload(self, data: str | bytes, content_type: str) -> str:
        """Upload the given data blob to the CDN and return the access URL. The content type should be specified
        as the second argument. Use upload_file or upload_image for convenience."""

        response = self._client.post(
            CDN_URL + "/files/upload",
            data=data,
            headers={"Content-Type": content_type},
        )
        response.raise_for_status()

        return response.json()["access_url"]

    def upload_file(self, path: os.PathLike) -> str:
        """Upload a file from the local filesystem to the CDN and return the access URL."""

        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(path, "rb") as file:
            return self.upload(file.read(), mime_type)

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
