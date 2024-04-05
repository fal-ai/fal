from __future__ import annotations

import io
import os
import httpx
import mimetypes
from httpx_sse import aconnect_sse, connect_sse
from dataclasses import dataclass, field
from functools import cached_property
from fal_client.auth import FAL_RUN_HOST, fetch_credentials
from typing import Any, AsyncIterator, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

AnyJSON = dict[str, Any]

RUN_URL_FORMAT = f"https://{FAL_RUN_HOST}/"
QUEUE_URL_FORMAT = f"https://queue.{FAL_RUN_HOST}/"
REALTIME_URL_FORMAT = f"wss://{FAL_RUN_HOST}/"
CDN_URL = "https://fal.media"


@dataclass
class Status: ...


@dataclass
class Queued(Status):
    position: int


@dataclass
class InProgress(Status):
    logs: list[dict[str, Any]] | None = field()


@dataclass
class Completed(Status):
    logs: list[dict[str, Any]] | None = field()
    metrics: dict[str, Any] = field()


@dataclass(frozen=True)
class _BaseRequestHandle:
    request_id: str
    response_url: str = field(repr=False)
    status_url: str = field(repr=False)
    cancel_url: str = field(repr=False)

    def parse_status(self, data: AnyJSON) -> Status:
        if data["status"] == "IN_QUEUE":
            return Queued(position=data["queue_position"])
        elif data["status"] == "IN_PROGRESS":
            return InProgress(logs=data["logs"])
        elif data["status"] == "COMPLETED":
            return Completed(logs=data["logs"], metrics=data["metrics"])
        else:
            raise ValueError(f"Unknown status: {data['status']}")


@dataclass(frozen=True)
class SyncRequestHandle(_BaseRequestHandle):
    client: httpx.Client = field(repr=False)

    def status(self, *, with_logs: bool = False) -> Status:
        """Checks the status of the request. If `with_logs` is True, logs will be
        included in the response."""

        response = self.client.get(
            self.status_url,
            params={
                "logs": with_logs,
            },
        )
        response.raise_for_status()

        return self.parse_status(response.json())

    def iter_events(self, *, with_logs: bool = False) -> Iterator[Status]:
        """Yield all events regarding the given task till its completed."""
        while True:
            status = self.status(with_logs=with_logs)
            yield status
            if isinstance(status, Completed):
                break

    def get(self) -> AnyJSON:
        """Wait till the request is completed and return the result."""
        for _ in self.iter_events(with_logs=False):
            continue

        response = self.client.get(self.response_url)
        response.raise_for_status()
        return response.json()


@dataclass(frozen=True)
class AsyncRequestHandle(_BaseRequestHandle):
    client: httpx.AsyncClient = field(repr=False)

    async def status(self, *, with_logs: bool = False) -> Status:
        """Checks the status of the request. If `with_logs` is True, logs will be
        included in the response."""

        response = await self.client.get(
            self.status_url,
            params={
                "logs": with_logs,
            },
        )
        response.raise_for_status()

        return self.parse_status(response.json())

    async def iter_events(self, *, with_logs: bool = False) -> AsyncIterator[Status]:
        """Yield all events regarding the given task till its completed."""
        while True:
            status = await self.status(with_logs=with_logs)
            yield status
            if isinstance(status, Completed):
                break

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
    default_timeout: float = 60.0

    @cached_property
    def _client(self) -> httpx.AsyncClient:
        if self.key is None:
            key = fetch_credentials()
        else:
            key = self.key

        return httpx.AsyncClient(headers={"Authorization": f"Key {key}"})

    async def run(
        self,
        application: str,
        arguments: AnyJSON,
        *,
        path: str = "",
    ) -> AnyJSON:
        url = RUN_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

        response = await self._client.post(
            url,
            json=arguments,
            timeout=self.default_timeout,
        )
        response.raise_for_status()
        return response.json()

    async def submit(
        self,
        application: str,
        arguments: AnyJSON,
        *,
        path: str = "",
    ) -> AsyncRequestHandle:
        url = QUEUE_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

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
    ) -> AsyncIterator[dict[str, Any]]:
        url = RUN_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

        async with aconnect_sse(self._client, "POST", url, json=arguments) as events:
            async for event in events.aiter_sse():
                yield event.json()

    async def upload(self, data: str | bytes, content_type: str) -> str:
        response = await self._client.post(
            CDN_URL + "/files/upload",
            data=data,
            headers={"Content-Type": content_type},
        )
        response.raise_for_status()

        return response.json()["access_url"]

    async def upload_file(self, path: os.PathLike) -> str:
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(path, "rb") as file:
            return await self.upload(file.read(), mime_type)

    async def upload_image(self, image: Image.Image, format: str = "jpeg") -> str:
        with io.BytesIO() as buffer:
            image.save(buffer, format=format)
            return await self.upload(buffer.getvalue(), f"image/{format}")


@dataclass(frozen=True)
class SyncClient:
    key: str | None = field(default=None, repr=False)
    default_timeout: float = 60.0

    @cached_property
    def _client(self) -> httpx.Client:
        if self.key is None:
            key = fetch_credentials()
        else:
            key = self.key
        return httpx.Client(
            headers={"Authorization": f"Key {key}"},
            timeout=self.default_timeout,
        )

    def run(
        self,
        application: str,
        arguments: AnyJSON,
        *,
        path: str = "",
        timeout: float | None = None,
    ) -> AnyJSON:
        url = RUN_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

        response = self._client.post(
            url,
            json=arguments,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    def submit(
        self,
        application: str,
        arguments: AnyJSON,
        *,
        path: str = "",
    ) -> SyncRequestHandle:
        url = QUEUE_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

        response = self._client.post(
            url,
            json=arguments,
            timeout=self.default_timeout,
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
    ) -> Iterator[dict[str, Any]]:
        url = RUN_URL_FORMAT + application
        if path:
            url += "/" + path.lstrip("/")

        with connect_sse(self._client, "POST", url, json=arguments) as events:
            for event in events.iter_sse():
                yield event.json()

    def upload(self, data: str | bytes, content_type: str) -> str:
        response = self._client.post(
            CDN_URL + "/files/upload",
            data=data,
            headers={"Content-Type": content_type},
        )
        response.raise_for_status()

        return response.json()["access_url"]

    def upload_file(self, path: os.PathLike) -> str:
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(path, "rb") as file:
            return self.upload(file.read(), mime_type)

    def upload_image(self, image: Image.Image, format: str = "jpeg") -> str:
        with io.BytesIO() as buffer:
            image.save(buffer, format=format)
            return self.upload(buffer.getvalue(), f"image/{format}")
