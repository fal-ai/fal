from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx

from fal import flags
from fal.sdk import Credentials, get_default_credentials

_URL_FORMAT = f"https://{{app_id}}.{flags.GATEWAY_HOST}/fal/queue"
_REALTIME_URL_FORMAT = f"wss://{{app_id}}.{flags.GATEWAY_HOST}"


@dataclass
class _Status:
    ...


@dataclass
class Queued(_Status):
    """Indicates the request is still in the queue, and provides the position
    in the queue for ETA calculation."""

    position: int


@dataclass
class InProgress(_Status):
    """Indicates the request is now being actively processed, and provides runtime
    logs for the inference task."""

    logs: list[dict[str, Any]] | None = field()


@dataclass
class Completed(_Status):
    """Indicates the request has been completed successfully and the result is
    ready to be retrieved."""

    logs: list[dict[str, Any]] | None = field()


@dataclass
class RequestHandle:
    """A handle to an async inference request."""

    app_id: str
    request_id: str

    # Use the credentials that were used to submit the request by default.
    _creds: Credentials = field(default_factory=get_default_credentials, repr=False)

    def status(self, *, logs: bool = False) -> _Status:
        """Check the status of an async inference request."""

        url = (
            _URL_FORMAT.format(app_id=self.app_id)
            + f"/requests/{self.request_id}/status/"
        )
        response = _HTTP_CLIENT.get(
            url,
            headers=self._creds.to_headers(),
            params={"logs": int(logs)},
        )
        response.raise_for_status()

        data = response.json()

        if response.status_code == 200:
            return Completed(logs=data["logs"])

        if data["status"] == "IN_QUEUE":
            return Queued(position=data["queue_position"])
        elif data["status"] == "IN_PROGRESS":
            return InProgress(logs=data["logs"])
        else:
            raise ValueError(f"Unknown status: {data['status']}")

    def iter_events(
        self,
        *,
        logs: bool = False,
        __poll_delay: float = 0.2,
    ) -> Iterator[_Status]:
        """Yield all events regarding the given task till its completed."""

        while True:
            status = self.status(logs=logs)

            if isinstance(status, Completed):
                return

            yield status
            time.sleep(__poll_delay)

    def fetch_result(self) -> dict[str, Any]:
        """Retrieve the result of an async inference request, raises an exception
        if the request is not completed yet."""
        url = (
            _URL_FORMAT.format(app_id=self.app_id)
            + f"/requests/{self.request_id}/response/"
        )
        response = _HTTP_CLIENT.get(url, headers=self._creds.to_headers())
        response.raise_for_status()

        data = response.json()
        return data

    def get(self) -> dict[str, Any]:
        """Retrieve the result of an async inference request, polling the status
        of the request until it is completed."""

        for event in self.iter_events(logs=False):
            continue

        return self.fetch_result()


_HTTP_CLIENT = httpx.Client(headers={"User-Agent": "Fal/Python"})


def run(app_id: str, arguments: dict[str, Any], *, path: str = "/") -> dict[str, Any]:
    """Run an inference task on a Fal app and return the result."""

    handle = submit(app_id, arguments, path=path)
    return handle.get()


def submit(app_id: str, arguments: dict[str, Any], *, path: str = "/") -> RequestHandle:
    """Submit an async inference task to the app. Returns a request handle
    which can be used to check the status of the request and retrieve the
    result."""

    url = _URL_FORMAT.format(app_id=app_id) + "/submit" + path
    creds = get_default_credentials()

    response = _HTTP_CLIENT.post(
        url,
        json=arguments,
        headers=creds.to_headers(),
    )
    response.raise_for_status()

    data = response.json()
    return RequestHandle(
        app_id=app_id,
        request_id=data["request_id"],
        _creds=creds,
    )


@dataclass
class _RealtimeConnection:
    """A realtime connection to a Fal app."""

    _ws: Any

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run an inference task on the app and return the result."""
        self.send(arguments)
        return self.recv()

    def send(self, arguments: dict[str, Any]) -> None:
        import msgpack

        """Send an inference task to the app."""
        payload = msgpack.packb(arguments)
        self._ws.send(payload)

    def recv(self) -> dict[str, Any]:
        import msgpack

        """Receive the result of an inference task."""
        while True:
            response = self._ws.recv()
            if isinstance(response, str):
                print(response)
                json_payload = json.loads(response)
                if json_payload.get("type") == "x-fal-error":
                    raise ValueError(json_payload["reason"])
                continue
            return msgpack.unpackb(response)


@contextmanager
def _connect(app_id: str, *, path: str = "/realtime") -> Iterator[_RealtimeConnection]:
    """Connect to a realtime endpoint. This is an internal and experimental API, use it
    at your own risk."""

    from websockets.sync import client

    url = _REALTIME_URL_FORMAT.format(app_id=app_id) + path
    creds = get_default_credentials()

    with client.connect(
        url, additional_headers=creds.to_headers(), open_timeout=90
    ) as ws:
        yield _RealtimeConnection(ws)
