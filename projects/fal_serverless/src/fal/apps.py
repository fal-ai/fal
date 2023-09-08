from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx
from fal import flags
from fal.sdk import Credentials, get_default_credentials

_URL_FORMAT = f"https://{{app_id}}.{flags.GATEWAY_HOST}/fal/queue"


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

    logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Completed(_Status):
    """Indicates the request has been completed successfully and the result is
    ready to be retrieved."""


@dataclass
class RequestHandle:
    """A handle to an async inference request."""

    app_id: str
    request_id: str

    # Use the credentials that were used to submit the request by default.
    _creds: Credentials = field(default_factory=get_default_credentials, repr=False)

    def status(self) -> _Status:
        """Check the status of an async inference request."""

        url = (
            _URL_FORMAT.format(app_id=self.app_id)
            + f"/requests/{self.request_id}/status/"
        )
        response = _HTTP_CLIENT.get(url, headers=self._creds.to_headers())
        response.raise_for_status()

        if response.status_code == 200:
            return Completed()

        data = response.json()
        if data["status"] == "IN_QUEUE":
            return Queued(position=data["queue_position"])
        elif data["status"] == "IN_PROGRESS":
            return InProgress(logs=data["logs"])
        else:
            raise ValueError(f"Unknown status: {data['status']}")

    def iter_events(
        self,
        *,
        __poll_delay: float = 0.2,
    ) -> Iterator[_Status]:
        """Yield all events regarding the given task till its completed."""

        while True:
            status = self.status()

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

        for event in self.iter_events():
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
