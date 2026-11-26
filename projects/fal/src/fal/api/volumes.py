from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional

import httpx

from fal.exceptions.volumes import volume_error_from_reason
from fal.volumes import Volume

from .api import FalServerlessError

if TYPE_CHECKING:
    from .client import SyncServerlessClient


@dataclass
class VolumeInfo:
    name: str
    created_at: datetime


def _http(client: "SyncServerlessClient") -> httpx.Client:
    # Hand-written like api/_metrics.py and cli/queue.py: the generated
    # openapi-fal-rest client predates the /volumes/ endpoints. Switch to
    # openapi_fal_rest.api.volumes once the client is regenerated.
    rest_client = client._create_rest_client()
    return httpx.Client(
        base_url=rest_client.base_url,
        headers=rest_client.get_headers(),
        timeout=rest_client.get_timeout(),
        verify=rest_client.verify_ssl,
        follow_redirects=rest_client.follow_redirects,
    )


def create_volume(client: "SyncServerlessClient", name: str) -> str:
    volume = Volume(name)
    with _http(client) as http:
        response = http.post("/volumes/", json={"name": volume.name})
    if response.status_code == HTTPStatus.CREATED:
        return response.json()["name"]
    raise _error_from_response(response, f"create volume {volume.name!r}", volume.name)


def list_volumes(client: "SyncServerlessClient") -> List[VolumeInfo]:
    with _http(client) as http:
        response = http.get("/volumes/")
    if response.status_code == HTTPStatus.OK:
        return [
            VolumeInfo(
                name=item["name"],
                # fromisoformat only accepts a trailing "Z" from Python 3.11.
                created_at=datetime.fromisoformat(
                    item["created_at"].replace("Z", "+00:00")
                ),
            )
            for item in response.json()
        ]
    raise _error_from_response(response, "list volumes")


def delete_volume(client: "SyncServerlessClient", name: str) -> None:
    volume = Volume(name)
    with _http(client) as http:
        response = http.delete(f"/volumes/{volume.name}")
    if response.status_code == HTTPStatus.NO_CONTENT:
        return
    raise _error_from_response(response, f"delete volume {volume.name!r}", volume.name)


def _error_from_response(
    response: httpx.Response, action: str, volume_name: Optional[str] = None
) -> Exception:
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        body = {}
    detail = body.get("detail")
    message = detail if isinstance(detail, str) else response.text
    reason = body.get("reason")
    if isinstance(reason, str):
        error = volume_error_from_reason(
            reason,
            message,
            http_status=response.status_code,
            volume_name=volume_name,
        )
        if error is not None:
            return error
    return FalServerlessError(f"Failed to {action}: {response.status_code} {message}")
