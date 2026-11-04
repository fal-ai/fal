from http import HTTPStatus
from typing import TYPE_CHECKING

import httpx

from fal.exceptions.volumes import volume_error_from_reason
from fal.volumes import Volume

from .api import FalServerlessError

if TYPE_CHECKING:
    from .client import SyncServerlessClient


def create_volume(client: "SyncServerlessClient", name: str) -> str:
    # Hand-written like api/_metrics.py and cli/queue.py: the generated
    # openapi-fal-rest client predates the /volumes/ endpoint. Switch to
    # openapi_fal_rest.api.volumes.create_volume once the client is regenerated.
    rest_client = client._create_rest_client()
    volume = Volume(name)
    with httpx.Client(
        base_url=rest_client.base_url,
        headers=rest_client.get_headers(),
        timeout=rest_client.get_timeout(),
        verify=rest_client.verify_ssl,
        follow_redirects=rest_client.follow_redirects,
    ) as http:
        response = http.post("/volumes/", json={"name": volume.name})
    if response.status_code == HTTPStatus.CREATED:
        return response.json()["name"]
    raise _error_from_response(response, volume.name)


def _error_from_response(response: httpx.Response, volume_name: str) -> Exception:
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
    return FalServerlessError(
        f"Failed to create volume {volume_name!r}: {response.status_code} {message}"
    )
