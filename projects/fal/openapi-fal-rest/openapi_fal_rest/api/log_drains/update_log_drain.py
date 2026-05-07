from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.log_drain_info import LogDrainInfo
from ...models.log_drain_update import LogDrainUpdate
from ...types import Response


def _get_kwargs(
    drain_id: UUID,
    *,
    body: LogDrainUpdate,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "patch",
        "url": f"/log-drains/{drain_id}",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, LogDrainInfo]]:
    if response.status_code == 200:
        response_200 = LogDrainInfo.from_dict(response.json())

        return response_200
    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[HTTPValidationError, LogDrainInfo]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    drain_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: LogDrainUpdate,
) -> Response[Union[HTTPValidationError, LogDrainInfo]]:
    """Update Log Drain

    Args:
        drain_id (UUID):
        body (LogDrainUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, LogDrainInfo]]
    """

    kwargs = _get_kwargs(
        drain_id=drain_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    drain_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: LogDrainUpdate,
) -> Optional[Union[HTTPValidationError, LogDrainInfo]]:
    """Update Log Drain

    Args:
        drain_id (UUID):
        body (LogDrainUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, LogDrainInfo]
    """

    return sync_detailed(
        drain_id=drain_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    drain_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: LogDrainUpdate,
) -> Response[Union[HTTPValidationError, LogDrainInfo]]:
    """Update Log Drain

    Args:
        drain_id (UUID):
        body (LogDrainUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, LogDrainInfo]]
    """

    kwargs = _get_kwargs(
        drain_id=drain_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    drain_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: LogDrainUpdate,
) -> Optional[Union[HTTPValidationError, LogDrainInfo]]:
    """Update Log Drain

    Args:
        drain_id (UUID):
        body (LogDrainUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, LogDrainInfo]
    """

    return (
        await asyncio_detailed(
            drain_id=drain_id,
            client=client,
            body=body,
        )
    ).parsed
