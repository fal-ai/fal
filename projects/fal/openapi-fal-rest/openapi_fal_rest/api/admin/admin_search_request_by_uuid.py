from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.gateway_request_item import GatewayRequestItem
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response


def _get_kwargs(
    *,
    uuid: UUID,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_uuid = str(uuid)
    params["uuid"] = json_uuid

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/admin/requests/search-uuid",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[GatewayRequestItem, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = GatewayRequestItem.from_dict(response.json())

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
) -> Response[Union[GatewayRequestItem, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    uuid: UUID,
) -> Response[Union[GatewayRequestItem, HTTPValidationError]]:
    """Search Request By Uuid

     Search for a UUID in request inputs from the past 15 minutes.

    Args:
        uuid (UUID): UUID to search for in json_input

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[GatewayRequestItem, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        uuid=uuid,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    uuid: UUID,
) -> Optional[Union[GatewayRequestItem, HTTPValidationError]]:
    """Search Request By Uuid

     Search for a UUID in request inputs from the past 15 minutes.

    Args:
        uuid (UUID): UUID to search for in json_input

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[GatewayRequestItem, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        uuid=uuid,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    uuid: UUID,
) -> Response[Union[GatewayRequestItem, HTTPValidationError]]:
    """Search Request By Uuid

     Search for a UUID in request inputs from the past 15 minutes.

    Args:
        uuid (UUID): UUID to search for in json_input

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[GatewayRequestItem, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        uuid=uuid,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    uuid: UUID,
) -> Optional[Union[GatewayRequestItem, HTTPValidationError]]:
    """Search Request By Uuid

     Search for a UUID in request inputs from the past 15 minutes.

    Args:
        uuid (UUID): UUID to search for in json_input

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[GatewayRequestItem, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            uuid=uuid,
        )
    ).parsed
