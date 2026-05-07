from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.gateway_request_item import GatewayRequestItem
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    share_id: UUID,
    *,
    logs: Union[Unset, bool] = False,
    include_billing_unit: Union[Unset, bool] = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["logs"] = logs

    params["include_billing_unit"] = include_billing_unit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/requests/share/{share_id}",
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
    share_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    logs: Union[Unset, bool] = False,
    include_billing_unit: Union[Unset, bool] = False,
) -> Response[Union[GatewayRequestItem, HTTPValidationError]]:
    """Get Gateway Request By Share Id

    Args:
        share_id (UUID):
        logs (Union[Unset, bool]):  Default: False.
        include_billing_unit (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[GatewayRequestItem, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        share_id=share_id,
        logs=logs,
        include_billing_unit=include_billing_unit,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    share_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    logs: Union[Unset, bool] = False,
    include_billing_unit: Union[Unset, bool] = False,
) -> Optional[Union[GatewayRequestItem, HTTPValidationError]]:
    """Get Gateway Request By Share Id

    Args:
        share_id (UUID):
        logs (Union[Unset, bool]):  Default: False.
        include_billing_unit (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[GatewayRequestItem, HTTPValidationError]
    """

    return sync_detailed(
        share_id=share_id,
        client=client,
        logs=logs,
        include_billing_unit=include_billing_unit,
    ).parsed


async def asyncio_detailed(
    share_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    logs: Union[Unset, bool] = False,
    include_billing_unit: Union[Unset, bool] = False,
) -> Response[Union[GatewayRequestItem, HTTPValidationError]]:
    """Get Gateway Request By Share Id

    Args:
        share_id (UUID):
        logs (Union[Unset, bool]):  Default: False.
        include_billing_unit (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[GatewayRequestItem, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        share_id=share_id,
        logs=logs,
        include_billing_unit=include_billing_unit,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    share_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    logs: Union[Unset, bool] = False,
    include_billing_unit: Union[Unset, bool] = False,
) -> Optional[Union[GatewayRequestItem, HTTPValidationError]]:
    """Get Gateway Request By Share Id

    Args:
        share_id (UUID):
        logs (Union[Unset, bool]):  Default: False.
        include_billing_unit (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[GatewayRequestItem, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            share_id=share_id,
            client=client,
            logs=logs,
            include_billing_unit=include_billing_unit,
        )
    ).parsed
