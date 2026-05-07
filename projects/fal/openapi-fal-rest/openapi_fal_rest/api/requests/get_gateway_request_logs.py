from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.gateway_request_log_item import GatewayRequestLogItem
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response


def _get_kwargs(
    *,
    request_ids: list[UUID],
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_request_ids = []
    for request_ids_item_data in request_ids:
        request_ids_item = str(request_ids_item_data)
        json_request_ids.append(request_ids_item)

    params["request_ids"] = json_request_ids

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/requests/logs",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list["GatewayRequestLogItem"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = GatewayRequestLogItem.from_dict(response_200_item_data)

            response_200.append(response_200_item)

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
) -> Response[Union[HTTPValidationError, list["GatewayRequestLogItem"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    request_ids: list[UUID],
) -> Response[Union[HTTPValidationError, list["GatewayRequestLogItem"]]]:
    """Get Gateway Request Logs

    Args:
        request_ids (list[UUID]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['GatewayRequestLogItem']]]
    """

    kwargs = _get_kwargs(
        request_ids=request_ids,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    request_ids: list[UUID],
) -> Optional[Union[HTTPValidationError, list["GatewayRequestLogItem"]]]:
    """Get Gateway Request Logs

    Args:
        request_ids (list[UUID]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['GatewayRequestLogItem']]
    """

    return sync_detailed(
        client=client,
        request_ids=request_ids,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    request_ids: list[UUID],
) -> Response[Union[HTTPValidationError, list["GatewayRequestLogItem"]]]:
    """Get Gateway Request Logs

    Args:
        request_ids (list[UUID]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['GatewayRequestLogItem']]]
    """

    kwargs = _get_kwargs(
        request_ids=request_ids,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    request_ids: list[UUID],
) -> Optional[Union[HTTPValidationError, list["GatewayRequestLogItem"]]]:
    """Get Gateway Request Logs

    Args:
        request_ids (list[UUID]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['GatewayRequestLogItem']]
    """

    return (
        await asyncio_detailed(
            client=client,
            request_ids=request_ids,
        )
    ).parsed
