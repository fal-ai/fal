from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.endpoint_price import EndpointPrice
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response


def _get_kwargs(
    *,
    endpoints: list[str],
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_endpoints = endpoints

    params["endpoints"] = json_endpoints

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/billing/v2/prices",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list["EndpointPrice"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = EndpointPrice.from_dict(response_200_item_data)

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
) -> Response[Union[HTTPValidationError, list["EndpointPrice"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    endpoints: list[str],
) -> Response[Union[HTTPValidationError, list["EndpointPrice"]]]:
    """Get Prices V2

     Return prices for the specified endpoints.

    This endpoint behaves similarly to `/prices`, but only returns entries
    for the requested endpoints and avoids dependency on the registry table.

    Args:
        endpoints (list[str]): List of endpoints to fetch prices for

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['EndpointPrice']]]
    """

    kwargs = _get_kwargs(
        endpoints=endpoints,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    endpoints: list[str],
) -> Optional[Union[HTTPValidationError, list["EndpointPrice"]]]:
    """Get Prices V2

     Return prices for the specified endpoints.

    This endpoint behaves similarly to `/prices`, but only returns entries
    for the requested endpoints and avoids dependency on the registry table.

    Args:
        endpoints (list[str]): List of endpoints to fetch prices for

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['EndpointPrice']]
    """

    return sync_detailed(
        client=client,
        endpoints=endpoints,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    endpoints: list[str],
) -> Response[Union[HTTPValidationError, list["EndpointPrice"]]]:
    """Get Prices V2

     Return prices for the specified endpoints.

    This endpoint behaves similarly to `/prices`, but only returns entries
    for the requested endpoints and avoids dependency on the registry table.

    Args:
        endpoints (list[str]): List of endpoints to fetch prices for

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['EndpointPrice']]]
    """

    kwargs = _get_kwargs(
        endpoints=endpoints,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    endpoints: list[str],
) -> Optional[Union[HTTPValidationError, list["EndpointPrice"]]]:
    """Get Prices V2

     Return prices for the specified endpoints.

    This endpoint behaves similarly to `/prices`, but only returns entries
    for the requested endpoints and avoids dependency on the registry table.

    Args:
        endpoints (list[str]): List of endpoints to fetch prices for

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['EndpointPrice']]
    """

    return (
        await asyncio_detailed(
            client=client,
            endpoints=endpoints,
        )
    ).parsed
