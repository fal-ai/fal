from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.endpoint_price import EndpointPrice
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    endpoint: str,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/billing/price/{endpoint}",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[EndpointPrice, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = EndpointPrice.from_dict(response.json())

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
) -> Response[Union[EndpointPrice, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    endpoint: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[EndpointPrice, HTTPValidationError]]:
    """Get Price

    Args:
        endpoint (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[EndpointPrice, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        endpoint=endpoint,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    endpoint: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[EndpointPrice, HTTPValidationError]]:
    """Get Price

    Args:
        endpoint (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[EndpointPrice, HTTPValidationError]
    """

    return sync_detailed(
        endpoint=endpoint,
        client=client,
    ).parsed


async def asyncio_detailed(
    endpoint: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[EndpointPrice, HTTPValidationError]]:
    """Get Price

    Args:
        endpoint (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[EndpointPrice, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        endpoint=endpoint,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    endpoint: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[EndpointPrice, HTTPValidationError]]:
    """Get Price

    Args:
        endpoint (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[EndpointPrice, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            endpoint=endpoint,
            client=client,
        )
    ).parsed
