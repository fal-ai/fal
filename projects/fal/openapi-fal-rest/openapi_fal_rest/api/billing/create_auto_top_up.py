from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.auto_top_up import AutoTopUp
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response


def _get_kwargs(
    *,
    threshold: int,
    amount: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["threshold"] = threshold

    params["amount"] = amount

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/billing/auto_top_up",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[AutoTopUp, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = AutoTopUp.from_dict(response.json())

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
) -> Response[Union[AutoTopUp, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    threshold: int,
    amount: int,
) -> Response[Union[AutoTopUp, HTTPValidationError]]:
    """Create Auto Top Up

    Args:
        threshold (int):
        amount (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AutoTopUp, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        threshold=threshold,
        amount=amount,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    threshold: int,
    amount: int,
) -> Optional[Union[AutoTopUp, HTTPValidationError]]:
    """Create Auto Top Up

    Args:
        threshold (int):
        amount (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AutoTopUp, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        threshold=threshold,
        amount=amount,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    threshold: int,
    amount: int,
) -> Response[Union[AutoTopUp, HTTPValidationError]]:
    """Create Auto Top Up

    Args:
        threshold (int):
        amount (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AutoTopUp, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        threshold=threshold,
        amount=amount,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    threshold: int,
    amount: int,
) -> Optional[Union[AutoTopUp, HTTPValidationError]]:
    """Create Auto Top Up

    Args:
        threshold (int):
        amount (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AutoTopUp, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            threshold=threshold,
            amount=amount,
        )
    ).parsed
