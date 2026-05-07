from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.current_contract import CurrentContract
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs() -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/billing/v2/current-contract",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[CurrentContract, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = CurrentContract.from_dict(response.json())

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
) -> Response[Union[CurrentContract, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[CurrentContract, HTTPValidationError]]:
    """Get Current Contract

     Return the currently active enterprise contract for the authenticated user.

    A contract is considered active if:
    - Status is ACTIVE
    - Current date is between start date and end date (or no end date)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CurrentContract, HTTPValidationError]]
    """

    kwargs = _get_kwargs()

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[CurrentContract, HTTPValidationError]]:
    """Get Current Contract

     Return the currently active enterprise contract for the authenticated user.

    A contract is considered active if:
    - Status is ACTIVE
    - Current date is between start date and end date (or no end date)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CurrentContract, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[CurrentContract, HTTPValidationError]]:
    """Get Current Contract

     Return the currently active enterprise contract for the authenticated user.

    A contract is considered active if:
    - Status is ACTIVE
    - Current date is between start date and end date (or no end date)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CurrentContract, HTTPValidationError]]
    """

    kwargs = _get_kwargs()

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[CurrentContract, HTTPValidationError]]:
    """Get Current Contract

     Return the currently active enterprise contract for the authenticated user.

    A contract is considered active if:
    - Status is ACTIVE
    - Current date is between start date and end date (or no end date)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CurrentContract, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
        )
    ).parsed
