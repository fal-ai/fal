from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.spending_lock_response import SpendingLockResponse
from ...models.spending_lock_update import SpendingLockUpdate
from ...types import Response


def _get_kwargs(
    lock_id: UUID,
    *,
    body: SpendingLockUpdate,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "patch",
        "url": f"/billing/v2/spending-locks/{lock_id}",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, SpendingLockResponse]]:
    if response.status_code == 200:
        response_200 = SpendingLockResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, SpendingLockResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    lock_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: SpendingLockUpdate,
) -> Response[Union[HTTPValidationError, SpendingLockResponse]]:
    """Update Spending Lock

    Args:
        lock_id (UUID):
        body (SpendingLockUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, SpendingLockResponse]]
    """

    kwargs = _get_kwargs(
        lock_id=lock_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    lock_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: SpendingLockUpdate,
) -> Optional[Union[HTTPValidationError, SpendingLockResponse]]:
    """Update Spending Lock

    Args:
        lock_id (UUID):
        body (SpendingLockUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, SpendingLockResponse]
    """

    return sync_detailed(
        lock_id=lock_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    lock_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: SpendingLockUpdate,
) -> Response[Union[HTTPValidationError, SpendingLockResponse]]:
    """Update Spending Lock

    Args:
        lock_id (UUID):
        body (SpendingLockUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, SpendingLockResponse]]
    """

    kwargs = _get_kwargs(
        lock_id=lock_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    lock_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: SpendingLockUpdate,
) -> Optional[Union[HTTPValidationError, SpendingLockResponse]]:
    """Update Spending Lock

    Args:
        lock_id (UUID):
        body (SpendingLockUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, SpendingLockResponse]
    """

    return (
        await asyncio_detailed(
            lock_id=lock_id,
            client=client,
            body=body,
        )
    ).parsed
