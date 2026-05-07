from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.spending_alert_response import SpendingAlertResponse
from ...models.spending_alert_update import SpendingAlertUpdate
from ...types import Response


def _get_kwargs(
    alert_id: UUID,
    *,
    body: SpendingAlertUpdate,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "patch",
        "url": f"/billing/v2/spending-alerts/{alert_id}",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, SpendingAlertResponse]]:
    if response.status_code == 200:
        response_200 = SpendingAlertResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, SpendingAlertResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    alert_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: SpendingAlertUpdate,
) -> Response[Union[HTTPValidationError, SpendingAlertResponse]]:
    """Update Spending Alert

    Args:
        alert_id (UUID):
        body (SpendingAlertUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, SpendingAlertResponse]]
    """

    kwargs = _get_kwargs(
        alert_id=alert_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    alert_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: SpendingAlertUpdate,
) -> Optional[Union[HTTPValidationError, SpendingAlertResponse]]:
    """Update Spending Alert

    Args:
        alert_id (UUID):
        body (SpendingAlertUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, SpendingAlertResponse]
    """

    return sync_detailed(
        alert_id=alert_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    alert_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: SpendingAlertUpdate,
) -> Response[Union[HTTPValidationError, SpendingAlertResponse]]:
    """Update Spending Alert

    Args:
        alert_id (UUID):
        body (SpendingAlertUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, SpendingAlertResponse]]
    """

    kwargs = _get_kwargs(
        alert_id=alert_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    alert_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: SpendingAlertUpdate,
) -> Optional[Union[HTTPValidationError, SpendingAlertResponse]]:
    """Update Spending Alert

    Args:
        alert_id (UUID):
        body (SpendingAlertUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, SpendingAlertResponse]
    """

    return (
        await asyncio_detailed(
            alert_id=alert_id,
            client=client,
            body=body,
        )
    ).parsed
