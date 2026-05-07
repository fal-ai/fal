from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.credit_balance_alert import CreditBalanceAlert
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response


def _get_kwargs(
    *,
    threshold_cents: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["threshold_cents"] = threshold_cents

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/billing/credit_balance_alert",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[CreditBalanceAlert, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = CreditBalanceAlert.from_dict(response.json())

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
) -> Response[Union[CreditBalanceAlert, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    threshold_cents: int,
) -> Response[Union[CreditBalanceAlert, HTTPValidationError]]:
    """Create Or Update Credit Balance Alert

     Create or update a credit balance alert for the authenticated user.

    Args:
        threshold_cents: The balance threshold in cents. When the balance drops to this value,
                        an alert will be triggered.

    Note:
        Orb stores credit balances in dollars, but the API accepts cents for consistency
        with other monetary amounts. We convert to USD before calling Orb functions.

    Args:
        threshold_cents (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CreditBalanceAlert, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        threshold_cents=threshold_cents,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    threshold_cents: int,
) -> Optional[Union[CreditBalanceAlert, HTTPValidationError]]:
    """Create Or Update Credit Balance Alert

     Create or update a credit balance alert for the authenticated user.

    Args:
        threshold_cents: The balance threshold in cents. When the balance drops to this value,
                        an alert will be triggered.

    Note:
        Orb stores credit balances in dollars, but the API accepts cents for consistency
        with other monetary amounts. We convert to USD before calling Orb functions.

    Args:
        threshold_cents (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CreditBalanceAlert, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        threshold_cents=threshold_cents,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    threshold_cents: int,
) -> Response[Union[CreditBalanceAlert, HTTPValidationError]]:
    """Create Or Update Credit Balance Alert

     Create or update a credit balance alert for the authenticated user.

    Args:
        threshold_cents: The balance threshold in cents. When the balance drops to this value,
                        an alert will be triggered.

    Note:
        Orb stores credit balances in dollars, but the API accepts cents for consistency
        with other monetary amounts. We convert to USD before calling Orb functions.

    Args:
        threshold_cents (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CreditBalanceAlert, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        threshold_cents=threshold_cents,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    threshold_cents: int,
) -> Optional[Union[CreditBalanceAlert, HTTPValidationError]]:
    """Create Or Update Credit Balance Alert

     Create or update a credit balance alert for the authenticated user.

    Args:
        threshold_cents: The balance threshold in cents. When the balance drops to this value,
                        an alert will be triggered.

    Note:
        Orb stores credit balances in dollars, but the API accepts cents for consistency
        with other monetary amounts. We convert to USD before calling Orb functions.

    Args:
        threshold_cents (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CreditBalanceAlert, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            threshold_cents=threshold_cents,
        )
    ).parsed
