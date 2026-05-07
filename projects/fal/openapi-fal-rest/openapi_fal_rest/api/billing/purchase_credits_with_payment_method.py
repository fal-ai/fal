from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.purchase_credits_with_payment_method_response_purchase_credits_with_payment_method import (
    PurchaseCreditsWithPaymentMethodResponsePurchaseCreditsWithPaymentMethod,
)
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    quantity: int,
    idempotency_key: Union[Unset, str] = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(idempotency_key, Unset):
        headers["Idempotency-Key"] = idempotency_key

    params: dict[str, Any] = {}

    params["quantity"] = quantity

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/billing/purchase_credits",
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, PurchaseCreditsWithPaymentMethodResponsePurchaseCreditsWithPaymentMethod]]:
    if response.status_code == 200:
        response_200 = PurchaseCreditsWithPaymentMethodResponsePurchaseCreditsWithPaymentMethod.from_dict(
            response.json()
        )

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
) -> Response[Union[HTTPValidationError, PurchaseCreditsWithPaymentMethodResponsePurchaseCreditsWithPaymentMethod]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    quantity: int,
    idempotency_key: Union[Unset, str] = UNSET,
) -> Response[Union[HTTPValidationError, PurchaseCreditsWithPaymentMethodResponsePurchaseCreditsWithPaymentMethod]]:
    """Purchase Credits With Payment Method

     Purchase credits using existing payment method via Orb auto-collection.

    This endpoint creates an Orb invoice with auto-collection enabled, which will
    charge the customer's default payment method immediately. Unlike /checkout,
    this does not redirect to Stripe - it processes the payment directly through Orb.

    Returns the invoice_id which can be used by the UI to poll Orb for payment status.

    Args:
        quantity: Amount in USD to purchase
        idempotency_key: Optional header to prevent duplicate charges

    Args:
        quantity (int):
        idempotency_key (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PurchaseCreditsWithPaymentMethodResponsePurchaseCreditsWithPaymentMethod]]
    """

    kwargs = _get_kwargs(
        quantity=quantity,
        idempotency_key=idempotency_key,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    quantity: int,
    idempotency_key: Union[Unset, str] = UNSET,
) -> Optional[Union[HTTPValidationError, PurchaseCreditsWithPaymentMethodResponsePurchaseCreditsWithPaymentMethod]]:
    """Purchase Credits With Payment Method

     Purchase credits using existing payment method via Orb auto-collection.

    This endpoint creates an Orb invoice with auto-collection enabled, which will
    charge the customer's default payment method immediately. Unlike /checkout,
    this does not redirect to Stripe - it processes the payment directly through Orb.

    Returns the invoice_id which can be used by the UI to poll Orb for payment status.

    Args:
        quantity: Amount in USD to purchase
        idempotency_key: Optional header to prevent duplicate charges

    Args:
        quantity (int):
        idempotency_key (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PurchaseCreditsWithPaymentMethodResponsePurchaseCreditsWithPaymentMethod]
    """

    return sync_detailed(
        client=client,
        quantity=quantity,
        idempotency_key=idempotency_key,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    quantity: int,
    idempotency_key: Union[Unset, str] = UNSET,
) -> Response[Union[HTTPValidationError, PurchaseCreditsWithPaymentMethodResponsePurchaseCreditsWithPaymentMethod]]:
    """Purchase Credits With Payment Method

     Purchase credits using existing payment method via Orb auto-collection.

    This endpoint creates an Orb invoice with auto-collection enabled, which will
    charge the customer's default payment method immediately. Unlike /checkout,
    this does not redirect to Stripe - it processes the payment directly through Orb.

    Returns the invoice_id which can be used by the UI to poll Orb for payment status.

    Args:
        quantity: Amount in USD to purchase
        idempotency_key: Optional header to prevent duplicate charges

    Args:
        quantity (int):
        idempotency_key (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PurchaseCreditsWithPaymentMethodResponsePurchaseCreditsWithPaymentMethod]]
    """

    kwargs = _get_kwargs(
        quantity=quantity,
        idempotency_key=idempotency_key,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    quantity: int,
    idempotency_key: Union[Unset, str] = UNSET,
) -> Optional[Union[HTTPValidationError, PurchaseCreditsWithPaymentMethodResponsePurchaseCreditsWithPaymentMethod]]:
    """Purchase Credits With Payment Method

     Purchase credits using existing payment method via Orb auto-collection.

    This endpoint creates an Orb invoice with auto-collection enabled, which will
    charge the customer's default payment method immediately. Unlike /checkout,
    this does not redirect to Stripe - it processes the payment directly through Orb.

    Returns the invoice_id which can be used by the UI to poll Orb for payment status.

    Args:
        quantity: Amount in USD to purchase
        idempotency_key: Optional header to prevent duplicate charges

    Args:
        quantity (int):
        idempotency_key (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PurchaseCreditsWithPaymentMethodResponsePurchaseCreditsWithPaymentMethod]
    """

    return (
        await asyncio_detailed(
            client=client,
            quantity=quantity,
            idempotency_key=idempotency_key,
        )
    ).parsed
