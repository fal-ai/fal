from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.credit_purchase_invoice import CreditPurchaseInvoice
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    user_id: str,
    amount_usd: int,
    effective_date: Union[Unset, str] = UNSET,
    expiration_in_days: Union[Unset, int] = 365,
    invoice_date: Union[Unset, str] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["user_id"] = user_id

    params["amount_usd"] = amount_usd

    params["effective_date"] = effective_date

    params["expiration_in_days"] = expiration_in_days

    params["invoice_date"] = invoice_date

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/admin/users/create_credit_invoice",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[CreditPurchaseInvoice, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = CreditPurchaseInvoice.from_dict(response.json())

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
) -> Response[Union[CreditPurchaseInvoice, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
    amount_usd: int,
    effective_date: Union[Unset, str] = UNSET,
    expiration_in_days: Union[Unset, int] = 365,
    invoice_date: Union[Unset, str] = UNSET,
) -> Response[Union[CreditPurchaseInvoice, HTTPValidationError]]:
    """Create Credit Invoice

    Args:
        user_id (str):
        amount_usd (int):
        effective_date (Union[Unset, str]):
        expiration_in_days (Union[Unset, int]):  Default: 365.
        invoice_date (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CreditPurchaseInvoice, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        amount_usd=amount_usd,
        effective_date=effective_date,
        expiration_in_days=expiration_in_days,
        invoice_date=invoice_date,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
    amount_usd: int,
    effective_date: Union[Unset, str] = UNSET,
    expiration_in_days: Union[Unset, int] = 365,
    invoice_date: Union[Unset, str] = UNSET,
) -> Optional[Union[CreditPurchaseInvoice, HTTPValidationError]]:
    """Create Credit Invoice

    Args:
        user_id (str):
        amount_usd (int):
        effective_date (Union[Unset, str]):
        expiration_in_days (Union[Unset, int]):  Default: 365.
        invoice_date (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CreditPurchaseInvoice, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        user_id=user_id,
        amount_usd=amount_usd,
        effective_date=effective_date,
        expiration_in_days=expiration_in_days,
        invoice_date=invoice_date,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
    amount_usd: int,
    effective_date: Union[Unset, str] = UNSET,
    expiration_in_days: Union[Unset, int] = 365,
    invoice_date: Union[Unset, str] = UNSET,
) -> Response[Union[CreditPurchaseInvoice, HTTPValidationError]]:
    """Create Credit Invoice

    Args:
        user_id (str):
        amount_usd (int):
        effective_date (Union[Unset, str]):
        expiration_in_days (Union[Unset, int]):  Default: 365.
        invoice_date (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CreditPurchaseInvoice, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        amount_usd=amount_usd,
        effective_date=effective_date,
        expiration_in_days=expiration_in_days,
        invoice_date=invoice_date,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
    amount_usd: int,
    effective_date: Union[Unset, str] = UNSET,
    expiration_in_days: Union[Unset, int] = 365,
    invoice_date: Union[Unset, str] = UNSET,
) -> Optional[Union[CreditPurchaseInvoice, HTTPValidationError]]:
    """Create Credit Invoice

    Args:
        user_id (str):
        amount_usd (int):
        effective_date (Union[Unset, str]):
        expiration_in_days (Union[Unset, int]):  Default: 365.
        invoice_date (Union[Unset, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CreditPurchaseInvoice, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            user_id=user_id,
            amount_usd=amount_usd,
            effective_date=effective_date,
            expiration_in_days=expiration_in_days,
            invoice_date=invoice_date,
        )
    ).parsed
