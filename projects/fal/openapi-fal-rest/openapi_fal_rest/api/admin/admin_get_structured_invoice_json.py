from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.structured_invoice_data import StructuredInvoiceData
from ...types import Response


def _get_kwargs(
    invoice_id: str,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/admin/billing/invoice/{invoice_id}/json",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, StructuredInvoiceData]]:
    if response.status_code == 200:
        response_200 = StructuredInvoiceData.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, StructuredInvoiceData]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    invoice_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[HTTPValidationError, StructuredInvoiceData]]:
    """Admin Get Structured Invoice Json

     Get structured invoice data in JSON format (admin endpoint).

    This endpoint returns a hierarchical JSON structure with:
    - Invoice metadata
    - Sections grouped and merged (Compute Seconds, Endpoint Output, etc.)
    - API key subsections with totals and database enrichment (alias, owner)
    - Playground usage breakdown by user with enrichment (nickname, email)
    - Platform discount tracking
    - Summary with total platform discount

    Admin endpoint - no user ownership verification required.

    Args:
        invoice_id (str): Orb invoice ID

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, StructuredInvoiceData]]
    """

    kwargs = _get_kwargs(
        invoice_id=invoice_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    invoice_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[HTTPValidationError, StructuredInvoiceData]]:
    """Admin Get Structured Invoice Json

     Get structured invoice data in JSON format (admin endpoint).

    This endpoint returns a hierarchical JSON structure with:
    - Invoice metadata
    - Sections grouped and merged (Compute Seconds, Endpoint Output, etc.)
    - API key subsections with totals and database enrichment (alias, owner)
    - Playground usage breakdown by user with enrichment (nickname, email)
    - Platform discount tracking
    - Summary with total platform discount

    Admin endpoint - no user ownership verification required.

    Args:
        invoice_id (str): Orb invoice ID

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, StructuredInvoiceData]
    """

    return sync_detailed(
        invoice_id=invoice_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    invoice_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[HTTPValidationError, StructuredInvoiceData]]:
    """Admin Get Structured Invoice Json

     Get structured invoice data in JSON format (admin endpoint).

    This endpoint returns a hierarchical JSON structure with:
    - Invoice metadata
    - Sections grouped and merged (Compute Seconds, Endpoint Output, etc.)
    - API key subsections with totals and database enrichment (alias, owner)
    - Playground usage breakdown by user with enrichment (nickname, email)
    - Platform discount tracking
    - Summary with total platform discount

    Admin endpoint - no user ownership verification required.

    Args:
        invoice_id (str): Orb invoice ID

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, StructuredInvoiceData]]
    """

    kwargs = _get_kwargs(
        invoice_id=invoice_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    invoice_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[HTTPValidationError, StructuredInvoiceData]]:
    """Admin Get Structured Invoice Json

     Get structured invoice data in JSON format (admin endpoint).

    This endpoint returns a hierarchical JSON structure with:
    - Invoice metadata
    - Sections grouped and merged (Compute Seconds, Endpoint Output, etc.)
    - API key subsections with totals and database enrichment (alias, owner)
    - Playground usage breakdown by user with enrichment (nickname, email)
    - Platform discount tracking
    - Summary with total platform discount

    Admin endpoint - no user ownership verification required.

    Args:
        invoice_id (str): Orb invoice ID

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, StructuredInvoiceData]
    """

    return (
        await asyncio_detailed(
            invoice_id=invoice_id,
            client=client,
        )
    ).parsed
