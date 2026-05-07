from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    invoice_id: str,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/admin/billing/invoice/{invoice_id}/csv",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = response.json()
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
) -> Response[Union[Any, HTTPValidationError]]:
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
) -> Response[Union[Any, HTTPValidationError]]:
    r"""Admin Get Structured Invoice Csv

     Get structured invoice data in CSV format (admin endpoint).

    This endpoint returns a CSV file with columns:
    - section: Main charge category
    - subsection_type: API Key, Playground, or empty
    - subsection_id: API key ID or \"playground\"
    - user_id: User ID for playground usage
    - description: Full item description
    - endpoint: Extracted endpoint name
    - quantity: Usage quantity
    - amount: Charge amount
    - platform_discount: \"Yes\" or \"No\"
    - subsection_total: Total for the subsection

    Admin endpoint - no user ownership verification required.

    Args:
        invoice_id (str): Orb invoice ID

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
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
) -> Optional[Union[Any, HTTPValidationError]]:
    r"""Admin Get Structured Invoice Csv

     Get structured invoice data in CSV format (admin endpoint).

    This endpoint returns a CSV file with columns:
    - section: Main charge category
    - subsection_type: API Key, Playground, or empty
    - subsection_id: API key ID or \"playground\"
    - user_id: User ID for playground usage
    - description: Full item description
    - endpoint: Extracted endpoint name
    - quantity: Usage quantity
    - amount: Charge amount
    - platform_discount: \"Yes\" or \"No\"
    - subsection_total: Total for the subsection

    Admin endpoint - no user ownership verification required.

    Args:
        invoice_id (str): Orb invoice ID

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        invoice_id=invoice_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    invoice_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[Any, HTTPValidationError]]:
    r"""Admin Get Structured Invoice Csv

     Get structured invoice data in CSV format (admin endpoint).

    This endpoint returns a CSV file with columns:
    - section: Main charge category
    - subsection_type: API Key, Playground, or empty
    - subsection_id: API key ID or \"playground\"
    - user_id: User ID for playground usage
    - description: Full item description
    - endpoint: Extracted endpoint name
    - quantity: Usage quantity
    - amount: Charge amount
    - platform_discount: \"Yes\" or \"No\"
    - subsection_total: Total for the subsection

    Admin endpoint - no user ownership verification required.

    Args:
        invoice_id (str): Orb invoice ID

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
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
) -> Optional[Union[Any, HTTPValidationError]]:
    r"""Admin Get Structured Invoice Csv

     Get structured invoice data in CSV format (admin endpoint).

    This endpoint returns a CSV file with columns:
    - section: Main charge category
    - subsection_type: API Key, Playground, or empty
    - subsection_id: API key ID or \"playground\"
    - user_id: User ID for playground usage
    - description: Full item description
    - endpoint: Extracted endpoint name
    - quantity: Usage quantity
    - amount: Charge amount
    - platform_discount: \"Yes\" or \"No\"
    - subsection_total: Total for the subsection

    Admin endpoint - no user ownership verification required.

    Args:
        invoice_id (str): Orb invoice ID

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            invoice_id=invoice_id,
            client=client,
        )
    ).parsed
