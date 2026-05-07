from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.default_plan_response import DefaultPlanResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    product_slug: str,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/vercel/v1/products/{product_slug}/plans",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[DefaultPlanResponse, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = DefaultPlanResponse.from_dict(response.json())

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
) -> Response[Union[DefaultPlanResponse, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    product_slug: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[DefaultPlanResponse, HTTPValidationError]]:
    """Get Product Plans

    Args:
        product_slug (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[DefaultPlanResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        product_slug=product_slug,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    product_slug: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[DefaultPlanResponse, HTTPValidationError]]:
    """Get Product Plans

    Args:
        product_slug (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[DefaultPlanResponse, HTTPValidationError]
    """

    return sync_detailed(
        product_slug=product_slug,
        client=client,
    ).parsed


async def asyncio_detailed(
    product_slug: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[DefaultPlanResponse, HTTPValidationError]]:
    """Get Product Plans

    Args:
        product_slug (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[DefaultPlanResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        product_slug=product_slug,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    product_slug: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[DefaultPlanResponse, HTTPValidationError]]:
    """Get Product Plans

    Args:
        product_slug (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[DefaultPlanResponse, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            product_slug=product_slug,
            client=client,
        )
    ).parsed
