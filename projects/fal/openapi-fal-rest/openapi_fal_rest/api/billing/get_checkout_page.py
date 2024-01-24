from http import HTTPStatus
from typing import Any, Dict, Optional, Union, cast

import httpx

from ... import errors
from ...client import Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    client: Client,
    quantity: int,
    product: Union[Unset, None, str] = "fal_credits",
    success_url: Union[Unset, None, str] = "https://fal.ai/dashboard/billing",
) -> Dict[str, Any]:
    url = "{}/billing/checkout".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["quantity"] = quantity

    params["product"] = product

    params["success_url"] = success_url

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "method": "post",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
        "params": params,
    }


def _parse_response(*, client: Client, response: httpx.Response) -> Optional[Union[HTTPValidationError, str]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = cast(str, response.json())
        return response_200
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: Client, response: httpx.Response) -> Response[Union[HTTPValidationError, str]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Client,
    quantity: int,
    product: Union[Unset, None, str] = "fal_credits",
    success_url: Union[Unset, None, str] = "https://fal.ai/dashboard/billing",
) -> Response[Union[HTTPValidationError, str]]:
    """Get Checkout Page

    Args:
        quantity (int):
        product (Union[Unset, None, str]):  Default: 'fal_credits'.
        success_url (Union[Unset, None, str]):  Default: 'https://fal.ai/dashboard/billing'.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, str]]
    """

    kwargs = _get_kwargs(
        client=client,
        quantity=quantity,
        product=product,
        success_url=success_url,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Client,
    quantity: int,
    product: Union[Unset, None, str] = "fal_credits",
    success_url: Union[Unset, None, str] = "https://fal.ai/dashboard/billing",
) -> Optional[Union[HTTPValidationError, str]]:
    """Get Checkout Page

    Args:
        quantity (int):
        product (Union[Unset, None, str]):  Default: 'fal_credits'.
        success_url (Union[Unset, None, str]):  Default: 'https://fal.ai/dashboard/billing'.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, str]
    """

    return sync_detailed(
        client=client,
        quantity=quantity,
        product=product,
        success_url=success_url,
    ).parsed


async def asyncio_detailed(
    *,
    client: Client,
    quantity: int,
    product: Union[Unset, None, str] = "fal_credits",
    success_url: Union[Unset, None, str] = "https://fal.ai/dashboard/billing",
) -> Response[Union[HTTPValidationError, str]]:
    """Get Checkout Page

    Args:
        quantity (int):
        product (Union[Unset, None, str]):  Default: 'fal_credits'.
        success_url (Union[Unset, None, str]):  Default: 'https://fal.ai/dashboard/billing'.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, str]]
    """

    kwargs = _get_kwargs(
        client=client,
        quantity=quantity,
        product=product,
        success_url=success_url,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Client,
    quantity: int,
    product: Union[Unset, None, str] = "fal_credits",
    success_url: Union[Unset, None, str] = "https://fal.ai/dashboard/billing",
) -> Optional[Union[HTTPValidationError, str]]:
    """Get Checkout Page

    Args:
        quantity (int):
        product (Union[Unset, None, str]):  Default: 'fal_credits'.
        success_url (Union[Unset, None, str]):  Default: 'https://fal.ai/dashboard/billing'.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, str]
    """

    return (
        await asyncio_detailed(
            client=client,
            quantity=quantity,
            product=product,
            success_url=success_url,
        )
    ).parsed
