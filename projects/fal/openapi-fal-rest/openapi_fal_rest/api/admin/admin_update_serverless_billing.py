from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    machine_type: str,
    *,
    unit_price: Union[Unset, float] = UNSET,
    is_visible: Union[Unset, bool] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["unit_price"] = unit_price

    params["is_visible"] = is_visible

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "patch",
        "url": f"/admin/serverless/billing/{machine_type}",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, str]]:
    if response.status_code == 200:
        response_200 = cast(str, response.json())
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
) -> Response[Union[HTTPValidationError, str]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    machine_type: str,
    *,
    client: Union[AuthenticatedClient, Client],
    unit_price: Union[Unset, float] = UNSET,
    is_visible: Union[Unset, bool] = UNSET,
) -> Response[Union[HTTPValidationError, str]]:
    """Update Serverless Billing

    Args:
        machine_type (str):
        unit_price (Union[Unset, float]):
        is_visible (Union[Unset, bool]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, str]]
    """

    kwargs = _get_kwargs(
        machine_type=machine_type,
        unit_price=unit_price,
        is_visible=is_visible,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    machine_type: str,
    *,
    client: Union[AuthenticatedClient, Client],
    unit_price: Union[Unset, float] = UNSET,
    is_visible: Union[Unset, bool] = UNSET,
) -> Optional[Union[HTTPValidationError, str]]:
    """Update Serverless Billing

    Args:
        machine_type (str):
        unit_price (Union[Unset, float]):
        is_visible (Union[Unset, bool]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, str]
    """

    return sync_detailed(
        machine_type=machine_type,
        client=client,
        unit_price=unit_price,
        is_visible=is_visible,
    ).parsed


async def asyncio_detailed(
    machine_type: str,
    *,
    client: Union[AuthenticatedClient, Client],
    unit_price: Union[Unset, float] = UNSET,
    is_visible: Union[Unset, bool] = UNSET,
) -> Response[Union[HTTPValidationError, str]]:
    """Update Serverless Billing

    Args:
        machine_type (str):
        unit_price (Union[Unset, float]):
        is_visible (Union[Unset, bool]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, str]]
    """

    kwargs = _get_kwargs(
        machine_type=machine_type,
        unit_price=unit_price,
        is_visible=is_visible,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    machine_type: str,
    *,
    client: Union[AuthenticatedClient, Client],
    unit_price: Union[Unset, float] = UNSET,
    is_visible: Union[Unset, bool] = UNSET,
) -> Optional[Union[HTTPValidationError, str]]:
    """Update Serverless Billing

    Args:
        machine_type (str):
        unit_price (Union[Unset, float]):
        is_visible (Union[Unset, bool]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, str]
    """

    return (
        await asyncio_detailed(
            machine_type=machine_type,
            client=client,
            unit_price=unit_price,
            is_visible=is_visible,
        )
    ).parsed
