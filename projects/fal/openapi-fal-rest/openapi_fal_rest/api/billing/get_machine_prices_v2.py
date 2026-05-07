from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.machine_type_price import MachineTypePrice
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    machine_types: Union[Unset, list[str]] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_machine_types: Union[Unset, list[str]] = UNSET
    if not isinstance(machine_types, Unset):
        json_machine_types = machine_types

    params["machine_types"] = json_machine_types

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/billing/v2/machine_prices",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list["MachineTypePrice"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = MachineTypePrice.from_dict(response_200_item_data)

            response_200.append(response_200_item)

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
) -> Response[Union[HTTPValidationError, list["MachineTypePrice"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    machine_types: Union[Unset, list[str]] = UNSET,
) -> Response[Union[HTTPValidationError, list["MachineTypePrice"]]]:
    """Get Machine Prices V2

     Return prices for the specified machine types.

    This endpoint returns machine type prices with user-specific discounts applied.
    If no machine types are specified, returns prices for all visible machine types.

    Args:
        machine_types (Union[Unset, list[str]]): List of machine types to fetch prices for

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['MachineTypePrice']]]
    """

    kwargs = _get_kwargs(
        machine_types=machine_types,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    machine_types: Union[Unset, list[str]] = UNSET,
) -> Optional[Union[HTTPValidationError, list["MachineTypePrice"]]]:
    """Get Machine Prices V2

     Return prices for the specified machine types.

    This endpoint returns machine type prices with user-specific discounts applied.
    If no machine types are specified, returns prices for all visible machine types.

    Args:
        machine_types (Union[Unset, list[str]]): List of machine types to fetch prices for

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['MachineTypePrice']]
    """

    return sync_detailed(
        client=client,
        machine_types=machine_types,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    machine_types: Union[Unset, list[str]] = UNSET,
) -> Response[Union[HTTPValidationError, list["MachineTypePrice"]]]:
    """Get Machine Prices V2

     Return prices for the specified machine types.

    This endpoint returns machine type prices with user-specific discounts applied.
    If no machine types are specified, returns prices for all visible machine types.

    Args:
        machine_types (Union[Unset, list[str]]): List of machine types to fetch prices for

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['MachineTypePrice']]]
    """

    kwargs = _get_kwargs(
        machine_types=machine_types,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    machine_types: Union[Unset, list[str]] = UNSET,
) -> Optional[Union[HTTPValidationError, list["MachineTypePrice"]]]:
    """Get Machine Prices V2

     Return prices for the specified machine types.

    This endpoint returns machine type prices with user-specific discounts applied.
    If no machine types are specified, returns prices for all visible machine types.

    Args:
        machine_types (Union[Unset, list[str]]): List of machine types to fetch prices for

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['MachineTypePrice']]
    """

    return (
        await asyncio_detailed(
            client=client,
            machine_types=machine_types,
        )
    ).parsed
