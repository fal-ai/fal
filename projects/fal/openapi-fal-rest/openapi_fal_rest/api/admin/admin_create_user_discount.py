import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.user_discount import UserDiscount
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    user_id: str,
    percent_discount: float,
    starts_at: datetime.datetime,
    ends_at: datetime.datetime,
    is_draft: Union[Unset, bool] = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["user_id"] = user_id

    params["percent_discount"] = percent_discount

    json_starts_at = starts_at.isoformat()
    params["starts_at"] = json_starts_at

    json_ends_at = ends_at.isoformat()
    params["ends_at"] = json_ends_at

    params["is_draft"] = is_draft

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/admin/billing/discounts",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, UserDiscount]]:
    if response.status_code == 200:
        response_200 = UserDiscount.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, UserDiscount]]:
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
    percent_discount: float,
    starts_at: datetime.datetime,
    ends_at: datetime.datetime,
    is_draft: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, UserDiscount]]:
    """Create User Discount

    Args:
        user_id (str):
        percent_discount (float):
        starts_at (datetime.datetime):
        ends_at (datetime.datetime):
        is_draft (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, UserDiscount]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        percent_discount=percent_discount,
        starts_at=starts_at,
        ends_at=ends_at,
        is_draft=is_draft,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
    percent_discount: float,
    starts_at: datetime.datetime,
    ends_at: datetime.datetime,
    is_draft: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, UserDiscount]]:
    """Create User Discount

    Args:
        user_id (str):
        percent_discount (float):
        starts_at (datetime.datetime):
        ends_at (datetime.datetime):
        is_draft (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, UserDiscount]
    """

    return sync_detailed(
        client=client,
        user_id=user_id,
        percent_discount=percent_discount,
        starts_at=starts_at,
        ends_at=ends_at,
        is_draft=is_draft,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
    percent_discount: float,
    starts_at: datetime.datetime,
    ends_at: datetime.datetime,
    is_draft: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, UserDiscount]]:
    """Create User Discount

    Args:
        user_id (str):
        percent_discount (float):
        starts_at (datetime.datetime):
        ends_at (datetime.datetime):
        is_draft (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, UserDiscount]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        percent_discount=percent_discount,
        starts_at=starts_at,
        ends_at=ends_at,
        is_draft=is_draft,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
    percent_discount: float,
    starts_at: datetime.datetime,
    ends_at: datetime.datetime,
    is_draft: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, UserDiscount]]:
    """Create User Discount

    Args:
        user_id (str):
        percent_discount (float):
        starts_at (datetime.datetime):
        ends_at (datetime.datetime):
        is_draft (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, UserDiscount]
    """

    return (
        await asyncio_detailed(
            client=client,
            user_id=user_id,
            percent_discount=percent_discount,
            starts_at=starts_at,
            ends_at=ends_at,
            is_draft=is_draft,
        )
    ).parsed
