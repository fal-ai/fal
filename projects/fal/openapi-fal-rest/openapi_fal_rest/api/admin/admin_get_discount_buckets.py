from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.customer_discount_bucket import CustomerDiscountBucket
from ...models.discount_bucket_type import DiscountBucketType
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    user_id: Union[Unset, str] = UNSET,
    type_: Union[Unset, DiscountBucketType] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["user_id"] = user_id

    json_type_: Union[Unset, str] = UNSET
    if not isinstance(type_, Unset):
        json_type_ = type_.value

    params["type"] = json_type_

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/admin/discount_buckets",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list["CustomerDiscountBucket"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = CustomerDiscountBucket.from_dict(response_200_item_data)

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
) -> Response[Union[HTTPValidationError, list["CustomerDiscountBucket"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: Union[Unset, str] = UNSET,
    type_: Union[Unset, DiscountBucketType] = UNSET,
) -> Response[Union[HTTPValidationError, list["CustomerDiscountBucket"]]]:
    """Get Discount Buckets

    Args:
        user_id (Union[Unset, str]):
        type_ (Union[Unset, DiscountBucketType]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['CustomerDiscountBucket']]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        type_=type_,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: Union[Unset, str] = UNSET,
    type_: Union[Unset, DiscountBucketType] = UNSET,
) -> Optional[Union[HTTPValidationError, list["CustomerDiscountBucket"]]]:
    """Get Discount Buckets

    Args:
        user_id (Union[Unset, str]):
        type_ (Union[Unset, DiscountBucketType]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['CustomerDiscountBucket']]
    """

    return sync_detailed(
        client=client,
        user_id=user_id,
        type_=type_,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: Union[Unset, str] = UNSET,
    type_: Union[Unset, DiscountBucketType] = UNSET,
) -> Response[Union[HTTPValidationError, list["CustomerDiscountBucket"]]]:
    """Get Discount Buckets

    Args:
        user_id (Union[Unset, str]):
        type_ (Union[Unset, DiscountBucketType]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['CustomerDiscountBucket']]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        type_=type_,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: Union[Unset, str] = UNSET,
    type_: Union[Unset, DiscountBucketType] = UNSET,
) -> Optional[Union[HTTPValidationError, list["CustomerDiscountBucket"]]]:
    """Get Discount Buckets

    Args:
        user_id (Union[Unset, str]):
        type_ (Union[Unset, DiscountBucketType]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['CustomerDiscountBucket']]
    """

    return (
        await asyncio_detailed(
            client=client,
            user_id=user_id,
            type_=type_,
        )
    ).parsed
