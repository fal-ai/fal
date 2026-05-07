import datetime
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
    user_id: str,
    product: str,
    discount_bucket: str,
    effective_date: datetime.date,
    type_: Union[Unset, DiscountBucketType] = UNSET,
    effective_end_date: Union[Unset, datetime.date] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["user_id"] = user_id

    params["product"] = product

    params["discount_bucket"] = discount_bucket

    json_effective_date = effective_date.isoformat()
    params["effective_date"] = json_effective_date

    json_type_: Union[Unset, str] = UNSET
    if not isinstance(type_, Unset):
        json_type_ = type_.value

    params["type"] = json_type_

    json_effective_end_date: Union[Unset, str] = UNSET
    if not isinstance(effective_end_date, Unset):
        json_effective_end_date = effective_end_date.isoformat()
    params["effective_end_date"] = json_effective_end_date

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/admin/discount_bucket",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[CustomerDiscountBucket, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = CustomerDiscountBucket.from_dict(response.json())

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
) -> Response[Union[CustomerDiscountBucket, HTTPValidationError]]:
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
    product: str,
    discount_bucket: str,
    effective_date: datetime.date,
    type_: Union[Unset, DiscountBucketType] = UNSET,
    effective_end_date: Union[Unset, datetime.date] = UNSET,
) -> Response[Union[CustomerDiscountBucket, HTTPValidationError]]:
    """Upsert Discount Bucket

    Args:
        user_id (str):
        product (str):
        discount_bucket (str):
        effective_date (datetime.date):
        type_ (Union[Unset, DiscountBucketType]):
        effective_end_date (Union[Unset, datetime.date]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CustomerDiscountBucket, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        product=product,
        discount_bucket=discount_bucket,
        effective_date=effective_date,
        type_=type_,
        effective_end_date=effective_end_date,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
    product: str,
    discount_bucket: str,
    effective_date: datetime.date,
    type_: Union[Unset, DiscountBucketType] = UNSET,
    effective_end_date: Union[Unset, datetime.date] = UNSET,
) -> Optional[Union[CustomerDiscountBucket, HTTPValidationError]]:
    """Upsert Discount Bucket

    Args:
        user_id (str):
        product (str):
        discount_bucket (str):
        effective_date (datetime.date):
        type_ (Union[Unset, DiscountBucketType]):
        effective_end_date (Union[Unset, datetime.date]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CustomerDiscountBucket, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        user_id=user_id,
        product=product,
        discount_bucket=discount_bucket,
        effective_date=effective_date,
        type_=type_,
        effective_end_date=effective_end_date,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
    product: str,
    discount_bucket: str,
    effective_date: datetime.date,
    type_: Union[Unset, DiscountBucketType] = UNSET,
    effective_end_date: Union[Unset, datetime.date] = UNSET,
) -> Response[Union[CustomerDiscountBucket, HTTPValidationError]]:
    """Upsert Discount Bucket

    Args:
        user_id (str):
        product (str):
        discount_bucket (str):
        effective_date (datetime.date):
        type_ (Union[Unset, DiscountBucketType]):
        effective_end_date (Union[Unset, datetime.date]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CustomerDiscountBucket, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        product=product,
        discount_bucket=discount_bucket,
        effective_date=effective_date,
        type_=type_,
        effective_end_date=effective_end_date,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
    product: str,
    discount_bucket: str,
    effective_date: datetime.date,
    type_: Union[Unset, DiscountBucketType] = UNSET,
    effective_end_date: Union[Unset, datetime.date] = UNSET,
) -> Optional[Union[CustomerDiscountBucket, HTTPValidationError]]:
    """Upsert Discount Bucket

    Args:
        user_id (str):
        product (str):
        discount_bucket (str):
        effective_date (datetime.date):
        type_ (Union[Unset, DiscountBucketType]):
        effective_end_date (Union[Unset, datetime.date]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CustomerDiscountBucket, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            user_id=user_id,
            product=product,
            discount_bucket=discount_bucket,
            effective_date=effective_date,
            type_=type_,
            effective_end_date=effective_end_date,
        )
    ).parsed
