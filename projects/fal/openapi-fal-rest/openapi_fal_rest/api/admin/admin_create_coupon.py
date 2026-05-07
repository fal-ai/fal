import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.credit_coupon_info import CreditCouponInfo
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    coupon_id: str,
    claim_limit: int,
    amount_dollars: int,
    start_date: Union[Unset, datetime.datetime] = UNSET,
    end_date: Union[Unset, datetime.datetime] = UNSET,
    expires_in_days: Union[Unset, int] = 90,
    skip_check: Union[Unset, bool] = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["coupon_id"] = coupon_id

    params["claim_limit"] = claim_limit

    params["amount_dollars"] = amount_dollars

    json_start_date: Union[Unset, str] = UNSET
    if not isinstance(start_date, Unset):
        json_start_date = start_date.isoformat()
    params["start_date"] = json_start_date

    json_end_date: Union[Unset, str] = UNSET
    if not isinstance(end_date, Unset):
        json_end_date = end_date.isoformat()
    params["end_date"] = json_end_date

    params["expires_in_days"] = expires_in_days

    params["skip_check"] = skip_check

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/admin/coupons",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[CreditCouponInfo, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = CreditCouponInfo.from_dict(response.json())

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
) -> Response[Union[CreditCouponInfo, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    coupon_id: str,
    claim_limit: int,
    amount_dollars: int,
    start_date: Union[Unset, datetime.datetime] = UNSET,
    end_date: Union[Unset, datetime.datetime] = UNSET,
    expires_in_days: Union[Unset, int] = 90,
    skip_check: Union[Unset, bool] = False,
) -> Response[Union[CreditCouponInfo, HTTPValidationError]]:
    """Create Coupons

    Args:
        coupon_id (str):
        claim_limit (int):
        amount_dollars (int):
        start_date (Union[Unset, datetime.datetime]):
        end_date (Union[Unset, datetime.datetime]):
        expires_in_days (Union[Unset, int]):  Default: 90.
        skip_check (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CreditCouponInfo, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        coupon_id=coupon_id,
        claim_limit=claim_limit,
        amount_dollars=amount_dollars,
        start_date=start_date,
        end_date=end_date,
        expires_in_days=expires_in_days,
        skip_check=skip_check,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    coupon_id: str,
    claim_limit: int,
    amount_dollars: int,
    start_date: Union[Unset, datetime.datetime] = UNSET,
    end_date: Union[Unset, datetime.datetime] = UNSET,
    expires_in_days: Union[Unset, int] = 90,
    skip_check: Union[Unset, bool] = False,
) -> Optional[Union[CreditCouponInfo, HTTPValidationError]]:
    """Create Coupons

    Args:
        coupon_id (str):
        claim_limit (int):
        amount_dollars (int):
        start_date (Union[Unset, datetime.datetime]):
        end_date (Union[Unset, datetime.datetime]):
        expires_in_days (Union[Unset, int]):  Default: 90.
        skip_check (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CreditCouponInfo, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        coupon_id=coupon_id,
        claim_limit=claim_limit,
        amount_dollars=amount_dollars,
        start_date=start_date,
        end_date=end_date,
        expires_in_days=expires_in_days,
        skip_check=skip_check,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    coupon_id: str,
    claim_limit: int,
    amount_dollars: int,
    start_date: Union[Unset, datetime.datetime] = UNSET,
    end_date: Union[Unset, datetime.datetime] = UNSET,
    expires_in_days: Union[Unset, int] = 90,
    skip_check: Union[Unset, bool] = False,
) -> Response[Union[CreditCouponInfo, HTTPValidationError]]:
    """Create Coupons

    Args:
        coupon_id (str):
        claim_limit (int):
        amount_dollars (int):
        start_date (Union[Unset, datetime.datetime]):
        end_date (Union[Unset, datetime.datetime]):
        expires_in_days (Union[Unset, int]):  Default: 90.
        skip_check (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CreditCouponInfo, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        coupon_id=coupon_id,
        claim_limit=claim_limit,
        amount_dollars=amount_dollars,
        start_date=start_date,
        end_date=end_date,
        expires_in_days=expires_in_days,
        skip_check=skip_check,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    coupon_id: str,
    claim_limit: int,
    amount_dollars: int,
    start_date: Union[Unset, datetime.datetime] = UNSET,
    end_date: Union[Unset, datetime.datetime] = UNSET,
    expires_in_days: Union[Unset, int] = 90,
    skip_check: Union[Unset, bool] = False,
) -> Optional[Union[CreditCouponInfo, HTTPValidationError]]:
    """Create Coupons

    Args:
        coupon_id (str):
        claim_limit (int):
        amount_dollars (int):
        start_date (Union[Unset, datetime.datetime]):
        end_date (Union[Unset, datetime.datetime]):
        expires_in_days (Union[Unset, int]):  Default: 90.
        skip_check (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CreditCouponInfo, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            coupon_id=coupon_id,
            claim_limit=claim_limit,
            amount_dollars=amount_dollars,
            start_date=start_date,
            end_date=end_date,
            expires_in_days=expires_in_days,
            skip_check=skip_check,
        )
    ).parsed
