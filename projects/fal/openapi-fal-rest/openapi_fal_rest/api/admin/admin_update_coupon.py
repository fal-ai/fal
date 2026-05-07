from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.credit_coupon_info import CreditCouponInfo
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    coupon_id: str,
    *,
    body: CreditCouponInfo,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": f"/admin/coupons/{coupon_id}",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
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
    coupon_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: CreditCouponInfo,
) -> Response[Union[CreditCouponInfo, HTTPValidationError]]:
    """Update Coupon

    Args:
        coupon_id (str):
        body (CreditCouponInfo):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CreditCouponInfo, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        coupon_id=coupon_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    coupon_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: CreditCouponInfo,
) -> Optional[Union[CreditCouponInfo, HTTPValidationError]]:
    """Update Coupon

    Args:
        coupon_id (str):
        body (CreditCouponInfo):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CreditCouponInfo, HTTPValidationError]
    """

    return sync_detailed(
        coupon_id=coupon_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    coupon_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: CreditCouponInfo,
) -> Response[Union[CreditCouponInfo, HTTPValidationError]]:
    """Update Coupon

    Args:
        coupon_id (str):
        body (CreditCouponInfo):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CreditCouponInfo, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        coupon_id=coupon_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    coupon_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: CreditCouponInfo,
) -> Optional[Union[CreditCouponInfo, HTTPValidationError]]:
    """Update Coupon

    Args:
        coupon_id (str):
        body (CreditCouponInfo):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CreditCouponInfo, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            coupon_id=coupon_id,
            client=client,
            body=body,
        )
    ).parsed
