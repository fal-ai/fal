import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.endpoint_billing_override import EndpointBillingOverride
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    user_id: str,
    endpoint: str,
    billing_unit: str,
    price: float,
    is_draft: Union[Unset, bool] = False,
    use_compute_seconds: Union[Unset, bool] = False,
    discountable: Union[Unset, bool] = False,
    percent_discount: Union[Unset, float] = UNSET,
    start_date: Union[Unset, datetime.datetime] = UNSET,
    end_date: Union[Unset, datetime.datetime] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["user_id"] = user_id

    params["endpoint"] = endpoint

    params["billing_unit"] = billing_unit

    params["price"] = price

    params["is_draft"] = is_draft

    params["use_compute_seconds"] = use_compute_seconds

    params["discountable"] = discountable

    params["percent_discount"] = percent_discount

    json_start_date: Union[Unset, str] = UNSET
    if not isinstance(start_date, Unset):
        json_start_date = start_date.isoformat()
    params["start_date"] = json_start_date

    json_end_date: Union[Unset, str] = UNSET
    if not isinstance(end_date, Unset):
        json_end_date = end_date.isoformat()
    params["end_date"] = json_end_date

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/admin/endpoints/billing_override",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[EndpointBillingOverride, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = EndpointBillingOverride.from_dict(response.json())

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
) -> Response[Union[EndpointBillingOverride, HTTPValidationError]]:
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
    endpoint: str,
    billing_unit: str,
    price: float,
    is_draft: Union[Unset, bool] = False,
    use_compute_seconds: Union[Unset, bool] = False,
    discountable: Union[Unset, bool] = False,
    percent_discount: Union[Unset, float] = UNSET,
    start_date: Union[Unset, datetime.datetime] = UNSET,
    end_date: Union[Unset, datetime.datetime] = UNSET,
) -> Response[Union[EndpointBillingOverride, HTTPValidationError]]:
    """Upsert Endpoint Billing Override

    Args:
        user_id (str):
        endpoint (str):
        billing_unit (str):
        price (float):
        is_draft (Union[Unset, bool]):  Default: False.
        use_compute_seconds (Union[Unset, bool]):  Default: False.
        discountable (Union[Unset, bool]):  Default: False.
        percent_discount (Union[Unset, float]):
        start_date (Union[Unset, datetime.datetime]):
        end_date (Union[Unset, datetime.datetime]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[EndpointBillingOverride, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        endpoint=endpoint,
        billing_unit=billing_unit,
        price=price,
        is_draft=is_draft,
        use_compute_seconds=use_compute_seconds,
        discountable=discountable,
        percent_discount=percent_discount,
        start_date=start_date,
        end_date=end_date,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
    endpoint: str,
    billing_unit: str,
    price: float,
    is_draft: Union[Unset, bool] = False,
    use_compute_seconds: Union[Unset, bool] = False,
    discountable: Union[Unset, bool] = False,
    percent_discount: Union[Unset, float] = UNSET,
    start_date: Union[Unset, datetime.datetime] = UNSET,
    end_date: Union[Unset, datetime.datetime] = UNSET,
) -> Optional[Union[EndpointBillingOverride, HTTPValidationError]]:
    """Upsert Endpoint Billing Override

    Args:
        user_id (str):
        endpoint (str):
        billing_unit (str):
        price (float):
        is_draft (Union[Unset, bool]):  Default: False.
        use_compute_seconds (Union[Unset, bool]):  Default: False.
        discountable (Union[Unset, bool]):  Default: False.
        percent_discount (Union[Unset, float]):
        start_date (Union[Unset, datetime.datetime]):
        end_date (Union[Unset, datetime.datetime]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[EndpointBillingOverride, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        user_id=user_id,
        endpoint=endpoint,
        billing_unit=billing_unit,
        price=price,
        is_draft=is_draft,
        use_compute_seconds=use_compute_seconds,
        discountable=discountable,
        percent_discount=percent_discount,
        start_date=start_date,
        end_date=end_date,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
    endpoint: str,
    billing_unit: str,
    price: float,
    is_draft: Union[Unset, bool] = False,
    use_compute_seconds: Union[Unset, bool] = False,
    discountable: Union[Unset, bool] = False,
    percent_discount: Union[Unset, float] = UNSET,
    start_date: Union[Unset, datetime.datetime] = UNSET,
    end_date: Union[Unset, datetime.datetime] = UNSET,
) -> Response[Union[EndpointBillingOverride, HTTPValidationError]]:
    """Upsert Endpoint Billing Override

    Args:
        user_id (str):
        endpoint (str):
        billing_unit (str):
        price (float):
        is_draft (Union[Unset, bool]):  Default: False.
        use_compute_seconds (Union[Unset, bool]):  Default: False.
        discountable (Union[Unset, bool]):  Default: False.
        percent_discount (Union[Unset, float]):
        start_date (Union[Unset, datetime.datetime]):
        end_date (Union[Unset, datetime.datetime]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[EndpointBillingOverride, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        endpoint=endpoint,
        billing_unit=billing_unit,
        price=price,
        is_draft=is_draft,
        use_compute_seconds=use_compute_seconds,
        discountable=discountable,
        percent_discount=percent_discount,
        start_date=start_date,
        end_date=end_date,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
    endpoint: str,
    billing_unit: str,
    price: float,
    is_draft: Union[Unset, bool] = False,
    use_compute_seconds: Union[Unset, bool] = False,
    discountable: Union[Unset, bool] = False,
    percent_discount: Union[Unset, float] = UNSET,
    start_date: Union[Unset, datetime.datetime] = UNSET,
    end_date: Union[Unset, datetime.datetime] = UNSET,
) -> Optional[Union[EndpointBillingOverride, HTTPValidationError]]:
    """Upsert Endpoint Billing Override

    Args:
        user_id (str):
        endpoint (str):
        billing_unit (str):
        price (float):
        is_draft (Union[Unset, bool]):  Default: False.
        use_compute_seconds (Union[Unset, bool]):  Default: False.
        discountable (Union[Unset, bool]):  Default: False.
        percent_discount (Union[Unset, float]):
        start_date (Union[Unset, datetime.datetime]):
        end_date (Union[Unset, datetime.datetime]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[EndpointBillingOverride, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            user_id=user_id,
            endpoint=endpoint,
            billing_unit=billing_unit,
            price=price,
            is_draft=is_draft,
            use_compute_seconds=use_compute_seconds,
            discountable=discountable,
            percent_discount=percent_discount,
            start_date=start_date,
            end_date=end_date,
        )
    ).parsed
