from http import HTTPStatus
from typing import Any, Dict, Optional, Union, cast

import httpx

from ... import errors
from ...client import Client
from ...models.billing_type import BillingType
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response


def _get_kwargs(
    *,
    client: Client,
    user_id: str,
    billing_type: BillingType,
) -> Dict[str, Any]:
    url = "{}/admin/users/set_billing_type".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["user_id"] = user_id

    json_billing_type = billing_type.value

    params["billing_type"] = json_billing_type

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


def _parse_response(*, client: Client, response: httpx.Response) -> Optional[Union[HTTPValidationError, bool]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = cast(bool, response.json())
        return response_200
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: Client, response: httpx.Response) -> Response[Union[HTTPValidationError, bool]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Client,
    user_id: str,
    billing_type: BillingType,
) -> Response[Union[HTTPValidationError, bool]]:
    """Set Billing Type

    Args:
        user_id (str):
        billing_type (BillingType): An enumeration.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, bool]]
    """

    kwargs = _get_kwargs(
        client=client,
        user_id=user_id,
        billing_type=billing_type,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Client,
    user_id: str,
    billing_type: BillingType,
) -> Optional[Union[HTTPValidationError, bool]]:
    """Set Billing Type

    Args:
        user_id (str):
        billing_type (BillingType): An enumeration.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, bool]
    """

    return sync_detailed(
        client=client,
        user_id=user_id,
        billing_type=billing_type,
    ).parsed


async def asyncio_detailed(
    *,
    client: Client,
    user_id: str,
    billing_type: BillingType,
) -> Response[Union[HTTPValidationError, bool]]:
    """Set Billing Type

    Args:
        user_id (str):
        billing_type (BillingType): An enumeration.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, bool]]
    """

    kwargs = _get_kwargs(
        client=client,
        user_id=user_id,
        billing_type=billing_type,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Client,
    user_id: str,
    billing_type: BillingType,
) -> Optional[Union[HTTPValidationError, bool]]:
    """Set Billing Type

    Args:
        user_id (str):
        billing_type (BillingType): An enumeration.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, bool]
    """

    return (
        await asyncio_detailed(
            client=client,
            user_id=user_id,
            billing_type=billing_type,
        )
    ).parsed
