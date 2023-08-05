from http import HTTPStatus
from typing import Any, Dict, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response


def _get_kwargs(
    *,
    client: AuthenticatedClient,
    soft_monthly_budget: int,
    hard_monthly_budget: int,
) -> Dict[str, Any]:
    url = "{}/billing/customer_budget".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["soft_monthly_budget"] = soft_monthly_budget

    params["hard_monthly_budget"] = hard_monthly_budget

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


def _parse_response(*, client: Client, response: httpx.Response) -> Optional[Union[Any, HTTPValidationError]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = cast(Any, response.json())
        return response_200
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: Client, response: httpx.Response) -> Response[Union[Any, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    soft_monthly_budget: int,
    hard_monthly_budget: int,
) -> Response[Union[Any, HTTPValidationError]]:
    """Update Customer Budget

    Args:
        soft_monthly_budget (int):
        hard_monthly_budget (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        client=client,
        soft_monthly_budget=soft_monthly_budget,
        hard_monthly_budget=hard_monthly_budget,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    soft_monthly_budget: int,
    hard_monthly_budget: int,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Update Customer Budget

    Args:
        soft_monthly_budget (int):
        hard_monthly_budget (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        soft_monthly_budget=soft_monthly_budget,
        hard_monthly_budget=hard_monthly_budget,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    soft_monthly_budget: int,
    hard_monthly_budget: int,
) -> Response[Union[Any, HTTPValidationError]]:
    """Update Customer Budget

    Args:
        soft_monthly_budget (int):
        hard_monthly_budget (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        client=client,
        soft_monthly_budget=soft_monthly_budget,
        hard_monthly_budget=hard_monthly_budget,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    soft_monthly_budget: int,
    hard_monthly_budget: int,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Update Customer Budget

    Args:
        soft_monthly_budget (int):
        hard_monthly_budget (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            soft_monthly_budget=soft_monthly_budget,
            hard_monthly_budget=hard_monthly_budget,
        )
    ).parsed
