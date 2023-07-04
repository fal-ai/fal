from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.usage_run_detail import UsageRunDetail
from ...types import Response


def _get_kwargs(
    machine_type: str,
    *,
    client: AuthenticatedClient,
) -> Dict[str, Any]:
    url = "{}/usage/machine_type/{machine_type}".format(client.base_url, machine_type=machine_type)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
    }


def _parse_response(
    *, client: Client, response: httpx.Response
) -> Optional[Union[HTTPValidationError, List["UsageRunDetail"]]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = UsageRunDetail.from_dict(response_200_item_data)

            response_200.append(response_200_item)

        return response_200
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Client, response: httpx.Response
) -> Response[Union[HTTPValidationError, List["UsageRunDetail"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    machine_type: str,
    *,
    client: AuthenticatedClient,
) -> Response[Union[HTTPValidationError, List["UsageRunDetail"]]]:
    """Per Machine Usage Details

    Args:
        machine_type (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, List['UsageRunDetail']]]
    """

    kwargs = _get_kwargs(
        machine_type=machine_type,
        client=client,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    machine_type: str,
    *,
    client: AuthenticatedClient,
) -> Optional[Union[HTTPValidationError, List["UsageRunDetail"]]]:
    """Per Machine Usage Details

    Args:
        machine_type (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, List['UsageRunDetail']]
    """

    return sync_detailed(
        machine_type=machine_type,
        client=client,
    ).parsed


async def asyncio_detailed(
    machine_type: str,
    *,
    client: AuthenticatedClient,
) -> Response[Union[HTTPValidationError, List["UsageRunDetail"]]]:
    """Per Machine Usage Details

    Args:
        machine_type (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, List['UsageRunDetail']]]
    """

    kwargs = _get_kwargs(
        machine_type=machine_type,
        client=client,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    machine_type: str,
    *,
    client: AuthenticatedClient,
) -> Optional[Union[HTTPValidationError, List["UsageRunDetail"]]]:
    """Per Machine Usage Details

    Args:
        machine_type (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, List['UsageRunDetail']]
    """

    return (
        await asyncio_detailed(
            machine_type=machine_type,
            client=client,
        )
    ).parsed
