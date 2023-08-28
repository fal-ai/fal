from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union

import httpx

from ... import errors
from ...client import Client
from ...models.http_validation_error import HTTPValidationError
from ...models.log_entry import LogEntry
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    client: Client,
    limit: Union[Unset, None, int] = 100,
    offset: Union[Unset, None, int] = 0,
    since: Union[Unset, None, str] = UNSET,
) -> Dict[str, Any]:
    url = "{}/logs/".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["limit"] = limit

    params["offset"] = offset

    params["since"] = since

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
        "params": params,
    }


def _parse_response(
    *, client: Client, response: httpx.Response
) -> Optional[Union[HTTPValidationError, List["LogEntry"]]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = LogEntry.from_dict(response_200_item_data)

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
) -> Response[Union[HTTPValidationError, List["LogEntry"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Client,
    limit: Union[Unset, None, int] = 100,
    offset: Union[Unset, None, int] = 0,
    since: Union[Unset, None, str] = UNSET,
) -> Response[Union[HTTPValidationError, List["LogEntry"]]]:
    """List Logs Since

    Args:
        limit (Union[Unset, None, int]):  Default: 100.
        offset (Union[Unset, None, int]):
        since (Union[Unset, None, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, List['LogEntry']]]
    """

    kwargs = _get_kwargs(
        client=client,
        limit=limit,
        offset=offset,
        since=since,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Client,
    limit: Union[Unset, None, int] = 100,
    offset: Union[Unset, None, int] = 0,
    since: Union[Unset, None, str] = UNSET,
) -> Optional[Union[HTTPValidationError, List["LogEntry"]]]:
    """List Logs Since

    Args:
        limit (Union[Unset, None, int]):  Default: 100.
        offset (Union[Unset, None, int]):
        since (Union[Unset, None, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, List['LogEntry']]
    """

    return sync_detailed(
        client=client,
        limit=limit,
        offset=offset,
        since=since,
    ).parsed


async def asyncio_detailed(
    *,
    client: Client,
    limit: Union[Unset, None, int] = 100,
    offset: Union[Unset, None, int] = 0,
    since: Union[Unset, None, str] = UNSET,
) -> Response[Union[HTTPValidationError, List["LogEntry"]]]:
    """List Logs Since

    Args:
        limit (Union[Unset, None, int]):  Default: 100.
        offset (Union[Unset, None, int]):
        since (Union[Unset, None, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, List['LogEntry']]]
    """

    kwargs = _get_kwargs(
        client=client,
        limit=limit,
        offset=offset,
        since=since,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Client,
    limit: Union[Unset, None, int] = 100,
    offset: Union[Unset, None, int] = 0,
    since: Union[Unset, None, str] = UNSET,
) -> Optional[Union[HTTPValidationError, List["LogEntry"]]]:
    """List Logs Since

    Args:
        limit (Union[Unset, None, int]):  Default: 100.
        offset (Union[Unset, None, int]):
        since (Union[Unset, None, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, List['LogEntry']]
    """

    return (
        await asyncio_detailed(
            client=client,
            limit=limit,
            offset=offset,
            since=since,
        )
    ).parsed
