import datetime
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union

import httpx

from ... import errors
from ...client import Client
from ...models.http_validation_error import HTTPValidationError
from ...models.usage_per_user import UsagePerUser
from ...types import UNSET, Response


def _get_kwargs(
    *,
    client: Client,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> Dict[str, Any]:
    url = "{}/admin/users/usage".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    json_start_time = start_time.isoformat()

    params["start_time"] = json_start_time

    json_end_time = end_time.isoformat()

    params["end_time"] = json_end_time

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
) -> Optional[Union[HTTPValidationError, List["UsagePerUser"]]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = UsagePerUser.from_dict(response_200_item_data)

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
) -> Response[Union[HTTPValidationError, List["UsagePerUser"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Client,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> Response[Union[HTTPValidationError, List["UsagePerUser"]]]:
    """Get Usage Per User

    Args:
        start_time (datetime.datetime):
        end_time (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, List['UsagePerUser']]]
    """

    kwargs = _get_kwargs(
        client=client,
        start_time=start_time,
        end_time=end_time,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Client,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> Optional[Union[HTTPValidationError, List["UsagePerUser"]]]:
    """Get Usage Per User

    Args:
        start_time (datetime.datetime):
        end_time (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, List['UsagePerUser']]
    """

    return sync_detailed(
        client=client,
        start_time=start_time,
        end_time=end_time,
    ).parsed


async def asyncio_detailed(
    *,
    client: Client,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> Response[Union[HTTPValidationError, List["UsagePerUser"]]]:
    """Get Usage Per User

    Args:
        start_time (datetime.datetime):
        end_time (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, List['UsagePerUser']]]
    """

    kwargs = _get_kwargs(
        client=client,
        start_time=start_time,
        end_time=end_time,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Client,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> Optional[Union[HTTPValidationError, List["UsagePerUser"]]]:
    """Get Usage Per User

    Args:
        start_time (datetime.datetime):
        end_time (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, List['UsagePerUser']]
    """

    return (
        await asyncio_detailed(
            client=client,
            start_time=start_time,
            end_time=end_time,
        )
    ).parsed
