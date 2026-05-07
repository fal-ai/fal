import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.stats_timeframe import StatsTimeframe
from ...models.user_concurrency_history import UserConcurrencyHistory
from ...types import UNSET, Response


def _get_kwargs(
    user_id: str,
    *,
    timeframe: StatsTimeframe,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_timeframe = timeframe.value
    params["timeframe"] = json_timeframe

    json_start_time = start_time.isoformat()
    params["start_time"] = json_start_time

    json_end_time = end_time.isoformat()
    params["end_time"] = json_end_time

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/admin/users/{user_id}/concurrency/history",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, UserConcurrencyHistory]]:
    if response.status_code == 200:
        response_200 = UserConcurrencyHistory.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, UserConcurrencyHistory]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    user_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    timeframe: StatsTimeframe,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> Response[Union[HTTPValidationError, UserConcurrencyHistory]]:
    """Admin Get User Concurrency History

     Admin endpoint to get the historical concurrency data for a specific user.

    Args:
        user_id (str):
        timeframe (StatsTimeframe):
        start_time (datetime.datetime):
        end_time (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, UserConcurrencyHistory]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    user_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    timeframe: StatsTimeframe,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> Optional[Union[HTTPValidationError, UserConcurrencyHistory]]:
    """Admin Get User Concurrency History

     Admin endpoint to get the historical concurrency data for a specific user.

    Args:
        user_id (str):
        timeframe (StatsTimeframe):
        start_time (datetime.datetime):
        end_time (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, UserConcurrencyHistory]
    """

    return sync_detailed(
        user_id=user_id,
        client=client,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
    ).parsed


async def asyncio_detailed(
    user_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    timeframe: StatsTimeframe,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> Response[Union[HTTPValidationError, UserConcurrencyHistory]]:
    """Admin Get User Concurrency History

     Admin endpoint to get the historical concurrency data for a specific user.

    Args:
        user_id (str):
        timeframe (StatsTimeframe):
        start_time (datetime.datetime):
        end_time (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, UserConcurrencyHistory]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    timeframe: StatsTimeframe,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> Optional[Union[HTTPValidationError, UserConcurrencyHistory]]:
    """Admin Get User Concurrency History

     Admin endpoint to get the historical concurrency data for a specific user.

    Args:
        user_id (str):
        timeframe (StatsTimeframe):
        start_time (datetime.datetime):
        end_time (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, UserConcurrencyHistory]
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            client=client,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
        )
    ).parsed
