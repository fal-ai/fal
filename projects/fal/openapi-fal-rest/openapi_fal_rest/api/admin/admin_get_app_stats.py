import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.gateway_usage_stats import GatewayUsageStats
from ...models.http_validation_error import HTTPValidationError
from ...models.stats_timeframe import StatsTimeframe
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    timeframe: StatsTimeframe,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    user_id: Union[Unset, str] = UNSET,
    app_id: str,
    subpaths: Union[Unset, list[str]] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_timeframe = timeframe.value
    params["timeframe"] = json_timeframe

    json_start_time = start_time.isoformat()
    params["start_time"] = json_start_time

    json_end_time = end_time.isoformat()
    params["end_time"] = json_end_time

    params["user_id"] = user_id

    params["app_id"] = app_id

    json_subpaths: Union[Unset, list[str]] = UNSET
    if not isinstance(subpaths, Unset):
        json_subpaths = subpaths

    params["subpaths"] = json_subpaths

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/admin/stats/app",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[GatewayUsageStats, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = GatewayUsageStats.from_dict(response.json())

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
) -> Response[Union[GatewayUsageStats, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    timeframe: StatsTimeframe,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    user_id: Union[Unset, str] = UNSET,
    app_id: str,
    subpaths: Union[Unset, list[str]] = UNSET,
) -> Response[Union[GatewayUsageStats, HTTPValidationError]]:
    """Get App Stats

    Args:
        timeframe (StatsTimeframe):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        user_id (Union[Unset, str]):
        app_id (str):
        subpaths (Union[Unset, list[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[GatewayUsageStats, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
        user_id=user_id,
        app_id=app_id,
        subpaths=subpaths,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    timeframe: StatsTimeframe,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    user_id: Union[Unset, str] = UNSET,
    app_id: str,
    subpaths: Union[Unset, list[str]] = UNSET,
) -> Optional[Union[GatewayUsageStats, HTTPValidationError]]:
    """Get App Stats

    Args:
        timeframe (StatsTimeframe):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        user_id (Union[Unset, str]):
        app_id (str):
        subpaths (Union[Unset, list[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[GatewayUsageStats, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
        user_id=user_id,
        app_id=app_id,
        subpaths=subpaths,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    timeframe: StatsTimeframe,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    user_id: Union[Unset, str] = UNSET,
    app_id: str,
    subpaths: Union[Unset, list[str]] = UNSET,
) -> Response[Union[GatewayUsageStats, HTTPValidationError]]:
    """Get App Stats

    Args:
        timeframe (StatsTimeframe):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        user_id (Union[Unset, str]):
        app_id (str):
        subpaths (Union[Unset, list[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[GatewayUsageStats, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
        user_id=user_id,
        app_id=app_id,
        subpaths=subpaths,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    timeframe: StatsTimeframe,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    user_id: Union[Unset, str] = UNSET,
    app_id: str,
    subpaths: Union[Unset, list[str]] = UNSET,
) -> Optional[Union[GatewayUsageStats, HTTPValidationError]]:
    """Get App Stats

    Args:
        timeframe (StatsTimeframe):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        user_id (Union[Unset, str]):
        app_id (str):
        subpaths (Union[Unset, list[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[GatewayUsageStats, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            user_id=user_id,
            app_id=app_id,
            subpaths=subpaths,
        )
    ).parsed
