import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.gateway_usage_stats import GatewayUsageStats
from ...models.http_validation_error import HTTPValidationError
from ...models.stats_timeframe import StatsTimeframe
from ...types import UNSET, Response


def _get_kwargs(
    *,
    timeframe: StatsTimeframe,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    app_ids: list[str],
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_timeframe = timeframe.value
    params["timeframe"] = json_timeframe

    json_start_time = start_time.isoformat()
    params["start_time"] = json_start_time

    json_end_time = end_time.isoformat()
    params["end_time"] = json_end_time

    json_app_ids = app_ids

    params["app_ids"] = json_app_ids

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/usage/owner/by_path/stats",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list["GatewayUsageStats"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = GatewayUsageStats.from_dict(response_200_item_data)

            response_200.append(response_200_item)

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
) -> Response[Union[HTTPValidationError, list["GatewayUsageStats"]]]:
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
    app_ids: list[str],
) -> Response[Union[HTTPValidationError, list["GatewayUsageStats"]]]:
    """Get Owner App Stats By Path

    Args:
        timeframe (StatsTimeframe):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        app_ids (list[str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['GatewayUsageStats']]]
    """

    kwargs = _get_kwargs(
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
        app_ids=app_ids,
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
    app_ids: list[str],
) -> Optional[Union[HTTPValidationError, list["GatewayUsageStats"]]]:
    """Get Owner App Stats By Path

    Args:
        timeframe (StatsTimeframe):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        app_ids (list[str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['GatewayUsageStats']]
    """

    return sync_detailed(
        client=client,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
        app_ids=app_ids,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    timeframe: StatsTimeframe,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    app_ids: list[str],
) -> Response[Union[HTTPValidationError, list["GatewayUsageStats"]]]:
    """Get Owner App Stats By Path

    Args:
        timeframe (StatsTimeframe):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        app_ids (list[str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['GatewayUsageStats']]]
    """

    kwargs = _get_kwargs(
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
        app_ids=app_ids,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    timeframe: StatsTimeframe,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    app_ids: list[str],
) -> Optional[Union[HTTPValidationError, list["GatewayUsageStats"]]]:
    """Get Owner App Stats By Path

    Args:
        timeframe (StatsTimeframe):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        app_ids (list[str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['GatewayUsageStats']]
    """

    return (
        await asyncio_detailed(
            client=client,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            app_ids=app_ids,
        )
    ).parsed
