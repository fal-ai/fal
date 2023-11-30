import datetime
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union

import httpx

from ... import errors
from ...client import Client
from ...models.get_gateway_request_stats_by_time_response_get_gateway_request_stats_by_time import (
    GetGatewayRequestStatsByTimeResponseGetGatewayRequestStatsByTime,
)
from ...models.http_validation_error import HTTPValidationError
from ...models.stats_timeframe import StatsTimeframe
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    client: Client,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    timeframe: StatsTimeframe,
    app_aliases: Union[Unset, None, List[str]] = UNSET,
) -> Dict[str, Any]:
    url = "{}/usage/stats/app/time".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    json_start_time = start_time.isoformat()

    params["start_time"] = json_start_time

    json_end_time = end_time.isoformat()

    params["end_time"] = json_end_time

    json_timeframe = timeframe.value

    params["timeframe"] = json_timeframe

    json_app_aliases: Union[Unset, None, List[str]] = UNSET
    if not isinstance(app_aliases, Unset):
        if app_aliases is None:
            json_app_aliases = None
        else:
            json_app_aliases = app_aliases

    params["app_aliases"] = json_app_aliases

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
) -> Optional[Union[GetGatewayRequestStatsByTimeResponseGetGatewayRequestStatsByTime, HTTPValidationError]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = GetGatewayRequestStatsByTimeResponseGetGatewayRequestStatsByTime.from_dict(response.json())

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
) -> Response[Union[GetGatewayRequestStatsByTimeResponseGetGatewayRequestStatsByTime, HTTPValidationError]]:
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
    timeframe: StatsTimeframe,
    app_aliases: Union[Unset, None, List[str]] = UNSET,
) -> Response[Union[GetGatewayRequestStatsByTimeResponseGetGatewayRequestStatsByTime, HTTPValidationError]]:
    """Get Gateway Request Stats By Time

    Args:
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        timeframe (StatsTimeframe): An enumeration.
        app_aliases (Union[Unset, None, List[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[GetGatewayRequestStatsByTimeResponseGetGatewayRequestStatsByTime, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        client=client,
        start_time=start_time,
        end_time=end_time,
        timeframe=timeframe,
        app_aliases=app_aliases,
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
    timeframe: StatsTimeframe,
    app_aliases: Union[Unset, None, List[str]] = UNSET,
) -> Optional[Union[GetGatewayRequestStatsByTimeResponseGetGatewayRequestStatsByTime, HTTPValidationError]]:
    """Get Gateway Request Stats By Time

    Args:
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        timeframe (StatsTimeframe): An enumeration.
        app_aliases (Union[Unset, None, List[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[GetGatewayRequestStatsByTimeResponseGetGatewayRequestStatsByTime, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        start_time=start_time,
        end_time=end_time,
        timeframe=timeframe,
        app_aliases=app_aliases,
    ).parsed


async def asyncio_detailed(
    *,
    client: Client,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    timeframe: StatsTimeframe,
    app_aliases: Union[Unset, None, List[str]] = UNSET,
) -> Response[Union[GetGatewayRequestStatsByTimeResponseGetGatewayRequestStatsByTime, HTTPValidationError]]:
    """Get Gateway Request Stats By Time

    Args:
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        timeframe (StatsTimeframe): An enumeration.
        app_aliases (Union[Unset, None, List[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[GetGatewayRequestStatsByTimeResponseGetGatewayRequestStatsByTime, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        client=client,
        start_time=start_time,
        end_time=end_time,
        timeframe=timeframe,
        app_aliases=app_aliases,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Client,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    timeframe: StatsTimeframe,
    app_aliases: Union[Unset, None, List[str]] = UNSET,
) -> Optional[Union[GetGatewayRequestStatsByTimeResponseGetGatewayRequestStatsByTime, HTTPValidationError]]:
    """Get Gateway Request Stats By Time

    Args:
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        timeframe (StatsTimeframe): An enumeration.
        app_aliases (Union[Unset, None, List[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[GetGatewayRequestStatsByTimeResponseGetGatewayRequestStatsByTime, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            start_time=start_time,
            end_time=end_time,
            timeframe=timeframe,
            app_aliases=app_aliases,
        )
    ).parsed
