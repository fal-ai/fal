import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.app_request_latency_request_type import AppRequestLatencyRequestType
from ...models.gateway_request_latency_stats import GatewayRequestLatencyStats
from ...models.http_validation_error import HTTPValidationError
from ...models.timing_granularity import TimingGranularity
from ...types import UNSET, Response, Unset


def _get_kwargs(
    app_user_id: str,
    app_alias_or_id: str,
    *,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: Union[Unset, TimingGranularity] = UNSET,
    request_type: Union[Unset, AppRequestLatencyRequestType] = AppRequestLatencyRequestType.HTTP,
    status_code: Union[Unset, list[str]] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_start_time = start_time.isoformat()
    params["start_time"] = json_start_time

    json_end_time = end_time.isoformat()
    params["end_time"] = json_end_time

    json_granularity: Union[Unset, str] = UNSET
    if not isinstance(granularity, Unset):
        json_granularity = granularity.value

    params["granularity"] = json_granularity

    json_request_type: Union[Unset, str] = UNSET
    if not isinstance(request_type, Unset):
        json_request_type = request_type.value

    params["request_type"] = json_request_type

    json_status_code: Union[Unset, list[str]] = UNSET
    if not isinstance(status_code, Unset):
        json_status_code = status_code

    params["status_code"] = json_status_code

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/applications/{app_user_id}/{app_alias_or_id}/request-latency",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[GatewayRequestLatencyStats, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = GatewayRequestLatencyStats.from_dict(response.json())

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
) -> Response[Union[GatewayRequestLatencyStats, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    app_user_id: str,
    app_alias_or_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: Union[Unset, TimingGranularity] = UNSET,
    request_type: Union[Unset, AppRequestLatencyRequestType] = AppRequestLatencyRequestType.HTTP,
    status_code: Union[Unset, list[str]] = UNSET,
) -> Response[Union[GatewayRequestLatencyStats, HTTPValidationError]]:
    """App Request Latency

    Args:
        app_user_id (str):
        app_alias_or_id (str):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (Union[Unset, TimingGranularity]):
        request_type (Union[Unset, AppRequestLatencyRequestType]):  Default:
            AppRequestLatencyRequestType.HTTP.
        status_code (Union[Unset, list[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[GatewayRequestLatencyStats, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        app_user_id=app_user_id,
        app_alias_or_id=app_alias_or_id,
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
        request_type=request_type,
        status_code=status_code,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    app_user_id: str,
    app_alias_or_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: Union[Unset, TimingGranularity] = UNSET,
    request_type: Union[Unset, AppRequestLatencyRequestType] = AppRequestLatencyRequestType.HTTP,
    status_code: Union[Unset, list[str]] = UNSET,
) -> Optional[Union[GatewayRequestLatencyStats, HTTPValidationError]]:
    """App Request Latency

    Args:
        app_user_id (str):
        app_alias_or_id (str):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (Union[Unset, TimingGranularity]):
        request_type (Union[Unset, AppRequestLatencyRequestType]):  Default:
            AppRequestLatencyRequestType.HTTP.
        status_code (Union[Unset, list[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[GatewayRequestLatencyStats, HTTPValidationError]
    """

    return sync_detailed(
        app_user_id=app_user_id,
        app_alias_or_id=app_alias_or_id,
        client=client,
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
        request_type=request_type,
        status_code=status_code,
    ).parsed


async def asyncio_detailed(
    app_user_id: str,
    app_alias_or_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: Union[Unset, TimingGranularity] = UNSET,
    request_type: Union[Unset, AppRequestLatencyRequestType] = AppRequestLatencyRequestType.HTTP,
    status_code: Union[Unset, list[str]] = UNSET,
) -> Response[Union[GatewayRequestLatencyStats, HTTPValidationError]]:
    """App Request Latency

    Args:
        app_user_id (str):
        app_alias_or_id (str):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (Union[Unset, TimingGranularity]):
        request_type (Union[Unset, AppRequestLatencyRequestType]):  Default:
            AppRequestLatencyRequestType.HTTP.
        status_code (Union[Unset, list[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[GatewayRequestLatencyStats, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        app_user_id=app_user_id,
        app_alias_or_id=app_alias_or_id,
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
        request_type=request_type,
        status_code=status_code,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    app_user_id: str,
    app_alias_or_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: Union[Unset, TimingGranularity] = UNSET,
    request_type: Union[Unset, AppRequestLatencyRequestType] = AppRequestLatencyRequestType.HTTP,
    status_code: Union[Unset, list[str]] = UNSET,
) -> Optional[Union[GatewayRequestLatencyStats, HTTPValidationError]]:
    """App Request Latency

    Args:
        app_user_id (str):
        app_alias_or_id (str):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (Union[Unset, TimingGranularity]):
        request_type (Union[Unset, AppRequestLatencyRequestType]):  Default:
            AppRequestLatencyRequestType.HTTP.
        status_code (Union[Unset, list[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[GatewayRequestLatencyStats, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            app_user_id=app_user_id,
            app_alias_or_id=app_alias_or_id,
            client=client,
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
            request_type=request_type,
            status_code=status_code,
        )
    ).parsed
