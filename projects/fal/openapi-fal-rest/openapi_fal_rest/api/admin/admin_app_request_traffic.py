import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.admin_app_request_traffic_request_type import AdminAppRequestTrafficRequestType
from ...models.gateway_request_traffic_stats import GatewayRequestTrafficStats
from ...models.http_validation_error import HTTPValidationError
from ...models.timing_granularity import TimingGranularity
from ...types import UNSET, Response, Unset


def _get_kwargs(
    user_id: str,
    app_name: str,
    *,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: Union[Unset, TimingGranularity] = UNSET,
    request_type: Union[Unset, AdminAppRequestTrafficRequestType] = AdminAppRequestTrafficRequestType.HTTP,
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

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/admin/apps/{user_id}/{app_name}/request-traffic",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[GatewayRequestTrafficStats, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = GatewayRequestTrafficStats.from_dict(response.json())

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
) -> Response[Union[GatewayRequestTrafficStats, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    user_id: str,
    app_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: Union[Unset, TimingGranularity] = UNSET,
    request_type: Union[Unset, AdminAppRequestTrafficRequestType] = AdminAppRequestTrafficRequestType.HTTP,
) -> Response[Union[GatewayRequestTrafficStats, HTTPValidationError]]:
    """App Request Traffic

    Args:
        user_id (str):
        app_name (str):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (Union[Unset, TimingGranularity]):
        request_type (Union[Unset, AdminAppRequestTrafficRequestType]):  Default:
            AdminAppRequestTrafficRequestType.HTTP.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[GatewayRequestTrafficStats, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        app_name=app_name,
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
        request_type=request_type,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    user_id: str,
    app_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: Union[Unset, TimingGranularity] = UNSET,
    request_type: Union[Unset, AdminAppRequestTrafficRequestType] = AdminAppRequestTrafficRequestType.HTTP,
) -> Optional[Union[GatewayRequestTrafficStats, HTTPValidationError]]:
    """App Request Traffic

    Args:
        user_id (str):
        app_name (str):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (Union[Unset, TimingGranularity]):
        request_type (Union[Unset, AdminAppRequestTrafficRequestType]):  Default:
            AdminAppRequestTrafficRequestType.HTTP.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[GatewayRequestTrafficStats, HTTPValidationError]
    """

    return sync_detailed(
        user_id=user_id,
        app_name=app_name,
        client=client,
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
        request_type=request_type,
    ).parsed


async def asyncio_detailed(
    user_id: str,
    app_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: Union[Unset, TimingGranularity] = UNSET,
    request_type: Union[Unset, AdminAppRequestTrafficRequestType] = AdminAppRequestTrafficRequestType.HTTP,
) -> Response[Union[GatewayRequestTrafficStats, HTTPValidationError]]:
    """App Request Traffic

    Args:
        user_id (str):
        app_name (str):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (Union[Unset, TimingGranularity]):
        request_type (Union[Unset, AdminAppRequestTrafficRequestType]):  Default:
            AdminAppRequestTrafficRequestType.HTTP.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[GatewayRequestTrafficStats, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        app_name=app_name,
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
        request_type=request_type,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: str,
    app_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: Union[Unset, TimingGranularity] = UNSET,
    request_type: Union[Unset, AdminAppRequestTrafficRequestType] = AdminAppRequestTrafficRequestType.HTTP,
) -> Optional[Union[GatewayRequestTrafficStats, HTTPValidationError]]:
    """App Request Traffic

    Args:
        user_id (str):
        app_name (str):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (Union[Unset, TimingGranularity]):
        request_type (Union[Unset, AdminAppRequestTrafficRequestType]):  Default:
            AdminAppRequestTrafficRequestType.HTTP.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[GatewayRequestTrafficStats, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            app_name=app_name,
            client=client,
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
            request_type=request_type,
        )
    ).parsed
