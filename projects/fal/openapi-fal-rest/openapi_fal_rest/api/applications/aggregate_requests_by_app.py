import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.aggregate_requests_by_app_metric import AggregateRequestsByAppMetric
from ...models.aggregate_requests_by_app_request_type import AggregateRequestsByAppRequestType
from ...models.http_validation_error import HTTPValidationError
from ...models.requests_by_app_stats import RequestsByAppStats
from ...models.timing_granularity import TimingGranularity
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: TimingGranularity,
    request_type: Union[Unset, AggregateRequestsByAppRequestType] = AggregateRequestsByAppRequestType.HTTP,
    metric: Union[Unset, AggregateRequestsByAppMetric] = AggregateRequestsByAppMetric.REQUESTS,
    app_ids: Union[Unset, list[str]] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_start_time = start_time.isoformat()
    params["start_time"] = json_start_time

    json_end_time = end_time.isoformat()
    params["end_time"] = json_end_time

    json_granularity = granularity.value
    params["granularity"] = json_granularity

    json_request_type: Union[Unset, str] = UNSET
    if not isinstance(request_type, Unset):
        json_request_type = request_type.value

    params["request_type"] = json_request_type

    json_metric: Union[Unset, str] = UNSET
    if not isinstance(metric, Unset):
        json_metric = metric.value

    params["metric"] = json_metric

    json_app_ids: Union[Unset, list[str]] = UNSET
    if not isinstance(app_ids, Unset):
        json_app_ids = app_ids

    params["app_ids"] = json_app_ids

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/applications/metrics/requests-by-app",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, RequestsByAppStats]]:
    if response.status_code == 200:
        response_200 = RequestsByAppStats.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, RequestsByAppStats]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: TimingGranularity,
    request_type: Union[Unset, AggregateRequestsByAppRequestType] = AggregateRequestsByAppRequestType.HTTP,
    metric: Union[Unset, AggregateRequestsByAppMetric] = AggregateRequestsByAppMetric.REQUESTS,
    app_ids: Union[Unset, list[str]] = UNSET,
) -> Response[Union[HTTPValidationError, RequestsByAppStats]]:
    """Aggregate Requests By App

    Args:
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (TimingGranularity):
        request_type (Union[Unset, AggregateRequestsByAppRequestType]):  Default:
            AggregateRequestsByAppRequestType.HTTP.
        metric (Union[Unset, AggregateRequestsByAppMetric]):  Default:
            AggregateRequestsByAppMetric.REQUESTS.
        app_ids (Union[Unset, list[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, RequestsByAppStats]]
    """

    kwargs = _get_kwargs(
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
        request_type=request_type,
        metric=metric,
        app_ids=app_ids,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: TimingGranularity,
    request_type: Union[Unset, AggregateRequestsByAppRequestType] = AggregateRequestsByAppRequestType.HTTP,
    metric: Union[Unset, AggregateRequestsByAppMetric] = AggregateRequestsByAppMetric.REQUESTS,
    app_ids: Union[Unset, list[str]] = UNSET,
) -> Optional[Union[HTTPValidationError, RequestsByAppStats]]:
    """Aggregate Requests By App

    Args:
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (TimingGranularity):
        request_type (Union[Unset, AggregateRequestsByAppRequestType]):  Default:
            AggregateRequestsByAppRequestType.HTTP.
        metric (Union[Unset, AggregateRequestsByAppMetric]):  Default:
            AggregateRequestsByAppMetric.REQUESTS.
        app_ids (Union[Unset, list[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, RequestsByAppStats]
    """

    return sync_detailed(
        client=client,
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
        request_type=request_type,
        metric=metric,
        app_ids=app_ids,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: TimingGranularity,
    request_type: Union[Unset, AggregateRequestsByAppRequestType] = AggregateRequestsByAppRequestType.HTTP,
    metric: Union[Unset, AggregateRequestsByAppMetric] = AggregateRequestsByAppMetric.REQUESTS,
    app_ids: Union[Unset, list[str]] = UNSET,
) -> Response[Union[HTTPValidationError, RequestsByAppStats]]:
    """Aggregate Requests By App

    Args:
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (TimingGranularity):
        request_type (Union[Unset, AggregateRequestsByAppRequestType]):  Default:
            AggregateRequestsByAppRequestType.HTTP.
        metric (Union[Unset, AggregateRequestsByAppMetric]):  Default:
            AggregateRequestsByAppMetric.REQUESTS.
        app_ids (Union[Unset, list[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, RequestsByAppStats]]
    """

    kwargs = _get_kwargs(
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
        request_type=request_type,
        metric=metric,
        app_ids=app_ids,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: TimingGranularity,
    request_type: Union[Unset, AggregateRequestsByAppRequestType] = AggregateRequestsByAppRequestType.HTTP,
    metric: Union[Unset, AggregateRequestsByAppMetric] = AggregateRequestsByAppMetric.REQUESTS,
    app_ids: Union[Unset, list[str]] = UNSET,
) -> Optional[Union[HTTPValidationError, RequestsByAppStats]]:
    """Aggregate Requests By App

    Args:
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (TimingGranularity):
        request_type (Union[Unset, AggregateRequestsByAppRequestType]):  Default:
            AggregateRequestsByAppRequestType.HTTP.
        metric (Union[Unset, AggregateRequestsByAppMetric]):  Default:
            AggregateRequestsByAppMetric.REQUESTS.
        app_ids (Union[Unset, list[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, RequestsByAppStats]
    """

    return (
        await asyncio_detailed(
            client=client,
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
            request_type=request_type,
            metric=metric,
            app_ids=app_ids,
        )
    ).parsed
