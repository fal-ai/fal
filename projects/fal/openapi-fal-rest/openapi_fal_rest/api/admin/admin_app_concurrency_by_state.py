import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.runner_state_series import RunnerStateSeries
from ...models.timing_granularity import TimingGranularity
from ...types import UNSET, Response, Unset


def _get_kwargs(
    user_id: str,
    app_name: str,
    *,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: Union[Unset, TimingGranularity] = UNSET,
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

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/admin/apps/{user_id}/{app_name}/concurrency-by-state",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, RunnerStateSeries]]:
    if response.status_code == 200:
        response_200 = RunnerStateSeries.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, RunnerStateSeries]]:
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
) -> Response[Union[HTTPValidationError, RunnerStateSeries]]:
    """Admin App Concurrency By State

    Args:
        user_id (str):
        app_name (str):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (Union[Unset, TimingGranularity]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, RunnerStateSeries]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        app_name=app_name,
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
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
) -> Optional[Union[HTTPValidationError, RunnerStateSeries]]:
    """Admin App Concurrency By State

    Args:
        user_id (str):
        app_name (str):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (Union[Unset, TimingGranularity]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, RunnerStateSeries]
    """

    return sync_detailed(
        user_id=user_id,
        app_name=app_name,
        client=client,
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
    ).parsed


async def asyncio_detailed(
    user_id: str,
    app_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: Union[Unset, TimingGranularity] = UNSET,
) -> Response[Union[HTTPValidationError, RunnerStateSeries]]:
    """Admin App Concurrency By State

    Args:
        user_id (str):
        app_name (str):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (Union[Unset, TimingGranularity]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, RunnerStateSeries]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        app_name=app_name,
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
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
) -> Optional[Union[HTTPValidationError, RunnerStateSeries]]:
    """Admin App Concurrency By State

    Args:
        user_id (str):
        app_name (str):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (Union[Unset, TimingGranularity]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, RunnerStateSeries]
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            app_name=app_name,
            client=client,
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
        )
    ).parsed
