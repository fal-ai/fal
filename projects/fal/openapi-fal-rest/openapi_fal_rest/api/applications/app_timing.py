import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.app_timing import AppTiming
from ...models.http_validation_error import HTTPValidationError
from ...models.timing_granularity import TimingGranularity
from ...types import UNSET, Response, Unset


def _get_kwargs(
    app_alias_or_id: str,
    *,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: Union[Unset, TimingGranularity] = UNSET,
    job_id: Union[Unset, str] = UNSET,
    use_precise: Union[Unset, bool] = UNSET,
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

    params["job_id"] = job_id

    params["use_precise"] = use_precise

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/applications/{app_alias_or_id}/timing",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[AppTiming, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = AppTiming.from_dict(response.json())

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
) -> Response[Union[AppTiming, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    app_alias_or_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: Union[Unset, TimingGranularity] = UNSET,
    job_id: Union[Unset, str] = UNSET,
    use_precise: Union[Unset, bool] = UNSET,
) -> Response[Union[AppTiming, HTTPValidationError]]:
    """App Timing

    Args:
        app_alias_or_id (str):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (Union[Unset, TimingGranularity]):
        job_id (Union[Unset, str]):
        use_precise (Union[Unset, bool]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AppTiming, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        app_alias_or_id=app_alias_or_id,
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
        job_id=job_id,
        use_precise=use_precise,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    app_alias_or_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: Union[Unset, TimingGranularity] = UNSET,
    job_id: Union[Unset, str] = UNSET,
    use_precise: Union[Unset, bool] = UNSET,
) -> Optional[Union[AppTiming, HTTPValidationError]]:
    """App Timing

    Args:
        app_alias_or_id (str):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (Union[Unset, TimingGranularity]):
        job_id (Union[Unset, str]):
        use_precise (Union[Unset, bool]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AppTiming, HTTPValidationError]
    """

    return sync_detailed(
        app_alias_or_id=app_alias_or_id,
        client=client,
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
        job_id=job_id,
        use_precise=use_precise,
    ).parsed


async def asyncio_detailed(
    app_alias_or_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: Union[Unset, TimingGranularity] = UNSET,
    job_id: Union[Unset, str] = UNSET,
    use_precise: Union[Unset, bool] = UNSET,
) -> Response[Union[AppTiming, HTTPValidationError]]:
    """App Timing

    Args:
        app_alias_or_id (str):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (Union[Unset, TimingGranularity]):
        job_id (Union[Unset, str]):
        use_precise (Union[Unset, bool]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AppTiming, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        app_alias_or_id=app_alias_or_id,
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
        job_id=job_id,
        use_precise=use_precise,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    app_alias_or_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    granularity: Union[Unset, TimingGranularity] = UNSET,
    job_id: Union[Unset, str] = UNSET,
    use_precise: Union[Unset, bool] = UNSET,
) -> Optional[Union[AppTiming, HTTPValidationError]]:
    """App Timing

    Args:
        app_alias_or_id (str):
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        granularity (Union[Unset, TimingGranularity]):
        job_id (Union[Unset, str]):
        use_precise (Union[Unset, bool]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AppTiming, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            app_alias_or_id=app_alias_or_id,
            client=client,
            start_time=start_time,
            end_time=end_time,
            granularity=granularity,
            job_id=job_id,
            use_precise=use_precise,
        )
    ).parsed
