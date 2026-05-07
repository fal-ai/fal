from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.get_log_volume_run_source import GetLogVolumeRunSource
from ...models.http_validation_error import HTTPValidationError
from ...models.label_filter import LabelFilter
from ...models.log_level import LogLevel
from ...models.log_volume_response import LogVolumeResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: list["LabelFilter"],
    since: str,
    until: str,
    step: Union[Unset, str] = UNSET,
    app_id: Union[Unset, str] = UNSET,
    job_id: Union[Unset, str] = UNSET,
    search: Union[Unset, str] = UNSET,
    level: Union[Unset, LogLevel] = UNSET,
    revision: Union[Unset, str] = UNSET,
    run_source: Union[Unset, GetLogVolumeRunSource] = UNSET,
    request_id: Union[Unset, str] = UNSET,
    endpoint: Union[Unset, str] = UNSET,
    traceback: Union[Unset, bool] = False,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    params: dict[str, Any] = {}

    params["since"] = since

    params["until"] = until

    params["step"] = step

    params["app_id"] = app_id

    params["job_id"] = job_id

    params["search"] = search

    json_level: Union[Unset, str] = UNSET
    if not isinstance(level, Unset):
        json_level = level.value

    params["level"] = json_level

    params["revision"] = revision

    json_run_source: Union[Unset, str] = UNSET
    if not isinstance(run_source, Unset):
        json_run_source = run_source.value

    params["run_source"] = json_run_source

    params["request_id"] = request_id

    params["endpoint"] = endpoint

    params["traceback"] = traceback

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/logs/volume",
        "params": params,
    }

    _body = []
    for body_item_data in body:
        body_item = body_item_data.to_dict()
        _body.append(body_item)

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, LogVolumeResponse]]:
    if response.status_code == 200:
        response_200 = LogVolumeResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, LogVolumeResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: list["LabelFilter"],
    since: str,
    until: str,
    step: Union[Unset, str] = UNSET,
    app_id: Union[Unset, str] = UNSET,
    job_id: Union[Unset, str] = UNSET,
    search: Union[Unset, str] = UNSET,
    level: Union[Unset, LogLevel] = UNSET,
    revision: Union[Unset, str] = UNSET,
    run_source: Union[Unset, GetLogVolumeRunSource] = UNSET,
    request_id: Union[Unset, str] = UNSET,
    endpoint: Union[Unset, str] = UNSET,
    traceback: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, LogVolumeResponse]]:
    """Get log volume over time

     Get log volume aggregated over time, grouped by log level.

    Returns a time series of log counts suitable for rendering a timeline chart.

    Args:
        since (str): Start time in ISO format
        until (str): End time in ISO format
        step (Union[Unset, str]): Bucket size (e.g., '10s', '30s', '2m', '10m', '2h', '8h'). Auto-
            calculated if not provided.
        app_id (Union[Unset, str]):
        job_id (Union[Unset, str]):
        search (Union[Unset, str]):
        level (Union[Unset, LogLevel]):
        revision (Union[Unset, str]):
        run_source (Union[Unset, GetLogVolumeRunSource]):
        request_id (Union[Unset, str]):
        endpoint (Union[Unset, str]):
        traceback (Union[Unset, bool]):  Default: False.
        body (list['LabelFilter']):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, LogVolumeResponse]]
    """

    kwargs = _get_kwargs(
        body=body,
        since=since,
        until=until,
        step=step,
        app_id=app_id,
        job_id=job_id,
        search=search,
        level=level,
        revision=revision,
        run_source=run_source,
        request_id=request_id,
        endpoint=endpoint,
        traceback=traceback,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    body: list["LabelFilter"],
    since: str,
    until: str,
    step: Union[Unset, str] = UNSET,
    app_id: Union[Unset, str] = UNSET,
    job_id: Union[Unset, str] = UNSET,
    search: Union[Unset, str] = UNSET,
    level: Union[Unset, LogLevel] = UNSET,
    revision: Union[Unset, str] = UNSET,
    run_source: Union[Unset, GetLogVolumeRunSource] = UNSET,
    request_id: Union[Unset, str] = UNSET,
    endpoint: Union[Unset, str] = UNSET,
    traceback: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, LogVolumeResponse]]:
    """Get log volume over time

     Get log volume aggregated over time, grouped by log level.

    Returns a time series of log counts suitable for rendering a timeline chart.

    Args:
        since (str): Start time in ISO format
        until (str): End time in ISO format
        step (Union[Unset, str]): Bucket size (e.g., '10s', '30s', '2m', '10m', '2h', '8h'). Auto-
            calculated if not provided.
        app_id (Union[Unset, str]):
        job_id (Union[Unset, str]):
        search (Union[Unset, str]):
        level (Union[Unset, LogLevel]):
        revision (Union[Unset, str]):
        run_source (Union[Unset, GetLogVolumeRunSource]):
        request_id (Union[Unset, str]):
        endpoint (Union[Unset, str]):
        traceback (Union[Unset, bool]):  Default: False.
        body (list['LabelFilter']):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, LogVolumeResponse]
    """

    return sync_detailed(
        client=client,
        body=body,
        since=since,
        until=until,
        step=step,
        app_id=app_id,
        job_id=job_id,
        search=search,
        level=level,
        revision=revision,
        run_source=run_source,
        request_id=request_id,
        endpoint=endpoint,
        traceback=traceback,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: list["LabelFilter"],
    since: str,
    until: str,
    step: Union[Unset, str] = UNSET,
    app_id: Union[Unset, str] = UNSET,
    job_id: Union[Unset, str] = UNSET,
    search: Union[Unset, str] = UNSET,
    level: Union[Unset, LogLevel] = UNSET,
    revision: Union[Unset, str] = UNSET,
    run_source: Union[Unset, GetLogVolumeRunSource] = UNSET,
    request_id: Union[Unset, str] = UNSET,
    endpoint: Union[Unset, str] = UNSET,
    traceback: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, LogVolumeResponse]]:
    """Get log volume over time

     Get log volume aggregated over time, grouped by log level.

    Returns a time series of log counts suitable for rendering a timeline chart.

    Args:
        since (str): Start time in ISO format
        until (str): End time in ISO format
        step (Union[Unset, str]): Bucket size (e.g., '10s', '30s', '2m', '10m', '2h', '8h'). Auto-
            calculated if not provided.
        app_id (Union[Unset, str]):
        job_id (Union[Unset, str]):
        search (Union[Unset, str]):
        level (Union[Unset, LogLevel]):
        revision (Union[Unset, str]):
        run_source (Union[Unset, GetLogVolumeRunSource]):
        request_id (Union[Unset, str]):
        endpoint (Union[Unset, str]):
        traceback (Union[Unset, bool]):  Default: False.
        body (list['LabelFilter']):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, LogVolumeResponse]]
    """

    kwargs = _get_kwargs(
        body=body,
        since=since,
        until=until,
        step=step,
        app_id=app_id,
        job_id=job_id,
        search=search,
        level=level,
        revision=revision,
        run_source=run_source,
        request_id=request_id,
        endpoint=endpoint,
        traceback=traceback,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: list["LabelFilter"],
    since: str,
    until: str,
    step: Union[Unset, str] = UNSET,
    app_id: Union[Unset, str] = UNSET,
    job_id: Union[Unset, str] = UNSET,
    search: Union[Unset, str] = UNSET,
    level: Union[Unset, LogLevel] = UNSET,
    revision: Union[Unset, str] = UNSET,
    run_source: Union[Unset, GetLogVolumeRunSource] = UNSET,
    request_id: Union[Unset, str] = UNSET,
    endpoint: Union[Unset, str] = UNSET,
    traceback: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, LogVolumeResponse]]:
    """Get log volume over time

     Get log volume aggregated over time, grouped by log level.

    Returns a time series of log counts suitable for rendering a timeline chart.

    Args:
        since (str): Start time in ISO format
        until (str): End time in ISO format
        step (Union[Unset, str]): Bucket size (e.g., '10s', '30s', '2m', '10m', '2h', '8h'). Auto-
            calculated if not provided.
        app_id (Union[Unset, str]):
        job_id (Union[Unset, str]):
        search (Union[Unset, str]):
        level (Union[Unset, LogLevel]):
        revision (Union[Unset, str]):
        run_source (Union[Unset, GetLogVolumeRunSource]):
        request_id (Union[Unset, str]):
        endpoint (Union[Unset, str]):
        traceback (Union[Unset, bool]):  Default: False.
        body (list['LabelFilter']):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, LogVolumeResponse]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            since=since,
            until=until,
            step=step,
            app_id=app_id,
            job_id=job_id,
            search=search,
            level=level,
            revision=revision,
            run_source=run_source,
            request_id=request_id,
            endpoint=endpoint,
            traceback=traceback,
        )
    ).parsed
