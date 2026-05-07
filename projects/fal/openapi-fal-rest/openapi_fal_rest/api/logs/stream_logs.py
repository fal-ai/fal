from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.label_filter import LabelFilter
from ...models.log_level import LogLevel
from ...models.stream_logs_run_source import StreamLogsRunSource
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: list["LabelFilter"],
    since: Union[Unset, str] = UNSET,
    until: Union[Unset, str] = UNSET,
    app_id: Union[Unset, list[str]] = UNSET,
    revision: Union[Unset, str] = UNSET,
    run_source: Union[Unset, StreamLogsRunSource] = UNSET,
    traceback: Union[Unset, bool] = False,
    search: Union[Unset, str] = UNSET,
    level: Union[Unset, LogLevel] = UNSET,
    job_id: Union[Unset, str] = UNSET,
    request_id: Union[Unset, str] = UNSET,
    endpoint: Union[Unset, str] = UNSET,
    expand_log_window: Union[Unset, bool] = False,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    params: dict[str, Any] = {}

    params["since"] = since

    params["until"] = until

    json_app_id: Union[Unset, list[str]] = UNSET
    if not isinstance(app_id, Unset):
        json_app_id = app_id

    params["app_id"] = json_app_id

    params["revision"] = revision

    json_run_source: Union[Unset, str] = UNSET
    if not isinstance(run_source, Unset):
        json_run_source = run_source.value

    params["run_source"] = json_run_source

    params["traceback"] = traceback

    params["search"] = search

    json_level: Union[Unset, str] = UNSET
    if not isinstance(level, Unset):
        json_level = level.value

    params["level"] = json_level

    params["job_id"] = job_id

    params["request_id"] = request_id

    params["endpoint"] = endpoint

    params["expand_log_window"] = expand_log_window

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/logs/stream",
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
) -> Optional[Union[Any, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = response.json()
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
) -> Response[Union[Any, HTTPValidationError]]:
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
    since: Union[Unset, str] = UNSET,
    until: Union[Unset, str] = UNSET,
    app_id: Union[Unset, list[str]] = UNSET,
    revision: Union[Unset, str] = UNSET,
    run_source: Union[Unset, StreamLogsRunSource] = UNSET,
    traceback: Union[Unset, bool] = False,
    search: Union[Unset, str] = UNSET,
    level: Union[Unset, LogLevel] = UNSET,
    job_id: Union[Unset, str] = UNSET,
    request_id: Union[Unset, str] = UNSET,
    endpoint: Union[Unset, str] = UNSET,
    expand_log_window: Union[Unset, bool] = False,
) -> Response[Union[Any, HTTPValidationError]]:
    """Stream logs (SSE)

    Args:
        since (Union[Unset, str]):
        until (Union[Unset, str]):
        app_id (Union[Unset, list[str]]):
        revision (Union[Unset, str]):
        run_source (Union[Unset, StreamLogsRunSource]):
        traceback (Union[Unset, bool]):  Default: False.
        search (Union[Unset, str]):
        level (Union[Unset, LogLevel]):
        job_id (Union[Unset, str]):
        request_id (Union[Unset, str]):
        endpoint (Union[Unset, str]):
        expand_log_window (Union[Unset, bool]):  Default: False.
        body (list['LabelFilter']):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
        since=since,
        until=until,
        app_id=app_id,
        revision=revision,
        run_source=run_source,
        traceback=traceback,
        search=search,
        level=level,
        job_id=job_id,
        request_id=request_id,
        endpoint=endpoint,
        expand_log_window=expand_log_window,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    body: list["LabelFilter"],
    since: Union[Unset, str] = UNSET,
    until: Union[Unset, str] = UNSET,
    app_id: Union[Unset, list[str]] = UNSET,
    revision: Union[Unset, str] = UNSET,
    run_source: Union[Unset, StreamLogsRunSource] = UNSET,
    traceback: Union[Unset, bool] = False,
    search: Union[Unset, str] = UNSET,
    level: Union[Unset, LogLevel] = UNSET,
    job_id: Union[Unset, str] = UNSET,
    request_id: Union[Unset, str] = UNSET,
    endpoint: Union[Unset, str] = UNSET,
    expand_log_window: Union[Unset, bool] = False,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Stream logs (SSE)

    Args:
        since (Union[Unset, str]):
        until (Union[Unset, str]):
        app_id (Union[Unset, list[str]]):
        revision (Union[Unset, str]):
        run_source (Union[Unset, StreamLogsRunSource]):
        traceback (Union[Unset, bool]):  Default: False.
        search (Union[Unset, str]):
        level (Union[Unset, LogLevel]):
        job_id (Union[Unset, str]):
        request_id (Union[Unset, str]):
        endpoint (Union[Unset, str]):
        expand_log_window (Union[Unset, bool]):  Default: False.
        body (list['LabelFilter']):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
        since=since,
        until=until,
        app_id=app_id,
        revision=revision,
        run_source=run_source,
        traceback=traceback,
        search=search,
        level=level,
        job_id=job_id,
        request_id=request_id,
        endpoint=endpoint,
        expand_log_window=expand_log_window,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: list["LabelFilter"],
    since: Union[Unset, str] = UNSET,
    until: Union[Unset, str] = UNSET,
    app_id: Union[Unset, list[str]] = UNSET,
    revision: Union[Unset, str] = UNSET,
    run_source: Union[Unset, StreamLogsRunSource] = UNSET,
    traceback: Union[Unset, bool] = False,
    search: Union[Unset, str] = UNSET,
    level: Union[Unset, LogLevel] = UNSET,
    job_id: Union[Unset, str] = UNSET,
    request_id: Union[Unset, str] = UNSET,
    endpoint: Union[Unset, str] = UNSET,
    expand_log_window: Union[Unset, bool] = False,
) -> Response[Union[Any, HTTPValidationError]]:
    """Stream logs (SSE)

    Args:
        since (Union[Unset, str]):
        until (Union[Unset, str]):
        app_id (Union[Unset, list[str]]):
        revision (Union[Unset, str]):
        run_source (Union[Unset, StreamLogsRunSource]):
        traceback (Union[Unset, bool]):  Default: False.
        search (Union[Unset, str]):
        level (Union[Unset, LogLevel]):
        job_id (Union[Unset, str]):
        request_id (Union[Unset, str]):
        endpoint (Union[Unset, str]):
        expand_log_window (Union[Unset, bool]):  Default: False.
        body (list['LabelFilter']):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
        since=since,
        until=until,
        app_id=app_id,
        revision=revision,
        run_source=run_source,
        traceback=traceback,
        search=search,
        level=level,
        job_id=job_id,
        request_id=request_id,
        endpoint=endpoint,
        expand_log_window=expand_log_window,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: list["LabelFilter"],
    since: Union[Unset, str] = UNSET,
    until: Union[Unset, str] = UNSET,
    app_id: Union[Unset, list[str]] = UNSET,
    revision: Union[Unset, str] = UNSET,
    run_source: Union[Unset, StreamLogsRunSource] = UNSET,
    traceback: Union[Unset, bool] = False,
    search: Union[Unset, str] = UNSET,
    level: Union[Unset, LogLevel] = UNSET,
    job_id: Union[Unset, str] = UNSET,
    request_id: Union[Unset, str] = UNSET,
    endpoint: Union[Unset, str] = UNSET,
    expand_log_window: Union[Unset, bool] = False,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Stream logs (SSE)

    Args:
        since (Union[Unset, str]):
        until (Union[Unset, str]):
        app_id (Union[Unset, list[str]]):
        revision (Union[Unset, str]):
        run_source (Union[Unset, StreamLogsRunSource]):
        traceback (Union[Unset, bool]):  Default: False.
        search (Union[Unset, str]):
        level (Union[Unset, LogLevel]):
        job_id (Union[Unset, str]):
        request_id (Union[Unset, str]):
        endpoint (Union[Unset, str]):
        expand_log_window (Union[Unset, bool]):  Default: False.
        body (list['LabelFilter']):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            since=since,
            until=until,
            app_id=app_id,
            revision=revision,
            run_source=run_source,
            traceback=traceback,
            search=search,
            level=level,
            job_id=job_id,
            request_id=request_id,
            endpoint=endpoint,
            expand_log_window=expand_log_window,
        )
    ).parsed
