import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_top_apps_by_requests_request_type import ListTopAppsByRequestsRequestType
from ...models.list_top_apps_by_requests_sort_by import ListTopAppsByRequestsSortBy
from ...models.top_apps_by_requests_response import TopAppsByRequestsResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    limit: Union[Unset, int] = 10,
    request_type: Union[Unset, ListTopAppsByRequestsRequestType] = ListTopAppsByRequestsRequestType.HTTP,
    sort_by: Union[Unset, ListTopAppsByRequestsSortBy] = ListTopAppsByRequestsSortBy.REQUESTS,
    status_code: Union[Unset, list[str]] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_start_time = start_time.isoformat()
    params["start_time"] = json_start_time

    json_end_time = end_time.isoformat()
    params["end_time"] = json_end_time

    params["limit"] = limit

    json_request_type: Union[Unset, str] = UNSET
    if not isinstance(request_type, Unset):
        json_request_type = request_type.value

    params["request_type"] = json_request_type

    json_sort_by: Union[Unset, str] = UNSET
    if not isinstance(sort_by, Unset):
        json_sort_by = sort_by.value

    params["sort_by"] = json_sort_by

    json_status_code: Union[Unset, list[str]] = UNSET
    if not isinstance(status_code, Unset):
        json_status_code = status_code

    params["status_code"] = json_status_code

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/applications/metrics/top-apps-by-requests",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, TopAppsByRequestsResponse]]:
    if response.status_code == 200:
        response_200 = TopAppsByRequestsResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, TopAppsByRequestsResponse]]:
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
    limit: Union[Unset, int] = 10,
    request_type: Union[Unset, ListTopAppsByRequestsRequestType] = ListTopAppsByRequestsRequestType.HTTP,
    sort_by: Union[Unset, ListTopAppsByRequestsSortBy] = ListTopAppsByRequestsSortBy.REQUESTS,
    status_code: Union[Unset, list[str]] = UNSET,
) -> Response[Union[HTTPValidationError, TopAppsByRequestsResponse]]:
    """List Top Apps By Requests

    Args:
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        limit (Union[Unset, int]):  Default: 10.
        request_type (Union[Unset, ListTopAppsByRequestsRequestType]):  Default:
            ListTopAppsByRequestsRequestType.HTTP.
        sort_by (Union[Unset, ListTopAppsByRequestsSortBy]):  Default:
            ListTopAppsByRequestsSortBy.REQUESTS.
        status_code (Union[Unset, list[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, TopAppsByRequestsResponse]]
    """

    kwargs = _get_kwargs(
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        request_type=request_type,
        sort_by=sort_by,
        status_code=status_code,
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
    limit: Union[Unset, int] = 10,
    request_type: Union[Unset, ListTopAppsByRequestsRequestType] = ListTopAppsByRequestsRequestType.HTTP,
    sort_by: Union[Unset, ListTopAppsByRequestsSortBy] = ListTopAppsByRequestsSortBy.REQUESTS,
    status_code: Union[Unset, list[str]] = UNSET,
) -> Optional[Union[HTTPValidationError, TopAppsByRequestsResponse]]:
    """List Top Apps By Requests

    Args:
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        limit (Union[Unset, int]):  Default: 10.
        request_type (Union[Unset, ListTopAppsByRequestsRequestType]):  Default:
            ListTopAppsByRequestsRequestType.HTTP.
        sort_by (Union[Unset, ListTopAppsByRequestsSortBy]):  Default:
            ListTopAppsByRequestsSortBy.REQUESTS.
        status_code (Union[Unset, list[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, TopAppsByRequestsResponse]
    """

    return sync_detailed(
        client=client,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        request_type=request_type,
        sort_by=sort_by,
        status_code=status_code,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    limit: Union[Unset, int] = 10,
    request_type: Union[Unset, ListTopAppsByRequestsRequestType] = ListTopAppsByRequestsRequestType.HTTP,
    sort_by: Union[Unset, ListTopAppsByRequestsSortBy] = ListTopAppsByRequestsSortBy.REQUESTS,
    status_code: Union[Unset, list[str]] = UNSET,
) -> Response[Union[HTTPValidationError, TopAppsByRequestsResponse]]:
    """List Top Apps By Requests

    Args:
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        limit (Union[Unset, int]):  Default: 10.
        request_type (Union[Unset, ListTopAppsByRequestsRequestType]):  Default:
            ListTopAppsByRequestsRequestType.HTTP.
        sort_by (Union[Unset, ListTopAppsByRequestsSortBy]):  Default:
            ListTopAppsByRequestsSortBy.REQUESTS.
        status_code (Union[Unset, list[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, TopAppsByRequestsResponse]]
    """

    kwargs = _get_kwargs(
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        request_type=request_type,
        sort_by=sort_by,
        status_code=status_code,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    limit: Union[Unset, int] = 10,
    request_type: Union[Unset, ListTopAppsByRequestsRequestType] = ListTopAppsByRequestsRequestType.HTTP,
    sort_by: Union[Unset, ListTopAppsByRequestsSortBy] = ListTopAppsByRequestsSortBy.REQUESTS,
    status_code: Union[Unset, list[str]] = UNSET,
) -> Optional[Union[HTTPValidationError, TopAppsByRequestsResponse]]:
    """List Top Apps By Requests

    Args:
        start_time (datetime.datetime):
        end_time (datetime.datetime):
        limit (Union[Unset, int]):  Default: 10.
        request_type (Union[Unset, ListTopAppsByRequestsRequestType]):  Default:
            ListTopAppsByRequestsRequestType.HTTP.
        sort_by (Union[Unset, ListTopAppsByRequestsSortBy]):  Default:
            ListTopAppsByRequestsSortBy.REQUESTS.
        status_code (Union[Unset, list[str]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, TopAppsByRequestsResponse]
    """

    return (
        await asyncio_detailed(
            client=client,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            request_type=request_type,
            sort_by=sort_by,
            status_code=status_code,
        )
    ).parsed
