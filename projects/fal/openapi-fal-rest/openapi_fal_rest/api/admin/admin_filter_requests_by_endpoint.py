import datetime
from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.admin_filter_requests_by_endpoint_sort_by import AdminFilterRequestsByEndpointSortBy
from ...models.admin_filter_requests_by_endpoint_status import AdminFilterRequestsByEndpointStatus
from ...models.http_validation_error import HTTPValidationError
from ...models.page_type_var_customized_gateway_request_item import PageTypeVarCustomizedGatewayRequestItem
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: list[int],
    endpoint: Union[Unset, str] = UNSET,
    user_nickname: Union[Unset, str] = UNSET,
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    status: Union[Unset, AdminFilterRequestsByEndpointStatus] = UNSET,
    request_id: Union[Unset, UUID] = UNSET,
    runner_id: Union[Unset, str] = UNSET,
    sort_by: Union[Unset, AdminFilterRequestsByEndpointSortBy] = AdminFilterRequestsByEndpointSortBy.ENDED_AT,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    params: dict[str, Any] = {}

    params["endpoint"] = endpoint

    params["user_nickname"] = user_nickname

    json_start_time: Union[Unset, str] = UNSET
    if not isinstance(start_time, Unset):
        json_start_time = start_time.isoformat()
    params["start_time"] = json_start_time

    json_end_time: Union[Unset, str] = UNSET
    if not isinstance(end_time, Unset):
        json_end_time = end_time.isoformat()
    params["end_time"] = json_end_time

    json_status: Union[Unset, str] = UNSET
    if not isinstance(status, Unset):
        json_status = status.value

    params["status"] = json_status

    json_request_id: Union[Unset, str] = UNSET
    if not isinstance(request_id, Unset):
        json_request_id = str(request_id)
    params["request_id"] = json_request_id

    params["runner_id"] = runner_id

    json_sort_by: Union[Unset, str] = UNSET
    if not isinstance(sort_by, Unset):
        json_sort_by = sort_by.value

    params["sort_by"] = json_sort_by

    params["page"] = page

    params["size"] = size

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/admin/requests/by-endpoint",
        "params": params,
    }

    _body = body

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, PageTypeVarCustomizedGatewayRequestItem]]:
    if response.status_code == 200:
        response_200 = PageTypeVarCustomizedGatewayRequestItem.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, PageTypeVarCustomizedGatewayRequestItem]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: list[int],
    endpoint: Union[Unset, str] = UNSET,
    user_nickname: Union[Unset, str] = UNSET,
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    status: Union[Unset, AdminFilterRequestsByEndpointStatus] = UNSET,
    request_id: Union[Unset, UUID] = UNSET,
    runner_id: Union[Unset, str] = UNSET,
    sort_by: Union[Unset, AdminFilterRequestsByEndpointSortBy] = AdminFilterRequestsByEndpointSortBy.ENDED_AT,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, PageTypeVarCustomizedGatewayRequestItem]]:
    """Filter Requests By Endpoint

    Args:
        endpoint (Union[Unset, str]):
        user_nickname (Union[Unset, str]):
        start_time (Union[Unset, datetime.datetime]):
        end_time (Union[Unset, datetime.datetime]):
        status (Union[Unset, AdminFilterRequestsByEndpointStatus]):
        request_id (Union[Unset, UUID]):
        runner_id (Union[Unset, str]):
        sort_by (Union[Unset, AdminFilterRequestsByEndpointSortBy]):  Default:
            AdminFilterRequestsByEndpointSortBy.ENDED_AT.
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.
        body (list[int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PageTypeVarCustomizedGatewayRequestItem]]
    """

    kwargs = _get_kwargs(
        body=body,
        endpoint=endpoint,
        user_nickname=user_nickname,
        start_time=start_time,
        end_time=end_time,
        status=status,
        request_id=request_id,
        runner_id=runner_id,
        sort_by=sort_by,
        page=page,
        size=size,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    body: list[int],
    endpoint: Union[Unset, str] = UNSET,
    user_nickname: Union[Unset, str] = UNSET,
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    status: Union[Unset, AdminFilterRequestsByEndpointStatus] = UNSET,
    request_id: Union[Unset, UUID] = UNSET,
    runner_id: Union[Unset, str] = UNSET,
    sort_by: Union[Unset, AdminFilterRequestsByEndpointSortBy] = AdminFilterRequestsByEndpointSortBy.ENDED_AT,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, PageTypeVarCustomizedGatewayRequestItem]]:
    """Filter Requests By Endpoint

    Args:
        endpoint (Union[Unset, str]):
        user_nickname (Union[Unset, str]):
        start_time (Union[Unset, datetime.datetime]):
        end_time (Union[Unset, datetime.datetime]):
        status (Union[Unset, AdminFilterRequestsByEndpointStatus]):
        request_id (Union[Unset, UUID]):
        runner_id (Union[Unset, str]):
        sort_by (Union[Unset, AdminFilterRequestsByEndpointSortBy]):  Default:
            AdminFilterRequestsByEndpointSortBy.ENDED_AT.
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.
        body (list[int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PageTypeVarCustomizedGatewayRequestItem]
    """

    return sync_detailed(
        client=client,
        body=body,
        endpoint=endpoint,
        user_nickname=user_nickname,
        start_time=start_time,
        end_time=end_time,
        status=status,
        request_id=request_id,
        runner_id=runner_id,
        sort_by=sort_by,
        page=page,
        size=size,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: list[int],
    endpoint: Union[Unset, str] = UNSET,
    user_nickname: Union[Unset, str] = UNSET,
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    status: Union[Unset, AdminFilterRequestsByEndpointStatus] = UNSET,
    request_id: Union[Unset, UUID] = UNSET,
    runner_id: Union[Unset, str] = UNSET,
    sort_by: Union[Unset, AdminFilterRequestsByEndpointSortBy] = AdminFilterRequestsByEndpointSortBy.ENDED_AT,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, PageTypeVarCustomizedGatewayRequestItem]]:
    """Filter Requests By Endpoint

    Args:
        endpoint (Union[Unset, str]):
        user_nickname (Union[Unset, str]):
        start_time (Union[Unset, datetime.datetime]):
        end_time (Union[Unset, datetime.datetime]):
        status (Union[Unset, AdminFilterRequestsByEndpointStatus]):
        request_id (Union[Unset, UUID]):
        runner_id (Union[Unset, str]):
        sort_by (Union[Unset, AdminFilterRequestsByEndpointSortBy]):  Default:
            AdminFilterRequestsByEndpointSortBy.ENDED_AT.
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.
        body (list[int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, PageTypeVarCustomizedGatewayRequestItem]]
    """

    kwargs = _get_kwargs(
        body=body,
        endpoint=endpoint,
        user_nickname=user_nickname,
        start_time=start_time,
        end_time=end_time,
        status=status,
        request_id=request_id,
        runner_id=runner_id,
        sort_by=sort_by,
        page=page,
        size=size,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: list[int],
    endpoint: Union[Unset, str] = UNSET,
    user_nickname: Union[Unset, str] = UNSET,
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    status: Union[Unset, AdminFilterRequestsByEndpointStatus] = UNSET,
    request_id: Union[Unset, UUID] = UNSET,
    runner_id: Union[Unset, str] = UNSET,
    sort_by: Union[Unset, AdminFilterRequestsByEndpointSortBy] = AdminFilterRequestsByEndpointSortBy.ENDED_AT,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, PageTypeVarCustomizedGatewayRequestItem]]:
    """Filter Requests By Endpoint

    Args:
        endpoint (Union[Unset, str]):
        user_nickname (Union[Unset, str]):
        start_time (Union[Unset, datetime.datetime]):
        end_time (Union[Unset, datetime.datetime]):
        status (Union[Unset, AdminFilterRequestsByEndpointStatus]):
        request_id (Union[Unset, UUID]):
        runner_id (Union[Unset, str]):
        sort_by (Union[Unset, AdminFilterRequestsByEndpointSortBy]):  Default:
            AdminFilterRequestsByEndpointSortBy.ENDED_AT.
        page (Union[Unset, int]): Page number Default: 1.
        size (Union[Unset, int]): Page size Default: 50.
        body (list[int]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, PageTypeVarCustomizedGatewayRequestItem]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            endpoint=endpoint,
            user_nickname=user_nickname,
            start_time=start_time,
            end_time=end_time,
            status=status,
            request_id=request_id,
            runner_id=runner_id,
            sort_by=sort_by,
            page=page,
            size=size,
        )
    ).parsed
