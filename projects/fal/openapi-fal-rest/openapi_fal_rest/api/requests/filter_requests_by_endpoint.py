import datetime
from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.filter_requests_by_endpoint_sort_by import FilterRequestsByEndpointSortBy
from ...models.filter_requests_by_endpoint_status import FilterRequestsByEndpointStatus
from ...models.http_validation_error import HTTPValidationError
from ...models.page_type_var_customized_gateway_request_item import PageTypeVarCustomizedGatewayRequestItem
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: list[int],
    endpoint: str,
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    status: Union[Unset, FilterRequestsByEndpointStatus] = UNSET,
    request_id: Union[Unset, UUID] = UNSET,
    error_type: Union[Unset, str] = UNSET,
    payloads: Union[Unset, bool] = False,
    sort_by: Union[Unset, FilterRequestsByEndpointSortBy] = FilterRequestsByEndpointSortBy.ENDED_AT,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    params: dict[str, Any] = {}

    params["endpoint"] = endpoint

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

    params["error_type"] = error_type

    params["payloads"] = payloads

    json_sort_by: Union[Unset, str] = UNSET
    if not isinstance(sort_by, Unset):
        json_sort_by = sort_by.value

    params["sort_by"] = json_sort_by

    params["page"] = page

    params["size"] = size

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/requests/by-endpoint",
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
    endpoint: str,
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    status: Union[Unset, FilterRequestsByEndpointStatus] = UNSET,
    request_id: Union[Unset, UUID] = UNSET,
    error_type: Union[Unset, str] = UNSET,
    payloads: Union[Unset, bool] = False,
    sort_by: Union[Unset, FilterRequestsByEndpointSortBy] = FilterRequestsByEndpointSortBy.ENDED_AT,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, PageTypeVarCustomizedGatewayRequestItem]]:
    """Filter Requests By Endpoint

    Args:
        endpoint (str):
        start_time (Union[Unset, datetime.datetime]):
        end_time (Union[Unset, datetime.datetime]):
        status (Union[Unset, FilterRequestsByEndpointStatus]):
        request_id (Union[Unset, UUID]):
        error_type (Union[Unset, str]):
        payloads (Union[Unset, bool]):  Default: False.
        sort_by (Union[Unset, FilterRequestsByEndpointSortBy]):  Default:
            FilterRequestsByEndpointSortBy.ENDED_AT.
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
        start_time=start_time,
        end_time=end_time,
        status=status,
        request_id=request_id,
        error_type=error_type,
        payloads=payloads,
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
    endpoint: str,
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    status: Union[Unset, FilterRequestsByEndpointStatus] = UNSET,
    request_id: Union[Unset, UUID] = UNSET,
    error_type: Union[Unset, str] = UNSET,
    payloads: Union[Unset, bool] = False,
    sort_by: Union[Unset, FilterRequestsByEndpointSortBy] = FilterRequestsByEndpointSortBy.ENDED_AT,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, PageTypeVarCustomizedGatewayRequestItem]]:
    """Filter Requests By Endpoint

    Args:
        endpoint (str):
        start_time (Union[Unset, datetime.datetime]):
        end_time (Union[Unset, datetime.datetime]):
        status (Union[Unset, FilterRequestsByEndpointStatus]):
        request_id (Union[Unset, UUID]):
        error_type (Union[Unset, str]):
        payloads (Union[Unset, bool]):  Default: False.
        sort_by (Union[Unset, FilterRequestsByEndpointSortBy]):  Default:
            FilterRequestsByEndpointSortBy.ENDED_AT.
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
        start_time=start_time,
        end_time=end_time,
        status=status,
        request_id=request_id,
        error_type=error_type,
        payloads=payloads,
        sort_by=sort_by,
        page=page,
        size=size,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: list[int],
    endpoint: str,
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    status: Union[Unset, FilterRequestsByEndpointStatus] = UNSET,
    request_id: Union[Unset, UUID] = UNSET,
    error_type: Union[Unset, str] = UNSET,
    payloads: Union[Unset, bool] = False,
    sort_by: Union[Unset, FilterRequestsByEndpointSortBy] = FilterRequestsByEndpointSortBy.ENDED_AT,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Response[Union[HTTPValidationError, PageTypeVarCustomizedGatewayRequestItem]]:
    """Filter Requests By Endpoint

    Args:
        endpoint (str):
        start_time (Union[Unset, datetime.datetime]):
        end_time (Union[Unset, datetime.datetime]):
        status (Union[Unset, FilterRequestsByEndpointStatus]):
        request_id (Union[Unset, UUID]):
        error_type (Union[Unset, str]):
        payloads (Union[Unset, bool]):  Default: False.
        sort_by (Union[Unset, FilterRequestsByEndpointSortBy]):  Default:
            FilterRequestsByEndpointSortBy.ENDED_AT.
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
        start_time=start_time,
        end_time=end_time,
        status=status,
        request_id=request_id,
        error_type=error_type,
        payloads=payloads,
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
    endpoint: str,
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    status: Union[Unset, FilterRequestsByEndpointStatus] = UNSET,
    request_id: Union[Unset, UUID] = UNSET,
    error_type: Union[Unset, str] = UNSET,
    payloads: Union[Unset, bool] = False,
    sort_by: Union[Unset, FilterRequestsByEndpointSortBy] = FilterRequestsByEndpointSortBy.ENDED_AT,
    page: Union[Unset, int] = 1,
    size: Union[Unset, int] = 50,
) -> Optional[Union[HTTPValidationError, PageTypeVarCustomizedGatewayRequestItem]]:
    """Filter Requests By Endpoint

    Args:
        endpoint (str):
        start_time (Union[Unset, datetime.datetime]):
        end_time (Union[Unset, datetime.datetime]):
        status (Union[Unset, FilterRequestsByEndpointStatus]):
        request_id (Union[Unset, UUID]):
        error_type (Union[Unset, str]):
        payloads (Union[Unset, bool]):  Default: False.
        sort_by (Union[Unset, FilterRequestsByEndpointSortBy]):  Default:
            FilterRequestsByEndpointSortBy.ENDED_AT.
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
            start_time=start_time,
            end_time=end_time,
            status=status,
            request_id=request_id,
            error_type=error_type,
            payloads=payloads,
            sort_by=sort_by,
            page=page,
            size=size,
        )
    ).parsed
