from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.basic_user_info import BasicUserInfo
from ...models.http_validation_error import HTTPValidationError
from ...models.lock_reason import LockReason
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    limit: Union[Unset, int] = 100,
    offset: Union[Unset, int] = 0,
    is_locked: Union[Unset, bool] = UNSET,
    lock_reason: Union[Unset, LockReason] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["limit"] = limit

    params["offset"] = offset

    params["is_locked"] = is_locked

    json_lock_reason: Union[Unset, str] = UNSET
    if not isinstance(lock_reason, Unset):
        json_lock_reason = lock_reason.value

    params["lock_reason"] = json_lock_reason

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/admin/users/",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list["BasicUserInfo"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = BasicUserInfo.from_dict(response_200_item_data)

            response_200.append(response_200_item)

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
) -> Response[Union[HTTPValidationError, list["BasicUserInfo"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    limit: Union[Unset, int] = 100,
    offset: Union[Unset, int] = 0,
    is_locked: Union[Unset, bool] = UNSET,
    lock_reason: Union[Unset, LockReason] = UNSET,
) -> Response[Union[HTTPValidationError, list["BasicUserInfo"]]]:
    """Get Users

    Args:
        limit (Union[Unset, int]):  Default: 100.
        offset (Union[Unset, int]):  Default: 0.
        is_locked (Union[Unset, bool]):
        lock_reason (Union[Unset, LockReason]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['BasicUserInfo']]]
    """

    kwargs = _get_kwargs(
        limit=limit,
        offset=offset,
        is_locked=is_locked,
        lock_reason=lock_reason,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    limit: Union[Unset, int] = 100,
    offset: Union[Unset, int] = 0,
    is_locked: Union[Unset, bool] = UNSET,
    lock_reason: Union[Unset, LockReason] = UNSET,
) -> Optional[Union[HTTPValidationError, list["BasicUserInfo"]]]:
    """Get Users

    Args:
        limit (Union[Unset, int]):  Default: 100.
        offset (Union[Unset, int]):  Default: 0.
        is_locked (Union[Unset, bool]):
        lock_reason (Union[Unset, LockReason]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['BasicUserInfo']]
    """

    return sync_detailed(
        client=client,
        limit=limit,
        offset=offset,
        is_locked=is_locked,
        lock_reason=lock_reason,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    limit: Union[Unset, int] = 100,
    offset: Union[Unset, int] = 0,
    is_locked: Union[Unset, bool] = UNSET,
    lock_reason: Union[Unset, LockReason] = UNSET,
) -> Response[Union[HTTPValidationError, list["BasicUserInfo"]]]:
    """Get Users

    Args:
        limit (Union[Unset, int]):  Default: 100.
        offset (Union[Unset, int]):  Default: 0.
        is_locked (Union[Unset, bool]):
        lock_reason (Union[Unset, LockReason]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['BasicUserInfo']]]
    """

    kwargs = _get_kwargs(
        limit=limit,
        offset=offset,
        is_locked=is_locked,
        lock_reason=lock_reason,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    limit: Union[Unset, int] = 100,
    offset: Union[Unset, int] = 0,
    is_locked: Union[Unset, bool] = UNSET,
    lock_reason: Union[Unset, LockReason] = UNSET,
) -> Optional[Union[HTTPValidationError, list["BasicUserInfo"]]]:
    """Get Users

    Args:
        limit (Union[Unset, int]):  Default: 100.
        offset (Union[Unset, int]):  Default: 0.
        is_locked (Union[Unset, bool]):
        lock_reason (Union[Unset, LockReason]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['BasicUserInfo']]
    """

    return (
        await asyncio_detailed(
            client=client,
            limit=limit,
            offset=offset,
            is_locked=is_locked,
            lock_reason=lock_reason,
        )
    ).parsed
