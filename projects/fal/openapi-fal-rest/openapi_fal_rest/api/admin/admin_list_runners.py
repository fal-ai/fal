import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.admin_list_runners_owner import AdminListRunnersOwner
from ...models.admin_list_runners_state import AdminListRunnersState
from ...models.admin_runner_list_response import AdminRunnerListResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    user_id: str,
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    application: Union[Unset, str] = UNSET,
    state: Union[Unset, AdminListRunnersState] = UNSET,
    owner: Union[Unset, AdminListRunnersOwner] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["user_id"] = user_id

    json_start_time: Union[Unset, str] = UNSET
    if not isinstance(start_time, Unset):
        json_start_time = start_time.isoformat()
    params["start_time"] = json_start_time

    json_end_time: Union[Unset, str] = UNSET
    if not isinstance(end_time, Unset):
        json_end_time = end_time.isoformat()
    params["end_time"] = json_end_time

    params["application"] = application

    json_state: Union[Unset, str] = UNSET
    if not isinstance(state, Unset):
        json_state = state.value

    params["state"] = json_state

    json_owner: Union[Unset, str] = UNSET
    if not isinstance(owner, Unset):
        json_owner = owner.value

    params["owner"] = json_owner

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/admin/runners",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[AdminRunnerListResponse, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = AdminRunnerListResponse.from_dict(response.json())

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
) -> Response[Union[AdminRunnerListResponse, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    application: Union[Unset, str] = UNSET,
    state: Union[Unset, AdminListRunnersState] = UNSET,
    owner: Union[Unset, AdminListRunnersOwner] = UNSET,
) -> Response[Union[AdminRunnerListResponse, HTTPValidationError]]:
    """List Runners

     Admin endpoint to list runners for a specific user.

    Args:
        user_id (str): The user ID to list runners for
        start_time (Union[Unset, datetime.datetime]): Start time for historical runners
        end_time (Union[Unset, datetime.datetime]): End time for historical runners
        application (Union[Unset, str]): Filter by application name
        state (Union[Unset, AdminListRunnersState]): Filter by state
        owner (Union[Unset, AdminListRunnersOwner]): Filter by owner

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AdminRunnerListResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        start_time=start_time,
        end_time=end_time,
        application=application,
        state=state,
        owner=owner,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    application: Union[Unset, str] = UNSET,
    state: Union[Unset, AdminListRunnersState] = UNSET,
    owner: Union[Unset, AdminListRunnersOwner] = UNSET,
) -> Optional[Union[AdminRunnerListResponse, HTTPValidationError]]:
    """List Runners

     Admin endpoint to list runners for a specific user.

    Args:
        user_id (str): The user ID to list runners for
        start_time (Union[Unset, datetime.datetime]): Start time for historical runners
        end_time (Union[Unset, datetime.datetime]): End time for historical runners
        application (Union[Unset, str]): Filter by application name
        state (Union[Unset, AdminListRunnersState]): Filter by state
        owner (Union[Unset, AdminListRunnersOwner]): Filter by owner

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AdminRunnerListResponse, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        user_id=user_id,
        start_time=start_time,
        end_time=end_time,
        application=application,
        state=state,
        owner=owner,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    application: Union[Unset, str] = UNSET,
    state: Union[Unset, AdminListRunnersState] = UNSET,
    owner: Union[Unset, AdminListRunnersOwner] = UNSET,
) -> Response[Union[AdminRunnerListResponse, HTTPValidationError]]:
    """List Runners

     Admin endpoint to list runners for a specific user.

    Args:
        user_id (str): The user ID to list runners for
        start_time (Union[Unset, datetime.datetime]): Start time for historical runners
        end_time (Union[Unset, datetime.datetime]): End time for historical runners
        application (Union[Unset, str]): Filter by application name
        state (Union[Unset, AdminListRunnersState]): Filter by state
        owner (Union[Unset, AdminListRunnersOwner]): Filter by owner

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AdminRunnerListResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        start_time=start_time,
        end_time=end_time,
        application=application,
        state=state,
        owner=owner,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    application: Union[Unset, str] = UNSET,
    state: Union[Unset, AdminListRunnersState] = UNSET,
    owner: Union[Unset, AdminListRunnersOwner] = UNSET,
) -> Optional[Union[AdminRunnerListResponse, HTTPValidationError]]:
    """List Runners

     Admin endpoint to list runners for a specific user.

    Args:
        user_id (str): The user ID to list runners for
        start_time (Union[Unset, datetime.datetime]): Start time for historical runners
        end_time (Union[Unset, datetime.datetime]): End time for historical runners
        application (Union[Unset, str]): Filter by application name
        state (Union[Unset, AdminListRunnersState]): Filter by state
        owner (Union[Unset, AdminListRunnersOwner]): Filter by owner

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AdminRunnerListResponse, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            application=application,
            state=state,
            owner=owner,
        )
    ).parsed
