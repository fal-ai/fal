import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_runners_owner import ListRunnersOwner
from ...models.list_runners_state import ListRunnersState
from ...models.runner_list_response import RunnerListResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    application: Union[Unset, str] = UNSET,
    state: Union[Unset, ListRunnersState] = UNSET,
    owner: Union[Unset, ListRunnersOwner] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

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
        "url": "/runners/",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, RunnerListResponse]]:
    if response.status_code == 200:
        response_200 = RunnerListResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, RunnerListResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    application: Union[Unset, str] = UNSET,
    state: Union[Unset, ListRunnersState] = UNSET,
    owner: Union[Unset, ListRunnersOwner] = UNSET,
) -> Response[Union[HTTPValidationError, RunnerListResponse]]:
    """List Runners

    Args:
        start_time (Union[Unset, datetime.datetime]):
        end_time (Union[Unset, datetime.datetime]):
        application (Union[Unset, str]): Filter by application name
        state (Union[Unset, ListRunnersState]): Filter by state
        owner (Union[Unset, ListRunnersOwner]): Filter by owner

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, RunnerListResponse]]
    """

    kwargs = _get_kwargs(
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
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    application: Union[Unset, str] = UNSET,
    state: Union[Unset, ListRunnersState] = UNSET,
    owner: Union[Unset, ListRunnersOwner] = UNSET,
) -> Optional[Union[HTTPValidationError, RunnerListResponse]]:
    """List Runners

    Args:
        start_time (Union[Unset, datetime.datetime]):
        end_time (Union[Unset, datetime.datetime]):
        application (Union[Unset, str]): Filter by application name
        state (Union[Unset, ListRunnersState]): Filter by state
        owner (Union[Unset, ListRunnersOwner]): Filter by owner

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, RunnerListResponse]
    """

    return sync_detailed(
        client=client,
        start_time=start_time,
        end_time=end_time,
        application=application,
        state=state,
        owner=owner,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    application: Union[Unset, str] = UNSET,
    state: Union[Unset, ListRunnersState] = UNSET,
    owner: Union[Unset, ListRunnersOwner] = UNSET,
) -> Response[Union[HTTPValidationError, RunnerListResponse]]:
    """List Runners

    Args:
        start_time (Union[Unset, datetime.datetime]):
        end_time (Union[Unset, datetime.datetime]):
        application (Union[Unset, str]): Filter by application name
        state (Union[Unset, ListRunnersState]): Filter by state
        owner (Union[Unset, ListRunnersOwner]): Filter by owner

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, RunnerListResponse]]
    """

    kwargs = _get_kwargs(
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
    start_time: Union[Unset, datetime.datetime] = UNSET,
    end_time: Union[Unset, datetime.datetime] = UNSET,
    application: Union[Unset, str] = UNSET,
    state: Union[Unset, ListRunnersState] = UNSET,
    owner: Union[Unset, ListRunnersOwner] = UNSET,
) -> Optional[Union[HTTPValidationError, RunnerListResponse]]:
    """List Runners

    Args:
        start_time (Union[Unset, datetime.datetime]):
        end_time (Union[Unset, datetime.datetime]):
        application (Union[Unset, str]): Filter by application name
        state (Union[Unset, ListRunnersState]): Filter by state
        owner (Union[Unset, ListRunnersOwner]): Filter by owner

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, RunnerListResponse]
    """

    return (
        await asyncio_detailed(
            client=client,
            start_time=start_time,
            end_time=end_time,
            application=application,
            state=state,
            owner=owner,
        )
    ).parsed
