from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.current_user import CurrentUser
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response


def _get_kwargs(
    *,
    user_id: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["user_id"] = user_id

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/admin/users/migrate",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[CurrentUser, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = CurrentUser.from_dict(response.json())

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
) -> Response[Union[CurrentUser, HTTPValidationError]]:
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
) -> Response[Union[CurrentUser, HTTPValidationError]]:
    """Migrate User Personal Team

     This function receives a user_id of a personal user and migrates it to become a
    team account. This helps users that have been using personal users for their setup
    for a while and want to invite people over but don't want to have to migrate things
    manually. The process is
    1. create a new personal user for the auth_user
    2. give control to the auth_user over that new personal user
    3. make the old personal user now a team user
    4. create a Stripe customer for the new personal user

    Args:
        user_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CurrentUser, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
) -> Optional[Union[CurrentUser, HTTPValidationError]]:
    """Migrate User Personal Team

     This function receives a user_id of a personal user and migrates it to become a
    team account. This helps users that have been using personal users for their setup
    for a while and want to invite people over but don't want to have to migrate things
    manually. The process is
    1. create a new personal user for the auth_user
    2. give control to the auth_user over that new personal user
    3. make the old personal user now a team user
    4. create a Stripe customer for the new personal user

    Args:
        user_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CurrentUser, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        user_id=user_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
) -> Response[Union[CurrentUser, HTTPValidationError]]:
    """Migrate User Personal Team

     This function receives a user_id of a personal user and migrates it to become a
    team account. This helps users that have been using personal users for their setup
    for a while and want to invite people over but don't want to have to migrate things
    manually. The process is
    1. create a new personal user for the auth_user
    2. give control to the auth_user over that new personal user
    3. make the old personal user now a team user
    4. create a Stripe customer for the new personal user

    Args:
        user_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CurrentUser, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    user_id: str,
) -> Optional[Union[CurrentUser, HTTPValidationError]]:
    """Migrate User Personal Team

     This function receives a user_id of a personal user and migrates it to become a
    team account. This helps users that have been using personal users for their setup
    for a while and want to invite people over but don't want to have to migrate things
    manually. The process is
    1. create a new personal user for the auth_user
    2. give control to the auth_user over that new personal user
    3. make the old personal user now a team user
    4. create a Stripe customer for the new personal user

    Args:
        user_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CurrentUser, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            user_id=user_id,
        )
    ).parsed
