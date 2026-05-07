from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    team_user_str: str,
    member_user_str: str,
    *,
    send_email: Union[Unset, bool] = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["send_email"] = send_email

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": f"/organizations/teams/{team_user_str}/members/{member_user_str}",
        "params": params,
    }

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
    team_user_str: str,
    member_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    send_email: Union[Unset, bool] = False,
) -> Response[Union[Any, HTTPValidationError]]:
    """Org Remove Team Member

     Remove a member from a team.

    Org admins can remove members from teams within their organization.
    Member is identified by personal user_id or nickname.
    Requires the caller to be an org admin.

    Args:
        team_user_str (str):
        member_user_str (str):
        send_email (Union[Unset, bool]): Send removal notification email Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        team_user_str=team_user_str,
        member_user_str=member_user_str,
        send_email=send_email,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    team_user_str: str,
    member_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    send_email: Union[Unset, bool] = False,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Org Remove Team Member

     Remove a member from a team.

    Org admins can remove members from teams within their organization.
    Member is identified by personal user_id or nickname.
    Requires the caller to be an org admin.

    Args:
        team_user_str (str):
        member_user_str (str):
        send_email (Union[Unset, bool]): Send removal notification email Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        team_user_str=team_user_str,
        member_user_str=member_user_str,
        client=client,
        send_email=send_email,
    ).parsed


async def asyncio_detailed(
    team_user_str: str,
    member_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    send_email: Union[Unset, bool] = False,
) -> Response[Union[Any, HTTPValidationError]]:
    """Org Remove Team Member

     Remove a member from a team.

    Org admins can remove members from teams within their organization.
    Member is identified by personal user_id or nickname.
    Requires the caller to be an org admin.

    Args:
        team_user_str (str):
        member_user_str (str):
        send_email (Union[Unset, bool]): Send removal notification email Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        team_user_str=team_user_str,
        member_user_str=member_user_str,
        send_email=send_email,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    team_user_str: str,
    member_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    send_email: Union[Unset, bool] = False,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Org Remove Team Member

     Remove a member from a team.

    Org admins can remove members from teams within their organization.
    Member is identified by personal user_id or nickname.
    Requires the caller to be an org admin.

    Args:
        team_user_str (str):
        member_user_str (str):
        send_email (Union[Unset, bool]): Send removal notification email Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            team_user_str=team_user_str,
            member_user_str=member_user_str,
            client=client,
            send_email=send_email,
        )
    ).parsed
