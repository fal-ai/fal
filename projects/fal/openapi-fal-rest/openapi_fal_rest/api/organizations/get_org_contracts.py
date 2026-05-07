from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.org_contract import OrgContract
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    team_user_str: Union[Unset, str] = UNSET,
    status: Union[Unset, Any] = UNSET,
    current_only: Union[Unset, bool] = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["team_user_str"] = team_user_str

    params["status"] = status

    params["current_only"] = current_only

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/organizations/contracts",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list["OrgContract"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = OrgContract.from_dict(response_200_item_data)

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
) -> Response[Union[HTTPValidationError, list["OrgContract"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    team_user_str: Union[Unset, str] = UNSET,
    status: Union[Unset, Any] = UNSET,
    current_only: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, list["OrgContract"]]]:
    """Get Org Contracts

     List enterprise contracts for the organization. Requires org admin or billing role.

    Args:
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)
        status (Union[Unset, Any]): Filter by contract status
        current_only (Union[Unset, bool]): Only return currently active contracts (started and not
            ended) Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['OrgContract']]]
    """

    kwargs = _get_kwargs(
        team_user_str=team_user_str,
        status=status,
        current_only=current_only,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    team_user_str: Union[Unset, str] = UNSET,
    status: Union[Unset, Any] = UNSET,
    current_only: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, list["OrgContract"]]]:
    """Get Org Contracts

     List enterprise contracts for the organization. Requires org admin or billing role.

    Args:
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)
        status (Union[Unset, Any]): Filter by contract status
        current_only (Union[Unset, bool]): Only return currently active contracts (started and not
            ended) Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['OrgContract']]
    """

    return sync_detailed(
        client=client,
        team_user_str=team_user_str,
        status=status,
        current_only=current_only,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    team_user_str: Union[Unset, str] = UNSET,
    status: Union[Unset, Any] = UNSET,
    current_only: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, list["OrgContract"]]]:
    """Get Org Contracts

     List enterprise contracts for the organization. Requires org admin or billing role.

    Args:
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)
        status (Union[Unset, Any]): Filter by contract status
        current_only (Union[Unset, bool]): Only return currently active contracts (started and not
            ended) Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['OrgContract']]]
    """

    kwargs = _get_kwargs(
        team_user_str=team_user_str,
        status=status,
        current_only=current_only,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    team_user_str: Union[Unset, str] = UNSET,
    status: Union[Unset, Any] = UNSET,
    current_only: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, list["OrgContract"]]]:
    """Get Org Contracts

     List enterprise contracts for the organization. Requires org admin or billing role.

    Args:
        team_user_str (Union[Unset, str]): Filter by team (user_id or nickname)
        status (Union[Unset, Any]): Filter by contract status
        current_only (Union[Unset, bool]): Only return currently active contracts (started and not
            ended) Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['OrgContract']]
    """

    return (
        await asyncio_detailed(
            client=client,
            team_user_str=team_user_str,
            status=status,
            current_only=current_only,
        )
    ).parsed
