from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.legacy_key_info import LegacyKeyInfo
from ...models.policy_key_info import PolicyKeyInfo
from ...models.preset_key_info import PresetKeyInfo
from ...types import UNSET, Response, Unset


def _get_kwargs(
    team_user_str: str,
    *,
    include_revoked: Union[Unset, bool] = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["include_revoked"] = include_revoked

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/organizations/teams/{team_user_str}/keys",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list[Union["LegacyKeyInfo", "PolicyKeyInfo", "PresetKeyInfo"]]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:

            def _parse_response_200_item(data: object) -> Union["LegacyKeyInfo", "PolicyKeyInfo", "PresetKeyInfo"]:
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    response_200_item_type_0 = LegacyKeyInfo.from_dict(data)

                    return response_200_item_type_0
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    response_200_item_type_1 = PresetKeyInfo.from_dict(data)

                    return response_200_item_type_1
                except:  # noqa: E722
                    pass
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_item_type_2 = PolicyKeyInfo.from_dict(data)

                return response_200_item_type_2

            response_200_item = _parse_response_200_item(response_200_item_data)

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
) -> Response[Union[HTTPValidationError, list[Union["LegacyKeyInfo", "PolicyKeyInfo", "PresetKeyInfo"]]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    team_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    include_revoked: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, list[Union["LegacyKeyInfo", "PolicyKeyInfo", "PresetKeyInfo"]]]]:
    """Org List Team Keys

     List API keys for a team within the organization.

    Org admins or billing role can list keys for any team in their organization.
    Team is identified by user_id or nickname.

    Args:
        team_user_str (str):
        include_revoked (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list[Union['LegacyKeyInfo', 'PolicyKeyInfo', 'PresetKeyInfo']]]]
    """

    kwargs = _get_kwargs(
        team_user_str=team_user_str,
        include_revoked=include_revoked,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    team_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    include_revoked: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, list[Union["LegacyKeyInfo", "PolicyKeyInfo", "PresetKeyInfo"]]]]:
    """Org List Team Keys

     List API keys for a team within the organization.

    Org admins or billing role can list keys for any team in their organization.
    Team is identified by user_id or nickname.

    Args:
        team_user_str (str):
        include_revoked (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list[Union['LegacyKeyInfo', 'PolicyKeyInfo', 'PresetKeyInfo']]]
    """

    return sync_detailed(
        team_user_str=team_user_str,
        client=client,
        include_revoked=include_revoked,
    ).parsed


async def asyncio_detailed(
    team_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    include_revoked: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, list[Union["LegacyKeyInfo", "PolicyKeyInfo", "PresetKeyInfo"]]]]:
    """Org List Team Keys

     List API keys for a team within the organization.

    Org admins or billing role can list keys for any team in their organization.
    Team is identified by user_id or nickname.

    Args:
        team_user_str (str):
        include_revoked (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list[Union['LegacyKeyInfo', 'PolicyKeyInfo', 'PresetKeyInfo']]]]
    """

    kwargs = _get_kwargs(
        team_user_str=team_user_str,
        include_revoked=include_revoked,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    team_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    include_revoked: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, list[Union["LegacyKeyInfo", "PolicyKeyInfo", "PresetKeyInfo"]]]]:
    """Org List Team Keys

     List API keys for a team within the organization.

    Org admins or billing role can list keys for any team in their organization.
    Team is identified by user_id or nickname.

    Args:
        team_user_str (str):
        include_revoked (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list[Union['LegacyKeyInfo', 'PolicyKeyInfo', 'PresetKeyInfo']]]
    """

    return (
        await asyncio_detailed(
            team_user_str=team_user_str,
            client=client,
            include_revoked=include_revoked,
        )
    ).parsed
