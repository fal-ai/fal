from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.sync_settings_request import SyncSettingsRequest
from ...models.sync_settings_response import SyncSettingsResponse
from ...types import Response


def _get_kwargs(
    org_user_str: str,
    *,
    body: SyncSettingsRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/admin/organizations/{org_user_str}/actions/sync-settings",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, SyncSettingsResponse]]:
    if response.status_code == 200:
        response_200 = SyncSettingsResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, SyncSettingsResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    org_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: SyncSettingsRequest,
) -> Response[Union[HTTPValidationError, SyncSettingsResponse]]:
    """Sync Org Settings

     Sync organization settings to all teams.

    Each setting can be individually opted-in for syncing via the request body.
    Settings not included or set to False will be skipped.

    Available settings to sync:
    - sync_account_type: Sync account_type from org to all teams
    - sync_model_access_controls_enabled: Sync the model access controls feature flag

    Teams already matching the org settings are counted as skipped.
    Supports lookup by user_id or nickname.

    Args:
        org_user_str (str):
        body (SyncSettingsRequest): Request to sync organization settings to teams.

            Each setting can be individually opted-in for syncing.
            Settings not included or set to False will be skipped.

            Attributes:
                sync_account_type: Sync the org's account_type to all child teams.
                sync_model_access_controls_enabled: Sync whether model access controls
                    are enabled (the feature flag), not the actual control entries.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, SyncSettingsResponse]]
    """

    kwargs = _get_kwargs(
        org_user_str=org_user_str,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    org_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: SyncSettingsRequest,
) -> Optional[Union[HTTPValidationError, SyncSettingsResponse]]:
    """Sync Org Settings

     Sync organization settings to all teams.

    Each setting can be individually opted-in for syncing via the request body.
    Settings not included or set to False will be skipped.

    Available settings to sync:
    - sync_account_type: Sync account_type from org to all teams
    - sync_model_access_controls_enabled: Sync the model access controls feature flag

    Teams already matching the org settings are counted as skipped.
    Supports lookup by user_id or nickname.

    Args:
        org_user_str (str):
        body (SyncSettingsRequest): Request to sync organization settings to teams.

            Each setting can be individually opted-in for syncing.
            Settings not included or set to False will be skipped.

            Attributes:
                sync_account_type: Sync the org's account_type to all child teams.
                sync_model_access_controls_enabled: Sync whether model access controls
                    are enabled (the feature flag), not the actual control entries.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, SyncSettingsResponse]
    """

    return sync_detailed(
        org_user_str=org_user_str,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    org_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: SyncSettingsRequest,
) -> Response[Union[HTTPValidationError, SyncSettingsResponse]]:
    """Sync Org Settings

     Sync organization settings to all teams.

    Each setting can be individually opted-in for syncing via the request body.
    Settings not included or set to False will be skipped.

    Available settings to sync:
    - sync_account_type: Sync account_type from org to all teams
    - sync_model_access_controls_enabled: Sync the model access controls feature flag

    Teams already matching the org settings are counted as skipped.
    Supports lookup by user_id or nickname.

    Args:
        org_user_str (str):
        body (SyncSettingsRequest): Request to sync organization settings to teams.

            Each setting can be individually opted-in for syncing.
            Settings not included or set to False will be skipped.

            Attributes:
                sync_account_type: Sync the org's account_type to all child teams.
                sync_model_access_controls_enabled: Sync whether model access controls
                    are enabled (the feature flag), not the actual control entries.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, SyncSettingsResponse]]
    """

    kwargs = _get_kwargs(
        org_user_str=org_user_str,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    org_user_str: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: SyncSettingsRequest,
) -> Optional[Union[HTTPValidationError, SyncSettingsResponse]]:
    """Sync Org Settings

     Sync organization settings to all teams.

    Each setting can be individually opted-in for syncing via the request body.
    Settings not included or set to False will be skipped.

    Available settings to sync:
    - sync_account_type: Sync account_type from org to all teams
    - sync_model_access_controls_enabled: Sync the model access controls feature flag

    Teams already matching the org settings are counted as skipped.
    Supports lookup by user_id or nickname.

    Args:
        org_user_str (str):
        body (SyncSettingsRequest): Request to sync organization settings to teams.

            Each setting can be individually opted-in for syncing.
            Settings not included or set to False will be skipped.

            Attributes:
                sync_account_type: Sync the org's account_type to all child teams.
                sync_model_access_controls_enabled: Sync whether model access controls
                    are enabled (the feature flag), not the actual control entries.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, SyncSettingsResponse]
    """

    return (
        await asyncio_detailed(
            org_user_str=org_user_str,
            client=client,
            body=body,
        )
    ).parsed
