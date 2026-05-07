from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.installation_response import InstallationResponse
from ...models.patch_vercel_installation_request import PatchVercelInstallationRequest
from ...types import Response


def _get_kwargs(
    installation_id: str,
    *,
    body: PatchVercelInstallationRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "patch",
        "url": f"/vercel/v1/installations/{installation_id}",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, InstallationResponse]]:
    if response.status_code == 200:
        response_200 = InstallationResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, InstallationResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    installation_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: PatchVercelInstallationRequest,
) -> Response[Union[HTTPValidationError, InstallationResponse]]:
    """Patch Installation Plan

    Args:
        installation_id (str):
        body (PatchVercelInstallationRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, InstallationResponse]]
    """

    kwargs = _get_kwargs(
        installation_id=installation_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    installation_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: PatchVercelInstallationRequest,
) -> Optional[Union[HTTPValidationError, InstallationResponse]]:
    """Patch Installation Plan

    Args:
        installation_id (str):
        body (PatchVercelInstallationRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, InstallationResponse]
    """

    return sync_detailed(
        installation_id=installation_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    installation_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: PatchVercelInstallationRequest,
) -> Response[Union[HTTPValidationError, InstallationResponse]]:
    """Patch Installation Plan

    Args:
        installation_id (str):
        body (PatchVercelInstallationRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, InstallationResponse]]
    """

    kwargs = _get_kwargs(
        installation_id=installation_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    installation_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: PatchVercelInstallationRequest,
) -> Optional[Union[HTTPValidationError, InstallationResponse]]:
    """Patch Installation Plan

    Args:
        installation_id (str):
        body (PatchVercelInstallationRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, InstallationResponse]
    """

    return (
        await asyncio_detailed(
            installation_id=installation_id,
            client=client,
            body=body,
        )
    ).parsed
