from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.application_modifiable_info import ApplicationModifiableInfo
from ...models.deployment_strategy import DeploymentStrategy
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    app_user_id: str,
    app_alias: str,
    revision_id: str,
    *,
    deployment_strategy: Union[Unset, DeploymentStrategy] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_deployment_strategy: Union[Unset, str] = UNSET
    if not isinstance(deployment_strategy, Unset):
        json_deployment_strategy = deployment_strategy.value

    params["deployment_strategy"] = json_deployment_strategy

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": f"/applications/{app_user_id}/{app_alias}/{revision_id}",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[ApplicationModifiableInfo, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = ApplicationModifiableInfo.from_dict(response.json())

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
) -> Response[Union[ApplicationModifiableInfo, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    app_user_id: str,
    app_alias: str,
    revision_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    deployment_strategy: Union[Unset, DeploymentStrategy] = UNSET,
) -> Response[Union[ApplicationModifiableInfo, HTTPValidationError]]:
    """Set Revision

    Args:
        app_user_id (str):
        app_alias (str):
        revision_id (str):
        deployment_strategy (Union[Unset, DeploymentStrategy]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[ApplicationModifiableInfo, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        app_user_id=app_user_id,
        app_alias=app_alias,
        revision_id=revision_id,
        deployment_strategy=deployment_strategy,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    app_user_id: str,
    app_alias: str,
    revision_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    deployment_strategy: Union[Unset, DeploymentStrategy] = UNSET,
) -> Optional[Union[ApplicationModifiableInfo, HTTPValidationError]]:
    """Set Revision

    Args:
        app_user_id (str):
        app_alias (str):
        revision_id (str):
        deployment_strategy (Union[Unset, DeploymentStrategy]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[ApplicationModifiableInfo, HTTPValidationError]
    """

    return sync_detailed(
        app_user_id=app_user_id,
        app_alias=app_alias,
        revision_id=revision_id,
        client=client,
        deployment_strategy=deployment_strategy,
    ).parsed


async def asyncio_detailed(
    app_user_id: str,
    app_alias: str,
    revision_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    deployment_strategy: Union[Unset, DeploymentStrategy] = UNSET,
) -> Response[Union[ApplicationModifiableInfo, HTTPValidationError]]:
    """Set Revision

    Args:
        app_user_id (str):
        app_alias (str):
        revision_id (str):
        deployment_strategy (Union[Unset, DeploymentStrategy]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[ApplicationModifiableInfo, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        app_user_id=app_user_id,
        app_alias=app_alias,
        revision_id=revision_id,
        deployment_strategy=deployment_strategy,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    app_user_id: str,
    app_alias: str,
    revision_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    deployment_strategy: Union[Unset, DeploymentStrategy] = UNSET,
) -> Optional[Union[ApplicationModifiableInfo, HTTPValidationError]]:
    """Set Revision

    Args:
        app_user_id (str):
        app_alias (str):
        revision_id (str):
        deployment_strategy (Union[Unset, DeploymentStrategy]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[ApplicationModifiableInfo, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            app_user_id=app_user_id,
            app_alias=app_alias,
            revision_id=revision_id,
            client=client,
            deployment_strategy=deployment_strategy,
        )
    ).parsed
