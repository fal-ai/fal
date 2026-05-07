from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.object_lifecycle_preference_data import ObjectLifecyclePreferenceData
from ...models.save_object_lifecycle_preference_response_save_object_lifecycle_preference import (
    SaveObjectLifecyclePreferenceResponseSaveObjectLifecyclePreference,
)
from ...types import Response


def _get_kwargs(
    *,
    body: ObjectLifecyclePreferenceData,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/settings/",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, SaveObjectLifecyclePreferenceResponseSaveObjectLifecyclePreference]]:
    if response.status_code == 200:
        response_200 = SaveObjectLifecyclePreferenceResponseSaveObjectLifecyclePreference.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, SaveObjectLifecyclePreferenceResponseSaveObjectLifecyclePreference]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: ObjectLifecyclePreferenceData,
) -> Response[Union[HTTPValidationError, SaveObjectLifecyclePreferenceResponseSaveObjectLifecyclePreference]]:
    """Submit Cdn Object Lifecycle Preference

    Args:
        body (ObjectLifecyclePreferenceData):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, SaveObjectLifecyclePreferenceResponseSaveObjectLifecyclePreference]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    body: ObjectLifecyclePreferenceData,
) -> Optional[Union[HTTPValidationError, SaveObjectLifecyclePreferenceResponseSaveObjectLifecyclePreference]]:
    """Submit Cdn Object Lifecycle Preference

    Args:
        body (ObjectLifecyclePreferenceData):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, SaveObjectLifecyclePreferenceResponseSaveObjectLifecyclePreference]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: ObjectLifecyclePreferenceData,
) -> Response[Union[HTTPValidationError, SaveObjectLifecyclePreferenceResponseSaveObjectLifecyclePreference]]:
    """Submit Cdn Object Lifecycle Preference

    Args:
        body (ObjectLifecyclePreferenceData):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, SaveObjectLifecyclePreferenceResponseSaveObjectLifecyclePreference]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: ObjectLifecyclePreferenceData,
) -> Optional[Union[HTTPValidationError, SaveObjectLifecyclePreferenceResponseSaveObjectLifecyclePreference]]:
    """Submit Cdn Object Lifecycle Preference

    Args:
        body (ObjectLifecyclePreferenceData):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, SaveObjectLifecyclePreferenceResponseSaveObjectLifecyclePreference]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
