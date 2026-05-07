from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.upsert_installation_input import UpsertInstallationInput
from ...types import Response


def _get_kwargs(
    installation_id: str,
    *,
    body: UpsertInstallationInput,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": f"/vercel/v1/installations/{installation_id}",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, HTTPValidationError]]:
    if response.status_code == 201:
        response_201 = response.json()
        return response_201
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
    installation_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: UpsertInstallationInput,
) -> Response[Union[Any, HTTPValidationError]]:
    """Upsert Installation

    Args:
        installation_id (str):
        body (UpsertInstallationInput):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
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
    body: UpsertInstallationInput,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Upsert Installation

    Args:
        installation_id (str):
        body (UpsertInstallationInput):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
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
    body: UpsertInstallationInput,
) -> Response[Union[Any, HTTPValidationError]]:
    """Upsert Installation

    Args:
        installation_id (str):
        body (UpsertInstallationInput):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
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
    body: UpsertInstallationInput,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Upsert Installation

    Args:
        installation_id (str):
        body (UpsertInstallationInput):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            installation_id=installation_id,
            client=client,
            body=body,
        )
    ).parsed
