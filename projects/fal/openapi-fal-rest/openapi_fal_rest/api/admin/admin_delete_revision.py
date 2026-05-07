from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    user_id: str,
    app_name: str,
    revision_id: str,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": f"/admin/apps/{user_id}/{app_name}/revisions/{revision_id}",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, None]]:
    if response.status_code == 200:
        response_200 = cast(None, response.json())
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
) -> Response[Union[HTTPValidationError, None]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    user_id: str,
    app_name: str,
    revision_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[HTTPValidationError, None]]:
    """Admin Delete Revision

     Admin endpoint to delete a non-active revision for an application.

    Args:
        user_id (str):
        app_name (str):
        revision_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, None]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        app_name=app_name,
        revision_id=revision_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    user_id: str,
    app_name: str,
    revision_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[HTTPValidationError, None]]:
    """Admin Delete Revision

     Admin endpoint to delete a non-active revision for an application.

    Args:
        user_id (str):
        app_name (str):
        revision_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, None]
    """

    return sync_detailed(
        user_id=user_id,
        app_name=app_name,
        revision_id=revision_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    user_id: str,
    app_name: str,
    revision_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[HTTPValidationError, None]]:
    """Admin Delete Revision

     Admin endpoint to delete a non-active revision for an application.

    Args:
        user_id (str):
        app_name (str):
        revision_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, None]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        app_name=app_name,
        revision_id=revision_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: str,
    app_name: str,
    revision_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[HTTPValidationError, None]]:
    """Admin Delete Revision

     Admin endpoint to delete a non-active revision for an application.

    Args:
        user_id (str):
        app_name (str):
        revision_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, None]
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            app_name=app_name,
            revision_id=revision_id,
            client=client,
        )
    ).parsed
