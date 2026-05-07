from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.auth_override_body import AuthOverrideBody
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    user_id: str,
    app_name: str,
    target_user: str,
    *,
    body: AuthOverrideBody,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": f"/admin/apps/{user_id}/{app_name}/auth_overrides/{target_user}",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[AuthOverrideBody, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = AuthOverrideBody.from_dict(response.json())

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
) -> Response[Union[AuthOverrideBody, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    user_id: str,
    app_name: str,
    target_user: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: AuthOverrideBody,
) -> Response[Union[AuthOverrideBody, HTTPValidationError]]:
    """Upsert App Auth Override

    Args:
        user_id (str):
        app_name (str):
        target_user (str):
        body (AuthOverrideBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AuthOverrideBody, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        app_name=app_name,
        target_user=target_user,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    user_id: str,
    app_name: str,
    target_user: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: AuthOverrideBody,
) -> Optional[Union[AuthOverrideBody, HTTPValidationError]]:
    """Upsert App Auth Override

    Args:
        user_id (str):
        app_name (str):
        target_user (str):
        body (AuthOverrideBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AuthOverrideBody, HTTPValidationError]
    """

    return sync_detailed(
        user_id=user_id,
        app_name=app_name,
        target_user=target_user,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    user_id: str,
    app_name: str,
    target_user: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: AuthOverrideBody,
) -> Response[Union[AuthOverrideBody, HTTPValidationError]]:
    """Upsert App Auth Override

    Args:
        user_id (str):
        app_name (str):
        target_user (str):
        body (AuthOverrideBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AuthOverrideBody, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        app_name=app_name,
        target_user=target_user,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: str,
    app_name: str,
    target_user: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: AuthOverrideBody,
) -> Optional[Union[AuthOverrideBody, HTTPValidationError]]:
    """Upsert App Auth Override

    Args:
        user_id (str):
        app_name (str):
        target_user (str):
        body (AuthOverrideBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AuthOverrideBody, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            app_name=app_name,
            target_user=target_user,
            client=client,
            body=body,
        )
    ).parsed
