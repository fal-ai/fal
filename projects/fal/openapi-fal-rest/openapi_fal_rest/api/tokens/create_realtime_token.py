from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.body_create_realtime_token import BodyCreateRealtimeToken
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: BodyCreateRealtimeToken,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/tokens/realtime",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, str]]:
    if response.status_code == 201:
        response_201 = cast(str, response.json())
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
) -> Response[Union[HTTPValidationError, str]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyCreateRealtimeToken,
) -> Response[Union[HTTPValidationError, str]]:
    r"""Create Realtime Token

     Create a token for an app or endpoint that has realtime capabilities.

    When `app` has more than 2 segments (e.g., \"owner/app/endpoint-path\"), the
    specific endpoint is validated as realtime and token_expiration allows up to
    120s. Otherwise, the entire app is checked for realtime endpoints and
    token_expiration is capped at 10s.

    Unlike the general token creation endpoint, this does not use the JWT_ALLOWED_APPS
    allowlist. Instead, it validates that the app/endpoint has realtime capabilities
    defined in its OpenAPI spec.

    Args:
        body (BodyCreateRealtimeToken):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, str]]
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
    body: BodyCreateRealtimeToken,
) -> Optional[Union[HTTPValidationError, str]]:
    r"""Create Realtime Token

     Create a token for an app or endpoint that has realtime capabilities.

    When `app` has more than 2 segments (e.g., \"owner/app/endpoint-path\"), the
    specific endpoint is validated as realtime and token_expiration allows up to
    120s. Otherwise, the entire app is checked for realtime endpoints and
    token_expiration is capped at 10s.

    Unlike the general token creation endpoint, this does not use the JWT_ALLOWED_APPS
    allowlist. Instead, it validates that the app/endpoint has realtime capabilities
    defined in its OpenAPI spec.

    Args:
        body (BodyCreateRealtimeToken):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, str]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyCreateRealtimeToken,
) -> Response[Union[HTTPValidationError, str]]:
    r"""Create Realtime Token

     Create a token for an app or endpoint that has realtime capabilities.

    When `app` has more than 2 segments (e.g., \"owner/app/endpoint-path\"), the
    specific endpoint is validated as realtime and token_expiration allows up to
    120s. Otherwise, the entire app is checked for realtime endpoints and
    token_expiration is capped at 10s.

    Unlike the general token creation endpoint, this does not use the JWT_ALLOWED_APPS
    allowlist. Instead, it validates that the app/endpoint has realtime capabilities
    defined in its OpenAPI spec.

    Args:
        body (BodyCreateRealtimeToken):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, str]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyCreateRealtimeToken,
) -> Optional[Union[HTTPValidationError, str]]:
    r"""Create Realtime Token

     Create a token for an app or endpoint that has realtime capabilities.

    When `app` has more than 2 segments (e.g., \"owner/app/endpoint-path\"), the
    specific endpoint is validated as realtime and token_expiration allows up to
    120s. Otherwise, the entire app is checked for realtime endpoints and
    token_expiration is capped at 10s.

    Unlike the general token creation endpoint, this does not use the JWT_ALLOWED_APPS
    allowlist. Instead, it validates that the app/endpoint has realtime capabilities
    defined in its OpenAPI spec.

    Args:
        body (BodyCreateRealtimeToken):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, str]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
