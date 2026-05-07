from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.tag_assignment_create import TagAssignmentCreate
from ...models.tag_info import TagInfo
from ...types import Response


def _get_kwargs(
    tag_id: UUID,
    *,
    body: TagAssignmentCreate,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/tags/{tag_id}/assign",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, TagInfo]]:
    if response.status_code == 201:
        response_201 = TagInfo.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, TagInfo]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    tag_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: TagAssignmentCreate,
) -> Response[Union[HTTPValidationError, TagInfo]]:
    """Assign Tag

     Assign a tag to an entity.

    Args:
        tag_id (UUID):
        body (TagAssignmentCreate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, TagInfo]]
    """

    kwargs = _get_kwargs(
        tag_id=tag_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    tag_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: TagAssignmentCreate,
) -> Optional[Union[HTTPValidationError, TagInfo]]:
    """Assign Tag

     Assign a tag to an entity.

    Args:
        tag_id (UUID):
        body (TagAssignmentCreate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, TagInfo]
    """

    return sync_detailed(
        tag_id=tag_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    tag_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: TagAssignmentCreate,
) -> Response[Union[HTTPValidationError, TagInfo]]:
    """Assign Tag

     Assign a tag to an entity.

    Args:
        tag_id (UUID):
        body (TagAssignmentCreate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, TagInfo]]
    """

    kwargs = _get_kwargs(
        tag_id=tag_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    tag_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
    body: TagAssignmentCreate,
) -> Optional[Union[HTTPValidationError, TagInfo]]:
    """Assign Tag

     Assign a tag to an entity.

    Args:
        tag_id (UUID):
        body (TagAssignmentCreate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, TagInfo]
    """

    return (
        await asyncio_detailed(
            tag_id=tag_id,
            client=client,
            body=body,
        )
    ).parsed
