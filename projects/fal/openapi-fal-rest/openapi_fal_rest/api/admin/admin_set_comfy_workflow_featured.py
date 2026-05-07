from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.comfy_workflow_detail import ComfyWorkflowDetail
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response


def _get_kwargs(
    user_id: str,
    workflow_name: str,
    *,
    body: list[str],
    is_featured: bool,
    thumbnail_url: str,
    description: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    params: dict[str, Any] = {}

    params["is_featured"] = is_featured

    params["thumbnail_url"] = thumbnail_url

    params["description"] = description

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/admin/comfy/{user_id}/{workflow_name}/featured",
        "params": params,
    }

    _body = body

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[ComfyWorkflowDetail, HTTPValidationError]]:
    if response.status_code == 201:
        response_201 = ComfyWorkflowDetail.from_dict(response.json())

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
) -> Response[Union[ComfyWorkflowDetail, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    user_id: str,
    workflow_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: list[str],
    is_featured: bool,
    thumbnail_url: str,
    description: str,
) -> Response[Union[ComfyWorkflowDetail, HTTPValidationError]]:
    """Set Comfy Workflow Featured

    Args:
        user_id (str):
        workflow_name (str):
        is_featured (bool):
        thumbnail_url (str):
        description (str):
        body (list[str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[ComfyWorkflowDetail, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        workflow_name=workflow_name,
        body=body,
        is_featured=is_featured,
        thumbnail_url=thumbnail_url,
        description=description,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    user_id: str,
    workflow_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: list[str],
    is_featured: bool,
    thumbnail_url: str,
    description: str,
) -> Optional[Union[ComfyWorkflowDetail, HTTPValidationError]]:
    """Set Comfy Workflow Featured

    Args:
        user_id (str):
        workflow_name (str):
        is_featured (bool):
        thumbnail_url (str):
        description (str):
        body (list[str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[ComfyWorkflowDetail, HTTPValidationError]
    """

    return sync_detailed(
        user_id=user_id,
        workflow_name=workflow_name,
        client=client,
        body=body,
        is_featured=is_featured,
        thumbnail_url=thumbnail_url,
        description=description,
    ).parsed


async def asyncio_detailed(
    user_id: str,
    workflow_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: list[str],
    is_featured: bool,
    thumbnail_url: str,
    description: str,
) -> Response[Union[ComfyWorkflowDetail, HTTPValidationError]]:
    """Set Comfy Workflow Featured

    Args:
        user_id (str):
        workflow_name (str):
        is_featured (bool):
        thumbnail_url (str):
        description (str):
        body (list[str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[ComfyWorkflowDetail, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        workflow_name=workflow_name,
        body=body,
        is_featured=is_featured,
        thumbnail_url=thumbnail_url,
        description=description,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: str,
    workflow_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: list[str],
    is_featured: bool,
    thumbnail_url: str,
    description: str,
) -> Optional[Union[ComfyWorkflowDetail, HTTPValidationError]]:
    """Set Comfy Workflow Featured

    Args:
        user_id (str):
        workflow_name (str):
        is_featured (bool):
        thumbnail_url (str):
        description (str):
        body (list[str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[ComfyWorkflowDetail, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            workflow_name=workflow_name,
            client=client,
            body=body,
            is_featured=is_featured,
            thumbnail_url=thumbnail_url,
            description=description,
        )
    ).parsed
