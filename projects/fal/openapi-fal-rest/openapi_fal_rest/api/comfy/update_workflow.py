from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
from ...client import Client
from ...models.comfy_workflow_detail import ComfyWorkflowDetail
from ...models.http_validation_error import HTTPValidationError
from ...models.typed_comfy_workflow_update import TypedComfyWorkflowUpdate
from ...types import Response


def _get_kwargs(
    user_id: str,
    name: str,
    *,
    client: Client,
    json_body: TypedComfyWorkflowUpdate,
) -> Dict[str, Any]:
    url = "{}/comfy/{user_id}/{name}".format(client.base_url, user_id=user_id, name=name)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    json_json_body = json_body.to_dict()

    return {
        "method": "patch",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
        "json": json_json_body,
    }


def _parse_response(
    *, client: Client, response: httpx.Response
) -> Optional[Union[ComfyWorkflowDetail, HTTPValidationError]]:
    if response.status_code == HTTPStatus.CREATED:
        response_201 = ComfyWorkflowDetail.from_dict(response.json())

        return response_201
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Client, response: httpx.Response
) -> Response[Union[ComfyWorkflowDetail, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    user_id: str,
    name: str,
    *,
    client: Client,
    json_body: TypedComfyWorkflowUpdate,
) -> Response[Union[ComfyWorkflowDetail, HTTPValidationError]]:
    """Update Workflow

    Args:
        user_id (str):
        name (str):
        json_body (TypedComfyWorkflowUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[ComfyWorkflowDetail, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        name=name,
        client=client,
        json_body=json_body,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    user_id: str,
    name: str,
    *,
    client: Client,
    json_body: TypedComfyWorkflowUpdate,
) -> Optional[Union[ComfyWorkflowDetail, HTTPValidationError]]:
    """Update Workflow

    Args:
        user_id (str):
        name (str):
        json_body (TypedComfyWorkflowUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[ComfyWorkflowDetail, HTTPValidationError]
    """

    return sync_detailed(
        user_id=user_id,
        name=name,
        client=client,
        json_body=json_body,
    ).parsed


async def asyncio_detailed(
    user_id: str,
    name: str,
    *,
    client: Client,
    json_body: TypedComfyWorkflowUpdate,
) -> Response[Union[ComfyWorkflowDetail, HTTPValidationError]]:
    """Update Workflow

    Args:
        user_id (str):
        name (str):
        json_body (TypedComfyWorkflowUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[ComfyWorkflowDetail, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        name=name,
        client=client,
        json_body=json_body,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: str,
    name: str,
    *,
    client: Client,
    json_body: TypedComfyWorkflowUpdate,
) -> Optional[Union[ComfyWorkflowDetail, HTTPValidationError]]:
    """Update Workflow

    Args:
        user_id (str):
        name (str):
        json_body (TypedComfyWorkflowUpdate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[ComfyWorkflowDetail, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            name=name,
            client=client,
            json_body=json_body,
        )
    ).parsed
