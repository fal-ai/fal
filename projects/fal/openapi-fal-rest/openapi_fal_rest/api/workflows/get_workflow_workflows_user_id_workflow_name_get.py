from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
from ...client import Client
from ...models.http_validation_error import HTTPValidationError
from ...models.workflow_detail import WorkflowDetail
from ...types import Response


def _get_kwargs(
    user_id: str,
    workflow_name: str,
    *,
    client: Client,
) -> Dict[str, Any]:
    url = "{}/workflows/{user_id}/{workflow_name}".format(client.base_url, user_id=user_id, workflow_name=workflow_name)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
    }


def _parse_response(
    *, client: Client, response: httpx.Response
) -> Optional[Union[HTTPValidationError, WorkflowDetail]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = WorkflowDetail.from_dict(response.json())

        return response_200
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Client, response: httpx.Response
) -> Response[Union[HTTPValidationError, WorkflowDetail]]:
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
    client: Client,
) -> Response[Union[HTTPValidationError, WorkflowDetail]]:
    """Get Workflow

    Args:
        user_id (str):
        workflow_name (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, WorkflowDetail]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        workflow_name=workflow_name,
        client=client,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    user_id: str,
    workflow_name: str,
    *,
    client: Client,
) -> Optional[Union[HTTPValidationError, WorkflowDetail]]:
    """Get Workflow

    Args:
        user_id (str):
        workflow_name (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, WorkflowDetail]
    """

    return sync_detailed(
        user_id=user_id,
        workflow_name=workflow_name,
        client=client,
    ).parsed


async def asyncio_detailed(
    user_id: str,
    workflow_name: str,
    *,
    client: Client,
) -> Response[Union[HTTPValidationError, WorkflowDetail]]:
    """Get Workflow

    Args:
        user_id (str):
        workflow_name (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, WorkflowDetail]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        workflow_name=workflow_name,
        client=client,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: str,
    workflow_name: str,
    *,
    client: Client,
) -> Optional[Union[HTTPValidationError, WorkflowDetail]]:
    """Get Workflow

    Args:
        user_id (str):
        workflow_name (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, WorkflowDetail]
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            workflow_name=workflow_name,
            client=client,
        )
    ).parsed
