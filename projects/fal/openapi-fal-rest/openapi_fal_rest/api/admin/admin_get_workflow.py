from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.workflow_detail import WorkflowDetail
from ...types import Response


def _get_kwargs(
    user_id: str,
    workflow_name: str,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/admin/workflows/{user_id}/{workflow_name}",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, WorkflowDetail]]:
    if response.status_code == 200:
        response_200 = WorkflowDetail.from_dict(response.json())

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
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[HTTPValidationError, WorkflowDetail]]:
    """Get Workflow

     Get any workflow by user and workflow name (admin only).

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
) -> Optional[Union[HTTPValidationError, WorkflowDetail]]:
    """Get Workflow

     Get any workflow by user and workflow name (admin only).

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
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[HTTPValidationError, WorkflowDetail]]:
    """Get Workflow

     Get any workflow by user and workflow name (admin only).

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
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: str,
    workflow_name: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[HTTPValidationError, WorkflowDetail]]:
    """Get Workflow

     Get any workflow by user and workflow name (admin only).

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
