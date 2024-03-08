from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union, cast

import httpx

from ... import errors
from ...client import Client
from ...models.execute_workflow_workflows_user_id_workflow_name_post_json_body_type_0 import (
    ExecuteWorkflowWorkflowsUserIdWorkflowNamePostJsonBodyType0,
)
from ...models.execute_workflow_workflows_user_id_workflow_name_post_response_200_type_0 import (
    ExecuteWorkflowWorkflowsUserIdWorkflowNamePostResponse200Type0,
)
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    user_id: str,
    workflow_name: str,
    *,
    client: Client,
    json_body: Union["ExecuteWorkflowWorkflowsUserIdWorkflowNamePostJsonBodyType0", List[Any], bool, float, int, str],
) -> Dict[str, Any]:
    url = "{}/workflows/{user_id}/{workflow_name}".format(client.base_url, user_id=user_id, workflow_name=workflow_name)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    json_json_body: Union[Dict[str, Any], List[Any], bool, float, int, str]

    if isinstance(json_body, ExecuteWorkflowWorkflowsUserIdWorkflowNamePostJsonBodyType0):
        json_json_body = json_body.to_dict()

    elif isinstance(json_body, list):
        json_json_body = json_body

    else:
        json_json_body = json_body

    return {
        "method": "post",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
        "json": json_json_body,
    }


def _parse_response(*, client: Client, response: httpx.Response) -> Optional[
    Union[
        HTTPValidationError,
        Union["ExecuteWorkflowWorkflowsUserIdWorkflowNamePostResponse200Type0", List[Any], bool, float, int, str],
    ]
]:
    if response.status_code == HTTPStatus.OK:

        def _parse_response_200(
            data: object,
        ) -> Union["ExecuteWorkflowWorkflowsUserIdWorkflowNamePostResponse200Type0", List[Any], bool, float, int, str]:
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_0 = ExecuteWorkflowWorkflowsUserIdWorkflowNamePostResponse200Type0.from_dict(data)

                return response_200_type_0
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, list):
                    raise TypeError()
                response_200_type_1 = cast(List[Any], data)

                return response_200_type_1
            except:  # noqa: E722
                pass
            return cast(
                Union[
                    "ExecuteWorkflowWorkflowsUserIdWorkflowNamePostResponse200Type0", List[Any], bool, float, int, str
                ],
                data,
            )

        response_200 = _parse_response_200(response.json())

        return response_200
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: Client, response: httpx.Response) -> Response[
    Union[
        HTTPValidationError,
        Union["ExecuteWorkflowWorkflowsUserIdWorkflowNamePostResponse200Type0", List[Any], bool, float, int, str],
    ]
]:
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
    json_body: Union["ExecuteWorkflowWorkflowsUserIdWorkflowNamePostJsonBodyType0", List[Any], bool, float, int, str],
) -> Response[
    Union[
        HTTPValidationError,
        Union["ExecuteWorkflowWorkflowsUserIdWorkflowNamePostResponse200Type0", List[Any], bool, float, int, str],
    ]
]:
    """Execute Workflow

    Args:
        user_id (str):
        workflow_name (str):
        json_body (Union['ExecuteWorkflowWorkflowsUserIdWorkflowNamePostJsonBodyType0', List[Any],
            bool, float, int, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, Union['ExecuteWorkflowWorkflowsUserIdWorkflowNamePostResponse200Type0', List[Any], bool, float, int, str]]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        workflow_name=workflow_name,
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
    workflow_name: str,
    *,
    client: Client,
    json_body: Union["ExecuteWorkflowWorkflowsUserIdWorkflowNamePostJsonBodyType0", List[Any], bool, float, int, str],
) -> Optional[
    Union[
        HTTPValidationError,
        Union["ExecuteWorkflowWorkflowsUserIdWorkflowNamePostResponse200Type0", List[Any], bool, float, int, str],
    ]
]:
    """Execute Workflow

    Args:
        user_id (str):
        workflow_name (str):
        json_body (Union['ExecuteWorkflowWorkflowsUserIdWorkflowNamePostJsonBodyType0', List[Any],
            bool, float, int, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, Union['ExecuteWorkflowWorkflowsUserIdWorkflowNamePostResponse200Type0', List[Any], bool, float, int, str]]
    """

    return sync_detailed(
        user_id=user_id,
        workflow_name=workflow_name,
        client=client,
        json_body=json_body,
    ).parsed


async def asyncio_detailed(
    user_id: str,
    workflow_name: str,
    *,
    client: Client,
    json_body: Union["ExecuteWorkflowWorkflowsUserIdWorkflowNamePostJsonBodyType0", List[Any], bool, float, int, str],
) -> Response[
    Union[
        HTTPValidationError,
        Union["ExecuteWorkflowWorkflowsUserIdWorkflowNamePostResponse200Type0", List[Any], bool, float, int, str],
    ]
]:
    """Execute Workflow

    Args:
        user_id (str):
        workflow_name (str):
        json_body (Union['ExecuteWorkflowWorkflowsUserIdWorkflowNamePostJsonBodyType0', List[Any],
            bool, float, int, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, Union['ExecuteWorkflowWorkflowsUserIdWorkflowNamePostResponse200Type0', List[Any], bool, float, int, str]]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        workflow_name=workflow_name,
        client=client,
        json_body=json_body,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: str,
    workflow_name: str,
    *,
    client: Client,
    json_body: Union["ExecuteWorkflowWorkflowsUserIdWorkflowNamePostJsonBodyType0", List[Any], bool, float, int, str],
) -> Optional[
    Union[
        HTTPValidationError,
        Union["ExecuteWorkflowWorkflowsUserIdWorkflowNamePostResponse200Type0", List[Any], bool, float, int, str],
    ]
]:
    """Execute Workflow

    Args:
        user_id (str):
        workflow_name (str):
        json_body (Union['ExecuteWorkflowWorkflowsUserIdWorkflowNamePostJsonBodyType0', List[Any],
            bool, float, int, str]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, Union['ExecuteWorkflowWorkflowsUserIdWorkflowNamePostResponse200Type0', List[Any], bool, float, int, str]]
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            workflow_name=workflow_name,
            client=client,
            json_body=json_body,
        )
    ).parsed
