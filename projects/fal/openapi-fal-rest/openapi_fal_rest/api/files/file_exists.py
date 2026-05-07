from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.file_spec import FileSpec
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    file: str,
    *,
    calculate_checksum: Union[Unset, bool] = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["calculate_checksum"] = calculate_checksum

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/files/file/exists/{file}",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[FileSpec, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = FileSpec.from_dict(response.json())

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
) -> Response[Union[FileSpec, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    file: str,
    *,
    client: Union[AuthenticatedClient, Client],
    calculate_checksum: Union[Unset, bool] = False,
) -> Response[Union[FileSpec, HTTPValidationError]]:
    """File Exists

    Args:
        file (str):
        calculate_checksum (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[FileSpec, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        file=file,
        calculate_checksum=calculate_checksum,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    file: str,
    *,
    client: Union[AuthenticatedClient, Client],
    calculate_checksum: Union[Unset, bool] = False,
) -> Optional[Union[FileSpec, HTTPValidationError]]:
    """File Exists

    Args:
        file (str):
        calculate_checksum (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[FileSpec, HTTPValidationError]
    """

    return sync_detailed(
        file=file,
        client=client,
        calculate_checksum=calculate_checksum,
    ).parsed


async def asyncio_detailed(
    file: str,
    *,
    client: Union[AuthenticatedClient, Client],
    calculate_checksum: Union[Unset, bool] = False,
) -> Response[Union[FileSpec, HTTPValidationError]]:
    """File Exists

    Args:
        file (str):
        calculate_checksum (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[FileSpec, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        file=file,
        calculate_checksum=calculate_checksum,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    file: str,
    *,
    client: Union[AuthenticatedClient, Client],
    calculate_checksum: Union[Unset, bool] = False,
) -> Optional[Union[FileSpec, HTTPValidationError]]:
    """File Exists

    Args:
        file (str):
        calculate_checksum (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[FileSpec, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            file=file,
            client=client,
            calculate_checksum=calculate_checksum,
        )
    ).parsed
